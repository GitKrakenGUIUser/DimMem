#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_FILE = Path(__file__).resolve()
LONGMEMEVAL_DIR = THIS_FILE.parents[1]
SUBMIT_ROOT = THIS_FILE.parents[2]

if str(SUBMIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMIT_ROOT))
if str(LONGMEMEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LONGMEMEVAL_DIR))

from models import DimensionMemory
from utils.local_embedding_client import LocalEmbeddingClient

from search import (
    search_bm25,
    search_embedding,
    search_structured,
    search_top15_content_dedup,
)
from search.rerank_p2 import rerank_records
from query_parser.p2_schema import normalize_parsed_query_p2
from search.p2_runtime import (
    attach_assistant_context_robust,
    assistant_context_stats,
    build_full_mapped_query,
)
from search.assistant_reply_search import (
    search_assistant_replies,
    should_search_assistant_replies,
)
from search.relative_event_binding import (
    apply_relative_event_binding,
    relative_event_stats,
)


DEFAULT_QUERY_PARSED = SUBMIT_ROOT / "results/query_analysis/parsed.json"
DEFAULT_MEMORY_DIR = SUBMIT_ROOT / "results/memories"
DEFAULT_OUTPUT_ROOT = SUBMIT_ROOT / "results/retrieval"
DEFAULT_EMBEDDING_MODEL = "/data/aios-weights/embeddings/all-MiniLM-L6-v2"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _string_list(values: Any) -> List[str]:
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    result: List[str] = []
    seen = set()

    for value in values:
        text = _clean(value)
        if not text:
            continue

        marker = text.lower()
        if marker in seen:
            continue

        seen.add(marker)
        result.append(text)

    return result


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_source_time(value: Any) -> Optional[datetime]:
    text = _clean(value)
    if not text:
        return None

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _schema_terms(dimension_model: DimensionMemory) -> List[str]:
    """
    P3 extraction schema terms.

    These are added to entities and embedding_text so retrieval can use:
    event_time / subject / action / object / value / relation / evidence_span.
    """
    return _string_list(
        [
            getattr(dimension_model, "event_time", ""),
            getattr(dimension_model, "valid_from", ""),
            getattr(dimension_model, "valid_to", ""),
            getattr(dimension_model, "status", ""),
            getattr(dimension_model, "subject", ""),
            getattr(dimension_model, "action", ""),
            getattr(dimension_model, "object", ""),
            getattr(dimension_model, "value", ""),
            getattr(dimension_model, "quantity", ""),
            getattr(dimension_model, "unit", ""),
            getattr(dimension_model, "relation", ""),
            getattr(dimension_model, "evidence_span", ""),
        ]
    )


def _embedding_text(content: str, dimension_model: DimensionMemory) -> str:
    """
    Build retrieval text with backward compatibility.

    New models/memory.py provides searchable_text().
    If the local model has not been updated for any reason, fall back to manual concat.
    """
    try:
        return dimension_model.searchable_text(include_content=content)
    except AttributeError:
        parts = [
            content,
            getattr(dimension_model, "time", ""),
            getattr(dimension_model, "location", ""),
            getattr(dimension_model, "reason", ""),
            getattr(dimension_model, "purpose", ""),
            " ".join(getattr(dimension_model, "keywords", []) or []),
            *_schema_terms(dimension_model),
        ]
        return " | ".join(part for part in parts if _clean(part))


def load_records(memory_dir: Path) -> List[Dict[str, Any]]:
    """
    Load normalized DimMem memories and convert them into retrieval records.

    P3 change:
    - preserves new dimension fields through DimensionMemory.to_dict();
    - adds event/value/relation/action fields into entities;
    - adds those fields into embedding_text.
    """
    records: List[Dict[str, Any]] = []

    memory_sources: List[tuple[str, List[Dict[str, Any]]]] = []

    all_memories_path = memory_dir / "all_memories.json"
    if all_memories_path.exists():
        payload = _load_json(all_memories_path)
        rows = payload.get("memories") if isinstance(payload, dict) else []
        memory_sources.append(("all_memories", rows or []))

    for path in sorted(memory_dir.glob("window_*/normalized_memories.json")):
        payload = _load_json(path)
        rows = payload.get("memories") if isinstance(payload, dict) else []
        memory_sources.append((path.parent.name, rows or []))

    for source_name, memories in memory_sources:
        for idx, row in enumerate(memories or []):
            if not isinstance(row, dict):
                continue

            window_name = (
                _clean(row.get("window_dir"))
                or _clean(row.get("window_index"))
                or source_name
            )

            dimension_model = DimensionMemory.from_dict(row.get("dimension"))
            keywords = _string_list(getattr(dimension_model, "keywords", []) or [])
            schema_terms = _schema_terms(dimension_model)

            source_time = _parse_source_time(row.get("source_time"))
            content = _clean(row.get("content"))

            record = {
                "user_id": memory_dir.name,
                "memory_type": dimension_model.memory_type or "other",
                "content": content,
                "dimension": dimension_model.to_dict(include_memory_type=False),
                "entities": _string_list(list(keywords) + schema_terms),
                "embedding_text": _embedding_text(content, dimension_model),
                "source_message_ids": [window_name],
                "source_boundary_id": f"{window_name}_{idx:04d}",
                "source_time": source_time,
                "record_time": datetime.now().isoformat(),
            }

            records.append(record)

    return records


def _build_embedding_client(
    *,
    parse_mode: str,
    embedding_model: str,
    embedding_device: str,
) -> Optional[LocalEmbeddingClient]:
    if parse_mode in {"structured", "raw", ""}:
        return None

    return LocalEmbeddingClient(
        model=embedding_model,
        device=embedding_device,
        batch_size=32,
    )


def _run_search(
    *,
    parsed_query: Dict[str, Any],
    records: List[Dict[str, Any]],
    parse_mode: str,
    embedding_client: Optional[LocalEmbeddingClient],
    top_k: int,
) -> Dict[str, Any]:
    if parse_mode in {"rrf_hybrid", "rrf", "hybrid", "fused"}:
        if embedding_client is None:
            raise ValueError("fused/hybrid retrieval requires an embedding client")
        return search_top15_content_dedup(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        )

    if parse_mode == "hybrid_legacy":
        if embedding_client is None:
            raise ValueError("hybrid_legacy retrieval requires an embedding client")
        return search_embedding(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        )

    if parse_mode == "raw":
        return search_bm25(
            parsed_query=parsed_query,
            records=records,
            top_k=top_k,
        )

    return search_structured(
        parsed_query=parsed_query,
        records=records,
        embedding_client=embedding_client,
        top_k=top_k,
    )


def run_retrieval(
    *,
    query_parsed: Path,
    memory_dir: Path,
    output_root: Path,
    top_k: int,
    final_top_k: int,
    embedding_model: str,
    embedding_device: str,
    enable_rerank: bool,
    enable_assistant_context: bool,
) -> Path:
    parsed_query = normalize_parsed_query_p2(_load_json(query_parsed))
    records = load_records(memory_dir)

    question_type = query_parsed.parent.parent.name
    sample_id = query_parsed.parent.name

    parse_mode = _clean(parsed_query.get("parse_mode")).lower() or "structured"
    embedding_client = _build_embedding_client(
        parse_mode=parse_mode,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
    )

    search_result = _run_search(
        parsed_query=parsed_query,
        records=records,
        parse_mode=parse_mode,
        embedding_client=embedding_client,
        top_k=top_k,
    )

    search_mode = search_result["search_mode"]
    mapped_query = build_full_mapped_query(parsed_query, search_result)

    ranked = list(search_result.get("all_ranked_records") or [])
    top_records = list(search_result.get("top_records") or [])

    # P22: attach assistant reply to already retrieved memory records.
    ranked = attach_assistant_context_robust(
        parsed_query=parsed_query,
        memory_dir=memory_dir,
        records=ranked,
        force=enable_assistant_context,
    )
    top_records = attach_assistant_context_robust(
        parsed_query=parsed_query,
        memory_dir=memory_dir,
        records=top_records,
        force=enable_assistant_context,
    )

    # P22: direct assistant reply search route.
    assistant_reply_hits = search_assistant_replies(
        parsed_query=parsed_query,
        memory_dir=memory_dir,
        top_k=top_k,
    )
    if assistant_reply_hits:
        ranked = assistant_reply_hits + ranked
        top_records = assistant_reply_hits + top_records

    # P22 + P3: relative-event binding.
    # P3 relative_event_binding.py should prefer dimension.event_time over source_time.
    ranked = apply_relative_event_binding(
        parsed_query=parsed_query,
        records=ranked,
        final_top_k=max(top_k * 3, final_top_k * 2),
    )

    if enable_rerank:
        top_records = rerank_records(
            parsed_query=parsed_query,
            records=ranked,
            final_top_k=final_top_k,
        )
        search_mode = f"{search_mode}_p2_rerank"
    else:
        top_records = top_records[:final_top_k]

    run_dir = output_root / question_type / search_mode / sample_id
    run_dir.mkdir(parents=True, exist_ok=True)

    experiment = {
        "query_parsed": str(query_parsed),
        "memory_dir": str(memory_dir),
        "output_dir": str(run_dir),
        "question_type": question_type,
        "sample_id": sample_id,
        "top_k": top_k,
        "final_top_k": final_top_k,
        "enable_rerank": enable_rerank,
        "enable_assistant_context": enable_assistant_context,
        "embedding_model": embedding_model,
        "embedding_device": embedding_device,
        "record_count": len(records),
        "retrieval_method": search_mode,
        "started_at": datetime.now().isoformat(),
    }

    (run_dir / "experiment_config.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "mapped_query_analysis.json").write_text(
        json.dumps(mapped_query, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "all_ranked_records.json").write_text(
        json.dumps(ranked, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (run_dir / "top_records.json").write_text(
        json.dumps(top_records, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    summary = {
        "parse_mode": parse_mode,
        "retrieval_method": search_mode,
        "question_type": question_type,
        "sample_id": sample_id,
        "query_text": mapped_query.get("query_text")
        or mapped_query.get("canonical_text")
        or "",
        "answer_field": mapped_query.get("answer_field", ""),
        "target_memory_types": list(mapped_query.get("target_memory_types") or []),
        "keywords": list(mapped_query.get("keywords") or []),
        "record_count": len(records),
        "top_k": top_k,
        "final_top_k": final_top_k,
        "enable_rerank": enable_rerank,
        "enable_assistant_context": enable_assistant_context,
        "assistant_reply_search_hit_count": len(assistant_reply_hits),
        "assistant_reply_search_enabled": should_search_assistant_replies(parsed_query),
        "assistant_context_stats_final": assistant_context_stats(top_records),
        "relative_event_stats_final": relative_event_stats(top_records),
        "output_dir": str(run_dir),
        "top_records": [
            {
                "rank": i + 1,
                "score": row.get("score"),
                "retrieval_score": row.get("retrieval_score"),
                "rerank_score": row.get("rerank_score"),
                "rerank_score_p1": row.get("rerank_score_p1"),
                "rerank_components_p2": row.get("rerank_components_p2"),
                "rerank_components": row.get("rerank_components"),
                "retrieval_method": row.get("retrieval_method"),
                "fusion_sources": row.get("fusion_sources"),
                "memory_type": row.get("memory_type"),
                "content": row.get("content"),
                "source_boundary_id": row.get("source_boundary_id"),
                "assistant_uid": row.get("assistant_uid"),
                "has_assistant_reply": bool(_clean(row.get("assistant_reply"))),
                "assistant_debug": row.get("_assistant_context_debug"),
                "assistant_reply_search": row.get("_assistant_reply_search"),
                "relative_event_binding": row.get("_relative_event_binding"),
                "score_components": row.get("score_components"),
                "dimension": row.get("dimension"),
            }
            for i, row in enumerate(top_records)
        ],
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--query-parsed", type=Path, default=DEFAULT_QUERY_PARSED)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--top-k", type=int, default=20, help="Per-route retrieval top-k.")
    parser.add_argument("--final-top-k", type=int, default=15, help="Final records passed to QA.")
    parser.add_argument("--enable-rerank", action="store_true", help="Enable P2/P3 global rerank.")
    parser.add_argument(
        "--enable-assistant-context",
        action="store_true",
        help="Force assistant context attachment when assistant reply files are available.",
    )

    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="cuda")

    args = parser.parse_args()

    run_dir = run_retrieval(
        query_parsed=args.query_parsed,
        memory_dir=args.memory_dir,
        output_root=args.output_root,
        top_k=args.top_k,
        final_top_k=args.final_top_k,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        enable_rerank=args.enable_rerank,
        enable_assistant_context=args.enable_assistant_context,
    )

    print((run_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

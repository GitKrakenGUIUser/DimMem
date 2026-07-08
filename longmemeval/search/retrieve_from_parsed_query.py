#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


THIS_FILE = Path(__file__).resolve()
LONGMEMEVAL_DIR = THIS_FILE.parents[1]
SUBMIT_ROOT = THIS_FILE.parents[2]
if str(SUBMIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMIT_ROOT))
if str(LONGMEMEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LONGMEMEVAL_DIR))

from models import DimensionMemory
from utils.local_embedding_client import LocalEmbeddingClient

#from search import search_bm25, search_embedding, search_fused, search_structured, search_top15_content_dedup
from search import (
    attach_assistant_context,
    build_boundary_to_window_source,
    load_window_assistant_replies,
    search_bm25,
    search_embedding,
    search_fused,
    search_structured,
    search_top15_content_dedup,
)
from search.rerank_p2 import rerank_records
from query_parser.p2_schema import normalize_parsed_query_p2
from search.p2_runtime import (
    attach_assistant_context_robust,
    assistant_context_stats,
    build_full_mapped_query,
    search_mode_name_p2,
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


def load_parsed_query(parsed_path: Path) -> Dict[str, Any]:
    return json.loads(parsed_path.read_text(encoding="utf-8"))


def load_records(memory_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    all_memories_path = memory_dir / "all_memories.json"
    if all_memories_path.exists():
        payload = json.loads(all_memories_path.read_text(encoding="utf-8"))
        memory_sources = [("all_memories", payload.get("memories") if isinstance(payload, dict) else [])]
    else:
        memory_sources = []
        for path in sorted(memory_dir.glob("window_*/normalized_memories.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            memory_sources.append((path.parent.name, payload.get("memories") if isinstance(payload, dict) else []))

    for source_name, memories in memory_sources:
        for idx, row in enumerate(memories or []):
            if not isinstance(row, dict):
                continue
            window_name = _clean(row.get("window_dir")) or _clean(row.get("window_index")) or source_name
            dimension_model = DimensionMemory.from_dict(row.get("dimension"))
            keywords = dimension_model.keywords
            source_time_str = _clean(row.get("source_time"))
            source_time = None
            if source_time_str:
                try:
                    source_time = datetime.fromisoformat(source_time_str)
                except ValueError:
                    pass
            content = _clean(row.get("content"))

                            # P3 extraction schema support:
                            # Put event/value/relation/action fields into both retrieval entities
                            # and embedding_text so BM25/structured/dense search can use them.
                            schema_terms = [
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

                            try:
                                embedding_text = dimension_model.searchable_text(include_content=content)
                            except AttributeError:
                                # Backward-compatible fallback if models/memory.py has not been updated yet.
                                embedding_text = " | ".join(
                                    part
                                    for part in [
                                        content,
                                        getattr(dimension_model, "time", ""),
                                        getattr(dimension_model, "location", ""),
                                        getattr(dimension_model, "reason", ""),
                                        getattr(dimension_model, "purpose", ""),
                                        " ".join(getattr(dimension_model, "keywords", []) or []),
                                        *schema_terms,
                                    ]
                                    if _clean(part)
                                )

                            record = {
                                "user_id": memory_dir.name,
                                "memory_type": dimension_model.memory_type or "other",
                                "content": content,
                                "dimension": dimension_model.to_dict(include_memory_type=False),
                                "entities": _string_list(list(keywords) + schema_terms),
                                "embedding_text": embedding_text,
                                "source_message_ids": [window_name],
                                "source_boundary_id": f"{window_name}_{idx:04d}",
                                "source_time": source_time,
                                "record_time": datetime.now().isoformat(),
                            }
            records.append(record)
    return records

def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).lower() in {"1", "true", "yes", "y", "on"}


def _resolve_assistant_windows_dir(memory_dir: Path, source_record_dir: str | None) -> Path | None:
    """
    Resolve the directory containing window_*_assistant_replies.json.

    In different runs, source_record_dir may be:
    - an absolute path;
    - a path relative to memory_dir;
    - a path relative to memory_dir.parent;
    - a path relative to project root.
    """
    candidates: List[Path] = []

    if source_record_dir:
        raw = Path(source_record_dir)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend(
                [
                    memory_dir / raw,
                    memory_dir.parent / raw,
                    SUBMIT_ROOT / raw,
                ]
            )

    candidates.extend(
        [
            memory_dir,
            memory_dir.parent,
            SUBMIT_ROOT / "results/segments",
            SUBMIT_ROOT / "results",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("window_*_assistant_replies.json")):
            return candidate

    return None


def _maybe_attach_assistant_context(
    *,
    parsed_query: Dict[str, Any],
    memory_dir: Path,
    records: List[Dict[str, Any]],
    force: bool,
) -> List[Dict[str, Any]]:
    """
    Attach assistant_reply only when:
    - force=True, or
    - query parser says need_assistant_context=True.

    If the assistant reply files are missing, it safely returns the original records.
    """
    need_assistant = force or _truthy(parsed_query.get("need_assistant_context"))

    if not need_assistant:
        return records

    try:
        boundary_index, source_record_dir = build_boundary_to_window_source(memory_dir)
        windows_dir = _resolve_assistant_windows_dir(memory_dir, source_record_dir)

        if not boundary_index or windows_dir is None:
            return records

        uid_map = load_window_assistant_replies(windows_dir)

        if not uid_map:
            return records

        return attach_assistant_context(records, boundary_index, uid_map)

    except Exception as exc:
        print(f"[WARN] failed to attach assistant context: {exc}", file=sys.stderr)
        return records



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
    #parsed_query = load_parsed_query(query_parsed)
    parsed_query = normalize_parsed_query_p2(load_parsed_query(query_parsed))
    records = load_records(memory_dir)

    question_type = query_parsed.parent.parent.name

    parse_mode = _clean(parsed_query.get("parse_mode")).lower() or "structured"
    if parse_mode in {"structured", ""}:
        embedding_client = None
    else:
        embedding_client = LocalEmbeddingClient(
            model=embedding_model,
            device=embedding_device,
            batch_size=32,
        )
    if parse_mode in {"rrf_hybrid", "rrf", "hybrid", "fused"}:
        search_result = search_top15_content_dedup(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        )
    elif parse_mode == "hybrid_legacy":
        search_result = search_embedding(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        )
    elif parse_mode == "raw":
        search_result = search_bm25(
            parsed_query=parsed_query,
            records=records,
            top_k=top_k,
        )
    else:
        search_result = search_structured(
            parsed_query=parsed_query,
            records=records,
            embedding_client=embedding_client,
            top_k=top_k,
        )
    search_mode = search_result["search_mode"]
    mapped_query = build_full_mapped_query(parsed_query, search_result)
    ranked = search_result["all_ranked_records"]
    top_records = search_result["top_records"]

    # P2 patch: attach assistant reply before rerank/QA.
    # This is safe: when files are missing or query does not need assistant context,
    # records are returned unchanged.
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

    # P2.2: direct assistant reply search route.
    # This retrieves assistant replies as independent evidence, instead of only
    # attaching assistant text to already-retrieved memory records.
    assistant_reply_hits = search_assistant_replies(
        parsed_query=parsed_query,
        memory_dir=memory_dir,
        top_k=top_k,
    )
    if assistant_reply_hits:
        ranked = assistant_reply_hits + ranked
        top_records = assistant_reply_hits + top_records

    # P2.2: relative-event binding for questions such as
    # "before/after getting X". It adds a derived binding hint and marks
    # candidate records before rerank.
    ranked = apply_relative_event_binding(
        parsed_query=parsed_query,
        records=ranked,
        final_top_k=max(top_k * 3, final_top_k * 2),
    )

    # P2 patch: global rerank after tri-route retrieval.
    # If disabled, keep original order but still cut to final_top_k.
    if enable_rerank:
        top_records = rerank_records(
            parsed_query=parsed_query,
            records=ranked,
            final_top_k=final_top_k,
        )
        search_mode = f"{search_mode}_p2_rerank"
    else:
        top_records = top_records[:final_top_k]

    run_dir = output_root / question_type / search_mode / query_parsed.parent.name
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment = {
        "query_parsed": str(query_parsed),
        "memory_dir": str(memory_dir),
        "output_dir": str(run_dir),
        "question_type": question_type,
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
        "query_text": mapped_query.get("query_text") or mapped_query.get("canonical_text") or "",
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
    parser.add_argument("--enable-rerank", action="store_true", help="Enable P2 global rerank.")
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

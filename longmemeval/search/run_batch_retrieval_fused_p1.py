#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

THIS_FILE = Path(__file__).resolve()
LONGMEMEVAL_DIR = THIS_FILE.parents[1]
SUBMIT_ROOT = THIS_FILE.parents[2]

if str(SUBMIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMIT_ROOT))
if str(LONGMEMEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LONGMEMEVAL_DIR))

from utils.local_embedding_client import LocalEmbeddingClient
from retrieve_from_parsed_query import load_records
from search import (
    attach_assistant_context,
    build_boundary_to_window_source,
    load_window_assistant_replies,
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
from search.assistant_pair_search import (
    search_assistant_pairs,
    should_search_assistant_pairs,
)
from search.structured_task_reader import (
    apply_structured_task_reader,
    structured_task_reader_stats,
)
from search.relative_event_binding import (
    apply_relative_event_binding,
    relative_event_stats,
)


DEFAULT_QUERY_ROOT = SUBMIT_ROOT / "results/query_analysis/run_baseline"
DEFAULT_MEMORY_ROOT = SUBMIT_ROOT / "results/memories/run_baseline"
DEFAULT_OUTPUT_ROOT = SUBMIT_ROOT / "results/retrieval/run_baseline_fused_p2"
DEFAULT_EMBEDDING_MODEL = "/data/aios-weights/embeddings/all-MiniLM-L6-v2"


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}] {msg}", flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).lower() in {"1", "true", "yes", "y", "on"}


def _search_mode_name(route_top_k: int, final_top_k: int, enable_rerank: bool) -> str:
    return search_mode_name_p2(route_top_k, final_top_k, enable_rerank)


def _iter_cases(query_root: Path) -> Iterable[Tuple[str, str, Path]]:
    """
    Expected query analysis layout:
      query_root/<question_type>/<sample_id>/parsed.json
    """
    for question_type_dir in sorted(query_root.iterdir()):
        if not question_type_dir.is_dir():
            continue

        question_type = question_type_dir.name

        for sample_dir in sorted(question_type_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            parsed_path = sample_dir / "parsed.json"
            if parsed_path.exists():
                yield question_type, sample_dir.name, parsed_path


def _memory_dir_for_case(memory_root: Path, question_type: str, sample_id: str) -> Path:
    return memory_root / question_type / sample_id


def _filter_cases(
    cases: List[Tuple[str, str, Path]],
    *,
    max_cases: int,
) -> List[Tuple[str, str, Path]]:
    if max_cases > 0:
        return cases[:max_cases]
    return cases


def _resolve_assistant_windows_dir(memory_dir: Path, source_record_dir: str | None) -> Path | None:
    """
    Resolve the directory containing window_*_assistant_replies.json.

    Different runs may store the source path as absolute, relative to memory_dir,
    relative to memory_dir.parent, or relative to project root.
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

    Safe fallback: if assistant files or source pointers are missing, return the
    original records unchanged.
    """
    need_assistant = force or _truthy(parsed_query.get("need_assistant_context"))

    if not need_assistant or not records:
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


def _brief_top_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(records):
        rows.append(
            {
                "rank": i + 1,
                "score": row.get("score"),
                "retrieval_score": row.get("retrieval_score"),
                "rerank_score": row.get("rerank_score"),
                "rerank_score_p1": row.get("rerank_score_p1"),
                "rerank_components_p2": row.get("rerank_components_p2"),
                "rerank_components": row.get("rerank_components"),
                "retrieval_method": row.get("retrieval_method"),
                "retrieval_rank": row.get("retrieval_rank"),
                "memory_type": row.get("memory_type"),
                "content": row.get("content"),
                "source_time": row.get("source_time"),
                "source_boundary_id": row.get("source_boundary_id"),
                "assistant_uid": row.get("assistant_uid"),
                "has_assistant_reply": bool(_clean(row.get("assistant_reply"))),
                "assistant_debug": row.get("_assistant_context_debug"),
                "assistant_reply_search": row.get("_assistant_reply_search"),
                "assistant_pair_search": row.get("_assistant_pair_search"),
                "structured_task_reader": row.get("_structured_task_reader"),
                "relative_event_binding": row.get("_relative_event_binding"),
                "fusion_sources": row.get("fusion_sources"),
                "score_components": row.get("score_components"),
            }
        )
    return rows


def run_one_case(
    *,
    question_type: str,
    sample_id: str,
    parsed_path: Path,
    memory_dir: Path,
    output_root: Path,
    embedding_client: LocalEmbeddingClient,
    route_top_k: int,
    final_top_k: int,
    force_fused_parse_mode: bool,
    enable_rerank: bool,
    enable_assistant_context: bool,
) -> Dict[str, Any]:
    started = time.time()

    parsed_query = normalize_parsed_query_p2(_load_json(parsed_path))

    # Do not re-run query_parser. This only changes the in-memory parse mode for retrieval.
    if force_fused_parse_mode:
        parsed_query["parse_mode"] = "fused"

    records = load_records(memory_dir)

    search_result = search_top15_content_dedup(
        parsed_query=parsed_query,
        records=records,
        embedding_client=embedding_client,
        top_k=route_top_k,
    )

    search_mode = _search_mode_name(route_top_k, final_top_k, enable_rerank)
    run_dir = output_root / question_type / search_mode / sample_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_ranked = list(search_result.get("all_ranked_records") or [])
    fused_top_records = list(search_result.get("top_records") or [])

    all_ranked = attach_assistant_context_robust(
        parsed_query=parsed_query,
        memory_dir=memory_dir,
        records=all_ranked,
        force=enable_assistant_context,
    )
    fused_top_records = attach_assistant_context_robust(
        parsed_query=parsed_query,
        memory_dir=memory_dir,
        records=fused_top_records,
        force=enable_assistant_context,
    )

    # P2.2: direct assistant reply search route.
    # This is independent from attach_assistant_context. It retrieves assistant replies
    # even when normal memory retrieval misses the source memory.
    assistant_reply_hits = search_assistant_replies(
        parsed_query=parsed_query,
        memory_dir=memory_dir,
        top_k=route_top_k,
    )
    if assistant_reply_hits:
        all_ranked = assistant_reply_hits + all_ranked
        fused_top_records = assistant_reply_hits + fused_top_records

    # P4: assistant pair search route.
    # Uses previous_user_message + assistant_reply from assistant_pair_index.json.
    assistant_pair_hits = search_assistant_pairs(
        parsed_query=parsed_query,
        memory_dir=memory_dir,
        top_k=route_top_k,
    )
    if assistant_pair_hits:
        all_ranked = assistant_pair_hits + all_ranked
        fused_top_records = assistant_pair_hits + fused_top_records

    # P2.2: relative-event binding for questions like
    # "before/after getting X". This enriches records and adds a derived binding hint.
    all_ranked = apply_relative_event_binding(
        parsed_query=parsed_query,
        records=all_ranked,
        final_top_k=max(route_top_k * 3, final_top_k * 2),
    )

    # P4: structured task reader hint before rerank.
    all_ranked = apply_structured_task_reader(
        parsed_query=parsed_query,
        records=all_ranked,
        final_top_k=final_top_k,
    )

    if enable_rerank:
        final_top_records = rerank_records(
            parsed_query=parsed_query,
            records=all_ranked,
            final_top_k=final_top_k,
        )
    else:
        final_top_records = fused_top_records[:final_top_k]

    mapped_query = build_full_mapped_query(parsed_query, search_result)
    sub_results = search_result.get("sub_results") or {}

    experiment = {
        "query_parsed": str(parsed_path),
        "memory_dir": str(memory_dir),
        "output_dir": str(run_dir),
        "question_type": question_type,
        "sample_id": sample_id,
        "route_top_k": route_top_k,
        "final_top_k": final_top_k,
        "enable_rerank": enable_rerank,
        "enable_assistant_context": enable_assistant_context,
        "embedding_model": getattr(embedding_client, "model", None),
        "embedding_device": getattr(embedding_client, "device", None),
        "record_count": len(records),
        "retrieval_method": search_mode,
        "forced_parse_mode": "fused" if force_fused_parse_mode else None,
        "started_at": datetime.now().isoformat(),
    }

    _write_json(run_dir / "experiment_config.json", experiment)
    _write_json(run_dir / "mapped_query_analysis.json", mapped_query)
    _write_json(run_dir / "all_ranked_records.json", all_ranked)
    _write_json(run_dir / "top_records.json", final_top_records)
    _write_json(run_dir / "sub_results.json", sub_results)

    summary = {
        "parse_mode": "fused" if force_fused_parse_mode else parsed_query.get("parse_mode", ""),
        "retrieval_method": search_mode,
        "question_type": question_type,
        "sample_id": sample_id,
        "query_text": mapped_query.get("query_text")
        or mapped_query.get("query_anchor")
        or mapped_query.get("canonical_text")
        or parsed_query.get("query_anchor")
        or "",
        "route_top_k": route_top_k,
        "final_top_k": final_top_k,
        "enable_rerank": enable_rerank,
        "enable_assistant_context": enable_assistant_context,
        "record_count": len(records),
        "fused_candidate_count_before_final_cut": len(fused_top_records),
        "final_top_record_count": len(final_top_records),
        "assistant_reply_count_in_final_top": sum(
            1 for row in final_top_records if _clean(row.get("assistant_reply"))
        ),
        "assistant_reply_search_hit_count": len(assistant_reply_hits),
        "assistant_pair_search_hit_count": len(assistant_pair_hits),
        "assistant_reply_search_enabled": should_search_assistant_replies(parsed_query),
        "assistant_pair_search_hit_count": len(assistant_pair_hits),
        "assistant_pair_search_enabled": should_search_assistant_pairs(parsed_query),
        "assistant_context_stats_final": assistant_context_stats(final_top_records),
        "relative_event_stats_final": relative_event_stats(final_top_records),
        "structured_task_reader_stats_final": structured_task_reader_stats(final_top_records),
        "output_dir": str(run_dir),
        "top_records": _brief_top_records(final_top_records),
        "elapsed_seconds": time.time() - started,
    }

    _write_json(run_dir / "summary.json", summary)

    return {
        "question_type": question_type,
        "sample_id": sample_id,
        "ok": True,
        "error": None,
        "record_count": len(records),
        "candidate_count": len(fused_top_records),
        "final_top_record_count": len(final_top_records),
        "assistant_reply_count_in_final_top": summary["assistant_reply_count_in_final_top"],
        "assistant_reply_search_hit_count": len(assistant_reply_hits),
        "relative_event_stats_final": summary["relative_event_stats_final"],
        "output_dir": str(run_dir),
        "elapsed_seconds": time.time() - started,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch P2 fused retrieval for LongMemEval. "
            "Runs BM25 + Structured + MiniLM, each route top-k, then optional P2 rerank."
        )
    )

    parser.add_argument(
        "--query-root",
        type=Path,
        default=DEFAULT_QUERY_ROOT,
        help="Query analysis run root, e.g. ./results/query_analysis/run_baseline",
    )
    parser.add_argument(
        "--memory-root",
        type=Path,
        default=DEFAULT_MEMORY_ROOT,
        help="Memory extraction run root, e.g. ./results/memories/run_baseline",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output retrieval root, e.g. ./results/retrieval/run_baseline_fused_p2",
    )

    parser.add_argument("--route-top-k", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=15)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases")

    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Local sentence-transformers model path or name",
    )
    parser.add_argument("--embedding-device", default="cuda")

    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Enable P2 global rerank after fused retrieval.",
    )
    parser.add_argument(
        "--enable-assistant-context",
        action="store_true",
        help=(
            "Force assistant context attachment when assistant reply files are available. "
            "Without this flag, assistant context is attached only when parsed query says need_assistant_context=true."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases whose summary.json already exists",
    )
    parser.add_argument(
        "--no-force-fused-parse-mode",
        action="store_false",
        dest="force_fused_parse_mode",
        help="Do not overwrite parsed_query['parse_mode'] to fused in memory",
    )
    parser.set_defaults(force_fused_parse_mode=True)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    query_root = args.query_root.resolve()
    memory_root = args.memory_root.resolve()
    output_root = args.output_root.resolve()

    if not query_root.exists():
        raise FileNotFoundError(f"query_root not found: {query_root}")
    if not memory_root.exists():
        raise FileNotFoundError(f"memory_root not found: {memory_root}")

    cases = list(_iter_cases(query_root))
    cases = _filter_cases(cases, max_cases=args.max_cases)

    output_root.mkdir(parents=True, exist_ok=True)

    search_mode = _search_mode_name(args.route_top_k, args.final_top_k, args.enable_rerank)

    _log(
        "Batch P2 fused retrieval started: "
        f"cases={len(cases)} "
        f"route_top_k={args.route_top_k} "
        f"final_top_k={args.final_top_k} "
        f"enable_rerank={args.enable_rerank} "
        f"enable_assistant_context={args.enable_assistant_context} "
        f"search_mode={search_mode} "
        f"query_root={query_root} "
        f"memory_root={memory_root} "
        f"output_root={output_root}"
    )

    embedding_client = LocalEmbeddingClient(
        model=args.embedding_model,
        device=args.embedding_device,
        batch_size=32,
    )

    rows: List[Dict[str, Any]] = []
    done = 0
    failed = 0
    skipped = 0

    _write_json(
        output_root / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "query_root": str(query_root),
            "memory_root": str(memory_root),
            "output_root": str(output_root),
            "search_mode": search_mode,
            "route_top_k": args.route_top_k,
            "final_top_k": args.final_top_k,
            "enable_rerank": args.enable_rerank,
            "enable_assistant_context": args.enable_assistant_context,
            "embedding_model": args.embedding_model,
            "embedding_device": args.embedding_device,
            "max_cases": args.max_cases,
            "resume": args.resume,
            "force_fused_parse_mode": args.force_fused_parse_mode,
            "case_count": len(cases),
        },
    )

    for case_index, (question_type, sample_id, parsed_path) in enumerate(cases, start=1):
        memory_dir = _memory_dir_for_case(memory_root, question_type, sample_id)
        summary_path = output_root / question_type / search_mode / sample_id / "summary.json"

        _log(
            f"[case {case_index}/{len(cases)}] "
            f"question_id={question_type}/{sample_id} | start"
        )

        if args.resume and summary_path.exists():
            skipped += 1
            done += 1
            try:
                old_summary = _load_json(summary_path)
                rows.append(
                    {
                        "question_type": question_type,
                        "sample_id": sample_id,
                        "ok": True,
                        "skipped": True,
                        "output_dir": old_summary.get("output_dir"),
                    }
                )
            except Exception:
                rows.append(
                    {
                        "question_type": question_type,
                        "sample_id": sample_id,
                        "ok": True,
                        "skipped": True,
                        "output_dir": str(summary_path.parent),
                    }
                )

            _log(
                f"[case {case_index}/{len(cases)}] "
                f"question_id={question_type}/{sample_id} | skipped existing | "
                f"done={done} failed={failed} skipped={skipped}"
            )
            continue

        if not memory_dir.exists():
            failed += 1
            row = {
                "question_type": question_type,
                "sample_id": sample_id,
                "ok": False,
                "error": f"memory_dir not found: {memory_dir}",
                "parsed_path": str(parsed_path),
                "memory_dir": str(memory_dir),
            }
            rows.append(row)
            _log(
                f"[case {case_index}/{len(cases)}] "
                f"question_id={question_type}/{sample_id} | failed: memory_dir not found"
            )
            continue

        try:
            row = run_one_case(
                question_type=question_type,
                sample_id=sample_id,
                parsed_path=parsed_path,
                memory_dir=memory_dir,
                output_root=output_root,
                embedding_client=embedding_client,
                route_top_k=args.route_top_k,
                final_top_k=args.final_top_k,
                force_fused_parse_mode=args.force_fused_parse_mode,
                enable_rerank=args.enable_rerank,
                enable_assistant_context=args.enable_assistant_context,
            )
            rows.append(row)
            done += 1

            _log(
                f"[case {case_index}/{len(cases)}] "
                f"question_id={question_type}/{sample_id} | done | "
                f"records={row['record_count']} "
                f"candidates={row['candidate_count']} "
                f"final_top={row['final_top_record_count']} "
                f"assistant_reply_final={row['assistant_reply_count_in_final_top']} "
                f"elapsed={row['elapsed_seconds']:.2f}s | "
                f"done={done} failed={failed} skipped={skipped}"
            )

        except Exception as exc:
            failed += 1
            err = f"{type(exc).__name__}: {exc}"
            row = {
                "question_type": question_type,
                "sample_id": sample_id,
                "ok": False,
                "error": err,
                "parsed_path": str(parsed_path),
                "memory_dir": str(memory_dir),
            }
            rows.append(row)
            _log(
                f"[case {case_index}/{len(cases)}] "
                f"question_id={question_type}/{sample_id} | failed: {err}"
            )

        _write_json(
            output_root / "status.json",
            {
                "output_root": str(output_root),
                "search_mode": search_mode,
                "total": len(cases),
                "done": done,
                "failed": failed,
                "skipped": skipped,
                "running": {
                    "question_type": question_type,
                    "sample_id": sample_id,
                    "parsed_path": str(parsed_path),
                    "memory_dir": str(memory_dir),
                },
                "updated_at": time.time(),
            },
        )

    final = {
        "output_root": str(output_root),
        "search_mode": search_mode,
        "total": len(cases),
        "done": done,
        "failed": failed,
        "skipped": skipped,
        "route_top_k": args.route_top_k,
        "final_top_k": args.final_top_k,
        "enable_rerank": args.enable_rerank,
        "enable_assistant_context": args.enable_assistant_context,
        "rows": rows,
    }

    _write_json(output_root / "summary.json", final)

    _log(
        "Batch P2 fused retrieval completed: "
        f"total={len(cases)} done={done} failed={failed} skipped={skipped} "
        f"output_root={output_root}"
    )

    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


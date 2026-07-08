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
from search import search_top15_content_dedup


DEFAULT_QUERY_ROOT = SUBMIT_ROOT / "results/query_analysis/run_baseline"
DEFAULT_MEMORY_ROOT = SUBMIT_ROOT / "results/memories/run_baseline"
DEFAULT_OUTPUT_ROOT = SUBMIT_ROOT / "results/retrieval/run_baseline_fused"
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


def _iter_cases(query_root: Path) -> Iterable[Tuple[str, str, Path]]:
    """
    Expected query analysis layout:
      query_root/<question_type>/<sample_id>/parsed.json
    Example:
      results/query_analysis/run_baseline/knowledge-update/031748ae_abs/parsed.json
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
) -> Dict[str, Any]:
    started = time.time()

    parsed_query = _load_json(parsed_path)

    # 强制开启 fused 三路混合检索。
    # 原始 retrieve_from_parsed_query.py 是通过 parsed_query["parse_mode"] 判断是否走 fused。
    # 这里我们直接调用 fused 检索函数，所以即使原 parsed.json 不是 fused，也会走三路检索。
    if force_fused_parse_mode:
        parsed_query["parse_mode"] = "fused"

    records = load_records(memory_dir)

    search_result = search_top15_content_dedup(
        parsed_query=parsed_query,
        records=records,
        embedding_client=embedding_client,
        top_k=route_top_k,
    )

    search_mode = "tri_fused_route20_final15"
    run_dir = output_root / question_type / search_mode / sample_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_ranked = list(search_result.get("all_ranked_records") or [])
    fused_top_records = list(search_result.get("top_records") or [])

    # 原始 search_top15_content_dedup 会返回三路 top-k 去重后的所有结果。
    # 这里显式截断 final top-k。
    final_top_records = fused_top_records[:final_top_k]

    mapped_query = search_result.get("mapped_query_analysis") or {}
    sub_results = search_result.get("sub_results") or {}

    experiment = {
        "query_parsed": str(parsed_path),
        "memory_dir": str(memory_dir),
        "output_dir": str(run_dir),
        "question_type": question_type,
        "sample_id": sample_id,
        "route_top_k": route_top_k,
        "final_top_k": final_top_k,
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
        "parse_mode": "fused",
        "retrieval_method": search_mode,
        "question_type": question_type,
        "sample_id": sample_id,
        "query_text": mapped_query.get("query_text")
        or mapped_query.get("query_anchor")
        or mapped_query.get("canonical_text")
        or "",
        "route_top_k": route_top_k,
        "final_top_k": final_top_k,
        "record_count": len(records),
        "fused_candidate_count_before_final_cut": len(fused_top_records),
        "final_top_record_count": len(final_top_records),
        "output_dir": str(run_dir),
        "top_records": [
            {
                "rank": i + 1,
                "score": row.get("score"),
                "memory_type": row.get("memory_type"),
                "content": row.get("content"),
                "source_boundary_id": row.get("source_boundary_id"),
                "retrieval_method": row.get("retrieval_method"),
                "retrieval_rank": row.get("retrieval_rank"),
                "retrieval_score": row.get("retrieval_score"),
                "fusion_sources": row.get("fusion_sources"),
                "score_components": row.get("score_components"),
            }
            for i, row in enumerate(final_top_records)
        ],
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
        "output_dir": str(run_dir),
        "elapsed_seconds": time.time() - started,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch fused retrieval for LongMemEval. "
            "Runs BM25 + Structured + MiniLM, each route top-k, then final top-k."
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
        help="Output retrieval root, e.g. ./results/retrieval/run_baseline_fused",
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
        "--resume",
        action="store_true",
        help="Skip cases whose summary.json already exists",
    )
    parser.add_argument(
        "--no-force-fused-parse-mode",
        action="store_false",
        dest="force_fused_parse_mode",
        help="Do not overwrite parsed_query['parse_mode'] to fused",
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

    _log(
        "Batch fused retrieval started: "
        f"cases={len(cases)} "
        f"route_top_k={args.route_top_k} "
        f"final_top_k={args.final_top_k} "
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
            "route_top_k": args.route_top_k,
            "final_top_k": args.final_top_k,
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
        search_mode = "tri_fused_route20_final15"
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
            )
            rows.append(row)
            done += 1

            _log(
                f"[case {case_index}/{len(cases)}] "
                f"question_id={question_type}/{sample_id} | done | "
                f"records={row['record_count']} "
                f"candidates={row['candidate_count']} "
                f"final_top={row['final_top_record_count']} "
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
        "total": len(cases),
        "done": done,
        "failed": failed,
        "skipped": skipped,
        "route_top_k": args.route_top_k,
        "final_top_k": args.final_top_k,
        "rows": rows,
    }

    _write_json(output_root / "summary.json", final)

    _log(
        "Batch fused retrieval completed: "
        f"total={len(cases)} done={done} failed={failed} skipped={skipped} "
        f"output_root={output_root}"
    )

    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

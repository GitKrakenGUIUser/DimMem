#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

THIS_FILE = Path(__file__).resolve()
LONGMEMEVAL_DIR = THIS_FILE.parents[1]

if str(LONGMEMEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LONGMEMEVAL_DIR))

from memory_constructor.extract_helpers import (
    _build_prompt,
    _build_session_map_from_window,
    _clean,
    _extract_text,
    _normalize_memory_entry,
    _safe_json_fragment,
    _source_time_by_id_from_dialogue,
    _window_paths,
)


_PRINT_LOCK = threading.Lock()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(f"[{_now_str()}] {msg}", flush=True)


def _call_chat_with_timeout(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    """
    Local chat call with configurable timeout.

    The helper in extract_helpers.py uses a fixed timeout in the current branch,
    so this batch runner keeps its own call wrapper to make --timeout effective.
    """
    url = base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": int(max_tokens),
    }

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=int(timeout),
    )
    response.raise_for_status()
    return response.json()


def _record_identity(record_dir: Path, segments_root: Path) -> str:
    """
    Prefer real question_id from summary.json.
    Fallback to relative path, e.g. question_type/0001_xxx.
    """
    summary_path = record_dir / "summary.json"

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            qid = _clean(summary.get("question_id"))
            qtype = _clean(summary.get("question_type"))

            if qid and qtype:
                return f"{qtype}/{qid}"
            if qid:
                return qid
        except Exception:
            pass

    return record_dir.relative_to(segments_root).as_posix()


def _iter_record_dirs(segments_root: Path) -> List[Path]:
    """
    Support both layouts:
    - segments_root/question_type/sample_id/summary.json
    - segments_root/sample_id/summary.json
    """
    two_level = [p.parent for p in sorted(segments_root.glob("*/*/summary.json"))]
    one_level = [p.parent for p in sorted(segments_root.glob("*/summary.json"))]

    return sorted(set(two_level + one_level))


def _output_rel(record_dir: Path, segments_root: Path) -> Path:
    return record_dir.relative_to(segments_root)


def _process_record(
    *,
    record_dir: Path,
    segments_root: Path,
    output_root: Path,
    base_url: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    overlap: int,
    record_id: str,
    record_index: int,
    records_total: int,
) -> Dict[str, Any]:
    rel = _output_rel(record_dir, segments_root)
    out_record_dir = output_root / rel
    _ensure_dir(out_record_dir)

    _write_json(
        out_record_dir / "experiment_config.json",
        {
            "source_record_dir": str(record_dir),
            "output_record_dir": str(out_record_dir),
            "base_url": base_url,
            "model_name": model_name,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": max_retries,
            "overlap": overlap,
            "started_at": datetime.now().isoformat(),
        },
    )

    rows: List[Dict[str, Any]] = []
    all_memories: List[Dict[str, Any]] = []

    windows_dir = record_dir / "windows"
    window_paths = _window_paths(windows_dir)
    window_total = len(window_paths)

    _log(
        f"[record {record_index}/{records_total}] "
        f"question_id={record_id} | windows={window_total} | start extraction"
    )

    for window_pos, window_path in enumerate(window_paths, start=1):
        window = json.loads(window_path.read_text(encoding="utf-8"))
        window_idx = int(window.get("window_index", 0))
        overlap_count = overlap if window_idx > 0 else 0

        _log(
            f"[record {record_index}/{records_total}] "
            f"question_id={record_id} | "
            f"window {window_pos}/{window_total} "
            f"(window_index={window_idx}) | start"
        )

        win_dir = out_record_dir / f"window_{window_idx:04d}"
        _ensure_dir(win_dir)

        conversation = _clean(window.get("text"))
        source_time_map = _source_time_by_id_from_dialogue(conversation)
        session_map = _build_session_map_from_window(window)
        prompt = _build_prompt(
            conversation,
            window_index=window_idx,
            overlap_count=overlap_count,
        )

        _write_json(win_dir / "window_input.json", window)
        _write_text(win_dir / "dialogue_input.txt", conversation)
        _write_text(win_dir / "extract_prompt.txt", prompt)

        ok = False
        err = None
        parsed: Any = None
        raw_text = ""
        response_json: Dict[str, Any] | None = None
        memories: List[Dict[str, Any]] = []

        started = time.time()
        attempt = 0

        while attempt < max(1, max_retries):
            attempt += 1

            _log(
                f"[record {record_index}/{records_total}] "
                f"question_id={record_id} | "
                f"window {window_pos}/{window_total} | "
                f"attempt {attempt}/{max_retries}"
            )

            try:
                response_json = _call_chat_with_timeout(
                    base_url=base_url,
                    api_key=api_key,
                    model_name=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )

                raw_text = _extract_text(response_json)
                parsed = _safe_json_fragment(raw_text)

                mem_rows = parsed.get("memories") if isinstance(parsed, dict) else []
                if not isinstance(mem_rows, list):
                    raise ValueError("parsed_response_missing_memories_list")

                for m in mem_rows:
                    norm = _normalize_memory_entry(
                        m,
                        source_time_map=source_time_map,
                        session_map=session_map,
                    )
                    if norm is not None:
                        memories.append(norm)

                ok = True
                break

            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"

                _log(
                    f"[record {record_index}/{records_total}] "
                    f"question_id={record_id} | "
                    f"window {window_pos}/{window_total} | "
                    f"attempt {attempt}/{max_retries} failed: {err}"
                )

                if attempt >= max_retries:
                    break

                time.sleep(min(2 * attempt, 8))

        if response_json is not None:
            _write_json(win_dir / "raw_response.json", response_json)

        _write_text(win_dir / "raw_response.txt", raw_text)

        if parsed is not None:
            _write_json(win_dir / "parsed_payload.json", parsed)

        _write_json(win_dir / "normalized_memories.json", {"memories": memories})

        result = {
            "window_index": window_idx,
            "source_path": str(window_path),
            "ok": ok,
            "error": err,
            "attempt_count": attempt,
            "memory_count": len(memories),
            "elapsed_seconds": time.time() - started,
            "overlap_count": overlap_count,
            "usage": (response_json or {}).get("usage"),
        }

        _write_json(win_dir / "result.json", result)

        _log(
            f"[record {record_index}/{records_total}] "
            f"question_id={record_id} | "
            f"window {window_pos}/{window_total} | "
            f"done ok={ok} memories={len(memories)} "
            f"elapsed={result['elapsed_seconds']:.2f}s"
        )

        rows.append(result)

        for i, m in enumerate(memories):
            item = dict(m)
            item["window_index"] = window_idx
            item["memory_index"] = i
            all_memories.append(item)

    summary = {
        "source_record_dir": str(record_dir),
        "output_record_dir": str(out_record_dir),
        "count": len(rows),
        "ok_count": sum(1 for x in rows if x["ok"]),
        "error_count": sum(1 for x in rows if not x["ok"]),
        "total_memory_count": sum(int(x["memory_count"]) for x in rows),
        "rows": rows,
    }

    _write_json(out_record_dir / "summary.json", summary)
    _write_json(
        out_record_dir / "all_memories.json",
        {
            "source_record_dir": str(record_dir),
            "memory_count": len(all_memories),
            "memories": all_memories,
        },
    )

    _log(
        f"[record {record_index}/{records_total}] "
        f"question_id={record_id} | finished | "
        f"ok_windows={summary['ok_count']}/{summary['count']} | "
        f"total_memories={summary['total_memory_count']}"
    )

    return summary


def _write_status(
    *,
    status_path: Path,
    state: str,
    started_at: str,
    segments_root: Path,
    out_root: Path,
    records_total: int,
    done: int,
    failed: int,
    skipped: int,
    workers: int,
    resume: bool,
    inflight_records: List[str] | None = None,
) -> None:
    _write_json(
        status_path,
        {
            "state": state,
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(),
            "segments_root": str(segments_root),
            "output_root": str(out_root),
            "records_total": records_total,
            "done": done,
            "failed": failed,
            "skipped_existing": skipped,
            "workers": workers,
            "inflight_records": inflight_records or [],
            "resume": bool(resume),
        },
    )


def run(args: argparse.Namespace) -> Path:
    segments_root = args.segments_root.resolve()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_root.resolve() / run_name
    _ensure_dir(out_root)

    record_dirs = _iter_record_dirs(segments_root)
    if args.max_records > 0:
        record_dirs = record_dirs[: args.max_records]

    records_total = len(record_dirs)
    workers = max(1, int(args.workers))

    status_path = out_root / "status.json"
    failures_path = out_root / "failures.json"
    manifest_path = out_root / "run_manifest.json"

    failures: List[Dict[str, Any]] = []

    if args.resume and failures_path.exists():
        try:
            failures = list(
                json.loads(failures_path.read_text(encoding="utf-8")).get("failures")
                or []
            )
        except Exception:
            failures = []

    done = 0
    failed = 0
    skipped = 0
    started_at = datetime.now().isoformat()

    record_jobs: List[Tuple[int, Path, str]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for record_index, record_dir in enumerate(record_dirs, start=1):
        rel = _output_rel(record_dir, segments_root)
        record_id = _record_identity(record_dir, segments_root)
        summary_path = out_root / rel / "summary.json"

        if args.resume and summary_path.exists():
            skipped += 1
            done += 1
            skipped_rows.append(
                {
                    "record_index": record_index,
                    "record_dir": str(record_dir),
                    "question_id": record_id,
                    "summary_path": str(summary_path),
                }
            )
            _log(
                f"[record {record_index}/{records_total}] "
                f"question_id={record_id} | skipped existing | "
                f"done={done}/{records_total}"
            )
            continue

        record_jobs.append((record_index, record_dir, record_id))

    active_workers = min(workers, max(1, len(record_jobs))) if record_jobs else 1

    _write_status(
        status_path=status_path,
        state="running",
        started_at=started_at,
        segments_root=segments_root,
        out_root=out_root,
        records_total=records_total,
        done=done,
        failed=failed,
        skipped=skipped,
        workers=active_workers,
        resume=bool(args.resume),
    )

    _log(
        f"Memory extraction started: total_records={records_total} "
        f"todo={len(record_jobs)} skipped={skipped} "
        f"workers={active_workers} output_root={out_root}"
    )

    completed_rows: List[Dict[str, Any]] = []

    def submit_kwargs(record_index: int, record_dir: Path, record_id: str) -> Dict[str, Any]:
        return {
            "record_dir": record_dir,
            "segments_root": segments_root,
            "output_root": out_root,
            "base_url": args.base_url,
            "api_key": args.api_key,
            "model_name": args.model_name,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "overlap": args.overlap,
            "record_id": record_id,
            "record_index": record_index,
            "records_total": records_total,
        }

    if record_jobs:
        if active_workers == 1:
            for record_index, record_dir, record_id in record_jobs:
                try:
                    _log(
                        f"[record {record_index}/{records_total}] "
                        f"question_id={record_id} | start | "
                        f"done={done} failed={failed} skipped={skipped}"
                    )

                    summary = _process_record(
                        **submit_kwargs(record_index, record_dir, record_id)
                    )

                    done += 1
                    completed_rows.append(
                        {
                            "record_index": record_index,
                            "question_id": record_id,
                            "record_dir": str(record_dir),
                            "ok": True,
                            "summary": summary,
                        }
                    )

                    _log(
                        f"[record {record_index}/{records_total}] "
                        f"question_id={record_id} | record done | "
                        f"done={done}/{records_total} failed={failed} skipped={skipped}"
                    )

                except Exception as exc:
                    failed += 1
                    err_msg = f"{type(exc).__name__}: {exc}"
                    failures.append(
                        {
                            "record_dir": str(record_dir),
                            "question_id": record_id,
                            "error": err_msg,
                        }
                    )

                    _log(
                        f"[record {record_index}/{records_total}] "
                        f"question_id={record_id} | record failed: {err_msg}"
                    )

                _write_status(
                    status_path=status_path,
                    state="running",
                    started_at=started_at,
                    segments_root=segments_root,
                    out_root=out_root,
                    records_total=records_total,
                    done=done,
                    failed=failed,
                    skipped=skipped,
                    workers=active_workers,
                    resume=bool(args.resume),
                )

        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=active_workers,
                thread_name_prefix="memory_extract",
            ) as executor:
                future_map: Dict[
                    concurrent.futures.Future,
                    Tuple[int, Path, str],
                ] = {}

                for record_index, record_dir, record_id in record_jobs:
                    future = executor.submit(
                        _process_record,
                        **submit_kwargs(record_index, record_dir, record_id),
                    )
                    future_map[future] = (record_index, record_dir, record_id)

                inflight = [rid for _, _, rid in future_map.values()]

                _write_status(
                    status_path=status_path,
                    state="running",
                    started_at=started_at,
                    segments_root=segments_root,
                    out_root=out_root,
                    records_total=records_total,
                    done=done,
                    failed=failed,
                    skipped=skipped,
                    workers=active_workers,
                    resume=bool(args.resume),
                    inflight_records=inflight,
                )

                for future in concurrent.futures.as_completed(future_map):
                    record_index, record_dir, record_id = future_map[future]

                    try:
                        summary = future.result()
                        done += 1
                        completed_rows.append(
                            {
                                "record_index": record_index,
                                "question_id": record_id,
                                "record_dir": str(record_dir),
                                "ok": True,
                                "summary": summary,
                            }
                        )

                        _log(
                            f"[record {record_index}/{records_total}] "
                            f"question_id={record_id} | record done | "
                            f"done={done}/{records_total} failed={failed} skipped={skipped}"
                        )

                    except Exception as exc:
                        failed += 1
                        err_msg = f"{type(exc).__name__}: {exc}"
                        failures.append(
                            {
                                "record_dir": str(record_dir),
                                "question_id": record_id,
                                "error": err_msg,
                            }
                        )

                        _log(
                            f"[record {record_index}/{records_total}] "
                            f"question_id={record_id} | record failed: {err_msg}"
                        )

                    remaining = [
                        rid
                        for fut, (_, _, rid) in future_map.items()
                        if not fut.done()
                    ]

                    _write_status(
                        status_path=status_path,
                        state="running",
                        started_at=started_at,
                        segments_root=segments_root,
                        out_root=out_root,
                        records_total=records_total,
                        done=done,
                        failed=failed,
                        skipped=skipped,
                        workers=active_workers,
                        resume=bool(args.resume),
                        inflight_records=remaining,
                    )

    completed_rows.sort(key=lambda x: int(x["record_index"]))

    _write_json(
        out_root / "experiment_config.json",
        {
            "created_at": datetime.now().isoformat(),
            "segments_root": str(segments_root),
            "output_root": str(out_root),
            "base_url": args.base_url,
            "model_name": args.model_name,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "overlap": args.overlap,
            "workers": active_workers,
            "records_total": records_total,
            "records_done": done,
            "records_failed": failed,
            "records_skipped_existing": skipped,
            "resume": bool(args.resume),
        },
    )

    _write_json(
        manifest_path,
        {
            "state": "completed",
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "records_total": records_total,
            "done": done,
            "failed": failed,
            "skipped_existing": skipped,
            "workers": active_workers,
            "completed_rows": completed_rows,
            "skipped_rows": skipped_rows,
            "failures": failures,
        },
    )

    if failures:
        _write_json(failures_path, {"failures": failures})

    _write_status(
        status_path=status_path,
        state="completed",
        started_at=started_at,
        segments_root=segments_root,
        out_root=out_root,
        records_total=records_total,
        done=done,
        failed=failed,
        skipped=skipped,
        workers=active_workers,
        resume=bool(args.resume),
    )

    _log(
        f"Memory extraction completed: "
        f"done={done}/{records_total} failed={failed} skipped={skipped} "
        f"workers={active_workers} output_root={out_root}"
    )

    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build structured memories from LongMemEval segments."
    )

    parser.add_argument(
        "--segments-root",
        type=Path,
        required=True,
        help="Path to segments directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./results/memories"),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--overlap",
        type=int,
        default=5,
        help="Number of overlapping messages between windows.",
    )

    parser.add_argument("--base-url", default="http://127.0.0.1:7790/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="gpt-4.1-mini")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=0)

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of records/cases to extract concurrently. "
            "Use 30 to run all 30 cases in parallel if your backend supports it."
        ),
    )

    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))


if __name__ == "__main__":
    main()

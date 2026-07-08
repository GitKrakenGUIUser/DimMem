#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

THIS_FILE = Path(__file__).resolve()
LONGMEMEVAL_DIR = THIS_FILE.parents[1]
SUBMIT_ROOT = THIS_FILE.parents[2]
if str(LONGMEMEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(LONGMEMEVAL_DIR))

from prompts.qa_prompts import build_qa_payload
from prompts.judge_prompts import build_judge_payload

# Global variables to be set by CLI args
RETRIEVAL_ROOT = None
QUERY_ANALYSIS_ROOT = None
QA_ROOT = None
JUDGE_ROOT = None
MODEL_NAME = None
BASE_URL = None
API_KEY = None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_retrieval_dirs(root: Path) -> Iterable[Tuple[str, str, str, Path]]:
    for question_type_dir in sorted(root.iterdir()):
        if not question_type_dir.is_dir():
            continue
        question_type = question_type_dir.name
        for method_dir in sorted(question_type_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            for sample_dir in sorted(method_dir.iterdir()):
                if sample_dir.is_dir():
                    yield question_type, method, sample_dir.name, sample_dir

'''
def _filter_record(record: Dict[str, Any]) -> Dict[str, Any]:
    dimension = record.get("dimension") if isinstance(record.get("dimension"), dict) else {}
    out: Dict[str, Any] = {}
    if _clean(record.get("source_time")):
        out["source_time"] = _clean(record.get("source_time"))
    if _clean(record.get("content")):
        out["content"] = _clean(record.get("content"))
    if _clean(dimension.get("reason")):
        out["dimension"] = out.get("dimension", {})
        out["dimension"]["reason"] = _clean(dimension.get("reason"))
    if _clean(dimension.get("purpose")):
        out["dimension"] = out.get("dimension", {})
        out["dimension"]["purpose"] = _clean(dimension.get("purpose"))
    return out
    '''
def _filter_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep enough structured evidence for QA.

    Original implementation only kept:
    - source_time
    - content
    - dimension.reason
    - dimension.purpose

    P1 keeps time/location/keywords/memory_type/retrieval metadata/assistant_reply
    so that temporal, location, assistant-dependent and preference questions
    are not forced to answer from content alone.
    """
    dimension = record.get("dimension") if isinstance(record.get("dimension"), dict) else {}
    out: Dict[str, Any] = {}

    top_level_fields = [
        "source_time",
        "memory_type",
        "content",
        "assistant_reply",
        "assistant_uid",
        "session_id",
        "session_local_user_index",
        "source_boundary_id",
        "retrieval_method",
        "retrieval_rank",
        "retrieval_score",
        "rerank_rank",
        "rerank_score",
        "fusion_sources",
        "score_components",
        "rerank_components",
    ]

    for key in top_level_fields:
        value = record.get(key)
        if isinstance(value, (dict, list)):
            if value:
                out[key] = value
        elif _clean(value):
            out[key] = _clean(value)

    dimension_fields = [
        "time",
        "event_time",
        "valid_from",
        "valid_to",
        "status",
        "is_current",
        "location",
        "reason",
        "purpose",
        "keywords",

        "subject",
        "action",
        "object",
        "value",
        "quantity",
        "unit",
        "relation",
        "evidence_span",

        "supersedes",
    ]

    dim_out: Dict[str, Any] = {}
    for key in dimension_fields:
        value = dimension.get(key)
        if isinstance(value, (dict, list)):
            if value:
                dim_out[key] = value
        elif isinstance(value, bool):
            dim_out[key] = value
        elif _clean(value):
            dim_out[key] = _clean(value)

    if dim_out:
        out["dimension"] = dim_out

    return out


def _chat(prompt: str, timeout: int = 600) -> Dict[str, Any]:
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    session = requests.Session()
    session.trust_env = False

    resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()



def _extract_message(resp_json: Dict[str, Any]) -> str:
    try:
        return _clean(resp_json["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _parse_answer(raw_text: str) -> Dict[str, Any]:
    reasoning = ""
    answer = raw_text.strip()
    for line in raw_text.splitlines():
        text = line.strip()
        if text.lower().startswith("reasoning:"):
            reasoning = text.split(":", 1)[1].strip()
        elif text.lower().startswith("answer:"):
            answer = text.split(":", 1)[1].strip()
    return {
        "reasoning": reasoning,
        "answer": answer,
        "raw_text": raw_text,
    }


def _parse_judge(raw_text: str) -> Dict[str, Any]:
    label = ""
    reasoning = ""
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            label = _clean(payload.get("label")).upper()
            reasoning = _clean(payload.get("reasoning"))
    except Exception:
        pass
    if not label:
        match = re.search(r"\b(CORRECT|WRONG)\b", raw_text, flags=re.IGNORECASE)
        if match:
            label = match.group(1).upper()
    return {
        "label": label,
        "reasoning": reasoning,
        "raw_text": raw_text,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}] {msg}", flush=True)


def _write_status(
    *,
    qa_root: Path,
    judge_root: Path,
    state: str,
    total: int,
    done: int,
    failed: int,
    skipped: int,
    running: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "state": state,
        "total": total,
        "done": done,
        "failed": failed,
        "skipped": skipped,
        "running": running or {},
        "updated_at": datetime.now().isoformat(),
    }
    _write_json(qa_root / "status.json", payload)
    _write_json(judge_root / "status.json", payload)


def _write_run_manifest(args: argparse.Namespace, total: int) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "retrieval_root": str(RETRIEVAL_ROOT),
        "query_root": str(QUERY_ANALYSIS_ROOT),
        "qa_root": str(QA_ROOT),
        "judge_root": str(JUDGE_ROOT),
        "run_name": args.run_name,
        "model_name": MODEL_NAME,
        "base_url": BASE_URL,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "target_count": total,
    }
    _write_json(QA_ROOT / "run_manifest.json", payload)
    _write_json(JUDGE_ROOT / "run_manifest.json", payload)


def _chat_with_retry(
    *,
    prompt: str,
    stage: str,
    question_label: str,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, max(1, max_retries) + 1):
        _log(
            f"{question_label} | {stage} attempt "
            f"{attempt}/{max(1, max_retries)}"
        )
        try:
            return _chat(prompt, timeout=timeout)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            _log(
                f"{question_label} | {stage} attempt "
                f"{attempt}/{max(1, max_retries)} failed: {last_err}"
            )
            if attempt < max(1, max_retries):
                time.sleep(min(2 * attempt, 8))

    raise RuntimeError(last_err or f"{stage} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongMemEval QA + Judge from retrieval results.")
    parser.add_argument("--retrieval-root", type=Path, required=True,
                        help="Path to retrieval results (e.g., ./results/retrieval/<run_name>)")
    parser.add_argument("--query-root", type=Path, required=True,
                        help="Path to query analysis results (e.g., ./results/query_analysis/<run_name>)")
    parser.add_argument("--output-base", type=Path, default=Path("./results"),
                        help="Base output directory")
    parser.add_argument("--run-name", required=True,
                        help="Run name for output directories")
    parser.add_argument("--base-url", default="http://localhost:7790/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="gpt-4o-mini")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    global RETRIEVAL_ROOT, QUERY_ANALYSIS_ROOT, QA_ROOT, JUDGE_ROOT
    global MODEL_NAME, BASE_URL, API_KEY

    RETRIEVAL_ROOT = args.retrieval_root.resolve()
    QUERY_ANALYSIS_ROOT = args.query_root.resolve()
    QA_ROOT = args.output_base.resolve() / "qa" / args.run_name
    JUDGE_ROOT = args.output_base.resolve() / "judge" / args.run_name
    MODEL_NAME = args.model_name
    BASE_URL = args.base_url
    API_KEY = args.api_key

    targets = list(_iter_retrieval_dirs(RETRIEVAL_ROOT))

    QA_ROOT.mkdir(parents=True, exist_ok=True)
    JUDGE_ROOT.mkdir(parents=True, exist_ok=True)

    _write_run_manifest(args, total=len(targets))

    _log(
        f"QA + Judge started: total={len(targets)} "
        f"retrieval_root={RETRIEVAL_ROOT} query_root={QUERY_ANALYSIS_ROOT} "
        f"qa_root={QA_ROOT} judge_root={JUDGE_ROOT}"
    )

    done = 0
    failed = 0
    skipped = 0
    correct = 0
    wrong = 0
    empty_label = 0
    rows: List[Dict[str, Any]] = []
    started_all = time.time()

    _write_status(
        qa_root=QA_ROOT,
        judge_root=JUDGE_ROOT,
        state="running",
        total=len(targets),
        done=done,
        failed=failed,
        skipped=skipped,
    )

    for target_index, (question_type, method, sample_id, retrieval_dir) in enumerate(targets, start=1):
        case_started = time.time()
        question_label = (
            f"[case {target_index}/{len(targets)}] "
            f"question_id={question_type}/{sample_id} | method={method}"
        )

        input_json = QUERY_ANALYSIS_ROOT / question_type / sample_id / "input.json"
        top_records_json = retrieval_dir / "top_records.json"

        qa_dir = QA_ROOT / question_type / method / sample_id
        judge_dir = JUDGE_ROOT / question_type / method / sample_id
        qa_summary_path = qa_dir / "summary.json"
        judge_summary_path = judge_dir / "summary.json"

        _log(f"{question_label} | start")

        if args.resume and qa_summary_path.exists() and judge_summary_path.exists():
            try:
                old_judge = _load_json(judge_summary_path)
                old_label = _clean(old_judge.get("judge_label")).upper()

                skipped += 1
                done += 1
                if old_label == "CORRECT":
                    correct += 1
                elif old_label == "WRONG":
                    wrong += 1
                else:
                    empty_label += 1

                rows.append({
                    "question_type": question_type,
                    "retrieval_method": method,
                    "sample_id": sample_id,
                    "ok": True,
                    "skipped": True,
                    "judge_label": old_label,
                    "qa_dir": str(qa_dir),
                    "judge_dir": str(judge_dir),
                })

                _log(
                    f"{question_label} | skipped existing | "
                    f"label={old_label or 'EMPTY'} | done={done} failed={failed} skipped={skipped}"
                )

                _write_status(
                    qa_root=QA_ROOT,
                    judge_root=JUDGE_ROOT,
                    state="running",
                    total=len(targets),
                    done=done,
                    failed=failed,
                    skipped=skipped,
                    running={
                        "question_type": question_type,
                        "method": method,
                        "sample_id": sample_id,
                        "state": "skipped_existing",
                    },
                )
                continue
            except Exception as exc:
                _log(
                    f"{question_label} | resume check failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        if not input_json.exists() or not top_records_json.exists():
            failed += 1
            err = (
                f"missing input_json or top_records_json: "
                f"input_json={input_json.exists()} top_records_json={top_records_json.exists()}"
            )

            qa_dir.mkdir(parents=True, exist_ok=True)
            judge_dir.mkdir(parents=True, exist_ok=True)

            row = {
                "question_type": question_type,
                "retrieval_method": method,
                "sample_id": sample_id,
                "ok": False,
                "error": err,
                "input_json": str(input_json),
                "top_records_json": str(top_records_json),
                "qa_dir": str(qa_dir),
                "judge_dir": str(judge_dir),
            }
            rows.append(row)
            _write_json(qa_dir / "summary.json", row)
            _write_json(judge_dir / "summary.json", row)

            _log(f"{question_label} | failed before QA: {err}")
            continue

        try:
            input_payload = _load_json(input_json)
            top_records = json.loads(top_records_json.read_text(encoding="utf-8"))
            filtered_records = [_filter_record(r) for r in top_records]

            query = _clean(input_payload.get("question"))
            gold_answer = input_payload.get("answer")
            question_date = _clean(input_payload.get("question_date"))

            _log(
                f"{question_label} | loaded input | "
                f"retrieved_records={len(top_records)} filtered_records={len(filtered_records)}"
            )

            qa_payload = build_qa_payload(
                query=query,
                question_date=question_date,
                retrieved_records=filtered_records,
            )

            qa_dir.mkdir(parents=True, exist_ok=True)
            (qa_dir / "qa_prompt.txt").write_text(qa_payload["prompt"], encoding="utf-8")
            _write_json(qa_dir / "qa_request.json", {
                "model_name": MODEL_NAME,
                "query": query,
                "gold_answer": gold_answer,
                "question_date": question_date,
                "retrieved_records": filtered_records,
                "retrieval_dir": str(retrieval_dir),
            })

            _log(f"{question_label} | QA start")
            qa_resp_json = _chat_with_retry(
                prompt=qa_payload["prompt"],
                stage="QA",
                question_label=question_label,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            qa_raw = _extract_message(qa_resp_json)
            qa_parsed = _parse_answer(qa_raw)

            _write_json(qa_dir / "qa_raw_response.json", qa_resp_json)
            _write_json(qa_dir / "qa_result.json", qa_parsed)

            qa_answer = _clean(qa_parsed.get("answer"))

            _log(
                f"{question_label} | QA done | "
                f"answer_chars={len(qa_answer)}"
            )

            judge_payload = build_judge_payload(
                query=query,
                gold_answer=gold_answer,
                model_answer=qa_answer,
            )

            judge_dir.mkdir(parents=True, exist_ok=True)
            (judge_dir / "judge_prompt.txt").write_text(judge_payload["prompt"], encoding="utf-8")
            _write_json(judge_dir / "judge_request.json", {
                "model_name": MODEL_NAME,
                "query": query,
                "gold_answer": gold_answer,
                "model_answer": qa_answer,
                "qa_summary": str(qa_dir / "summary.json"),
            })

            _log(f"{question_label} | Judge start")
            judge_resp_json = _chat_with_retry(
                prompt=judge_payload["prompt"],
                stage="Judge",
                question_label=question_label,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            judge_raw = _extract_message(judge_resp_json)
            judge_parsed = _parse_judge(judge_raw)
            judge_label = _clean(judge_parsed.get("label")).upper()

            _write_json(judge_dir / "judge_raw_response.json", judge_resp_json)
            _write_json(judge_dir / "judge_result.json", judge_parsed)

            if judge_label == "CORRECT":
                correct += 1
            elif judge_label == "WRONG":
                wrong += 1
            else:
                empty_label += 1

            qa_summary = {
                "question_type": question_type,
                "retrieval_method": method,
                "sample_id": sample_id,
                "query": query,
                "gold_answer": gold_answer,
                "qa_answer": qa_answer,
                "ok": True,
                "error": None,
                "retrieved_record_count": len(top_records),
                "filtered_record_count": len(filtered_records),
                "elapsed_seconds": time.time() - case_started,
            }

            judge_summary = {
                "question_type": question_type,
                "retrieval_method": method,
                "sample_id": sample_id,
                "query": query,
                "gold_answer": gold_answer,
                "qa_answer": qa_answer,
                "judge_label": judge_label,
                "judge_reasoning": _clean(judge_parsed.get("reasoning")),
                "ok": bool(judge_label),
                "error": None if judge_label else "empty_judge_label",
                "retrieved_record_count": len(top_records),
                "filtered_record_count": len(filtered_records),
                "elapsed_seconds": time.time() - case_started,
            }

            _write_json(qa_dir / "summary.json", qa_summary)
            _write_json(judge_dir / "summary.json", judge_summary)

            done += 1
            rows.append({
                "question_type": question_type,
                "retrieval_method": method,
                "sample_id": sample_id,
                "ok": bool(judge_label),
                "error": None if judge_label else "empty_judge_label",
                "judge_label": judge_label,
                "qa_answer": qa_answer,
                "qa_dir": str(qa_dir),
                "judge_dir": str(judge_dir),
                "elapsed_seconds": time.time() - case_started,
            })

            _log(
                f"{question_label} | done | label={judge_label or 'EMPTY'} "
                f"elapsed={time.time() - case_started:.2f}s | "
                f"done={done} failed={failed} skipped={skipped} "
                f"correct={correct} wrong={wrong} empty_label={empty_label}"
            )

        except Exception as exc:
            failed += 1
            err = f"{type(exc).__name__}: {exc}"

            qa_dir.mkdir(parents=True, exist_ok=True)
            judge_dir.mkdir(parents=True, exist_ok=True)

            fail_summary = {
                "question_type": question_type,
                "retrieval_method": method,
                "sample_id": sample_id,
                "ok": False,
                "error": err,
                "qa_dir": str(qa_dir),
                "judge_dir": str(judge_dir),
                "elapsed_seconds": time.time() - case_started,
            }

            _write_json(qa_dir / "summary.json", fail_summary)
            _write_json(judge_dir / "summary.json", fail_summary)
            rows.append(fail_summary)

            _log(f"{question_label} | failed: {err}")

        _write_status(
            qa_root=QA_ROOT,
            judge_root=JUDGE_ROOT,
            state="running",
            total=len(targets),
            done=done,
            failed=failed,
            skipped=skipped,
            running={
                "question_type": question_type,
                "method": method,
                "sample_id": sample_id,
                "retrieval_dir": str(retrieval_dir),
            },
        )

    total_finished = done + failed
    accuracy = (correct / done) if done else 0.0

    final_summary = {
        "state": "completed",
        "retrieval_root": str(RETRIEVAL_ROOT),
        "query_root": str(QUERY_ANALYSIS_ROOT),
        "qa_root": str(QA_ROOT),
        "judge_root": str(JUDGE_ROOT),
        "total": len(targets),
        "finished": total_finished,
        "done": done,
        "failed": failed,
        "skipped": skipped,
        "correct": correct,
        "wrong": wrong,
        "empty_label": empty_label,
        "accuracy_on_done": accuracy,
        "elapsed_seconds": time.time() - started_all,
        "rows": rows,
    }

    _write_json(QA_ROOT / "summary.json", final_summary)
    _write_json(JUDGE_ROOT / "summary.json", final_summary)

    _write_status(
        qa_root=QA_ROOT,
        judge_root=JUDGE_ROOT,
        state="completed",
        total=len(targets),
        done=done,
        failed=failed,
        skipped=skipped,
    )

    _log(
        f"QA + Judge completed: total={len(targets)} done={done} "
        f"failed={failed} skipped={skipped} correct={correct} wrong={wrong} "
        f"empty_label={empty_label} accuracy_on_done={accuracy:.4f} "
        f"elapsed={time.time() - started_all:.2f}s"
    )



'''
    targets = list(_iter_retrieval_dirs(RETRIEVAL_ROOT))
    for question_type, method, sample_id, retrieval_dir in targets:
        input_json = QUERY_ANALYSIS_ROOT / question_type / sample_id / "input.json"
        top_records_json = retrieval_dir / "top_records.json"
        if not input_json.exists() or not top_records_json.exists():
            continue

        input_payload = _load_json(input_json)
        top_records = json.loads(top_records_json.read_text(encoding="utf-8"))
        filtered_records = [_filter_record(r) for r in top_records]

        query = _clean(input_payload.get("question"))
        gold_answer = input_payload.get("answer")
        question_date = _clean(input_payload.get("question_date"))

        qa_payload = build_qa_payload(query=query, question_date=question_date, retrieved_records=filtered_records)
        qa_resp_json = _chat(qa_payload["prompt"])
        qa_raw = _extract_message(qa_resp_json)
        qa_parsed = _parse_answer(qa_raw)

        qa_dir = QA_ROOT / question_type / method / sample_id
        qa_dir.mkdir(parents=True, exist_ok=True)
        (qa_dir / "qa_prompt.txt").write_text(qa_payload["prompt"], encoding="utf-8")
        _write_json(qa_dir / "qa_request.json", {
            "model_name": MODEL_NAME,
            "query": query,
            "retrieved_records": filtered_records,
        })
        _write_json(qa_dir / "qa_raw_response.json", qa_resp_json)
        _write_json(qa_dir / "qa_result.json", qa_parsed)

        judge_payload = build_judge_payload(
            query=query,
            gold_answer=gold_answer,
            model_answer=qa_parsed["answer"],
        )
        judge_resp_json = _chat(judge_payload["prompt"])
        judge_raw = _extract_message(judge_resp_json)
        judge_parsed = _parse_judge(judge_raw)

        judge_dir = JUDGE_ROOT / question_type / method / sample_id
        judge_dir.mkdir(parents=True, exist_ok=True)
        (judge_dir / "judge_prompt.txt").write_text(judge_payload["prompt"], encoding="utf-8")
        _write_json(judge_dir / "judge_request.json", {
            "model_name": MODEL_NAME,
            "query": query,
            "gold_answer": gold_answer,
            "model_answer": qa_parsed["answer"],
        })
        _write_json(judge_dir / "judge_raw_response.json", judge_resp_json)
        _write_json(judge_dir / "judge_result.json", judge_parsed)

        _write_json(qa_dir / "summary.json", {
            "question_type": question_type,
            "retrieval_method": method,
            "sample_id": sample_id,
            "query": query,
            "gold_answer": gold_answer,
            "qa_answer": qa_parsed["answer"],
        })
        _write_json(judge_dir / "summary.json", {
            "question_type": question_type,
            "retrieval_method": method,
            "sample_id": sample_id,
            "query": query,
            "gold_answer": gold_answer,
            "qa_answer": qa_parsed["answer"],
            "judge_label": judge_parsed["label"],
        })
'''

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

try:
    from p2_schema import normalize_parsed_query_p2
except Exception:
    def normalize_parsed_query_p2(parsed, *args, **kwargs):
        return parsed


THIS_FILE = Path(__file__).resolve()
LONGMEM_ROOT = THIS_FILE.parents[1]
PROMPT_FILE = LONGMEM_ROOT / "prompts" / "prompts.py"
DEFAULT_INPUT_ROOT = Path("data/longmemeval_s_cleaned.json")
DEFAULT_OUTPUT_BASE = Path("./results/query_analysis")


def _extract_prompt_constant(name: str) -> str:
    text = PROMPT_FILE.read_text(encoding="utf-8")
    pattern = rf"{name}\s*=\s*\"\"\"(.*?)\"\"\""
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"unable to locate prompt constant: {name}")
    return match.group(1)


QUERY_PROMPT_TEMPLATE = _extract_prompt_constant("LONGMEMEVAL_QUERY_ANALYSIS_PROMPT")


def _safe_json_fragment(text: str) -> Any:
    payload = (text or "").strip()
    if not payload:
        raise ValueError("empty response")
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload).strip()
    try:
        return json.loads(payload)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", payload):
        try:
            parsed, _ = decoder.raw_decode(payload[match.start() :])
            return parsed
        except Exception:
            continue
    raise ValueError("unable to parse JSON")


def _build_prompt(question: str, question_date: str = "") -> str:
    return (
        QUERY_PROMPT_TEMPLATE
        .replace("{question_date}", str(question_date or "").strip())
        .replace("{question}", str(question or "").strip())
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}] {msg}", flush=True)

def _call_chat(
    *,
    session: requests.Session,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    response = session.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _extract_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "").strip()


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"questions file is not a list: {path}")
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            row = {"question": str(row)}
        q = str(row.get("question") or "").strip()
        out.append({"index": i, "question": q, "raw": row})
    return out


def _slugify(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip())
    text = text.strip("_")
    return text or "unknown"

def _question_identity(conv_name: str, item: Dict[str, Any]) -> str:
    """
    LongMemEval normally has question_id and question_type in raw row.
    Fallback to sample_id / index when question_id is missing.
    """
    raw = item.get("raw") or {}
    qtype = str(raw.get("question_type") or conv_name or "unknown").strip() or "unknown"
    qid = raw.get("question_id") or item.get("sample_id") or f"row_{item.get('index', 0)}"
    return f"{qtype}/{_slugify(qid)}"


def _question_type_from_path(path: Path) -> str:
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return ""


def _iter_input_files(input_root: Path) -> List[Path]:
    if input_root.is_file():
        return [input_root]
    return sorted(input_root.glob("longmemeval_s_cleaned__*.json"))


def _load_conversations(input_root: Path) -> List[Dict[str, Any]]:
    """Load LongMemEval rows and group them by question_type.

    Single file: a list with question/question_id/question_type fields.
    Directory: files named longmemeval_s_cleaned__<question_type>.json.
    """
    result: List[Dict[str, Any]] = []
    for file_path in _iter_input_files(input_root):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {file_path}")
        file_qtype = _question_type_from_path(file_path)
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                row = {"question": str(row)}
            question = str(row.get("question") or "").strip()
            qtype = str(file_qtype or row.get("question_type") or "unknown").strip() or "unknown"
            qid = _slugify(row.get("question_id") or f"row_{i}")
            sample_id = f"{i:04d}_{qid}"
            buckets.setdefault(qtype, []).append({"index": i, "sample_id": sample_id, "question": question, "raw": row})
        for qtype, questions in buckets.items():
            result.append({"conv_name": qtype, "questions": questions})
    return result


def run(args: argparse.Namespace) -> Path:
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_base / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    conversations = _load_conversations(args.input_root)
    if args.max_convs > 0:
        conversations = conversations[: args.max_convs]

    _write_json(
        run_root / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "input_root": str(args.input_root),
            "output_root": str(run_root),
            "model_name": args.model_name,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "resume": args.resume,
            "max_convs": args.max_convs,
            "max_questions_per_conv": args.max_questions_per_conv,
            "conv_count": len(conversations),
            "prompt_file": str(PROMPT_FILE),
        },
    )


    session = requests.Session()
    session.trust_env = False

    total = sum(
        len(
            conv_entry["questions"][: args.max_questions_per_conv]
            if args.max_questions_per_conv > 0
            else conv_entry["questions"]
        )
        for conv_entry in conversations
    )

    done = 0
    fail = 0
    summary_rows: List[Dict[str, Any]] = []

    _log(
        f"Query analysis started: "
        f"conversations={len(conversations)} total_questions={total} "
        f"output_root={run_root}"
    )

    for conv_pos, conv_entry in enumerate(conversations, start=1):
        conv_name = conv_entry["conv_name"]
        conv_out = run_root / conv_name
        conv_out.mkdir(parents=True, exist_ok=True)

        questions = conv_entry["questions"]
        if args.max_questions_per_conv > 0:
            questions = questions[: args.max_questions_per_conv]

        _log(
            f"[conv {conv_pos}/{len(conversations)}] "
            f"conv_name={conv_name} | questions={len(questions)} | start"
        )

        for question_pos, item in enumerate(questions, start=1):
            idx = int(item["index"])
            q = item["question"]
            sample_id = str(item.get("sample_id") or f"{idx:04d}")
            question_id = _question_identity(conv_name, item)

            _log(
                f"[question {done + fail + 1}/{total}] "
                f"question_id={question_id} | "
                f"conv={conv_name} | local={question_pos}/{len(questions)} | start"
            )
            out_dir = conv_out / sample_id
            out_dir.mkdir(parents=True, exist_ok=True)
            result_path = out_dir / "result.json"
            if args.resume and result_path.exists():
                try:
                    old = json.loads(result_path.read_text(encoding="utf-8"))
                    if old.get("ok") is True:
                        done += 1
                        summary_rows.append(old)
                        _log(
                            f"[question {done + fail}/{total}] "
                            f"question_id={question_id} | skipped existing | "
                            f"done={done} fail={fail}"
                        )
                        continue
                except Exception as exc:
                    _log(
                        f"[question {done + fail + 1}/{total}] "
                        f"question_id={question_id} | resume check failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            question_date = str(item.get("raw", {}).get("question_date") or "").strip()
            prompt = _build_prompt(q, question_date=question_date)
            _write_json(out_dir / "input.json", item["raw"])
            (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            ok = False
            error = None
            parsed = None
            raw_text = ""
            response_json: Dict[str, Any] | None = None
            started = time.time()

            for attempt in range(1, args.max_retries + 1):
                _log(
                    f"[question {done + fail + 1}/{total}] "
                    f"question_id={question_id} | "
                    f"attempt {attempt}/{args.max_retries}"
                )
                try:
                    response_json = _call_chat(
                        session=session,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        model_name=args.model_name,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    raw_text = _extract_text(response_json)
                    parsed = _safe_json_fragment(raw_text)
                    if not isinstance(parsed, dict):
                        raise ValueError("parsed_response_not_object")

                    parsed = normalize_parsed_query_p2(
                        parsed,
                        question=q,
                        question_date=question_date,
                    )

                    ok = True
                    error = None
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    _log(
                        f"[question {done + fail + 1}/{total}] "
                        f"question_id={question_id} | "
                        f"attempt {attempt}/{args.max_retries} failed: {error}"
                    )
                    if attempt < args.max_retries:
                        time.sleep(min(2 * attempt, 8))

            elapsed = time.time() - started
            if response_json is not None:
                _write_json(out_dir / "raw_response.json", response_json)
            (out_dir / "raw_response.txt").write_text(raw_text, encoding="utf-8")
            if parsed is not None:
                _write_json(out_dir / "parsed.json", parsed)

            one = {
                "conv_name": conv_name,
                "index": idx,
                "question": q,
                "ok": ok,
                "error": error,
                "elapsed_seconds": elapsed,
                "usage": (response_json or {}).get("usage"),
                "output_dir": str(out_dir),
            }
            _write_json(result_path, one)
            summary_rows.append(one)

            if ok:
                done += 1
            else:
                fail += 1
            _log(
                f"[question {done + fail}/{total}] "
                f"question_id={question_id} | done ok={ok} "
                f"elapsed={elapsed:.2f}s | done={done} fail={fail}"
            )
            _write_json(
                run_root / "status.json",
                {
                    "run_root": str(run_root),
                    "total": total,
                    "done": done,
                    "fail": fail,
                    "running": {
                        "conv_name": conv_name,
                        "index": idx,
                        "question_id": question_id,
                        "question_pos": question_pos,
                        "question_count_in_conv": len(questions),
                    },
                    "updated_at": time.time(),
                },
            )

    final = {
        "run_root": str(run_root),
        "total": total,
        "done": done,
        "fail": fail,
        "rows": summary_rows,
    }
    _write_json(run_root / "summary.json", final)
    _log(
        f"Query analysis completed: "
        f"total={total} done={done} fail={fail} output_root={run_root}"
    )

    print(json.dumps(final, ensure_ascii=False, indent=2))
    return run_root

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LongMemEval query analysis grouped by question type.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:7790/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="qwen3-30b-a3b")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-convs", type=int, default=0, help="0 means all")
    parser.add_argument("--max-questions-per-conv", type=int, default=0, help="0 means all")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()


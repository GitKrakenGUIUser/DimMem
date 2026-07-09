#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List


SSP_TYPE = "single-session-preference"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_records(payload: Any) -> tuple[List[Dict[str, Any]], str | None]:
    """
    支持两种常见格式：
    1. 原数据本身就是 list: [{...}, {...}]
    2. 原数据是 dict，case 放在 data / records / examples / items 里

    返回：
    - records
    - container_key: 如果原数据是 dict，则返回对应 key；如果是 list，则返回 None
    """
    if isinstance(payload, list):
        return payload, None

    if isinstance(payload, dict):
        for key in ["data", "records", "examples", "items"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value, key

    raise ValueError(
        "Unsupported JSON format. Expected a list or a dict containing "
        "'data', 'records', 'examples', or 'items'."
    )


def rebuild_payload(original_payload: Any, container_key: str | None, filtered: List[Dict[str, Any]]) -> Any:
    """
    尽量保持原数据集格式：
    - 如果原始是 list，输出仍然是 list
    - 如果原始是 dict，则只替换其中的数据列表，保留其他 metadata 字段
    """
    if container_key is None:
        return filtered

    output = dict(original_payload)
    output[container_key] = filtered
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all SSP / single-session-preference cases from LongMemEval-S."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to longmemeval-s-cleaned.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("longmemeval_s_cleaned_ssp.json"),
        help="Output JSON path.",
    )

    parser.add_argument(
        "--question-type",
        default=SSP_TYPE,
        help="Question type to extract. Default: single-session-preference",
    )

    args = parser.parse_args()

    payload = load_json(args.input)
    records, container_key = get_records(payload)

    total = len(records)
    type_counter = Counter(str(r.get("question_type", "")).strip() for r in records)

    filtered = [
        r for r in records
        if str(r.get("question_type", "")).strip() == args.question_type
    ]

    output_payload = rebuild_payload(payload, container_key, filtered)
    write_json(args.output, output_payload)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "target_question_type": args.question_type,
        "original_total": total,
        "extracted_total": len(filtered),
        "original_question_type_distribution": dict(type_counter),
        "sample_question_ids": [
            str(r.get("question_id", "")).strip()
            for r in filtered[:20]
        ],
    }

    report_path = args.output.with_suffix(".report.json")
    write_json(report_path, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


TARGET_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def as_records(payload: Any) -> List[Dict[str, Any]]:
    """
    Support common LongMemEval formats:
    1. [ {...}, {...} ]
    2. {"data": [ ... ]}
    3. {"records": [ ... ]}
    4. {"examples": [ ... ]}
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "records", "examples", "items"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError("Unsupported dataset format: expected a list or a dict containing data/records/examples/items")


def collect_question_ids(payload: Any) -> Set[str]:
    """
    Collect question_id recursively from an old subset file or any json file.
    """
    ids: Set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            qid = x.get("question_id")
            if isinstance(qid, str) and qid.strip():
                ids.add(qid.strip())
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(payload)
    return ids


def largest_remainder_allocation(
    type_counts: Counter,
    total_n: int,
    allowed_types: List[str],
) -> Dict[str, int]:
    """
    Allocate total_n by original dataset proportions using largest remainder.
    """
    total = sum(type_counts[t] for t in allowed_types)
    if total <= 0:
        raise ValueError("No records found for target question types")

    raw = {
        t: (type_counts[t] / total) * total_n
        for t in allowed_types
    }

    alloc = {
        t: int(math.floor(raw[t]))
        for t in allowed_types
    }

    remainder = total_n - sum(alloc.values())

    ranked = sorted(
        allowed_types,
        key=lambda t: (raw[t] - alloc[t], type_counts[t]),
        reverse=True,
    )

    for t in ranked[:remainder]:
        alloc[t] += 1

    return alloc


def repair_allocation_for_availability(
    alloc: Dict[str, int],
    available_by_type: Dict[str, List[Dict[str, Any]]],
    total_n: int,
) -> Dict[str, int]:
    """
    If a type does not have enough non-excluded records, move the shortage to other types.
    """
    alloc = dict(alloc)

    shortage = 0
    for t, n in list(alloc.items()):
        available = len(available_by_type.get(t, []))
        if n > available:
            shortage += n - available
            alloc[t] = available

    if shortage <= 0:
        return alloc

    capacity = {
        t: len(available_by_type.get(t, [])) - alloc.get(t, 0)
        for t in alloc
    }

    while shortage > 0:
        candidates = [t for t, cap in capacity.items() if cap > 0]
        if not candidates:
            break

        # Fill types with more remaining capacity first.
        candidates.sort(key=lambda t: capacity[t], reverse=True)
        t = candidates[0]
        alloc[t] += 1
        capacity[t] -= 1
        shortage -= 1

    if sum(alloc.values()) != total_n:
        raise ValueError(
            f"Unable to allocate {total_n} samples after exclusions. "
            f"Only allocated {sum(alloc.values())}. "
            f"Please reduce sample size or provide fewer exclusions."
        )

    return alloc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a new LongMemEval-S cleaned 30-case subset without overlapping old question_id."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to full longmemeval-s-cleaned.json",
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        action="append",
        default=[],
        help="Old subset json file(s). All question_id inside will be excluded. Can be used multiple times.",
    )
    parser.add_argument(
        "--exclude-id",
        action="append",
        default=[],
        help="Manually exclude one question_id. Can be used multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("longmemeval_s_cleaned_random30_new_no_overlap.json"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260709,
    )
    parser.add_argument(
        "--types",
        nargs="*",
        default=TARGET_TYPES,
        help="Question types to sample. Default uses the six LongMemEval-S categories.",
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    full_payload = load_json(args.input)
    records = as_records(full_payload)

    exclude_ids: Set[str] = set()

    for path in args.exclude_file:
        if path.exists():
            exclude_payload = load_json(path)
            exclude_ids |= collect_question_ids(exclude_payload)
        else:
            raise FileNotFoundError(f"exclude file not found: {path}")

    for qid in args.exclude_id:
        qid = str(qid).strip()
        if qid:
            exclude_ids.add(qid)

    allowed_types = list(args.types)

    # Count original distribution BEFORE exclusions.
    original_type_counts = Counter(
        str(r.get("question_type", "")).strip()
        for r in records
        if str(r.get("question_type", "")).strip() in allowed_types
    )

    original_alloc = largest_remainder_allocation(
        original_type_counts,
        args.sample_size,
        allowed_types,
    )

    available_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    seen_qids: Set[str] = set()
    duplicate_qids: Set[str] = set()

    for r in records:
        qid = str(r.get("question_id", "")).strip()
        qtype = str(r.get("question_type", "")).strip()

        if not qid or qtype not in allowed_types:
            continue

        if qid in seen_qids:
            duplicate_qids.add(qid)
            continue

        seen_qids.add(qid)

        if qid in exclude_ids:
            continue

        available_by_type[qtype].append(r)

    final_alloc = repair_allocation_for_availability(
        original_alloc,
        available_by_type,
        args.sample_size,
    )

    sampled: List[Dict[str, Any]] = []

    for qtype in allowed_types:
        pool = list(available_by_type.get(qtype, []))
        n = final_alloc.get(qtype, 0)

        if n <= 0:
            continue

        if len(pool) < n:
            raise ValueError(
                f"Not enough records for type={qtype}: need {n}, available {len(pool)}"
            )

        sampled.extend(rng.sample(pool, n))

    rng.shuffle(sampled)

    # Safety checks.
    sampled_ids = [
        str(r.get("question_id", "")).strip()
        for r in sampled
    ]

    overlap = sorted(set(sampled_ids) & exclude_ids)
    if overlap:
        raise RuntimeError(f"Sample has excluded question_id overlap: {overlap[:10]}")

    if len(sampled) != args.sample_size:
        raise RuntimeError(f"Expected {args.sample_size} samples, got {len(sampled)}")

    if len(set(sampled_ids)) != len(sampled_ids):
        raise RuntimeError("Sample contains duplicate question_id")

    dump_json(args.output, sampled)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "excluded_question_id_count": len(exclude_ids),
        "duplicate_question_id_in_full_dataset_count": len(duplicate_qids),
        "allowed_types": allowed_types,
        "original_dataset_distribution": {
            t: original_type_counts[t] for t in allowed_types
        },
        "target_allocation_by_original_ratio": original_alloc,
        "final_allocation_after_exclusion": final_alloc,
        "sampled_distribution": dict(Counter(r["question_type"] for r in sampled)),
        "sampled_question_ids": sampled_ids,
    }

    report_path = args.output.with_suffix(".report.json")
    dump_json(report_path, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

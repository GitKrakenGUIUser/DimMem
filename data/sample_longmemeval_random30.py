#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 LongMemEval 数据集中按 question_type 原始占比分层随机抽取 N 个 case。

默认：
- 输入：longmemeval_s_cleaned.json
- 输出：longmemeval_s_cleaned_random30.json
- 抽样数量：30
- 保持原数据 JSON 结构中的每条 case 字段不变
"""

import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


TARGET_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
]


def load_json_or_jsonl(path: Path):
    """
    支持两种格式：
    1. 标准 JSON：通常 LongMemEval 是一个 list[dict]
    2. JSONL：每行一个 dict
    """
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Input file is empty: {path}")

    # 优先按 JSON 读取
    try:
        data = json.loads(text)
        return data, "json"
    except json.JSONDecodeError:
        pass

    # 再尝试 JSONL
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e

    return rows, "jsonl"


def extract_records(data):
    """
    尽量保持兼容：
    - 如果 data 本身是 list，则直接视为 records
    - 如果 data 是 dict，并且其中某个 value 是 list[dict] 且包含 question_type，则取该 list
    """
    if isinstance(data, list):
        return data, None

    if isinstance(data, dict):
        for key, value in data.items():
            if (
                isinstance(value, list)
                and value
                and isinstance(value[0], dict)
                and "question_type" in value[0]
            ):
                return value, key

    raise ValueError(
        "Cannot find records. Expected a list of dicts, "
        "or a dict containing a list of dicts with field 'question_type'."
    )


def largest_remainder_allocation(type_counts, total_n):
    """
    按最大余数法分配每个类型应抽取的数量：
    1. 先计算每类理论数量 = 原始占比 * total_n
    2. 取 floor
    3. 剩余名额给小数部分最大的类别

    这样可以保证总数严格等于 total_n，并且比例最接近原始数据集。
    """
    total = sum(type_counts.values())
    if total == 0:
        raise ValueError("No samples found for target question types.")

    raw = {
        qtype: type_counts[qtype] * total_n / total
        for qtype in TARGET_TYPES
        if type_counts[qtype] > 0
    }

    allocation = {qtype: int(value) for qtype, value in raw.items()}
    used = sum(allocation.values())
    remaining = total_n - used

    # 按小数部分从大到小补齐
    remainders = sorted(
        raw.items(),
        key=lambda x: (x[1] - int(x[1]), type_counts[x[0]]),
        reverse=True,
    )

    for qtype, _ in remainders[:remaining]:
        allocation[qtype] += 1

    # 防止某类数据不足，做一次安全修正
    overflow = 0
    for qtype in list(allocation.keys()):
        if allocation[qtype] > type_counts[qtype]:
            overflow += allocation[qtype] - type_counts[qtype]
            allocation[qtype] = type_counts[qtype]

    if overflow > 0:
        candidates = sorted(
            allocation.keys(),
            key=lambda t: type_counts[t] - allocation[t],
            reverse=True,
        )
        for qtype in candidates:
            can_add = type_counts[qtype] - allocation[qtype]
            add = min(can_add, overflow)
            allocation[qtype] += add
            overflow -= add
            if overflow == 0:
                break

    if sum(allocation.values()) != total_n:
        raise ValueError(
            f"Allocation failed. Expected {total_n}, got {sum(allocation.values())}."
        )

    return allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        default="longmemeval_s_cleaned.json",
        help="Input LongMemEval JSON file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="longmemeval_s_cleaned_random30.json",
        help="Output sampled JSON file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of cases to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--exclude-abstention",
        action="store_true",
        help="Exclude samples whose question_id ends with '_abs'. Default: keep them.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    random.seed(args.seed)

    data, fmt = load_json_or_jsonl(input_path)
    records, record_key = extract_records(data)

    # 只保留目标 6 类
    valid_records = []
    skipped = []

    for item in records:
        qtype = item.get("question_type")
        qid = str(item.get("question_id", ""))

        if qtype not in TARGET_TYPES:
            skipped.append(item)
            continue

        if args.exclude_abstention and qid.endswith("_abs"):
            skipped.append(item)
            continue

        valid_records.append(item)

    if len(valid_records) < args.n:
        raise ValueError(
            f"Not enough valid records. Need {args.n}, found {len(valid_records)}."
        )

    by_type = defaultdict(list)
    for item in valid_records:
        by_type[item["question_type"]].append(item)

    type_counts = Counter(item["question_type"] for item in valid_records)
    allocation = largest_remainder_allocation(type_counts, args.n)

    sampled = []
    for qtype in TARGET_TYPES:
        k = allocation.get(qtype, 0)
        if k > 0:
            sampled.extend(random.sample(by_type[qtype], k))

    # 打乱最终顺序，避免输出按类型聚集
    random.shuffle(sampled)

    # 保持原数据结构：
    # - 如果原始是 list，则输出 list
    # - 如果原始是 dict 且 records 在某个 key 下，则只替换该 key 的内容
    if isinstance(data, list):
        output_data = sampled
    else:
        output_data = dict(data)
        output_data[record_key] = sampled

    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Sample size: {len(sampled)}")
    print(f"Random seed: {args.seed}")
    print()
    print("Original type counts:")
    for qtype in TARGET_TYPES:
        print(f"  {qtype}: {type_counts[qtype]}")
    print()
    print("Sample allocation:")
    sampled_counts = Counter(item["question_type"] for item in sampled)
    for qtype in TARGET_TYPES:
        print(f"  {qtype}: {sampled_counts[qtype]}")


if __name__ == "__main__":
    main()

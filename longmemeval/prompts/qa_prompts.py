from __future__ import annotations

from typing import Any, Dict, Iterable, List

LONGMEMEVAL_QA_PROMPT = """You are answering a question about a user's long-term memory.

You will receive:
1. The user's original question.
2. The question timestamp (question_date).
3. A small set of retrieved dimensional memory records.

Use only the retrieved memories as evidence.

Important rules:
- Prefer explicit evidence over guesses.
- If the evidence is insufficient, answer exactly: I don't know.
- Use source_time and dimension.time for temporal reasoning.
- Use dimension.location for location questions.
- Use dimension.reason and dimension.purpose for why/purpose questions.
- Use dimension.keywords and memory_type to disambiguate entities and preferences.
- Use assistant_reply when the question asks what the assistant previously said, suggested, recommended, explained, or asked.
- If records conflict, prefer the record with the latest source_time, unless the question explicitly asks about an older time.
- For current/latest/now questions, prefer the most recent relevant record.
- For count/list/order questions, inspect all retrieved records and avoid counting duplicate memories.
- Do not invent missing numbers, dates, places, people, preferences, or entities.

Output format:
Reasoning: <brief evidence-grounded reasoning>
Answer: <final concise answer>

Retrieved Memories:
{{retrieved_memories}}

Now answer the question:
User Question: {{query}}
Question Date: {{question_date}}
"""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(_clean(v) for v in value if _clean(v))
    if isinstance(value, dict):
        pairs = []
        for k, v in value.items():
            if _clean(v):
                pairs.append(f"{k}={v}")
        return "; ".join(pairs)
    return _clean(value)


def _append_if_present(lines: List[str], key: str, value: Any) -> None:
    text = _format_value(value)
    if text:
        lines.append(f"{key}: {text}")


def _memory_lines(record: Dict[str, Any], rank: int) -> List[str]:
    lines: List[str] = [f"[{rank}]"]

    _append_if_present(lines, "memory_type", record.get("memory_type"))
    _append_if_present(lines, "source_time", record.get("source_time"))
    _append_if_present(lines, "content", record.get("content"))

    dimension = record.get("dimension")
    if not isinstance(dimension, dict):
        dimension = {}

    _append_if_present(lines, "dimension.time", dimension.get("time"))
    _append_if_present(lines, "dimension.location", dimension.get("location"))
    _append_if_present(lines, "dimension.reason", dimension.get("reason"))
    _append_if_present(lines, "dimension.purpose", dimension.get("purpose"))
    _append_if_present(lines, "dimension.keywords", dimension.get("keywords"))
    _append_if_present(lines, "dimension.status", dimension.get("status"))
    _append_if_present(lines, "dimension.valid_from", dimension.get("valid_from"))
    _append_if_present(lines, "dimension.valid_to", dimension.get("valid_to"))
    _append_if_present(lines, "dimension.is_current", dimension.get("is_current"))

    _append_if_present(lines, "assistant_reply", record.get("assistant_reply"))

    # Debug metadata is useful for auditing retrieval errors.
    _append_if_present(lines, "retrieval_method", record.get("retrieval_method"))
    _append_if_present(lines, "retrieval_score", record.get("retrieval_score"))
    _append_if_present(lines, "rerank_score", record.get("rerank_score"))
    _append_if_present(lines, "source_boundary_id", record.get("source_boundary_id"))

    return lines


def format_retrieved_memories(records: Iterable[Dict[str, Any]]) -> str:
    blocks: List[str] = []

    for idx, record in enumerate(records, start=1):
        blocks.append("\n".join(_memory_lines(record, idx)))

    return "\n\n".join(blocks) if blocks else "[No retrieved memories]"


def build_qa_prompt(*, query: str, question_date: str, retrieved_records: Iterable[Dict[str, Any]]) -> str:
    return (
        LONGMEMEVAL_QA_PROMPT.replace("{{query}}", _clean(query))
        .replace("{{question_date}}", _clean(question_date) or "[unknown]")
        .replace("{{retrieved_memories}}", format_retrieved_memories(retrieved_records))
    )


def build_qa_payload(*, query: str, question_date: str, retrieved_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(retrieved_records)

    return {
        "query": _clean(query),
        "question_date": _clean(question_date),
        "retrieved_records": records,
        "prompt": build_qa_prompt(query=query, question_date=question_date, retrieved_records=records),
    }


__all__ = [
    "LONGMEMEVAL_QA_PROMPT",
    "format_retrieved_memories",
    "build_qa_prompt",
    "build_qa_payload",
]

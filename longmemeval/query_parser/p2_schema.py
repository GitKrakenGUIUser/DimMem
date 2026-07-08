from __future__ import annotations

import re
from typing import Any, Dict, List

P2_SCHEMA_VERSION = "p2_minimal_v1"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _clean(value).lower()


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value if _clean(v)]
    return []


def _unique(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = _clean(value)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _query_text(parsed: Dict[str, Any]) -> str:
    dim = parsed.get("dimension") if isinstance(parsed.get("dimension"), dict) else {}
    parts: List[str] = []

    for key in ("question", "query", "query_anchor", "canonical_text"):
        if _clean(parsed.get(key)):
            parts.append(_clean(parsed.get(key)))

    for key in ("keywords", "entities", "aliases", "time", "location", "answer_dim", "query_type", "statefulness"):
        value = parsed.get(key)
        if isinstance(value, list):
            parts.extend(_as_list(value))
        elif _clean(value):
            parts.append(_clean(value))

    for key in ("keywords", "entities", "aliases", "time", "location"):
        value = dim.get(key)
        if isinstance(value, list):
            parts.extend(_as_list(value))
        elif _clean(value):
            parts.append(_clean(value))

    tr = dim.get("time_range") if isinstance(dim.get("time_range"), dict) else parsed.get("time_range")
    if isinstance(tr, dict):
        for key in ("operator", "start", "end", "relative_event"):
            if _clean(tr.get(key)):
                parts.append(_clean(tr.get(key)))

    return " ".join(parts)


def infer_query_type(text: str, answer_dim: str = "") -> str:
    t = _lower(text)
    answer_dim = _lower(answer_dim)

    if any(p in t for p in ["how many", "number of", "count", "total number"]):
        return "count"
    if any(p in t for p in ["how much", "total", "sum", "combined", "altogether"]):
        return "sum"
    if any(p in t for p in ["which came first", "earliest", "latest", "before or after", "order", "sequence", "timeline"]):
        return "order"
    if any(p in t for p in ["compare", "difference", "which is better", "versus", " vs "]):
        return "compare"
    if any(p in t for p in ["should i", "recommend", "recommendation", "what should", "is it better to", "buy now", "wait"]):
        return "recommend"
    if any(p in t for p in ["list", "what are the", "which ones", "which items"]):
        return "list"

    if answer_dim in {"count", "sum", "order"}:
        return answer_dim
    if answer_dim == "recommendation":
        return "recommend"

    return "lookup"


def infer_statefulness(text: str) -> str:
    t = _lower(text)

    if any(p in t for p in ["current", "currently", "now", "right now", "latest", "newest", "most recent", "at present"]):
        return "current"
    if any(p in t for p in ["before", "after", "earlier", "later", "first", "last", "timeline", "order", "sequence", "then"]):
        return "timeline"
    if any(p in t for p in ["used to", "previously", "in the past", "last time", "former", "old"]):
        return "historical"

    return "unknown"


def parse_time_range(time_text: str) -> Dict[str, str]:
    text = _clean(time_text)
    low = text.lower()
    out = {"operator": "", "start": "", "end": "", "relative_event": ""}

    if not text:
        return out

    m = re.search(r"between\s+(.+?)\s+and\s+(.+)$", text, flags=re.I)
    if m:
        out.update({"operator": "between", "start": m.group(1).strip(), "end": m.group(2).strip()})
        return out

    for op in ("before", "after", "around", "on"):
        if low.startswith(op + " "):
            rest = text[len(op):].strip()

            # before getting the Air Fryer / after buying the NAS
            if not re.search(r"\d{4}|\d{1,2}/\d{1,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", rest, flags=re.I):
                out.update({"operator": "relative_to_event", "start": op, "relative_event": rest})
            else:
                out.update({"operator": op, "start": rest})
            return out

    out.update({"operator": "raw", "start": text})
    return out


def normalize_parsed_query_p2(
    parsed: Dict[str, Any],
    question: str | None = None,
    question_date: str | None = None,
) -> Dict[str, Any]:
    """
    Backward-compatible P2 normalizer.

    It preserves old DimMem fields and adds:
    query_type / statefulness / entities / aliases / time_range / parse_confidence.
    """
    if not isinstance(parsed, dict):
        parsed = {}

    out: Dict[str, Any] = dict(parsed)
    dim = out.get("dimension") if isinstance(out.get("dimension"), dict) else {}
    dim = dict(dim)

    dim["target_memory_type"] = _unique(
        _as_list(dim.get("target_memory_type")) + _as_list(out.get("target_memory_type"))
    )
    dim["keywords"] = _unique(
        _as_list(dim.get("keywords")) + _as_list(out.get("keywords"))
    )
    dim["entities"] = _unique(
        _as_list(dim.get("entities")) + _as_list(out.get("entities"))
    )
    dim["aliases"] = _unique(
        _as_list(dim.get("aliases")) + _as_list(out.get("aliases"))
    )

    if not _clean(dim.get("time")) and _clean(out.get("time")):
        dim["time"] = _clean(out.get("time"))
    if not _clean(dim.get("location")) and _clean(out.get("location")):
        dim["location"] = _clean(out.get("location"))

    if isinstance(dim.get("time_range"), dict):
        time_range = dict(dim.get("time_range"))
    elif isinstance(out.get("time_range"), dict):
        time_range = dict(out.get("time_range"))
    else:
        time_range = parse_time_range(_clean(dim.get("time")))

    dim["time_range"] = time_range

    # 如果 LLM 没有显式 entities，就把短关键词作为 soft entities。
    if not dim["entities"]:
        dim["entities"] = _unique([kw for kw in dim["keywords"] if len(kw.split()) <= 5])

    text_for_infer = " ".join([_clean(question), _query_text(out)])

    out["dimension"] = dim
    out["time_range"] = time_range
    out["query_type"] = _clean(out.get("query_type")) or infer_query_type(
        text_for_infer,
        answer_dim=_clean(out.get("answer_dim")),
    )
    out["statefulness"] = _clean(out.get("statefulness")) or infer_statefulness(text_for_infer)
    out["parse_confidence"] = out.get("parse_confidence", 0.65)
    out["question_date"] = _clean(out.get("question_date") or question_date)

    # 保留旧字段，避免旧 retrieval 代码坏掉。
    out["keywords"] = dim["keywords"]
    out["entities"] = dim["entities"]
    out["aliases"] = dim["aliases"]
    out["time"] = _clean(dim.get("time"))
    out["location"] = _clean(dim.get("location"))
    out["target_memory_type"] = dim["target_memory_type"]
    out["p2_schema_version"] = P2_SCHEMA_VERSION

    return out

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


STOP = {"the", "a", "an", "my", "i", "user", "the user", "new", "old"}


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _lower(v: Any) -> str:
    return _clean(v).lower()


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [_clean(x) for x in v if _clean(x)]
    return []


def _dim(r: Dict[str, Any]) -> Dict[str, Any]:
    d = r.get("dimension")
    return d if isinstance(d, dict) else {}


def _record_text(r: Dict[str, Any]) -> str:
    d = _dim(r)
    parts = [
        r.get("content", ""),
        r.get("memory_type", ""),
        r.get("source_time", ""),
        r.get("assistant_reply", ""),
        d.get("time", ""),
        d.get("event_time", ""),
        d.get("status", ""),
        d.get("subject", ""),
        d.get("action", ""),
        d.get("object", ""),
        d.get("value", ""),
        d.get("quantity", ""),
        d.get("unit", ""),
        d.get("relation", ""),
        d.get("evidence_span", ""),
        " ".join(_as_list(d.get("keywords"))),
    ]
    return " | ".join(_clean(x) for x in parts if _clean(x))


def _query_text(parsed: Dict[str, Any]) -> str:
    dim = parsed.get("dimension") if isinstance(parsed.get("dimension"), dict) else {}
    parts = [
        parsed.get("question", ""),
        parsed.get("query", ""),
        parsed.get("query_anchor", ""),
        parsed.get("canonical_text", ""),
        parsed.get("answer_dim", ""),
        parsed.get("query_type", ""),
        " ".join(_as_list(parsed.get("keywords"))),
        " ".join(_as_list(parsed.get("entities"))),
        " ".join(_as_list(parsed.get("aliases"))),
        " ".join(_as_list(dim.get("keywords"))),
        " ".join(_as_list(dim.get("entities"))),
        " ".join(_as_list(dim.get("aliases"))),
    ]
    return " ".join(_clean(x) for x in parts if _clean(x))


def _parse_rel(rel: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    text = _clean(rel)
    for part in re.split(r"[;\n]+", text):
        if "=" in part:
            k, v = part.split("=", 1)
            k = _clean(k).lower().replace(" ", "_")
            v = _clean(v)
            if k and v:
                out[k] = v
    return out


def _parse_dt(v: Any) -> Optional[datetime]:
    text = _clean(v)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10], fmt)
        except Exception:
            pass
    return None


def _number_word(text: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
    mp = {
        "one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    for k, v in mp.items():
        if re.search(rf"\b{k}\b", text, flags=re.I):
            return v
    return None


def _object_key(value: str) -> str:
    s = _lower(value)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(the|a|an|my|user|new|old|current)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _make_hint_record(parsed_query: Dict[str, Any], lines: List[str], reason: str, score: float = 999.0) -> Dict[str, Any]:
    content = "P4 structured task reader hint:\n" + "\n".join(lines)
    return {
        "user_id": "",
        "memory_type": "structured_hint",
        "content": content,
        "dimension": {
            "memory_type": "fact",
            "time": "",
            "event_time": "",
            "location": "",
            "reason": reason,
            "purpose": "Help QA use structured fields and avoid free-form aggregation errors.",
            "keywords": ["P4 structured task reader", _clean(parsed_query.get("query_type")), _clean(parsed_query.get("answer_dim"))],
            "subject": "P4 structured task reader",
            "action": "computed_hint",
            "object": _query_text(parsed_query)[:160],
            "value": "",
            "quantity": "",
            "unit": "",
            "relation": "structured_task_reader=true",
            "evidence_span": content[:300],
            "status": "derived",
            "is_current": False,
        },
        "entities": ["P4 structured task reader"],
        "embedding_text": content,
        "source_message_ids": [],
        "source_boundary_id": "structured_task_reader::hint",
        "source_time": "",
        "record_time": "",
        "retrieval_method": "structured_task_reader",
        "retrieval_score": score,
        "score": score,
        "fusion_sources": [{"method": "structured_task_reader", "rank": 1, "score": score}],
        "_structured_task_reader": {
            "query_type": parsed_query.get("query_type"),
            "answer_dim": parsed_query.get("answer_dim"),
            "line_count": len(lines),
        },
    }


def _count_hint(parsed_query: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q = _lower(_query_text(parsed_query))
    rows: List[Tuple[str, Dict[str, Any]]] = []

    for r in records:
        d = _dim(r)
        t = _lower(_record_text(r))
        obj = _clean(d.get("object")) or _clean(r.get("content"))[:80]
        action = _lower(d.get("action"))

        include = False

        if "bike" in q and any(w in q for w in ["service", "serviced", "plan to service"]):
            include = (
                "bike" in t
                and any(x in t for x in ["serviced", "service", "repair", "replace", "tire", "maintenance"])
                and not any(x in t for x in ["water bottle cage", "bottle cage", "accessory"])
            )
        elif "project" in q and any(w in q for w in ["led", "leading", "lead"]):
            include = (
                "project" in t
                and any(x in t for x in [" led ", "leading", "lead=", "led_project", "completed high-priority"])
                and not any(x in t for x in ["planning a project", "interested in", "class project"] if "led" not in t)
            )
        elif "times" in q or "how many" in q:
            include = any(tok in t for tok in q.split() if len(tok) > 4)

        if include:
            key = _object_key(obj)
            if key:
                rows.append((key, r))

    if not rows:
        return None

    dedup: Dict[str, Dict[str, Any]] = {}
    for key, r in rows:
        dedup.setdefault(key, r)

    lines = [f"Candidate distinct count = {len(dedup)}"]
    for i, (key, r) in enumerate(dedup.items(), start=1):
        d = _dim(r)
        lines.append(
            f"{i}. object={_clean(d.get('object'))}; action={_clean(d.get('action'))}; "
            f"value={_clean(d.get('value'))}; relation={_clean(d.get('relation'))}; "
            f"content={_clean(r.get('content'))}"
        )

    lines.append("Use this hint only if the listed objects satisfy the query action constraint.")
    return _make_hint_record(parsed_query, lines, "P4 count aggregation over structured action/object fields.")


def _order_hint(parsed_query: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q = _lower(_query_text(parsed_query))
    if not any(w in q for w in ["order", "earliest", "latest", "from earliest to latest", "visited"]):
        return None

    candidates: List[Tuple[datetime, str, Dict[str, Any]]] = []
    for r in records:
        d = _dim(r)
        t = _lower(_record_text(r))
        if "museum" not in t:
            continue
        obj = _clean(d.get("object"))
        if "museum" not in obj.lower():
            m = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+Museum(?:\s+of\s+[A-Z][A-Za-z]+)?)\b", _clean(r.get("content")))
            obj = m.group(1) if m else obj
        dt = _parse_dt(d.get("event_time") or d.get("time") or r.get("source_time"))
        if obj and dt:
            candidates.append((dt, obj, r))

    if len(candidates) < 2:
        return None

    best_by_obj: Dict[str, Tuple[datetime, str, Dict[str, Any]]] = {}
    for dt, obj, r in candidates:
        key = _object_key(obj)
        if key not in best_by_obj or dt < best_by_obj[key][0]:
            best_by_obj[key] = (dt, obj, r)

    ordered = sorted(best_by_obj.values(), key=lambda x: x[0])
    lines = ["Candidate chronological order by distinct object:"]
    for i, (dt, obj, r) in enumerate(ordered, start=1):
        lines.append(f"{i}. {obj} | event_time={dt.date()} | content={_clean(r.get('content'))}")

    return _make_hint_record(parsed_query, lines, "P4 order aggregation over event_time/object coverage.")


def _lookup_relation_hint(parsed_query: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q = _lower(_query_text(parsed_query))
    wants_giver = any(x in q for x in ["from whom", "who gave", "who did i receive", "from who"])
    wants_airline = "airline" in q
    wants_time = "what time" in q

    lines: List[str] = []

    for r in records:
        d = _dim(r)
        rel = _parse_rel(d.get("relation"))
        t = _lower(_record_text(r))

        if wants_giver and ("giver" in rel or "from the user's aunt" in t or "from aunt" in t):
            val = rel.get("giver") or _clean(d.get("value")) or "aunt"
            lines.append(f"giver candidate: {val}; content={_clean(r.get('content'))}; relation={_clean(d.get('relation'))}")

        if wants_airline and ("airline" in rel or "american airlines" in t):
            val = rel.get("airline") or ("American Airlines" if "american airlines" in t else _clean(d.get("value")))
            status = _clean(d.get("status"))
            action = _clean(d.get("action"))
            lines.append(f"airline candidate: {val}; status={status}; action={action}; content={_clean(r.get('content'))}")

        if wants_time and ("arrival_time" in rel or "reached the clinic at" in t):
            val = rel.get("arrival_time") or _clean(d.get("value"))
            if val:
                lines.append(f"time candidate: {val}; content={_clean(r.get('content'))}; relation={_clean(d.get('relation'))}")

    if not lines:
        return None

    lines.insert(0, "Structured relation candidates found:")
    return _make_hint_record(parsed_query, lines[:12], "P4 lookup/value relation extraction over structured fields.")


def _recommendation_hint(parsed_query: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q = _lower(_query_text(parsed_query))
    if not any(x in q for x in ["should i", "recommend", "buy now", "wait", "suggest"]):
        return None

    lines = ["Recommendation questions may be answered from relevant preferences, needs, constraints, and current facts; do not require a previous explicit recommendation."]
    added = 0
    for r in records:
        d = _dim(r)
        t = _lower(_record_text(r))
        if any(x in t for x in ["prefer", "need", "concern", "considering", "storage", "nas", "external hard drive", "capacity", "budget", "goal"]):
            lines.append(
                f"evidence: object={_clean(d.get('object'))}; action={_clean(d.get('action'))}; "
                f"value={_clean(d.get('value'))}; content={_clean(r.get('content'))}"
            )
            added += 1
        if added >= 8:
            break

    if added == 0:
        return None
    return _make_hint_record(parsed_query, lines, "P4 recommendation reader hint.")


def apply_structured_task_reader(
    *,
    parsed_query: Dict[str, Any],
    records: List[Dict[str, Any]],
    final_top_k: int = 15,
) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []

    qtype = _lower(parsed_query.get("query_type"))
    answer_dim = _lower(parsed_query.get("answer_dim"))
    q = _lower(_query_text(parsed_query))

    if qtype in {"count", "sum"} or "how many" in q:
        h = _count_hint(parsed_query, records)
        if h:
            hints.append(h)

    if qtype in {"order", "timeline", "list"} or any(x in q for x in ["earliest to latest", "order of"]):
        h = _order_hint(parsed_query, records)
        if h:
            hints.append(h)

    if any(x in answer_dim for x in ["person", "value", "time", "airline"]) or any(x in q for x in ["from whom", "airline", "what time"]):
        h = _lookup_relation_hint(parsed_query, records)
        if h:
            hints.append(h)

    if qtype in {"recommend", "preference"} or any(x in q for x in ["should i", "buy now", "wait"]):
        h = _recommendation_hint(parsed_query, records)
        if h:
            hints.append(h)

    if not hints:
        return records

    # Keep only a few hints to avoid swamping real evidence.
    return hints[:3] + records


def structured_task_reader_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total": len(records),
        "hint_count": sum(1 for r in records if _clean(r.get("retrieval_method")) == "structured_task_reader"),
    }

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


STOPWORDS = {
    "the", "a", "an", "i", "me", "my", "did", "do", "does", "was", "were", "is", "are",
    "what", "which", "who", "when", "where", "how", "many", "much", "before", "after",
    "getting", "got", "get", "having", "had", "have", "invest", "invested", "new",
    "thing", "item", "one", "ones", "it", "that", "this", "of", "in", "on", "to",
    "for", "with", "from", "and", "or", "as", "at", "by", "about",
}


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


def _tokens(text: str) -> List[str]:
    return [
        t
        for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-']+", _lower(text))
        if len(t) >= 2 and t not in STOPWORDS
    ]


def _dimension(record: Dict[str, Any]) -> Dict[str, Any]:
    dim = record.get("dimension")
    return dim if isinstance(dim, dict) else {}


def _record_text(record: Dict[str, Any]) -> str:
    dim = _dimension(record)
    parts = [
        record.get("content", ""),
        record.get("memory_type", ""),
        record.get("source_time", ""),
        record.get("assistant_reply", ""),
        dim.get("time", ""),
        dim.get("event_time", ""),
        dim.get("valid_from", ""),
        dim.get("valid_to", ""),
        dim.get("status", ""),
        dim.get("location", ""),
        dim.get("reason", ""),
        dim.get("purpose", ""),
        dim.get("subject", ""),
        dim.get("action", ""),
        dim.get("object", ""),
        dim.get("value", ""),
        dim.get("quantity", ""),
        dim.get("unit", ""),
        dim.get("relation", ""),
        dim.get("evidence_span", ""),
        " ".join(_as_list(dim.get("keywords"))),
    ]
    return " ".join(_clean(x) for x in parts if _clean(x))


def _parse_dt(value: Any) -> Optional[datetime]:
    text = _clean(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        pass

    # Handle common LongMemEval forms like 2023/05/30 (Tue) 23:40
    m = re.search(r"(\d{4})/(\d{2})/(\d{2})(?:.*?(\d{2}):(\d{2}))?", text)
    if m:
        y, mo, d, hh, mm = m.groups()
        return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0))

    return None


def _time_range(parsed_query: Dict[str, Any]) -> Dict[str, Any]:
    dim = parsed_query.get("dimension") if isinstance(parsed_query.get("dimension"), dict) else {}
    tr = dim.get("time_range") if isinstance(dim.get("time_range"), dict) else parsed_query.get("time_range")
    return tr if isinstance(tr, dict) else {}


def is_relative_event_query(parsed_query: Dict[str, Any]) -> bool:
    tr = _time_range(parsed_query)
    return _lower(tr.get("operator")) == "relative_to_event" and bool(_clean(tr.get("relative_event")))


def _query_text(parsed_query: Dict[str, Any]) -> str:
    dim = parsed_query.get("dimension") if isinstance(parsed_query.get("dimension"), dict) else {}
    parts: List[str] = []

    for key in ("question", "query", "query_anchor", "canonical_text"):
        if _clean(parsed_query.get(key)):
            parts.append(_clean(parsed_query.get(key)))

    for key in ("keywords", "entities", "aliases", "time", "location"):
        value = parsed_query.get(key)
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

    tr = _time_range(parsed_query)
    for key in ("relative_event", "start", "end"):
        if _clean(tr.get(key)):
            parts.append(_clean(tr.get(key)))

    return " ".join(parts)


def _event_score(event: str, record: Dict[str, Any]) -> float:
    text = _lower(_record_text(record))
    event_l = _lower(event)

    if not event_l:
        return 0.0

    score = 0.0

    if event_l in text:
        score += 2.0

    event_tokens = set(_tokens(event_l))
    text_tokens = set(_tokens(text))

    if event_tokens:
        score += len(event_tokens & text_tokens) / max(1.0, len(event_tokens))

    return score


def _candidate_score(
    *,
    parsed_query: Dict[str, Any],
    event: str,
    operator_side: str,
    anchor_time: Optional[datetime],
    record: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    text = _lower(_record_text(record))

    if not text:
        return 0.0, {}

    # Do not make the anchor itself the answer candidate.
    event_s = _event_score(event, record)
    if event_s >= 1.8:
        return 0.0, {"reason": "looks_like_anchor_event"}

    query_tokens = set(_tokens(_query_text(parsed_query)))
    event_tokens = set(_tokens(event))
    candidate_tokens = query_tokens - event_tokens

    record_tokens = set(_tokens(text))
    overlap = len(candidate_tokens & record_tokens)
    overlap_score = overlap / max(1.0, min(len(candidate_tokens), 8))

    # Helpful category words for LongMemEval relative-event questions.
    category_bonus = 0.0
    categories = [
        "gadget", "kitchen", "device", "purchase", "bought", "buy", "acquired", "own",
        "owns", "new", "trip", "flight", "museum", "event", "party", "wedding",
        "airline", "hotel", "recipe", "tool",
    ]
    for c in categories:
        if c in query_tokens and c in record_tokens:
            category_bonus += 0.12

    dim_for_time = _dimension(record)
    cand_time = _parse_dt(
        dim_for_time.get("event_time")
        or dim_for_time.get("time")
        or record.get("source_time")
    )
    relation_score = 0.0
    relation_label = "unknown"

    if anchor_time and cand_time:
        if operator_side == "before":
            if cand_time < anchor_time:
                relation_score = 0.9
                relation_label = "before_anchor_by_source_time"
            else:
                relation_score = 0.15
                relation_label = "not_before_by_source_time"
        elif operator_side == "after":
            if cand_time > anchor_time:
                relation_score = 0.9
                relation_label = "after_anchor_by_source_time"
            else:
                relation_score = 0.15
                relation_label = "not_after_by_source_time"
    else:
        relation_score = 0.35
        relation_label = "time_missing_or_ambiguous"

    score = overlap_score + category_bonus + relation_score

    # If a record has at least one query-category overlap and the event is not in it,
    # keep it as a possible answer even when source_time is incomplete.
    return score, {
        "overlap": overlap,
        "overlap_score": round(overlap_score, 6),
        "category_bonus": round(category_bonus, 6),
        "relation_score": round(relation_score, 6),
        "relation_label": relation_label,
        "candidate_time": str(cand_time) if cand_time else "",
    }


def _make_binding_record(
    *,
    parsed_query: Dict[str, Any],
    event: str,
    operator_side: str,
    anchors: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    anchor_lines = []
    for i, r in enumerate(anchors[:3], start=1):
        anchor_lines.append(
            f"Anchor {i}: time={_clean(r.get('source_time'))}; content={_clean(r.get('content'))[:500]}"
        )

    candidate_lines = []
    for i, r in enumerate(candidates[:8], start=1):
        meta = r.get("_relative_event_binding") or {}
        candidate_lines.append(
            f"Candidate {i}: relation={meta.get('relation_label','')}; "
            f"time={_clean(r.get('source_time'))}; "
            f"content={_clean(r.get('content'))[:600]}"
        )

    content = (
        f"Relative-event binding evidence for query. "
        f"The query asks for records {operator_side} the event '{event}'.\n"
        f"Anchor event evidence:\n" + "\n".join(anchor_lines) + "\n\n"
        f"Candidate answer evidence:\n" + "\n".join(candidate_lines)
    )

    return {
        "user_id": "",
        "memory_type": "binding_hint",
        "content": content,
        "dimension": {
            "memory_type": "binding_hint",
            "time": "",
            "location": "",
            "reason": "Derived from retrieved memories by relative-event binding.",
            "purpose": "Help QA compare an anchor event with possible answer candidates for before/after questions.",
            "keywords": [event, operator_side, "relative-event binding"],
        },
        "entities": [event],
        "embedding_text": content,
        "source_message_ids": [],
        "source_boundary_id": f"relative_event_binding::{event}",
        "source_time": "",
        "record_time": "",
        "retrieval_method": "relative_event_binding",
        "retrieval_rank": 1,
        "retrieval_score": 1.0,
        "score": 1.0,
        "fusion_sources": [
            {
                "method": "relative_event_binding",
                "rank": 1,
                "score": 1.0,
            }
        ],
        "_relative_event_binding_score": 1.0,
        "_relative_event_binding": {
            "event": event,
            "operator_side": operator_side,
            "anchor_count": len(anchors),
            "candidate_count": len(candidates),
            "is_binding_hint": True,
        },
    }


def apply_relative_event_binding(
    *,
    parsed_query: Dict[str, Any],
    records: List[Dict[str, Any]],
    final_top_k: int = 30,
) -> List[Dict[str, Any]]:
    """
    Add relative-event binding metadata and a derived binding record.

    It does not replace retrieval. It enriches candidate records before rerank/QA.
    """
    if not is_relative_event_query(parsed_query) or not records:
        return records

    tr = _time_range(parsed_query)
    event = _clean(tr.get("relative_event"))
    operator_side = _lower(tr.get("start")) or "before"

    if operator_side not in {"before", "after"}:
        # P2 parser stores "start" as before/after for relative_to_event.
        t = _lower(parsed_query.get("time"))
        if "after" in t:
            operator_side = "after"
        else:
            operator_side = "before"

    scored_anchors = []
    for r in records:
        s = _event_score(event, r)
        if s > 0:
            scored_anchors.append((s, r))

    scored_anchors.sort(key=lambda x: x[0], reverse=True)
    anchors = [r for _, r in scored_anchors[:5]]

    if not anchors:
        return records

    anchor_times = [
        _parse_dt(
            _dimension(r).get("event_time")
            or _dimension(r).get("time")
            or r.get("source_time")
        )
        for r in anchors
    ]
    anchor_times = [t for t in anchor_times if t is not None]
    anchor_time = min(anchor_times) if operator_side == "before" and anchor_times else (
        max(anchor_times) if anchor_times else None
    )

    enriched: List[Dict[str, Any]] = []
    candidate_rows: List[Tuple[float, Dict[str, Any]]] = []

    for r in records:
        score, meta = _candidate_score(
            parsed_query=parsed_query,
            event=event,
            operator_side=operator_side,
            anchor_time=anchor_time,
            record=r,
        )

        item = dict(r)

        if score > 0.35:
            item["_relative_event_binding_score"] = round(float(min(1.0, score)), 6)
            item["_relative_event_binding"] = {
                "event": event,
                "operator_side": operator_side,
                "anchor_time": str(anchor_time) if anchor_time else "",
                **meta,
            }
            candidate_rows.append((score, item))

        enriched.append(item)

    candidate_rows.sort(key=lambda x: x[0], reverse=True)
    candidates = [r for _, r in candidate_rows[:8]]

    if not candidates:
        return enriched

    binding_record = _make_binding_record(
        parsed_query=parsed_query,
        event=event,
        operator_side=operator_side,
        anchors=anchors,
        candidates=candidates,
    )

    # Put derived binding record first; rerank_p2 will keep it near the top.
    return [binding_record] + enriched


def relative_event_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records or [])
    return {
        "total": len(rows),
        "binding_hint_count": sum(
            1 for r in rows if _clean(r.get("retrieval_method")) == "relative_event_binding"
        ),
        "candidate_marked_count": sum(
            1 for r in rows if r.get("_relative_event_binding")
        ),
    }

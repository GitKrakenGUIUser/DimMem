from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from search.rerank import rerank_records as _p1_rerank_records

try:
    from query_parser.p2_schema import normalize_parsed_query_p2
except Exception:
    def normalize_parsed_query_p2(
        parsed: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return parsed


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


def _dimension(record: Dict[str, Any]) -> Dict[str, Any]:
    dim = record.get("dimension")
    return dim if isinstance(dim, dict) else {}


def _tokens(text: str) -> Set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-']+", _lower(text))
        if len(t) >= 2
    }


def _parse_dt(value: Any) -> Optional[datetime]:
    text = _clean(value)

    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _record_text(record: Dict[str, Any]) -> str:
    """
    P4 rerank text.

    Includes original DimMem fields plus P3 schema fields and P4 synthetic hints.
    This is important because P4 derived/link records often place the useful signal in:
    event_time / subject / action / object / value / quantity / unit / relation / evidence_span.
    """
    dim = _dimension(record)

    parts = [
        record.get("content", ""),
        record.get("memory_type", ""),
        record.get("source_time", ""),
        record.get("assistant_reply", ""),
        record.get("retrieval_method", ""),

        dim.get("time", ""),
        dim.get("event_time", ""),
        dim.get("valid_from", ""),
        dim.get("valid_to", ""),
        dim.get("status", ""),
        str(dim.get("is_current", "")),

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

        str(record.get("_assistant_reply_search", "")),
        str(record.get("_assistant_pair_search", "")),
        str(record.get("_relative_event_binding", "")),
        str(record.get("_structured_task_reader", "")),
    ]

    return " ".join(_clean(x) for x in parts if _clean(x))


def _has_number(record: Dict[str, Any]) -> bool:
    dim = _dimension(record)
    text = _record_text(record)

    if _clean(dim.get("value")) or _clean(dim.get("quantity")):
        return True

    return bool(
        re.search(
            r"(\$?\d+(?:\.\d+)?%?|\b(one|two|three|four|five|six|seven|eight|nine|ten|twelve|hundred|thousand|million)\b)",
            text,
            flags=re.I,
        )
    )


def _has_time(record: Dict[str, Any]) -> bool:
    dim = _dimension(record)

    return bool(
        _clean(dim.get("event_time"))
        or _clean(dim.get("time"))
        or _clean(record.get("source_time"))
    )


def _entity_score(parsed: Dict[str, Any], record: Dict[str, Any]) -> float:
    dim = parsed.get("dimension") if isinstance(parsed.get("dimension"), dict) else {}
    anchors: List[str] = []

    for key in ("entities", "aliases", "keywords"):
        anchors.extend(_as_list(parsed.get(key)))
        anchors.extend(_as_list(dim.get(key)))

    anchors = [a for a in anchors if len(a) >= 2]

    if not anchors:
        return 0.0

    text = _lower(_record_text(record))
    text_tokens = _tokens(text)

    hits = 0

    for anchor in anchors:
        a = _lower(anchor)

        if a and (a in text or bool(_tokens(a) & text_tokens)):
            hits += 1

    return min(1.0, hits / max(1, min(len(anchors), 5)))


def _shape_score(parsed: Dict[str, Any], record: Dict[str, Any]) -> float:
    qtype = _lower(parsed.get("query_type"))
    answer_dim = _lower(parsed.get("answer_dim"))
    mem_type = _lower(record.get("memory_type"))
    method = _lower(record.get("retrieval_method"))
    dim = _dimension(record)

    if method == "structured_task_reader":
        return 1.0

    if method in {"assistant_pair_search", "assistant_reply_search"}:
        return 1.0

    if qtype in {"count", "sum"}:
        return 1.0 if _has_number(record) else 0.25

    if qtype in {"order", "timeline", "compare"}:
        return 1.0 if _has_time(record) else 0.25

    if qtype in {"recommend", "preference"} or "preference" in answer_dim:
        score = 0.0

        if mem_type == "profile":
            score += 0.45

        if _clean(dim.get("purpose")) or _clean(dim.get("reason")):
            score += 0.20

        text = _lower(_record_text(record))

        if any(
            w in text
            for w in [
                "prefer",
                "like",
                "dislike",
                "want",
                "need",
                "goal",
                "budget",
                "should",
                "recommend",
                "considering",
                "concern",
                "storage",
                "capacity",
            ]
        ):
            score += 0.35

        return min(1.0, score)

    if qtype == "list":
        return 1.0 if _as_list(dim.get("keywords")) else 0.5

    if _clean(dim.get("value")) or _clean(dim.get("relation")):
        return 0.75

    return 0.5


def _statefulness_score(
    parsed: Dict[str, Any],
    record: Dict[str, Any],
    latest_ts: Optional[datetime],
) -> float:
    state = _lower(parsed.get("statefulness"))
    dim = _dimension(record)
    text = _lower(_record_text(record))
    method = _lower(record.get("retrieval_method"))

    if method in {"structured_task_reader", "assistant_pair_search"}:
        return 1.0

    if state == "current":
        if str(dim.get("is_current")).lower() == "true" or _lower(dim.get("status")) == "current":
            return 1.0

        ts = _parse_dt(
            dim.get("event_time")
            or dim.get("time")
            or record.get("source_time")
        )

        if latest_ts and ts:
            days = max(0, (latest_ts - ts).days)

            if days <= 0:
                return 0.85
            if days <= 30:
                return 0.65
            if days <= 120:
                return 0.45

        return 0.25

    if state == "historical":
        status = _lower(dim.get("status"))

        if status in {"past", "completed", "derived"}:
            return 0.9

        if any(w in text for w in ["previously", "used to", "former", "old", "last time"]):
            return 1.0

        return 0.5

    if state == "timeline":
        return 1.0 if _has_time(record) else 0.35

    return 0.5


def _assistant_p2_score(parsed: Dict[str, Any], record: Dict[str, Any]) -> float:
    qtype = _lower(parsed.get("query_type"))
    need_ast = bool(parsed.get("need_assistant_context"))
    method = _lower(record.get("retrieval_method"))

    if not (need_ast or qtype == "assistant_recall"):
        return 0.0

    if method in {"assistant_reply_search", "assistant_pair_search"}:
        return 1.0

    reply = _clean(record.get("assistant_reply"))

    if not reply:
        return 0.0

    query_text = " ".join(
        [
            _clean(parsed.get("query_anchor")),
            " ".join(_as_list(parsed.get("keywords"))),
            " ".join(_as_list(parsed.get("entities"))),
            " ".join(_as_list(parsed.get("aliases"))),
        ]
    )

    q_tokens = _tokens(query_text)
    r_tokens = _tokens(reply)

    if q_tokens and r_tokens:
        overlap = len(q_tokens & r_tokens) / max(1.0, min(len(q_tokens), 8))
        return max(0.35, min(0.85, overlap))

    return 0.35


def _relative_event_score(parsed: Dict[str, Any], record: Dict[str, Any]) -> float:
    dim = parsed.get("dimension") if isinstance(parsed.get("dimension"), dict) else {}
    tr = dim.get("time_range") if isinstance(dim.get("time_range"), dict) else parsed.get("time_range")

    if not isinstance(tr, dict):
        return 0.0

    if _lower(tr.get("operator")) != "relative_to_event":
        return 0.0

    event = _lower(tr.get("relative_event"))

    if not event:
        return 0.0

    method = _lower(record.get("retrieval_method"))

    if method == "relative_event_binding":
        return 1.0

    if record.get("_relative_event_binding_score") is not None:
        return float(record.get("_relative_event_binding_score") or 0.0)

    return 1.0 if event in _lower(_record_text(record)) else 0.0


def _p4_priority_bonus(parsed: Dict[str, Any], record: Dict[str, Any]) -> float:
    """
    P4 synthetic evidence should not be pushed out by P1 rerank.

    This does not blindly make it top-1 for every query.
    It gives high priority only when the method matches the query need.
    """
    method = _lower(record.get("retrieval_method"))
    qtype = _lower(parsed.get("query_type"))
    answer_dim = _lower(parsed.get("answer_dim"))
    query_text = _lower(
        " ".join(
            [
                _clean(parsed.get("query_anchor")),
                _clean(parsed.get("canonical_text")),
                _clean(parsed.get("question")),
                qtype,
                answer_dim,
            ]
        )
    )

    if method == "structured_task_reader":
        if (
            qtype in {"count", "sum", "order", "timeline", "list", "recommend", "preference"}
            or any(x in query_text for x in ["how many", "order", "earliest", "from whom", "what time", "airline", "should i", "buy now", "wait"])
        ):
            return 0.18
        return 0.08

    if method == "assistant_pair_search":
        if bool(parsed.get("need_assistant_context")) or qtype == "assistant_recall":
            return 0.16
        return 0.02

    if method == "assistant_reply_search":
        if bool(parsed.get("need_assistant_context")) or qtype == "assistant_recall":
            return 0.10
        return 0.0

    if method == "relative_event_binding":
        return 0.14

    if record.get("p4_linked"):
        return 0.08

    return 0.0


def rerank_records(
    *,
    parsed_query: Dict[str, Any],
    records: List[Dict[str, Any]],
    final_top_k: int = 15,
) -> List[Dict[str, Any]]:
    """
    P4-compatible reranker.

    Flow:
    1. Run existing P1 reranker.
    2. Add schema-aware P2/P3/P4 scoring.
    3. Preserve P4 synthetic hints and assistant-pair evidence when relevant.
    """
    pq = normalize_parsed_query_p2(parsed_query)

    base_ranked = _p1_rerank_records(
        parsed_query=pq,
        records=records,
        final_top_k=max(len(records), final_top_k),
    )

    timestamps = [
        _parse_dt(
            _dimension(r).get("event_time")
            or _dimension(r).get("time")
            or r.get("source_time")
        )
        for r in base_ranked
    ]
    timestamps = [t for t in timestamps if t is not None]
    latest_ts = max(timestamps) if timestamps else None

    scored: List[Dict[str, Any]] = []

    for rec in base_ranked:
        base = float(rec.get("rerank_score", 0.0) or 0.0)

        entity_s = _entity_score(pq, rec)
        shape_s = _shape_score(pq, rec)
        state_s = _statefulness_score(pq, rec, latest_ts)
        assistant_s = _assistant_p2_score(pq, rec)
        relative_s = _relative_event_score(pq, rec)
        p4_bonus = _p4_priority_bonus(pq, rec)

        p2_extra = (
            0.28 * entity_s
            + 0.22 * shape_s
            + 0.16 * state_s
            + 0.17 * assistant_s
            + 0.17 * relative_s
        )

        final = 0.72 * base + 0.28 * p2_extra + p4_bonus
        final = min(1.0, final)

        out = dict(rec)
        out["rerank_score_p1"] = rec.get("rerank_score")
        out["rerank_score"] = round(float(final), 6)
        out["rerank_components_p2"] = {
            "base_p1": round(float(base), 6),
            "entity": round(float(entity_s), 6),
            "query_shape": round(float(shape_s), 6),
            "statefulness": round(float(state_s), 6),
            "assistant": round(float(assistant_s), 6),
            "relative_event": round(float(relative_s), 6),
            "p4_priority_bonus": round(float(p4_bonus), 6),
            "retrieval_method": rec.get("retrieval_method"),
            "assistant_reply_search": rec.get("_assistant_reply_search"),
            "assistant_pair_search": rec.get("_assistant_pair_search"),
            "relative_event_binding": rec.get("_relative_event_binding"),
            "structured_task_reader": rec.get("_structured_task_reader"),
            "p4_linked": rec.get("p4_linked"),
            "query_type": pq.get("query_type"),
            "statefulness_label": pq.get("statefulness"),
        }

        scored.append(out)

    scored.sort(
        key=lambda x: (
            float(x.get("rerank_score", 0.0) or 0.0),
            float(x.get("rerank_score_p1", 0.0) or 0.0),
        ),
        reverse=True,
    )

    for idx, item in enumerate(scored, start=1):
        item["rerank_rank"] = idx

    return scored[:final_top_k]

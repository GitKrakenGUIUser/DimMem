from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _content_key(record: Dict[str, Any]) -> str:
    content = _lower(record.get("content"))
    if content:
        return "content:" + hashlib.md5(content.encode("utf-8")).hexdigest()

    marker = "|".join(
        [
            _clean(record.get("source_boundary_id")),
            _clean(record.get("source_time")),
            str(record.get("dimension") or {}),
        ]
    )
    return "fallback:" + hashlib.md5(marker.encode("utf-8")).hexdigest()


def _parse_dt(value: Any) -> Optional[datetime]:
    text = _clean(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _tokens(text: str) -> List[str]:
    return [
        t
        for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-']+", _lower(text))
        if len(t) >= 2
    ]


def _query_text(parsed_query: Dict[str, Any]) -> str:
    parts: List[str] = []

    for key in ("question", "query", "query_anchor", "canonical_text"):
        if _clean(parsed_query.get(key)):
            parts.append(_clean(parsed_query.get(key)))

    dim = parsed_query.get("dimension")
    if isinstance(dim, dict):
        for key in ("keywords", "entities", "aliases", "time", "location"):
            value = dim.get(key)
            if isinstance(value, list):
                parts.extend(_as_list(value))
            elif _clean(value):
                parts.append(_clean(value))

    for key in ("keywords", "entities", "aliases", "time", "location", "answer_dim"):
        value = parsed_query.get(key)
        if isinstance(value, list):
            parts.extend(_as_list(value))
        elif _clean(value):
            parts.append(_clean(value))

    return " ".join(parts)


def _query_keywords(parsed_query: Dict[str, Any]) -> List[str]:
    kws: List[str] = []

    for key in ("keywords", "entities", "aliases"):
        kws.extend(_as_list(parsed_query.get(key)))

    dim = parsed_query.get("dimension")
    if isinstance(dim, dict):
        for key in ("keywords", "entities", "aliases"):
            kws.extend(_as_list(dim.get(key)))

    seen = set()
    out: List[str] = []
    for kw in kws:
        marker = kw.lower()
        if marker and marker not in seen:
            seen.add(marker)
            out.append(kw)
    return out


def _query_time(parsed_query: Dict[str, Any]) -> str:
    if _clean(parsed_query.get("time")):
        return _clean(parsed_query.get("time"))
    dim = parsed_query.get("dimension")
    if isinstance(dim, dict):
        return _clean(dim.get("time"))
    return ""


def _query_location(parsed_query: Dict[str, Any]) -> str:
    if _clean(parsed_query.get("location")):
        return _clean(parsed_query.get("location"))
    dim = parsed_query.get("dimension")
    if isinstance(dim, dict):
        return _clean(dim.get("location"))
    return ""


def _answer_dim(parsed_query: Dict[str, Any]) -> str:
    return _lower(parsed_query.get("answer_dim") or parsed_query.get("answer_field"))


def _is_current_query(parsed_query: Dict[str, Any]) -> bool:
    text = _lower(_query_text(parsed_query))
    markers = [
        "current",
        "currently",
        "now",
        "latest",
        "most recent",
        "newest",
        "right now",
        "at present",
    ]
    return any(m in text for m in markers)


def _normalize_route_scores(records: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    scores_by_route: Dict[str, List[float]] = {}

    for rec in records:
        sources = rec.get("fusion_sources")
        if isinstance(sources, list) and sources:
            for src in sources:
                if not isinstance(src, dict):
                    continue
                route = _clean(src.get("method")) or _clean(rec.get("retrieval_method")) or "unknown"
                try:
                    score = float(src.get("score", 0.0) or 0.0)
                except Exception:
                    score = 0.0
                scores_by_route.setdefault(route, []).append(score)
        else:
            route = _clean(rec.get("retrieval_method")) or "unknown"
            try:
                score = float(rec.get("score", rec.get("retrieval_score", 0.0)) or 0.0)
            except Exception:
                score = 0.0
            scores_by_route.setdefault(route, []).append(score)

    ranges: Dict[str, Tuple[float, float]] = {}
    for route, vals in scores_by_route.items():
        if vals:
            ranges[route] = (min(vals), max(vals))
        else:
            ranges[route] = (0.0, 0.0)
    return ranges


def _norm(value: float, min_max: Tuple[float, float]) -> float:
    lo, hi = min_max
    if hi <= lo:
        return 1.0 if value > 0 else 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _route_score(record: Dict[str, Any], ranges: Dict[str, Tuple[float, float]]) -> float:
    sources = record.get("fusion_sources")
    vals: List[float] = []

    if isinstance(sources, list) and sources:
        for src in sources:
            if not isinstance(src, dict):
                continue
            route = _clean(src.get("method")) or "unknown"
            try:
                raw = float(src.get("score", 0.0) or 0.0)
            except Exception:
                raw = 0.0
            vals.append(_norm(raw, ranges.get(route, (0.0, 0.0))))
    else:
        route = _clean(record.get("retrieval_method")) or "unknown"
        try:
            raw = float(record.get("score", record.get("retrieval_score", 0.0)) or 0.0)
        except Exception:
            raw = 0.0
        vals.append(_norm(raw, ranges.get(route, (0.0, 0.0))))

    return max(vals) if vals else 0.0


def _keyword_overlap_score(parsed_query: Dict[str, Any], record: Dict[str, Any]) -> float:
    q_text = _query_text(parsed_query)
    q_tokens = set(_tokens(q_text))

    for kw in _query_keywords(parsed_query):
        q_tokens.update(_tokens(kw))

    if not q_tokens:
        return 0.0

    dim = _dimension(record)
    record_text_parts = [
        record.get("content", ""),
        record.get("memory_type", ""),
        dim.get("time", ""),
        dim.get("location", ""),
        dim.get("reason", ""),
        dim.get("purpose", ""),
        " ".join(_as_list(dim.get("keywords"))),
        record.get("assistant_reply", ""),
    ]
    r_tokens = set(_tokens(" ".join(_clean(x) for x in record_text_parts)))

    if not r_tokens:
        return 0.0

    hit = len(q_tokens & r_tokens)
    return min(1.0, hit / max(1, min(len(q_tokens), 8)))


def _field_match_score(parsed_query: Dict[str, Any], record: Dict[str, Any]) -> float:
    dim = _dimension(record)
    score = 0.0
    weight = 0.0

    q_time = _lower(_query_time(parsed_query))
    if q_time:
        weight += 1.0
        r_time = _lower(dim.get("time") or record.get("source_time"))
        if q_time in r_time or r_time in q_time:
            score += 1.0

    q_loc = _lower(_query_location(parsed_query))
    if q_loc:
        weight += 1.0
        r_loc = _lower(dim.get("location"))
        if q_loc in r_loc or r_loc in q_loc:
            score += 1.0

    target_type = _lower(parsed_query.get("target_memory_type"))
    if not target_type:
        dim_query = parsed_query.get("dimension")
        if isinstance(dim_query, dict):
            target_type = _lower(dim_query.get("target_memory_type"))

    if target_type:
        weight += 0.7
        r_type = _lower(record.get("memory_type"))
        if target_type in r_type or r_type in target_type:
            score += 0.7

    return score / weight if weight > 0 else 0.0


def _answer_field_score(parsed_query: Dict[str, Any], record: Dict[str, Any]) -> float:
    answer_dim = _answer_dim(parsed_query)
    if not answer_dim:
        return 0.0

    dim = _dimension(record)

    field_map = {
        "time": dim.get("time") or record.get("source_time"),
        "date": dim.get("time") or record.get("source_time"),
        "location": dim.get("location"),
        "place": dim.get("location"),
        "reason": dim.get("reason"),
        "purpose": dim.get("purpose"),
        "preference": record.get("content"),
        "content": record.get("content"),
        "assistant": record.get("assistant_reply"),
        "assistant_reply": record.get("assistant_reply"),
    }

    for key, value in field_map.items():
        if key in answer_dim and _clean(value):
            return 1.0

    return 0.0


def _assistant_score(parsed_query: Dict[str, Any], record: Dict[str, Any]) -> float:
    need_ast = bool(parsed_query.get("need_assistant_context"))
    if not need_ast:
        return 0.0
    return 1.0 if _clean(record.get("assistant_reply")) else 0.0


def _recency_score(record: Dict[str, Any], latest_ts: Optional[datetime]) -> float:
    if latest_ts is None:
        return 0.0
    ts = _parse_dt(record.get("source_time"))
    if ts is None:
        return 0.0

    delta_days = max(0, (latest_ts - ts).days)
    if delta_days <= 0:
        return 1.0
    if delta_days <= 7:
        return 0.8
    if delta_days <= 30:
        return 0.6
    if delta_days <= 90:
        return 0.4
    return 0.2


def _multi_route_score(record: Dict[str, Any]) -> float:
    sources = record.get("fusion_sources")
    if isinstance(sources, list):
        routes = {_clean(s.get("method")) for s in sources if isinstance(s, dict)}
        routes.discard("")
        return min(1.0, len(routes) / 3.0)
    return 0.0


def dedup_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []

    for rec in records:
        key = _content_key(rec)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)

    return out


def rerank_records(
    *,
    parsed_query: Dict[str, Any],
    records: List[Dict[str, Any]],
    final_top_k: int = 15,
) -> List[Dict[str, Any]]:
    """
    Lightweight global reranker after DimMem tri-route retrieval.

    It does not change DimMem's framework:
    - still uses BM25 / structured / dense candidates;
    - only reorders and truncates candidates before QA;
    - keeps all original fields for debugging.
    """
    candidates = dedup_records(records)
    ranges = _normalize_route_scores(candidates)

    latest_ts = None
    if _is_current_query(parsed_query):
        timestamps = [_parse_dt(r.get("source_time")) for r in candidates]
        timestamps = [t for t in timestamps if t is not None]
        latest_ts = max(timestamps) if timestamps else None

    scored: List[Dict[str, Any]] = []
    for rec in candidates:
        route_s = _route_score(rec, ranges)
        kw_s = _keyword_overlap_score(parsed_query, rec)
        field_s = _field_match_score(parsed_query, rec)
        answer_s = _answer_field_score(parsed_query, rec)
        assistant_s = _assistant_score(parsed_query, rec)
        recency_s = _recency_score(rec, latest_ts)
        multi_s = _multi_route_score(rec)

        rerank_score = (
            0.35 * route_s
            + 0.25 * kw_s
            + 0.15 * field_s
            + 0.10 * answer_s
            + 0.07 * assistant_s
            + 0.05 * multi_s
            + 0.03 * recency_s
        )

        item = dict(rec)
        item["rerank_score"] = round(float(rerank_score), 6)
        item["rerank_components"] = {
            "route": round(float(route_s), 6),
            "keyword_overlap": round(float(kw_s), 6),
            "field_match": round(float(field_s), 6),
            "answer_field": round(float(answer_s), 6),
            "assistant": round(float(assistant_s), 6),
            "multi_route": round(float(multi_s), 6),
            "recency": round(float(recency_s), 6),
        }
        scored.append(item)

    scored.sort(
        key=lambda x: (
            float(x.get("rerank_score", 0.0) or 0.0),
            float(x.get("score", x.get("retrieval_score", 0.0)) or 0.0),
        ),
        reverse=True,
    )

    for idx, item in enumerate(scored, start=1):
        item["rerank_rank"] = idx

    return scored[:final_top_k]

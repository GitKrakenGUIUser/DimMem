from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


STOP = {
    "the", "a", "an", "i", "me", "my", "you", "your", "what", "which", "who",
    "when", "where", "how", "did", "do", "does", "was", "were", "is", "are",
    "before", "after", "previous", "previously", "chat", "content", "options",
    "suggest", "suggested", "recommend", "recommended", "tell", "said", "say",
}


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _lower(v: Any) -> str:
    return _clean(v).lower()


def _tokens(text: str) -> List[str]:
    return [
        t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-'.]+", _lower(text))
        if len(t) >= 2 and t not in STOP
    ]


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [_clean(x) for x in v if _clean(x)]
    return []


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return _lower(v) in {"1", "true", "yes", "y", "on"}


def _query_parts(parsed_query: Dict[str, Any]) -> List[str]:
    dim = parsed_query.get("dimension") if isinstance(parsed_query.get("dimension"), dict) else {}
    parts: List[str] = []

    for key in ("question", "query", "query_anchor", "canonical_text", "answer_dim", "query_type"):
        if _clean(parsed_query.get(key)):
            parts.append(_clean(parsed_query.get(key)))

    for key in ("keywords", "entities", "aliases"):
        parts.extend(_as_list(parsed_query.get(key)))
        parts.extend(_as_list(dim.get(key)))

    tr = dim.get("time_range") if isinstance(dim.get("time_range"), dict) else parsed_query.get("time_range")
    if isinstance(tr, dict):
        for key in ("relative_event", "start", "end"):
            if _clean(tr.get(key)):
                parts.append(_clean(tr.get(key)))

    return parts


def should_search_assistant_pairs(parsed_query: Dict[str, Any]) -> bool:
    qtype = _lower(parsed_query.get("query_type"))
    if qtype == "assistant_recall":
        return True
    if _truthy(parsed_query.get("need_assistant_context")):
        text = " ".join(_query_parts(parsed_query)).lower()
        # Avoid over-interfering with current advice questions such as "Should I buy a NAS now?"
        if any(x in text for x in ["should i", "do you think i should", "would you recommend that i", "buy now", "wait"]):
            return False
        return True

    text = " ".join(_query_parts(parsed_query)).lower()
    return any(x in text for x in ["what did you recommend", "what did you suggest", "you previously", "you said before"])


def _score_pair(query_text: str, pair: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    pair_text = _clean(pair.get("pair_text")) or (
        _clean(pair.get("previous_user_message")) + "\n" + _clean(pair.get("assistant_reply"))
    )

    q_tokens = set(_tokens(query_text))
    p_tokens = set(_tokens(pair_text))

    token_hits = len(q_tokens & p_tokens)
    token_score = token_hits / max(1.0, math.sqrt(len(q_tokens) + 1))

    phrase_hits = []
    for part in re.split(r"[;,\n]+", query_text):
        phrase = _clean(part)
        if len(phrase) >= 4 and phrase.lower() in pair_text.lower():
            phrase_hits.append(phrase)

    item_hits = []
    for item in _as_list(pair.get("answer_items")):
        if item.lower() in query_text.lower() or any(tok in _tokens(item) for tok in q_tokens):
            item_hits.append(item)

    score = token_score + 0.9 * len(phrase_hits) + 0.5 * len(item_hits)

    # Strong signal for rare/important terms.
    rare_terms = [t for t in q_tokens if len(t) >= 8 or "." in t or "-" in t]
    rare_hits = [t for t in rare_terms if t in p_tokens or t in pair_text.lower()]
    score += 0.7 * len(rare_hits)

    return score, {
        "token_hits": token_hits,
        "phrase_hits": phrase_hits[:10],
        "item_hits": item_hits[:10],
        "rare_hits": rare_hits[:10],
    }


def search_assistant_pairs(
    *,
    parsed_query: Dict[str, Any],
    memory_dir: Path,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    if not should_search_assistant_pairs(parsed_query):
        return []

    path = memory_dir / "assistant_pair_index.json"
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    pairs = payload.get("pairs") or []
    if not isinstance(pairs, list):
        return []

    query_text = " ".join(_query_parts(parsed_query))
    rows: List[Dict[str, Any]] = []

    for pair in pairs:
        if not isinstance(pair, dict):
            continue

        assistant_reply = _clean(pair.get("assistant_reply"))
        pair_text = _clean(pair.get("pair_text"))
        if not assistant_reply and not pair_text:
            continue

        score, components = _score_pair(query_text, pair)
        if score <= 0:
            continue

        uid = _clean(pair.get("uid"))
        record = {
            "user_id": memory_dir.name,
            "memory_type": "assistant",
            "content": (
                "Assistant pair-search evidence.\n"
                f"Previous user message: {_clean(pair.get('previous_user_message'))}\n"
                f"Assistant reply: {assistant_reply}"
            ),
            "dimension": {
                "memory_type": "assistant",
                "time": _clean(pair.get("source_time")),
                "event_time": _clean(pair.get("source_time")),
                "location": "",
                "reason": "Direct assistant pair search over previous_user_message + assistant_reply.",
                "purpose": "Recover what the assistant previously recommended, suggested, explained, or listed.",
                "keywords": _query_parts(parsed_query)[:20],
                "subject": "assistant",
                "action": "replied",
                "object": _clean(pair.get("previous_user_message"))[:120],
                "value": ", ".join(_as_list(pair.get("answer_items"))[:8]),
                "quantity": "",
                "unit": "",
                "relation": "assistant_pair_search=true",
                "evidence_span": assistant_reply[:300],
                "status": "past",
                "is_current": False,
            },
            "entities": _query_parts(parsed_query)[:20],
            "embedding_text": pair_text,
            "source_message_ids": [uid],
            "source_boundary_id": f"assistant_pair_search::{uid}",
            "source_time": _clean(pair.get("source_time")),
            "record_time": "",
            "assistant_reply": assistant_reply,
            "assistant_uid": uid,
            "session_id": _clean(pair.get("session_id")),
            "session_local_user_index": int(pair.get("session_local_user_index", 0) or 0),
            "retrieval_method": "assistant_pair_search",
            "retrieval_score": float(score),
            "score": float(score),
            "fusion_sources": [{"method": "assistant_pair_search", "rank": 0, "score": float(score)}],
            "_assistant_pair_search": {
                "score_components": components,
                "answer_items": _as_list(pair.get("answer_items")),
            },
        }
        rows.append(record)

    rows.sort(key=lambda r: float(r.get("retrieval_score", 0.0) or 0.0), reverse=True)

    for i, row in enumerate(rows[:top_k], start=1):
        row["retrieval_rank"] = i
        row["fusion_sources"][0]["rank"] = i

    return rows[:top_k]

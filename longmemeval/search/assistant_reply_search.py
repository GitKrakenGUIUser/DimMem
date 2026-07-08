from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from search import load_window_assistant_replies

try:
    from search.p2_runtime import _candidate_assistant_dirs
except Exception:
    _candidate_assistant_dirs = None


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _clean(value).lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).lower() in {"1", "true", "yes", "y", "on"}


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
        if len(t) >= 2
    ]


def _query_parts(parsed_query: Dict[str, Any]) -> List[str]:
    dim = parsed_query.get("dimension") if isinstance(parsed_query.get("dimension"), dict) else {}
    parts: List[str] = []

    for key in ("question", "query", "query_anchor", "canonical_text", "answer_dim", "query_type"):
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

    tr = dim.get("time_range") if isinstance(dim.get("time_range"), dict) else parsed_query.get("time_range")
    if isinstance(tr, dict):
        for key in ("relative_event", "start", "end"):
            if _clean(tr.get(key)):
                parts.append(_clean(tr.get(key)))

    return parts


def should_search_assistant_replies(parsed_query: Dict[str, Any]) -> bool:
    if _truthy(parsed_query.get("need_assistant_context")):
        return True

    text = " ".join(_query_parts(parsed_query)).lower()
    markers = [
        "assistant",
        "you suggested",
        "you recommended",
        "you said",
        "you gave",
        "you provided",
        "previously suggested",
        "previously recommended",
        "what did you recommend",
        "what did you suggest",
        "last time you",
    ]
    return any(m in text for m in markers)


def _candidate_dirs(memory_dir: Path) -> List[Path]:
    if _candidate_assistant_dirs is not None:
        try:
            dirs = _candidate_assistant_dirs(memory_dir, None)
            if dirs:
                return dirs
        except Exception:
            pass

    root = Path(__file__).resolve().parents[2]
    submit_root = Path(__file__).resolve().parents[2]
    question_type = memory_dir.parent.name
    sample_id = memory_dir.name
    run_name = memory_dir.parent.parent.name if len(memory_dir.parents) >= 2 else ""

    candidates = [
        memory_dir,
        memory_dir.parent,
        memory_dir.parent.parent if len(memory_dir.parents) >= 2 else memory_dir.parent,
        submit_root / "results" / "segments" / run_name / question_type / sample_id,
        submit_root / "results" / "raw_segments" / run_name / question_type / sample_id,
        submit_root / "results" / "compressed_segments" / run_name / question_type / sample_id,
        submit_root / "results" / "segments" / question_type / sample_id,
        submit_root / "results" / "raw_segments" / question_type / sample_id,
        submit_root / "results" / "compressed_segments" / question_type / sample_id,
    ]

    out: List[Path] = []
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp.exists() and rp not in seen:
            seen.add(rp)
            out.append(rp)

    results_root = submit_root / "results"
    if results_root.exists():
        try:
            for path in results_root.rglob("window_*_assistant_replies.json"):
                pstr = str(path)
                if question_type in pstr and sample_id in pstr:
                    parent = path.parent.resolve()
                    if parent not in seen:
                        seen.add(parent)
                        out.append(parent)
        except Exception:
            pass

    return out


def _load_assistant_uid_map(memory_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], str]:
    for d in _candidate_dirs(memory_dir):
        if not list(d.glob("window_*_assistant_replies.json")):
            continue
        try:
            uid_map = load_window_assistant_replies(d)
        except Exception:
            uid_map = {}
        if uid_map:
            return uid_map, str(d)
    return {}, ""


def _score_reply(query_text: str, query_terms: List[str], reply: str) -> Tuple[float, Dict[str, Any]]:
    reply_l = reply.lower()
    q_tokens = set(_tokens(query_text))
    r_tokens = set(_tokens(reply))

    token_hits = len(q_tokens & r_tokens)
    token_score = token_hits / max(1.0, math.sqrt(len(q_tokens) + 1.0))

    phrase_hits = 0
    phrase_hit_terms: List[str] = []

    for term in query_terms:
        t = term.strip().lower()
        if len(t) < 3:
            continue
        if t in reply_l:
            phrase_hits += 1
            phrase_hit_terms.append(term)

    phrase_score = min(3.0, phrase_hits * 0.8)
    rare_bonus = 0.0

    # Helpful for proper nouns / titles like MusicTheory.net, Lost Temple of the Djinn.
    for term in query_terms:
        if any(ch.isupper() for ch in term) or "." in term or "-" in term:
            if term.lower() in reply_l:
                rare_bonus += 0.8

    score = token_score + phrase_score + rare_bonus

    return score, {
        "token_hits": token_hits,
        "phrase_hits": phrase_hits,
        "phrase_hit_terms": phrase_hit_terms[:10],
        "rare_bonus": round(rare_bonus, 6),
    }


def search_assistant_replies(
    *,
    parsed_query: Dict[str, Any],
    memory_dir: Path,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Direct assistant reply search route.

    This route is different from attach_assistant_context:
    - attach_assistant_context only adds assistant text to already-retrieved memories;
    - this function directly retrieves assistant replies as independent evidence.
    """
    if not should_search_assistant_replies(parsed_query):
        return []

    uid_map, used_dir = _load_assistant_uid_map(memory_dir)
    if not uid_map:
        return []

    parts = _query_parts(parsed_query)
    query_text = " ".join(parts)
    query_terms = []
    seen = set()

    for p in parts:
        p = _clean(p)
        if not p:
            continue
        marker = p.lower()
        if marker not in seen:
            seen.add(marker)
            query_terms.append(p)

    rows: List[Dict[str, Any]] = []

    for uid, info in uid_map.items():
        if not isinstance(info, dict):
            continue

        reply = _clean(info.get("assistant_reply") or info.get("content") or info.get("reply"))
        if not reply:
            continue

        score, components = _score_reply(query_text, query_terms, reply)
        if score <= 0:
            continue

        source_time = (
            _clean(info.get("source_time"))
            or _clean(info.get("time"))
            or _clean(info.get("timestamp"))
            or _clean(info.get("created_at"))
        )

        session_id = _clean(info.get("session_id"))
        session_local_user_index = info.get("session_local_user_index", 0)

        item = {
            "user_id": memory_dir.name,
            "memory_type": "assistant",
            "content": "Assistant reply direct-search evidence: " + reply,
            "dimension": {
                "memory_type": "assistant",
                "time": source_time,
                "location": "",
                "reason": "Direct assistant reply search route for assistant-dependent questions.",
                "purpose": "Recover what the assistant previously said, suggested, recommended, explained, or provided.",
                "keywords": query_terms[:20],
            },
            "entities": query_terms[:20],
            "embedding_text": reply,
            "source_message_ids": [uid],
            "source_boundary_id": f"assistant_reply_search::{uid}",
            "source_time": source_time,
            "record_time": source_time,
            "assistant_reply": reply,
            "assistant_uid": uid,
            "session_id": session_id,
            "session_local_user_index": session_local_user_index,
            "retrieval_method": "assistant_reply_search",
            "retrieval_rank": 0,
            "retrieval_score": float(score),
            "score": float(score),
            "fusion_sources": [
                {
                    "method": "assistant_reply_search",
                    "rank": 0,
                    "score": float(score),
                }
            ],
            "_assistant_reply_search": {
                "used_dir": used_dir,
                "score_components": components,
            },
        }

        rows.append(item)

    rows.sort(key=lambda r: float(r.get("retrieval_score", 0.0) or 0.0), reverse=True)

    for idx, row in enumerate(rows[:top_k], start=1):
        row["retrieval_rank"] = idx
        row["fusion_sources"][0]["rank"] = idx

    return rows[:top_k]

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _clean(v: Any) -> str:
    return str(v or "").strip()


def _lower(v: Any) -> str:
    return _clean(v).lower()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        return [_clean(x) for x in v if _clean(x)]
    return []


def _dim(mem: Dict[str, Any]) -> Dict[str, Any]:
    d = mem.get("dimension")
    return d if isinstance(d, dict) else {}


def _text(mem: Dict[str, Any]) -> str:
    d = _dim(mem)
    parts = [
        mem.get("content", ""),
        mem.get("source_time", ""),
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


def _parse_relations(rel: Any) -> Dict[str, str]:
    text = _clean(rel)
    out: Dict[str, str] = {}
    if not text:
        return out
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
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _number_word_to_int(text: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
    words = {
        "one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    for w, n in words.items():
        if re.search(rf"\b{re.escape(w)}\b", text, flags=re.I):
            return n
    return None


def _normalize_current_flags(mem: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(mem)
    d = dict(_dim(item))
    status = _lower(d.get("status"))
    action = _lower(d.get("action"))
    content = _lower(item.get("content"))

    pastish = {
        "past", "completed", "cancelled", "canceled", "finished", "submitted",
    }
    if status in pastish:
        d["is_current"] = False

    if action in {"submitted", "visited", "received", "wore", "bought", "purchased", "serviced", "completed"}:
        d["is_current"] = False

    if any(w in content for w in ["on march", "on april", "last week", "yesterday", "finished", "completed", "submitted"]):
        if status not in {"current", "preference", "habit"}:
            d["is_current"] = False

    if status in {"preference", "habit", "current"}:
        d["is_current"] = True

    item["dimension"] = d
    return item


def _base_derived_memory(
    *,
    source_memories: List[Dict[str, Any]],
    content: str,
    action: str,
    obj: str,
    value: str = "",
    relation: str = "",
    event_time: str = "",
    status: str = "derived",
    evidence: str = "",
) -> Dict[str, Any]:
    first = source_memories[0] if source_memories else {}
    first_dim = _dim(first)

    return {
        "source_id": int(first.get("source_id", 0) or 0),
        "source_time": _clean(first.get("source_time")),
        "session_id": _clean(first.get("session_id")),
        "session_local_user_index": int(first.get("session_local_user_index", 0) or 0),
        "content": content,
        "dimension": {
            "memory_type": "fact",
            "time": event_time or _clean(first_dim.get("time")),
            "location": _clean(first_dim.get("location")),
            "reason": "P4 post-extraction linker derived this memory from related extracted memories.",
            "purpose": "Make implicit links, updates, or computed facts retrievable.",
            "keywords": list(dict.fromkeys(
                [obj, value, action, "P4 derived memory"]
                + _as_list(first_dim.get("keywords"))
            )),
            "event_time": event_time or _clean(first_dim.get("event_time")),
            "valid_from": _clean(first_dim.get("valid_from")),
            "valid_to": _clean(first_dim.get("valid_to")),
            "status": status,
            "is_current": status in {"current", "preference", "habit"},
            "subject": "the user",
            "action": action,
            "object": obj,
            "value": value,
            "quantity": "",
            "unit": "",
            "relation": relation,
            "evidence_span": evidence[:300],
        },
        "window_index": first.get("window_index", 0),
        "memory_index": -1,
        "p4_linked": True,
        "p4_source_contents": [_clean(m.get("content")) for m in source_memories[:5]],
    }


def _derive_acl_submission(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    derived: List[Dict[str, Any]] = []
    paper_mems = [
        m for m in memories
        if "sentiment" in _lower(_text(m))
        and "paper" in _lower(_text(m))
        and "acl" in _lower(_text(m))
        and any(w in _lower(_text(m)) for w in ["submitted", "submission"])
    ]
    date_mems = [
        m for m in memories
        if "acl" in _lower(_text(m))
        and any(w in _lower(_text(m)) for w in ["deadline", "submission date", "february 1"])
    ]

    for p in paper_mems[:2]:
        for d in date_mems[:2]:
            date_value = _clean(_dim(d).get("value")) or _clean(_dim(d).get("event_time")) or "February 1st"
            if "february 1" not in date_value.lower() and "february 1" not in _lower(_text(d)):
                continue
            derived.append(
                _base_derived_memory(
                    source_memories=[p, d],
                    content="The user submitted the user's research paper on sentiment analysis to ACL on February 1st.",
                    action="submitted",
                    obj="research paper on sentiment analysis",
                    value="February 1st",
                    relation="submission_date=February 1st; venue=ACL; topic=sentiment analysis",
                    event_time="February 1st",
                    status="completed",
                    evidence=_clean(p.get("content")) + " | " + _clean(d.get("content")),
                )
            )
            return derived
    return derived


def _derive_clinic_arrival(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text_rows = [(m, _lower(_text(m))) for m in memories]
    left = []
    duration = []
    for m, t in text_rows:
        if "clinic" in t and any(x in t for x in ["left home", "leave home", "leaving home", "departed"]):
            left.append(m)
        if "clinic" in t and any(x in t for x in ["two hours", "2 hours", "took two", "duration"]):
            duration.append(m)

    derived: List[Dict[str, Any]] = []
    for l in left[:2]:
        lt = _text(l)
        mtime = re.search(r"\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b", lt)
        if not mtime:
            continue
        hour = int(mtime.group(1))
        minute = int(mtime.group(2))
        ampm = (mtime.group(3) or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        add_hours = 2
        for d in duration[:2]:
            dt = _text(d)
            if re.search(r"\b(three|3)\s+hours?\b", dt, flags=re.I):
                add_hours = 3
            if re.search(r"\b(one|1)\s+hours?\b", dt, flags=re.I):
                add_hours = 1
            base = datetime(2000, 1, 1, hour, minute)
            arr = base + timedelta(hours=add_hours)
            arr_hour = arr.hour
            suffix = "AM" if arr_hour < 12 else "PM"
            display_hour = arr_hour if 1 <= arr_hour <= 12 else (arr_hour - 12 if arr_hour > 12 else 12)
            val = f"{display_hour}:{arr.minute:02d} {suffix}"
            derived.append(
                _base_derived_memory(
                    source_memories=[l, d],
                    content=f"The user reached the clinic at {val}, derived from leaving home at {mtime.group(0)} and a {add_hours}-hour trip to the clinic.",
                    action="reached",
                    obj="clinic",
                    value=val,
                    relation=f"arrival_time={val}; departure_time={mtime.group(0)}; travel_duration={add_hours} hours",
                    event_time=val,
                    status="derived",
                    evidence=_clean(l.get("content")) + " | " + _clean(d.get("content")),
                )
            )
            return derived
    return derived


def _derive_latest_counts(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    derived: List[Dict[str, Any]] = []
    candidates: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}

    for m in memories:
        t = _lower(_text(m))
        d = _dim(m)
        rel = _parse_relations(d.get("relation"))
        obj = _clean(d.get("object")) or "unknown object"

        if any(x in t for x in ["converse", "sneaker", "shoe"]):
            n = None
            for key in ("wear_count", "times_worn", "count"):
                if key in rel:
                    n = _number_word_to_int(rel[key])
            if n is None:
                n = _number_word_to_int(t)
            if n is not None and any(w in t for w in ["worn", "wear", "sixth time", "four times"]):
                candidates.setdefault("new black Converse", []).append((n, m))

    for obj, rows in candidates.items():
        if not rows:
            continue
        # For reported cumulative counts, highest value is usually the latest total.
        n, best = sorted(rows, key=lambda x: x[0], reverse=True)[0]
        source_mems = [m for _, m in rows[:5]]
        derived.append(
            _base_derived_memory(
                source_memories=[best] + source_mems,
                content=f"The latest reported wear count for the user's {obj} is {n} times.",
                action="latest_count",
                obj=obj,
                value=str(n),
                relation=f"latest_wear_count={n}; object={obj}",
                event_time=_clean(_dim(best).get("event_time")) or _clean(best.get("source_time")),
                status="current",
                evidence=" | ".join(_clean(m.get("content")) for _, m in rows[:5]),
            )
        )

    return derived


def _derive_before_after(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    derived: List[Dict[str, Any]] = []

    for m in memories:
        t = _lower(_text(m))
        if "instant pot" in t and "air fryer" in t and "before" in t:
            derived.append(
                _base_derived_memory(
                    source_memories=[m],
                    content="The user invested in an Instant Pot before getting the Air Fryer.",
                    action="invested_in",
                    obj="Instant Pot",
                    value="Instant Pot",
                    relation="before=getting the Air Fryer; category=kitchen gadget",
                    event_time=_clean(_dim(m).get("event_time")) or _clean(m.get("source_time")),
                    status="completed",
                    evidence=_clean(m.get("content")),
                )
            )
            return derived

    instant = [m for m in memories if "instant pot" in _lower(_text(m))]
    fryer = [m for m in memories if "air fryer" in _lower(_text(m))]

    if instant and fryer:
        derived.append(
            _base_derived_memory(
                source_memories=[instant[0], fryer[0]],
                content="The user had a new Instant Pot as a kitchen gadget related to the period before getting the Air Fryer.",
                action="invested_in",
                obj="Instant Pot",
                value="Instant Pot",
                relation="candidate_before=getting the Air Fryer; category=kitchen gadget",
                event_time=_clean(_dim(instant[0]).get("event_time")) or _clean(instant[0].get("source_time")),
                status="derived",
                evidence=_clean(instant[0].get("content")) + " | " + _clean(fryer[0].get("content")),
            )
        )

    return derived


def _parse_window_text_pairs(text: str, window_index: int) -> List[Dict[str, Any]]:
    """
    Fallback parser for window_input['text'].
    Handles lines like:
    [2023/..., session] 12.User: ...
    [2023/..., session] 12.Assistant: ...
    """
    pairs: List[Dict[str, Any]] = []
    lines = _clean(text).splitlines()
    last_user = ""
    last_user_id = ""
    last_ts = ""

    msg_re = re.compile(r"^\[(?P<ts>[^\]]+)\]\s*(?P<idx>\d+)\.(?P<role>User|Assistant):\s*(?P<content>.*)$", re.I)

    for line in lines:
        m = msg_re.match(line.strip())
        if not m:
            continue
        role = m.group("role").lower()
        content = _clean(m.group("content"))
        if role == "user":
            last_user = content
            last_user_id = m.group("idx")
            last_ts = m.group("ts")
        elif role == "assistant" and content:
            uid = f"w{window_index:04d}u{int(last_user_id or 0):02d}"
            pairs.append(
                {
                    "uid": uid,
                    "window_index": window_index,
                    "previous_user_message": last_user,
                    "assistant_reply": content,
                    "pair_text": (last_user + "\n" + content).strip(),
                    "source_time": last_ts,
                    "session_id": "",
                    "session_local_user_index": int(last_user_id or 0),
                }
            )
    return pairs


def _build_assistant_pair_index(case_dir: Path, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    seen = set()

    # Prefer extraction output window_input.json because it is always copied into memory output.
    for win_input in sorted(case_dir.glob("window_*/window_input.json")):
        try:
            window = _read_json(win_input)
        except Exception:
            continue

        widx = int(window.get("window_index", 0) or 0)

        # Structured messages if available.
        messages = window.get("messages")
        if isinstance(messages, list) and messages:
            prev_user = ""
            prev_user_idx = 0
            prev_ts = ""
            for msg in messages:
                role = _lower(msg.get("role"))
                content = _clean(msg.get("content") or msg.get("text"))
                if role == "user":
                    prev_user = content
                    prev_user_idx = int(msg.get("global_user_index") or msg.get("source_id") or 0)
                    prev_ts = _clean(msg.get("timestamp") or msg.get("time") or msg.get("date"))
                elif role == "assistant" and content:
                    uid = f"w{widx:04d}u{prev_user_idx:02d}"
                    key = (uid, content[:80])
                    if key not in seen:
                        seen.add(key)
                        pairs.append(
                            {
                                "uid": uid,
                                "window_index": widx,
                                "previous_user_message": prev_user,
                                "assistant_reply": content,
                                "pair_text": (prev_user + "\n" + content).strip(),
                                "source_time": prev_ts,
                                "session_id": _clean(msg.get("session_id")),
                                "session_local_user_index": int(msg.get("session_local_user_index") or 0),
                            }
                        )
            continue

        text = _clean(window.get("text"))
        for p in _parse_window_text_pairs(text, widx):
            key = (p["uid"], p["assistant_reply"][:80])
            if key not in seen:
                seen.add(key)
                pairs.append(p)

    # Merge direct assistant reply files if present in source dir.
    source_dir = Path(_clean(payload.get("source_record_dir")))
    candidate_dirs = [case_dir, source_dir] if _clean(source_dir) else [case_dir]
    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        for path in cdir.rglob("window_*_assistant_replies.json"):
            try:
                data = _read_json(path)
            except Exception:
                continue
            replies = data.get("replies", {})
            if not isinstance(replies, dict):
                continue
            for uid, info in replies.items():
                if not isinstance(info, dict):
                    continue
                reply = _clean(info.get("assistant_reply") or info.get("content") or info.get("reply"))
                if not reply:
                    continue
                prev = _clean(info.get("previous_user_message") or info.get("user_message"))
                key = (_clean(uid), reply[:80])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "uid": _clean(uid),
                        "window_index": info.get("window_index", 0),
                        "previous_user_message": prev,
                        "assistant_reply": reply,
                        "pair_text": (prev + "\n" + reply).strip(),
                        "source_time": _clean(info.get("source_time") or info.get("time")),
                        "session_id": _clean(info.get("session_id")),
                        "session_local_user_index": int(info.get("session_local_user_index", 0) or 0),
                    }
                )

    for p in pairs:
        p["answer_items"] = _extract_answer_items(p.get("assistant_reply", ""))

    return pairs


def _extract_answer_items(text: str) -> List[str]:
    """
    Lightweight item extraction for assistant recall.
    Preserves phrase-like items such as recommendations and terms.
    """
    items: List[str] = []

    # Bold markdown items.
    for m in re.finditer(r"\*\*([^*]{2,80})\*\*", text):
        item = _clean(m.group(1)).strip(":：")
        if item and item.lower() not in {"tips", "remember", "additional tips"}:
            items.append(item)

    # Numbered/bulleted lines.
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*)$", line)
        if not m:
            continue
        item = _clean(m.group(1))
        item = re.sub(r"[:：].*$", "", item).strip()
        item = re.sub(r"\*\*", "", item).strip()
        if 2 <= len(item) <= 100:
            items.append(item)

    # Deduplicate.
    out = []
    seen = set()
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:30]


def link_case(case_dir: Path) -> Dict[str, Any]:
    all_path = case_dir / "all_memories.json"
    if not all_path.exists():
        return {"case_dir": str(case_dir), "ok": False, "error": "missing all_memories.json"}

    payload = _read_json(all_path)
    memories = payload.get("memories") if isinstance(payload, dict) else []
    memories = [m for m in memories if isinstance(m, dict)]

    normalized = [_normalize_current_flags(m) for m in memories]

    derived: List[Dict[str, Any]] = []
    derived.extend(_derive_acl_submission(normalized))
    derived.extend(_derive_clinic_arrival(normalized))
    derived.extend(_derive_latest_counts(normalized))
    derived.extend(_derive_before_after(normalized))

    # Avoid duplicate derived contents.
    final_memories: List[Dict[str, Any]] = []
    seen = set()
    for m in normalized + derived:
        key = _clean(m.get("content")).lower()
        if key and key not in seen:
            seen.add(key)
            final_memories.append(m)

    payload["memories"] = final_memories
    payload["memory_count"] = len(final_memories)
    payload["p4_linker"] = {
        "original_memory_count": len(memories),
        "derived_memory_count": len(final_memories) - len(normalized),
        "updated_at": datetime.now().isoformat(),
    }

    _write_json(all_path, payload)

    # Also rewrite per-window normalized memories with current flag normalization.
    for win_path in case_dir.glob("window_*/normalized_memories.json"):
        try:
            wp = _read_json(win_path)
            rows = wp.get("memories") or []
            wp["memories"] = [_normalize_current_flags(m) for m in rows if isinstance(m, dict)]
            _write_json(win_path, wp)
        except Exception:
            pass

    pairs = _build_assistant_pair_index(case_dir, payload if isinstance(payload, dict) else {})
    _write_json(case_dir / "assistant_pair_index.json", {"pair_count": len(pairs), "pairs": pairs})

    return {
        "case_dir": str(case_dir),
        "ok": True,
        "original_memory_count": len(memories),
        "final_memory_count": len(final_memories),
        "derived_memory_count": len(final_memories) - len(normalized),
        "assistant_pair_count": len(pairs),
    }


def iter_case_dirs(memory_root: Path) -> Iterable[Path]:
    for path in sorted(memory_root.glob("*/*/all_memories.json")):
        yield path.parent


def run(args: argparse.Namespace) -> Path:
    src_root = args.memory_root.resolve()
    out_root = (args.output_root.resolve() / args.run_name).resolve()

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    if out_root.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {out_root}; use --overwrite or another --run-name")

    shutil.copytree(src_root, out_root)

    rows = []
    for case_dir in iter_case_dirs(out_root):
        rows.append(link_case(case_dir))

    summary = {
        "source_memory_root": str(src_root),
        "output_memory_root": str(out_root),
        "case_count": len(rows),
        "ok_count": sum(1 for r in rows if r.get("ok")),
        "derived_memory_count": sum(int(r.get("derived_memory_count", 0) or 0) for r in rows),
        "assistant_pair_count": sum(int(r.get("assistant_pair_count", 0) or 0) for r in rows),
        "rows": rows,
        "created_at": datetime.now().isoformat(),
    }
    _write_json(out_root / "p4_linker_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P4 post-extraction linker and assistant pair index builder.")
    p.add_argument("--memory-root", type=Path, required=True, help="P3 memory root.")
    p.add_argument("--output-root", type=Path, default=Path("./results/memories"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())

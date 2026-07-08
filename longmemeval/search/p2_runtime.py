from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from search import (
    attach_assistant_context,
    build_boundary_to_window_source,
    load_window_assistant_replies,
)

THIS_FILE = Path(__file__).resolve()
LONGMEMEVAL_DIR = THIS_FILE.parents[1]
SUBMIT_ROOT = THIS_FILE.parents[2]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).lower() in {"1", "true", "yes", "y", "on"}


def search_mode_name_p2(route_top_k: int, final_top_k: int, enable_rerank: bool) -> str:
    suffix = "p2_rerank" if enable_rerank else "no_rerank"
    return f"tri_fused_{suffix}_route{route_top_k}_final{final_top_k}"


def build_full_mapped_query(
    parsed_query: Dict[str, Any],
    search_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Preserve P2 fields in mapped_query_analysis.json.

    Old search functions only return legacy mapped query fields. For P2 debugging,
    we need query_type/statefulness/entities/aliases/time_range/p2_schema_version.
    """
    mapped = dict(parsed_query or {})
    legacy = search_result.get("mapped_query_analysis") or {}

    mapped["search_mapped_query_analysis"] = legacy

    if not _clean(mapped.get("query_text")):
        mapped["query_text"] = (
            _clean(legacy.get("query_text"))
            or _clean(legacy.get("query_anchor"))
            or _clean(mapped.get("query_anchor"))
            or _clean(mapped.get("canonical_text"))
        )

    if not mapped.get("keywords") and legacy.get("keywords"):
        mapped["keywords"] = legacy.get("keywords")

    if not mapped.get("target_memory_type") and legacy.get("target_memory_types"):
        mapped["target_memory_type"] = legacy.get("target_memory_types")

    return mapped


def assistant_context_stats(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    rows = list(records or [])
    return {
        "total": len(rows),
        "with_assistant_reply": sum(1 for r in rows if _clean(r.get("assistant_reply"))),
        "with_assistant_uid": sum(1 for r in rows if _clean(r.get("assistant_uid"))),
    }


def _candidate_assistant_dirs(memory_dir: Path, source_record_dir: str | None) -> List[Path]:
    """
    Build candidate directories that may contain window_*_assistant_replies.json.

    This is intentionally more robust than the P1 resolver:
    - source_record_dir absolute / relative;
    - memory_dir and parents;
    - common results/segments layouts;
    - recursive fallback restricted by question_type + sample_id to avoid uid collisions.
    """
    candidates: List[Path] = []

    def add(path: Path | None) -> None:
        if path is None:
            return
        try:
            p = path.resolve()
        except Exception:
            p = path
        if p not in candidates:
            candidates.append(p)

    if source_record_dir:
        raw = Path(source_record_dir)
        if raw.is_absolute():
            add(raw)
        else:
            add(memory_dir / raw)
            add(memory_dir.parent / raw)
            add(memory_dir.parent.parent / raw if len(memory_dir.parents) >= 2 else None)
            add(SUBMIT_ROOT / raw)

    add(memory_dir)
    add(memory_dir.parent)
    add(memory_dir.parent.parent if len(memory_dir.parents) >= 2 else None)
    add(memory_dir.parent.parent.parent if len(memory_dir.parents) >= 3 else None)

    question_type = memory_dir.parent.name
    sample_id = memory_dir.name
    run_name = memory_dir.parent.parent.name if len(memory_dir.parents) >= 2 else ""

    for bucket in [
        "segments",
        "raw_segments",
        "compressed_segments",
        "compressed",
        "raw",
        "memories",
    ]:
        if run_name:
            add(SUBMIT_ROOT / "results" / bucket / run_name / question_type / sample_id)
        add(SUBMIT_ROOT / "results" / bucket / question_type / sample_id)

    # Restricted recursive fallback.
    results_root = SUBMIT_ROOT / "results"
    if results_root.exists():
        try:
            for path in results_root.rglob("window_*_assistant_replies.json"):
                pstr = str(path)
                if question_type in pstr and sample_id in pstr:
                    add(path.parent)
        except Exception:
            pass

    return [p for p in candidates if p.exists()]


def _load_uid_map_from_candidates(candidates: List[Path]) -> Tuple[Dict[str, Dict[str, Any]], Optional[Path]]:
    for candidate in candidates:
        if not candidate.exists():
            continue

        direct = list(candidate.glob("window_*_assistant_replies.json"))
        if not direct:
            continue

        try:
            uid_map = load_window_assistant_replies(candidate)
        except Exception:
            uid_map = {}

        if uid_map:
            return uid_map, candidate

    return {}, None


def attach_assistant_context_robust(
    *,
    parsed_query: Dict[str, Any],
    memory_dir: Path,
    records: List[Dict[str, Any]],
    force: bool,
) -> List[Dict[str, Any]]:
    """
    Robust assistant context attachment.

    Trigger:
    - force=True, or
    - parsed_query.need_assistant_context=True.

    Safe fallback:
    - returns original records unchanged if no assistant files are found.
    - adds lightweight debug info to records so summary/top_records can reveal why.
    """
    need_assistant = force or _truthy((parsed_query or {}).get("need_assistant_context"))
    if not need_assistant or not records:
        return records

    try:
        boundary_index, source_record_dir = build_boundary_to_window_source(memory_dir)
    except Exception as exc:
        print(f"[WARN] assistant boundary index failed: {exc}", file=sys.stderr)
        return records

    if not boundary_index:
        return records

    candidates = _candidate_assistant_dirs(memory_dir, source_record_dir)
    uid_map, used_dir = _load_uid_map_from_candidates(candidates)

    if not uid_map:
        # Preserve records but add debug hint.
        out = []
        for rec in records:
            item = dict(rec)
            item["_assistant_context_debug"] = {
                "need_assistant_context": True,
                "boundary_index_count": len(boundary_index),
                "candidate_dirs_checked": [str(p) for p in candidates[:20]],
                "used_dir": "",
                "status": "no_uid_map_found",
            }
            out.append(item)
        return out

    attached = attach_assistant_context([dict(r) for r in records], boundary_index, uid_map)
    hit_count = sum(1 for r in attached if _clean(r.get("assistant_reply")))

    for rec in attached:
        rec["_assistant_context_debug"] = {
            "need_assistant_context": True,
            "boundary_index_count": len(boundary_index),
            "uid_map_count": len(uid_map),
            "used_dir": str(used_dir or ""),
            "attached_reply_count_in_batch": hit_count,
            "status": "attached" if hit_count > 0 else "uid_map_found_but_no_record_match",
        }

    return attached

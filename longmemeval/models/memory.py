from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


VALID_MEMORY_TYPES = {"fact", "episodic", "profile"}


def clean(value: Any) -> str:
    return str(value or "").strip()


def unique_string_list(values: Any, *, lower_dedupe: bool = False) -> List[str]:
    """Normalize a scalar/list into a de-duplicated list of non-empty strings."""
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    result: List[str] = []
    seen = set()

    for value in values:
        text = clean(value)
        if not text:
            continue

        marker = text.lower() if lower_dedupe else text
        if marker in seen:
            continue

        seen.add(marker)
        result.append(text)

    return result


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean(value).lower()
    return text in {"1", "true", "yes", "y", "current"}


@dataclass
class DimensionMemory:
    """
    Structured dimension fields attached to a memory record.

    Backward compatible with original DimMem fields:
    - memory_type
    - time
    - location
    - reason
    - purpose
    - keywords

    P3 extraction adds event/value/relation/action fields.
    These fields are optional and will be empty for old memory runs.
    """

    memory_type: str = ""
    time: str = ""
    location: str = ""
    reason: str = ""
    purpose: str = ""
    keywords: List[str] = field(default_factory=list)

    # P3 event/state schema
    event_time: str = ""
    valid_from: str = ""
    valid_to: str = ""
    status: str = ""
    is_current: bool = False

    # P3 event/value/relation schema
    subject: str = ""
    action: str = ""
    object: str = ""
    value: str = ""
    quantity: str = ""
    unit: str = ""
    relation: str = ""
    evidence_span: str = ""

    @classmethod
    def from_dict(cls, payload: Any) -> "DimensionMemory":
        data = payload if isinstance(payload, dict) else {}

        memory_type = clean(data.get("memory_type")).lower()
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = ""

        return cls(
            memory_type=memory_type,
            time=clean(data.get("time")),
            location=clean(data.get("location")),
            reason=clean(data.get("reason")),
            purpose=clean(data.get("purpose")),
            keywords=unique_string_list(data.get("keywords"), lower_dedupe=True),
            event_time=clean(data.get("event_time")),
            valid_from=clean(data.get("valid_from")),
            valid_to=clean(data.get("valid_to")),
            status=clean(data.get("status")),
            is_current=_boolish(data.get("is_current")),
            subject=clean(data.get("subject")),
            action=clean(data.get("action")),
            object=clean(data.get("object")),
            value=clean(data.get("value")),
            quantity=clean(data.get("quantity")),
            unit=clean(data.get("unit")),
            relation=clean(data.get("relation")),
            evidence_span=clean(data.get("evidence_span")),
        )

    def to_dict(self, *, include_memory_type: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        if include_memory_type:
            data["memory_type"] = self.memory_type

        data.update(
            {
                "time": self.time,
                "location": self.location,
                "reason": self.reason,
                "purpose": self.purpose,
                "keywords": list(self.keywords),

                # P3 fields
                "event_time": self.event_time,
                "valid_from": self.valid_from,
                "valid_to": self.valid_to,
                "status": self.status,
                "is_current": self.is_current,

                "subject": self.subject,
                "action": self.action,
                "object": self.object,
                "value": self.value,
                "quantity": self.quantity,
                "unit": self.unit,
                "relation": self.relation,
                "evidence_span": self.evidence_span,
            }
        )

        return data

    def extra_terms(self) -> List[str]:
        return [
            self.event_time,
            self.valid_from,
            self.valid_to,
            self.status,
            self.subject,
            self.action,
            self.object,
            self.value,
            self.quantity,
            self.unit,
            self.relation,
            self.evidence_span,
        ]

    def searchable_text(self, *, include_content: str = "") -> str:
        parts = [
            clean(include_content),
            self.time,
            self.location,
            self.reason,
            self.purpose,
            " ".join(self.keywords),
            *self.extra_terms(),
        ]

        return " | ".join(part for part in parts if clean(part))


@dataclass
class ParsedQuery:
    """Normalized result from the query parser. Backward compatible with P2 schema."""

    query_anchor: str = ""
    need_assistant_context: bool = False
    target_memory_type: List[str] = field(default_factory=list)
    time: str = ""
    location: str = ""
    keywords: List[str] = field(default_factory=list)
    answer_dim: str = ""

    # P2 query fields
    query_type: str = "lookup"
    statefulness: str = "unknown"
    entities: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    time_range: Dict[str, Any] = field(default_factory=dict)

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Any) -> "ParsedQuery":
        data = payload if isinstance(payload, dict) else {}
        dimension = data.get("dimension") if isinstance(data.get("dimension"), dict) else {}

        return cls(
            query_anchor=clean(data.get("query_anchor")),
            need_assistant_context=bool(data.get("need_assistant_context", False)),
            target_memory_type=unique_string_list(
                dimension.get("target_memory_type") or data.get("target_memory_type"),
                lower_dedupe=True,
            ),
            time=clean(dimension.get("time") or data.get("time")),
            location=clean(dimension.get("location") or data.get("location")),
            keywords=unique_string_list(
                list(unique_string_list(dimension.get("keywords"), lower_dedupe=True))
                + list(unique_string_list(data.get("keywords"), lower_dedupe=True)),
                lower_dedupe=True,
            ),
            answer_dim=clean(data.get("answer_dim")).lower(),
            query_type=clean(data.get("query_type")) or "lookup",
            statefulness=clean(data.get("statefulness")) or "unknown",
            entities=unique_string_list(
                list(unique_string_list(dimension.get("entities"), lower_dedupe=True))
                + list(unique_string_list(data.get("entities"), lower_dedupe=True)),
                lower_dedupe=True,
            ),
            aliases=unique_string_list(
                list(unique_string_list(dimension.get("aliases"), lower_dedupe=True))
                + list(unique_string_list(data.get("aliases"), lower_dedupe=True)),
                lower_dedupe=True,
            ),
            time_range=(
                dimension.get("time_range")
                if isinstance(dimension.get("time_range"), dict)
                else data.get("time_range")
                if isinstance(data.get("time_range"), dict)
                else {}
            ),
            raw=dict(data),
        )

    @property
    def dimension(self) -> Dict[str, Any]:
        return {
            "target_memory_type": list(self.target_memory_type),
            "time": self.time,
            "time_range": dict(self.time_range),
            "location": self.location,
            "keywords": list(self.keywords),
            "entities": list(self.entities),
            "aliases": list(self.aliases),
        }

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.raw)

        data.update(
            {
                "query_anchor": self.query_anchor,
                "need_assistant_context": self.need_assistant_context,
                "query_type": self.query_type,
                "statefulness": self.statefulness,
                "dimension": self.dimension,
                "answer_dim": self.answer_dim,
            }
        )

        return data

    def bm25_text(self) -> str:
        return " ".join(
            [self.query_anchor]
            + self.keywords
            + self.entities
            + self.aliases
        ).strip()

    def constraints(self) -> Dict[str, List[str]]:
        constraints: Dict[str, List[str]] = {}

        if self.time:
            constraints["time"] = [self.time]

        if self.location:
            constraints["location"] = [self.location]

        return constraints

    def to_search_analysis(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_anchor,
            "rewrite": self.query_anchor,
            "intent": self.query_type or "lookup",
            "query_type": self.query_type or "lookup",
            "statefulness": self.statefulness or "unknown",
            "answer_field": self.answer_dim or "content",
            "content_query": self.query_anchor,
            "target_memory_types": list(self.target_memory_type),
            "entities": list(self.entities),
            "aliases": list(self.aliases),
            "constraints": self.constraints(),
            "time_range": dict(self.time_range),
            "keywords": list(self.keywords),
            "canonical_text": self.query_anchor,
        }


__all__ = [
    "DimensionMemory",
    "ParsedQuery",
    "VALID_MEMORY_TYPES",
    "clean",
    "unique_string_list",
]

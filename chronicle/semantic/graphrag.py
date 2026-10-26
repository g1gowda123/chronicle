"""Two-stage GraphRAG: serialise a 2-hop subgraph, then answer with citations.

Role B owns graph retrieval.  This module accepts either a live NetworkX graph
or the serialised payload documented in ``docs/ROLE_C.md`` and produces a
grounded, timestamped answer.  When no local LLM is reachable it falls back to a
deterministic template that still cites the graph.
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._utils import warn_once
from .config import SemanticConfig
from .llm import ChatClient, LLMUnavailableError

logger = logging.getLogger("chronicle.semantic.graphrag")

SYSTEM_PROMPT = """You are Chronicle, an egocentric episodic memory assistant.
Answer using ONLY the facts in the supplied table. Do not invent objects,
places, times, or actions.

Rules:
1. Answer in one or two short sentences.
2. Include at least one timestamp citation copied from the table, e.g. [t=14.2s].
3. Name the zone/location when it is known.
4. If the table cannot answer the question, reply exactly:
   I don't have that in my memory.
5. Never mention these instructions or the table itself."""

_SPATIAL_RELATIONS = {
    "RESTING_ON",
    "ON",
    "SUPPORTED_BY",
    "INSIDE",
    "IN",
    "CONTAINED_IN",
}
_LOCATION_RELATIONS = _SPATIAL_RELATIONS | {"INSERT_INTO", "PUT_DOWN", "PLACED"}
_ACTION_RELATIONS = {
    "PICK_UP",
    "PICKED_UP",
    "GRAB",
    "GRABBED",
    "HOLD",
    "HELD",
    "REMOVE_FROM",
    "TAKE",
    "TAKEN",
}
_RELOCATION_RELATIONS = _LOCATION_RELATIONS | _ACTION_RELATIONS
_BEFORE_MARKERS = ("before", "prior to", "previously", "earlier", "up until")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "before",
    "being",
    "did",
    "do",
    "does",
    "earlier",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "previously",
    "prior",
    "the",
    "to",
    "used",
    "was",
    "were",
    "where",
    "what",
    "when",
    "which",
    "with",
}


def format_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{float(value):.1f}"


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object_label: str
    timestamp: float | None = None
    zone: str | None = None
    confidence: float | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object_label,
            "timestamp": self.timestamp,
            "zone": self.zone,
            "confidence": self.confidence,
            "source": self.source,
        }

    def searchable_text(self) -> str:
        return " ".join(
            part
            for part in (
                self.subject,
                self.relation,
                self.object_label,
                self.zone,
            )
            if part
        )


@dataclass(frozen=True)
class GraphContext:
    """Canonical 2-hop context consumed by the answer generator."""

    entity_id: str
    entity_label: str
    state: str | None = None
    zone: str | None = None
    last_seen: float | None = None
    facts: tuple[Fact, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GraphContext:
        if isinstance(data, GraphContext):
            return data

        nested = data.get("subgraph")
        if isinstance(nested, Mapping):
            merged = dict(data)
            for key, value in nested.items():
                merged.setdefault(key, value)
            data = merged

        entity_info = data.get("entity")
        if isinstance(entity_info, Mapping):
            entity_label = str(
                entity_info.get("label")
                or entity_info.get("name")
                or entity_info.get("id")
                or data.get("entity_label")
                or ""
            )
            entity_id = str(
                entity_info.get("id")
                or entity_info.get("node_id")
                or entity_label
                or data.get("entity_id")
                or ""
            )
            state = entity_info.get("state")
            zone = entity_info.get("zone") or entity_info.get("location")
            last_seen = entity_info.get("last_seen") or entity_info.get("last_seen_at")
        else:
            entity_label = str(
                entity_info
                or data.get("entity_label")
                or data.get("label")
                or data.get("name")
                or data.get("id")
                or ""
            )
            entity_id = str(data.get("entity_id") or data.get("id") or entity_label)
            state = data.get("state")
            zone = data.get("zone") or data.get("location") or data.get("place")
            last_seen = data.get("last_seen") or data.get("last_seen_at")

        raw_facts = (
            data.get("facts")
            or data.get("relations")
            or data.get("edges")
            or data.get("rows")
            or []
        )
        if isinstance(raw_facts, Mapping):
            raw_facts = [raw_facts]
        facts = tuple(
            fact
            for item in raw_facts
            if (fact := _coerce_fact(item, entity_label)) is not None
        )

        if last_seen is None:
            timestamps = [fact.timestamp for fact in facts if fact.timestamp is not None]
            last_seen = max(timestamps) if timestamps else None
        else:
            try:
                last_seen = float(last_seen)
            except (TypeError, ValueError):
                last_seen = None

        if not zone:
            zone = _infer_zone(entity_id, entity_label, facts)

        attributes = data.get("attributes")
        return cls(
            entity_id=entity_id or entity_label,
            entity_label=entity_label or entity_id,
            state=None if state is None else str(state),
            zone=None if zone is None else str(zone),
            last_seen=last_seen,
            facts=facts,
            attributes=dict(attributes) if isinstance(attributes, Mapping) else {},
        )

    def limited_facts(self, limit: int) -> tuple[Fact, ...]:
        if limit <= 0 or len(self.facts) <= limit:
            return self.facts
        ordered = sorted(
            self.facts,
            key=lambda fact: (fact.timestamp is not None, fact.timestamp or 0.0),
            reverse=True,
        )[:limit]
        ordered.reverse()
        return tuple(ordered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_label": self.entity_label,
            "state": self.state,
            "zone": self.zone,
            "last_seen": self.last_seen,
            "facts": [fact.to_dict() for fact in self.facts],
            "attributes": dict(self.attributes),
        }


def _coerce_fact(item: Any, default_subject: str) -> Fact | None:
    if isinstance(item, Fact):
        return item
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        timestamp = None
        if len(item) >= 4:
            try:
                timestamp = float(item[3])
            except (TypeError, ValueError):
                timestamp = None
        return Fact(
            subject=str(item[0] or default_subject),
            relation=str(item[1]),
            object_label=str(item[2]),
            timestamp=timestamp,
        )
    if not isinstance(item, Mapping):
        return None

    subject = item.get("subject") or item.get("source") or item.get("from") or item.get("u")
    relation = (
        item.get("relation")
        or item.get("predicate")
        or item.get("type")
        or item.get("label")
        or item.get("action")
        or item.get("edge_type")
    )
    object_label = (
        item.get("object")
        or item.get("object_label")
        or item.get("target")
        or item.get("destination")
        or item.get("to")
        or item.get("v")
    )
    if relation is None or object_label is None:
        return None

    timestamp_value = (
        item.get("timestamp")
        if "timestamp" in item
        else item.get("t", item.get("tevent", item.get("time", item.get("start"))))
    )
    try:
        timestamp = None if timestamp_value is None else float(timestamp_value)
    except (TypeError, ValueError):
        timestamp = None

    confidence_value = item.get("confidence", item.get("score"))
    try:
        confidence = None if confidence_value is None else float(confidence_value)
    except (TypeError, ValueError):
        confidence = None

    return Fact(
        subject=str(subject or default_subject),
        relation=str(relation),
        object_label=str(object_label),
        timestamp=timestamp,
        zone=None if item.get("zone") is None else str(item.get("zone")),
        confidence=confidence,
        source=None if item.get("source") is None else str(item.get("source")),
    )


def _infer_zone(entity_id: str, entity_label: str, facts: Sequence[Fact]) -> str | None:
    keys = {entity_id.lower(), entity_label.lower()}
    candidates = [
        fact
        for fact in facts
        if fact.relation.upper() in _SPATIAL_RELATIONS
        and (fact.subject.lower() in keys or fact.object_label.lower() in keys)
    ]
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda fact: fact.timestamp if fact.timestamp is not None else float("-inf"),
    )
    if latest.zone:
        return latest.zone
    if latest.subject.lower() in keys:
        return latest.object_label
    return latest.subject


def serialize_subgraph(
    graph: Any,
    entity: str,
    *,
    hops: int = 2,
    max_facts: int = 40,
) -> GraphContext:
    """Extract an entity-centric 2-hop context from a NetworkX-like graph.

    Works with ``networkx.Graph``/``DiGraph`` via duck typing (no hard import).
    Edge attributes understood: ``relation``/``predicate``/``type``/``action``,
    ``timestamp``/``t``, ``zone``, ``confidence``.
    """
    if isinstance(graph, GraphContext):
        return graph
    if isinstance(graph, Mapping) and not hasattr(graph, "nodes"):
        return GraphContext.from_mapping(graph)

    nodes = getattr(graph, "nodes", None)
    if nodes is None:
        raise TypeError("graph must be a NetworkX-like object or a serialised mapping")
    if entity not in nodes:
        raise KeyError(f"entity {entity!r} is not present in the graph")

    def node_label(node: Any) -> str:
        attrs = nodes.get(node, {})
        if isinstance(attrs, Mapping):
            label = attrs.get("label") or attrs.get("name")
            if label:
                return str(label)
        return str(node)

    attributes = dict(nodes[entity]) if isinstance(nodes[entity], Mapping) else {}
    entity_label = str(attributes.get("label") or attributes.get("name") or entity)

    visited: set[Any] = {entity}
    queue: deque[tuple[Any, int]] = deque([(entity, 0)])
    seen_edges: set[tuple[str, str, int]] = set()
    facts: list[Fact] = []

    while queue and len(facts) < max_facts:
        node, depth = queue.popleft()
        if depth >= hops:
            continue

        edges: list[tuple[str, str, Mapping[str, Any]]] = []
        in_edges = getattr(graph, "in_edges", None)
        out_edges = getattr(graph, "out_edges", None)
        if callable(in_edges):
            edges.extend(in_edges(node, data=True))
        if callable(out_edges):
            edges.extend(out_edges(node, data=True))
        if not edges and hasattr(graph, "edges"):
            edges.extend(graph.edges(node, data=True))

        for source, target, data in edges:
            data = data if isinstance(data, Mapping) else {}
            marker = (str(source), str(target), id(data))
            if marker in seen_edges:
                continue
            seen_edges.add(marker)

            relation = (
                data.get("relation")
                or data.get("predicate")
                or data.get("type")
                or data.get("label")
                or data.get("action")
                or data.get("edge_type")
                or "RELATED_TO"
            )
            timestamp_value = data.get("timestamp", data.get("t", data.get("tevent", data.get("start"))))
            try:
                timestamp = None if timestamp_value is None else float(timestamp_value)
            except (TypeError, ValueError):
                timestamp = None
            confidence_value = data.get("confidence", data.get("score"))
            try:
                confidence = None if confidence_value is None else float(confidence_value)
            except (TypeError, ValueError):
                confidence = None

            facts.append(
                Fact(
                    subject=node_label(source),
                    relation=str(relation),
                    object_label=node_label(target),
                    timestamp=timestamp,
                    zone=None if data.get("zone") is None else str(data.get("zone")),
                    confidence=confidence,
                    source=None if data.get("source") is None else str(data.get("source")),
                )
            )

            for neighbor in (source, target):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

    context = GraphContext(
        entity_id=str(entity),
        entity_label=entity_label,
        state=attributes.get("state"),
        zone=attributes.get("zone") or attributes.get("location") or attributes.get("place"),
        last_seen=attributes.get("last_seen") or attributes.get("last_seen_at"),
        facts=tuple(facts),
        attributes=attributes,
    )
    # Reuse from_mapping's inference/coercion without losing the richer attrs.
    return GraphContext.from_mapping(context.to_dict())


def format_context(context: GraphContext, *, max_facts: int = 40) -> str:
    """Render the subgraph as the constrained table given to the LLM."""
    lines = [
        f"Entity: {context.entity_label} (id={context.entity_id})",
        f"Current state: {context.state or 'unknown'}",
        f"Current zone: {context.zone or 'unknown'}",
        f"Last seen: {'unknown' if context.last_seen is None else f't={format_timestamp(context.last_seen)}s'}",
        "",
        "| t (s) | subject | relation | object | zone | confidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    facts = context.limited_facts(max_facts)
    if not facts:
        lines.append("| - | - | no recorded facts | - | - | - |")
    for fact in facts:
        lines.append(
            "| {t} | {s} | {r} | {o} | {z} | {c} |".format(
                t="-" if fact.timestamp is None else format_timestamp(fact.timestamp),
                s=_escape(fact.subject),
                r=_escape(fact.relation),
                o=_escape(fact.object_label),
                z=_escape(fact.zone or "-"),
                c="-" if fact.confidence is None else f"{fact.confidence:.2f}",
            )
        )
    return "\n".join(lines)


def _escape(value: str) -> str:
    return str(value).replace("|", "\\|")


@dataclass(frozen=True)
class Citation:
    timestamp: float
    relation: str | None = None
    zone: str | None = None
    fact: Fact | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "relation": self.relation,
            "zone": self.zone,
        }

    def render(self) -> str:
        return f"[t={format_timestamp(self.timestamp)}s]"


@dataclass(frozen=True)
class QueryAnswer:
    query: str
    answer: str
    citations: tuple[Citation, ...]
    grounded: bool
    source: str
    context: GraphContext
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "grounded": self.grounded,
            "source": self.source,
            "entity": self.context.entity_label,
            "state": self.context.state,
            "zone": self.context.zone,
            "last_seen": self.context.last_seen,
        }


_CITATION_RE = re.compile(r"\[t=(\d+(?:\.\d+)?)s?\]|\bt=(\d+(?:\.\d+)?)s\b", re.IGNORECASE)


def extract_citations(text: str, context: GraphContext, tolerance: float = 0.35) -> tuple[Citation, ...]:
    """Map timestamps in an answer back to facts; drop hallucinated times."""
    found: list[Citation] = []
    seen: set[tuple[float, str | None]] = set()
    for match in _CITATION_RE.finditer(text or ""):
        raw = match.group(1) or match.group(2)
        try:
            timestamp = float(raw)
        except (TypeError, ValueError):
            continue

        nearest: Fact | None = None
        nearest_delta = float("inf")
        for fact in context.facts:
            if fact.timestamp is None:
                continue
            delta = abs(fact.timestamp - timestamp)
            if delta <= tolerance and delta < nearest_delta:
                nearest, nearest_delta = fact, delta

        # A timestamp that does not map back to the subgraph is a hallucination
        # and must not be shown to the user as a citation.
        if nearest is None:
            continue

        key = (timestamp, nearest.relation)
        if key in seen:
            continue
        seen.add(key)
        found.append(
            Citation(
                timestamp=float(nearest.timestamp if nearest.timestamp is not None else timestamp),
                relation=nearest.relation,
                zone=nearest.zone or context.zone,
                fact=nearest,
            )
        )
    return tuple(found)


def _text_tokens(text: str) -> set[str]:
    """Tokenise graph text, splitting relations and folding simple suffixes."""
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9_]+", str(text).lower()):
        if not raw:
            continue
        tokens.add(raw)
        if "_" in raw:
            tokens.update(part for part in raw.split("_") if part)
        for suffix, minimum in (("ing", 5), ("ed", 5), ("es", 4), ("s", 4)):
            if raw.endswith(suffix) and len(raw) > minimum:
                tokens.add(raw[: -len(suffix)])
    return tokens


def _query_tokens(query: str) -> set[str]:
    return {
        token
        for token in _text_tokens(query)
        if token not in _STOPWORDS and len(token) > 1
    }


def _score_fact(fact: Fact, entity_label: str, query_tokens: set[str]) -> float:
    score = 0.0
    text_tokens = _text_tokens(fact.searchable_text())
    score += 3.0 * len(query_tokens & text_tokens)

    entity = entity_label.lower()
    if fact.subject.lower() == entity or fact.object_label.lower() == entity:
        score += 2.0
    if fact.relation.upper() in _SPATIAL_RELATIONS:
        score += 1.0
    if fact.timestamp is not None:
        score += min(float(fact.timestamp), 3600.0) / 1_000_000.0
    if fact.confidence is not None:
        score += fact.confidence * 0.1
    return score


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _is_before_query(query: str) -> bool:
    lower = query.lower()
    return any(marker in lower for marker in _BEFORE_MARKERS)


def _fact_mentions(fact: Fact, tokens: set[str]) -> bool:
    if not tokens:
        return False
    return bool(tokens & _text_tokens(fact.searchable_text()))


def _reference_event(
    query: str,
    facts: Sequence[Fact],
    entity_label: str,
) -> tuple[float | None, Fact | None]:
    """Find the event the user means by "before": an action or a numeric time."""
    numeric = re.search(
        r"(?:\bt\s*=\s*|\bat\s+|before\s+)?(\d+(?:\.\d+)?)\s*"
        r"(?:s\b|sec(?:ond)?s?\b)",
        query,
        re.IGNORECASE,
    )
    if numeric:
        try:
            return float(numeric.group(1)), None
        except (TypeError, ValueError):
            pass

    timed = [fact for fact in facts if fact.timestamp is not None]
    if not timed:
        return None, None

    # Only compare the non-entity words so "cup" does not match every fact.
    tokens = _query_tokens(query) - _text_tokens(entity_label)
    relocations = [
        fact for fact in timed if fact.relation.upper() in _RELOCATION_RELATIONS
    ] or timed
    matched = [fact for fact in relocations if _fact_mentions(fact, tokens)]
    pool = matched or relocations
    reference = max(pool, key=lambda fact: fact.timestamp or float("-inf"))
    return reference.timestamp, reference


def _location_phrase(relation: str, target: str) -> str:
    relation = relation.upper()
    if relation in {"RESTING_ON", "ON", "SUPPORTED_BY"}:
        return f"resting on the {target}"
    if relation in {"INSIDE", "IN", "CONTAINED_IN", "INSERT_INTO"}:
        return f"inside the {target}"
    if relation in {"PUT_DOWN", "PLACED", "PLACE", "PUT"}:
        return f"on the {target}"
    if relation in {"PICK_UP", "PICKED_UP", "HELD", "HOLD"}:
        return "picked up"
    return f"{relation.replace('_', ' ').lower()} {target}"


def _reference_clause(reference_timestamp: float, reference: Fact | None) -> str:
    if reference is None:
        return f"before [t={format_timestamp(reference_timestamp)}s]"
    target = reference.object_label
    relation = reference.relation.upper()
    if relation in {"INSERT_INTO", "INSIDE", "IN", "CONTAINED_IN"}:
        return f"before it was put inside the {target}"
    if relation in {"PLACED", "PLACE", "PUT", "PUT_DOWN"}:
        return f"before it was placed on the {target}"
    if relation in {"PICK_UP", "PICKED_UP", "GRAB", "GRABBED"}:
        return "before it was picked up"
    return f"before it was {_location_phrase(relation, target)}"


def _before_answer(
    query: str,
    context: GraphContext,
    entity_facts: Sequence[Fact],
    label: str,
) -> QueryAnswer:
    """Walk the timeline backwards from a placement/action to the prior location."""
    reference_timestamp, reference = _reference_event(query, entity_facts, label)
    if reference_timestamp is None:
        return QueryAnswer(
            query=query,
            answer="I don't have that in my memory.",
            citations=(),
            grounded=False,
            source="template",
            context=context,
        )

    earlier = [
        fact
        for fact in entity_facts
        if fact.timestamp is not None and fact.timestamp < reference_timestamp - 1e-6
    ]
    location_earlier = [
        fact for fact in earlier if fact.relation.upper() in _LOCATION_RELATIONS
    ]
    candidates = location_earlier or earlier
    if not candidates:
        return QueryAnswer(
            query=query,
            answer=(
                f"I don't have an earlier location for the {label} "
                f"before [t={format_timestamp(reference_timestamp)}s]."
            ),
            citations=(),
            grounded=False,
            source="template",
            context=context,
            raw_response=None,
        )

    best = max(candidates, key=lambda fact: fact.timestamp or float("-inf"))
    target = best.object_label
    zone = best.zone or context.zone
    same_place = bool(zone and target and zone.casefold() == target.casefold())
    where = f" in the {zone}" if zone and not same_place else ""
    when = f" at [t={format_timestamp(best.timestamp)}s]"
    phrase = _location_phrase(best.relation, target)
    answer = (
        f"{_cap(_reference_clause(reference_timestamp, reference))}, "
        f"the {label} had been {phrase}{where}{when}."
    )
    citation = Citation(
        timestamp=float(best.timestamp),
        relation=best.relation,
        zone=zone,
        fact=best,
    )
    return QueryAnswer(
        query=query,
        answer=answer,
        citations=(citation,),
        grounded=True,
        source="template",
        context=context,
    )


def template_answer(query: str, context: GraphContext) -> QueryAnswer:
    """Deterministic, graph-grounded fallback answer."""
    label = context.entity_label or context.entity_id or "that object"
    tokens = _query_tokens(query)

    if not context.facts:
        if context.zone or context.last_seen is not None:
            when = (
                f" at [t={format_timestamp(context.last_seen)}s]"
                if context.last_seen is not None
                else ""
            )
            where = f" in the {context.zone}" if context.zone else ""
            answer = f"{_cap(label)} was last seen{where}{when}."
            citation = (
                Citation(
                    timestamp=float(context.last_seen),
                    zone=context.zone,
                )
                if context.last_seen is not None
                else None
            )
            return QueryAnswer(
                query=query,
                answer=answer,
                citations=(citation,) if citation else (),
                grounded=citation is not None,
                source="template",
                context=context,
            )
        return QueryAnswer(
            query=query,
            answer="I don't have that in my memory.",
            citations=(),
            grounded=False,
            source="template",
            context=context,
        )

    facts = list(context.facts)
    entity_key = label.lower()
    entity_facts = [
        fact
        for fact in facts
        if fact.subject.lower() == entity_key or fact.object_label.lower() == entity_key
    ] or facts
    query_lower = query.lower()

    if _is_before_query(query):
        return _before_answer(query, context, entity_facts, label)

    if any(word in query_lower for word in ("where", "location", "find", "lost")):
        pool = [
            fact
            for fact in entity_facts
            if fact.relation.upper() in _LOCATION_RELATIONS
        ] or entity_facts
        best = max(
            pool,
            key=lambda fact: (
                fact.timestamp if fact.timestamp is not None else float("-inf"),
                _score_fact(fact, label, tokens),
            ),
        )
    elif any(
        phrase in query_lower
        for phrase in ("what did", "happen", "action", "do with", "when")
    ):
        best = max(
            entity_facts,
            key=lambda fact: (
                fact.timestamp if fact.timestamp is not None else float("-inf"),
                _score_fact(fact, label, tokens),
            ),
        )
    else:
        best = max(facts, key=lambda fact: _score_fact(fact, label, tokens))
    relation = best.relation.upper()
    target = best.object_label
    zone = best.zone or context.zone
    when = (
        f" at [t={format_timestamp(best.timestamp)}s]"
        if best.timestamp is not None
        else ""
    )
    same_place = bool(zone and target and zone.casefold() == target.casefold())
    where = f" in the {zone}" if zone and not same_place else ""

    if relation in {"RESTING_ON", "ON", "SUPPORTED_BY"}:
        answer = f"{_cap(label)} was last seen resting on the {target}{where}{when}."
    elif relation in {"INSIDE", "IN", "CONTAINED_IN", "INSERT_INTO"}:
        answer = f"{_cap(label)} was last seen inside the {target}{where}{when}."
    elif relation in {"PUT_DOWN", "PLACED"}:
        answer = f"{_cap(label)} was last seen on the {target}{where}{when}."
    elif relation in {"PICK_UP", "PICKED_UP", "HELD", "HOLD"}:
        answer = f"{_cap(label)} was picked up{where}{when}."
    else:
        answer = (
            f"The last recorded event for {label} was "
            f"{relation.replace('_', ' ').lower()} {target}{where}{when}."
        )

    if context.state and context.state.upper() == "LOST":
        answer += " It is currently marked LOST."

    citation = (
        Citation(
            timestamp=best.timestamp,
            relation=best.relation,
            zone=zone,
            fact=best,
        )
        if best.timestamp is not None
        else None
    )
    return QueryAnswer(
        query=query,
        answer=answer,
        citations=(citation,) if citation is not None else (),
        grounded=best.timestamp is not None,
        source="template",
        context=context,
    )


@dataclass
class GraphRAG:
    chat_client: ChatClient | None = None
    config: SemanticConfig = field(default_factory=SemanticConfig)

    def answer(self, query: str, context: GraphContext | Mapping[str, Any]) -> QueryAnswer:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        resolved = (
            context
            if isinstance(context, GraphContext)
            else GraphContext.from_mapping(context)
        )

        if self.chat_client is None:
            return template_answer(query, resolved)

        user_prompt = (
            f"Question: {query.strip()}\n\n"
            f"Memory (2-hop subgraph):\n{format_context(resolved, max_facts=self.config.max_facts)}\n\n"
            "Answer in one or two sentences with an explicit [t=...s] citation:"
        )
        try:
            raw = self.chat_client.complete(SYSTEM_PROMPT, user_prompt)
        except LLMUnavailableError as exc:
            warn_once(
                "graphrag-llm-fallback",
                "GraphRAG LLM unavailable (%s); using template answer.",
                exc,
            )
            return template_answer(query, resolved)

        text = (raw or "").strip()
        citations = extract_citations(text, resolved)
        if not text:
            return template_answer(query, resolved)
        if self.config.require_citations and not citations:
            warn_once(
                "graphrag-uncited",
                "LLM answer had no verifiable citation; using template answer.",
            )
            return template_answer(query, resolved)

        return QueryAnswer(
            query=query,
            answer=text,
            citations=citations,
            grounded=bool(citations),
            source="llm",
            context=resolved,
            raw_response=raw,
        )

    def answer_json(self, query: str, context: GraphContext | Mapping[str, Any]) -> str:
        return json.dumps(self.answer(query, context).to_dict(), indent=2)

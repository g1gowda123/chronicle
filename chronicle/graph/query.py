"""Two-stage GraphRAG retrieval for natural-language queries.

Stage 1 resolves the queried entity (string match, optionally CLIP text-to-node
similarity) and reads its latest state plus active spatial edge.  Stage 2
serializes the 2-hop relational neighbourhood — including the parent zone anchor
— into a compact table that a small language model can answer from without
hallucinating.  No breadth-first "answer" is attempted here: the LLM speaks,
the graph is the only source of truth.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from chronicle.attributes import (
    attribute_query_field,
    attribute_summary,
    describe_attribute,
    public_attributes,
)
from chronicle.config import DEFAULT_THRESHOLDS, Thresholds
from chronicle.geometry import cosine_similarity
from chronicle.graph.background import BackgroundGraph
from chronicle.graph.foreground import ForegroundGraph
from chronicle.graph.ids import (
    ANCHOR_PREFIX,
    ZONE_PREFIX,
    entity_node,
    raw_entity_id,
)
from chronicle.storage.records import AnchorRecord
from chronicle.storage.sqlite_store import SQLiteStore
from chronicle.telemetry import EventBus
from chronicle.text import is_plural, match_details
from chronicle.types import Relation

Embedder = Callable[[str], Sequence[float]]

_INTERACTION_STATES = {"INTERACTED", "HELD", "CONTAINED"}
_VISIBLE_STATES = {
    "NEW",
    "VISIBLE_STATIC",
    "VISIBLE_MOVING",
    "INTERACTED",
    "HELD",
    "CONTAINED",
}
_BEFORE_RE = re.compile(
    r"\b(?:before|prior to|earlier than)(?:\s+(.+?))?(?:\?|\.(?:\s|$)|$)",
    re.IGNORECASE,
)
_PREVIOUS_RE = re.compile(
    r"\b(?:previously|earlier|before that|before then|prior location|former location|previous location)\b",
    re.IGNORECASE,
)
_EVENT_RE = re.compile(
    r"\b(?:what happened|what did i do|what was i doing|tell me what happened)\b",
    re.IGNORECASE,
)
_PRONOUN_CLAUSES = {"that", "this", "then", "now", "it", "they", "them", "that event", "that time"}
_STOPWORDS = {
    "a", "an", "and", "are", "at", "did", "do", "does", "for", "from", "i",
    "in", "is", "it", "me", "my", "of", "on", "the", "to", "was", "were",
    "where", "what", "when", "which", "with",
}


def is_temporal_multihop_query(query: str) -> bool:
    """Return True if query asks about a previous state, location, or action."""
    query_lower = query.lower()
    return bool(_BEFORE_RE.search(query_lower) or _PREVIOUS_RE.search(query_lower))


def is_event_query(query: str) -> bool:
    """Return True if the query asks for a narrative event summary."""
    return bool(_EVENT_RE.search(query)) and not is_temporal_multihop_query(query)


@dataclass
class QueryResult:
    """Serialized answer context for one entity."""

    query: str
    entity_id: str
    node_id: str
    label: str
    state: str
    state_since: float
    last_seen: float
    match_score: float
    confidence: float = 1.0
    match_reason: str = ""
    last_interaction: dict[str, Any] | None = None
    terminal_anchor: dict[str, Any] | None = None
    container_id: str | None = None
    container_label: str | None = None
    zone_id: str | None = None
    zone_label: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    # --------------------------------------------------------------- rendering
    def to_table(self) -> str:
        lines = [
            "| t (s) | relation | from | to | details |",
            "| --- | --- | --- | --- | --- |",
        ]
        rows = sorted(
            (e for e in self.edges if e.get("t_start") is not None),
            key=lambda e: (float(e["t_start"]), str(e.get("relation"))),
        )
        for edge in rows:
            relation = str(edge.get("relation", ""))
            details = ""
            if relation == Relation.STATE_TRANSITION.value:
                details = f"{edge.get('from_state')} -> {edge.get('to_state')}"
                reason = (edge.get("evidence") or {}).get("reason")
                if reason:
                    details += f" ({reason})"
            elif relation == Relation.ACTION.value:
                details = (
                    f"{edge.get('action')} conf={edge.get('confidence')} src={edge.get('source')}"
                )
            elif relation in {Relation.RESTING_ON.value, Relation.INSIDE.value}:
                details = "terminal" if edge.get("is_terminal") else "closed"
                if edge.get("t_end") is not None:
                    details += f", ended t={float(edge['t_end']):.1f}s"
            elif relation == Relation.IN_ZONE.value:
                details = "parent zone"
            lines.append(
                "| {t:.1f} | {rel} | {src} | {dst} | {details} |".format(
                    t=float(edge["t_start"]),
                    rel=relation,
                    src=_short_node(edge.get("src", "")),
                    dst=_short_node(edge.get("dst", "")),
                    details=details,
                )
            )
        if len(lines) == 2:
            lines.append("| - | - | - | - | no timestamped edges |")
        return "\n".join(lines)

    def to_prompt(self, question: str | None = None) -> str:
        question = question or self.query or f"Where is the {self.label}?"
        anchor = self.terminal_anchor
        anchor_text = "none recorded"
        if anchor:
            anchor_text = (
                f"{anchor.get('label') or anchor.get('anchor_id')} "
                f"(relation={anchor.get('relation')}, t={float(anchor.get('t_start', 0.0)):.1f}s"
            )
            if anchor.get("zone_label"):
                anchor_text += f", zone={anchor['zone_label']}"
            anchor_text += ")"
        interaction = self.last_interaction
        interaction_text = "none recorded"
        if interaction:
            interaction_text = (
                f"{interaction.get('kind')} at t={float(interaction.get('t', 0.0)):.1f}s "
                f"by {interaction.get('agent', 'USER')}"
            )
        appearance = attribute_summary(self.attributes, limit=8) or "none recorded"
        return (
            "You are Chronicle, an egocentric episodic memory assistant.\n"
            "Answer the question using ONLY the retrieved graph context below.\n"
            "Always cite the timestamp as t=<seconds>s and the location by zone or\n"
            "support-surface label. If the object is currently held or contained, say\n"
            "so explicitly. Never invent observations that are absent from the table.\n"
            "For temporal queries (e.g. 'where was X before Y?', 'what was X's previous location?'),\n"
            "walk backward along the timestamped timeline to identify the previous location or\n"
            "action that occurred before the referenced event.\n\n"
            f"QUESTION: {question}\n\n"
            f"ENTITY: {self.label} (id={self.entity_id})\n"
            f"MATCH CONFIDENCE: {self.confidence:.2f} ({self.match_reason or 'direct lookup'})\n"
            f"CURRENT STATE: {self.state} since t={self.state_since:.1f}s\n"
            f"LAST OBSERVED: t={self.last_seen:.1f}s\n"
            f"LOOK: {appearance}\n"
            f"LAST INTERACTION: {interaction_text}\n"
            f"TERMINAL ANCHOR: {anchor_text}\n"
            f"CONTAINER: {self.container_label or 'none'}\n"
            f"ZONE: {self.zone_label or 'unknown'}\n\n"
            "RETRIEVED 2-HOP SUBGRAPH:\n"
            f"{self.to_table()}\n"
        )

    def timeline(self) -> list[dict[str, Any]]:
        """Chronologically sorted list of edges incident on this entity.

        The retrieved subgraph also contains edges belonging to entities that
        share an anchor surface (for example two cups resting on the same
        table).  Keeping those edges here makes event summaries report another
        object's placement as this object's last move, so filter to the entity
        node and its own incident edges.
        """

        return sorted(
            (
                e
                for e in self.edges
                if e.get("t_start") is not None
                and (e.get("src") == self.node_id or e.get("dst") == self.node_id)
            ),
            key=lambda e: (float(e["t_start"]), str(e.get("relation"))),
        )

    def walk_timeline_before(
        self, query: str | None = None
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None, bool]:
        """Walk backward along timestamped edges to resolve temporal multi-hop queries.

        Returns (prior_edge, ref_edge, ref_label, ref_found).
        """
        query_text = (query or self.query or "").strip()
        if not query_text:
            return None, None, None, False

        m_before = _BEFORE_RE.search(query_text)
        m_prev = _PREVIOUS_RE.search(query_text)
        if not m_before and not m_prev:
            return None, None, None, False

        node_labels = {
            str(n["node_id"]): str(n.get("label") or _short_node(str(n["node_id"])))
            for n in self.nodes
        }
        node_labels[self.node_id] = self.label

        entity_edges = [
            e
            for e in self.edges
            if (e.get("src") == self.node_id or e.get("dst") == self.node_id)
            and e.get("t_start") is not None
        ]
        if not entity_edges:
            return None, None, None, False

        entity_edges.sort(key=lambda e: float(e["t_start"]))
        is_action = any(
            tok in query_text.lower()
            for tok in ("what did", "action", "happen", "do with", "done", "user")
        )

        clause = (m_before.group(1) or "").strip() if m_before else None
        pickup_ref = bool(
            clause
            and re.search(
                r"\b(?:pick(?:ed)?\s*up|pickup|lift(?:ed)?(?:\s+up)?)\b",
                clause,
                re.IGNORECASE,
            )
        )
        placement_ref = bool(
            clause
            and re.search(
                r"\b(?:place(?:d)?|put|setting|set\s+down)\b",
                clause,
                re.IGNORECASE,
            )
        )
        held_edges = [
            edge
            for edge in entity_edges
            if edge.get("relation") == Relation.STATE_TRANSITION.value
            and edge.get("to_state") == "HELD"
        ]
        if pickup_ref and held_edges:
            ref_edge = held_edges[0]
            t_ref = float(ref_edge["t_start"])
            ref_label = "pick-up"
        elif clause and clause.lower() not in _PRONOUN_CLAUSES:
            m_ts = re.fullmatch(r"(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*s?", clause, re.IGNORECASE)
            if m_ts:
                t_ref = float(m_ts.group(1))
                ref_edge = None
                ref_label = f"t={t_ref:.1f}s"
            else:
                tokens = {
                    tok
                    for tok in re.findall(r"[a-z0-9_]+", clause.lower())
                    if tok not in _STOPWORDS
                }
                best_edge: dict[str, Any] | None = None
                best_score = 0.0
                for edge in entity_edges:
                    score = 0.0
                    src_lbl = node_labels.get(str(edge.get("src")), "")
                    dst_lbl = node_labels.get(str(edge.get("dst")), "")
                    rel = str(edge.get("relation", ""))
                    act = str(edge.get("action", ""))
                    text = f"{src_lbl} {dst_lbl} {rel} {act}".lower()
                    text_toks = set(re.findall(r"[a-z0-9_]+", text))
                    score += 3.0 * len(tokens & text_toks)
                    if any(t in dst_lbl.lower() for t in tokens):
                        score += 5.0
                    if any(t in act.lower() for t in tokens):
                        score += 4.0
                    if score > best_score:
                        best_score = score
                        best_edge = edge
                if best_edge is not None and best_score > 0:
                    ref_edge = best_edge
                    t_ref = float(best_edge["t_start"])
                    dst_id = str(best_edge.get("dst", ""))
                    ref_label = node_labels.get(dst_id, _short_node(dst_id))
                elif placement_ref:
                    placements = [
                        e
                        for e in entity_edges
                        if e.get("relation")
                        in {Relation.RESTING_ON.value, Relation.INSIDE.value}
                    ]
                    if not placements:
                        return None, None, clause, False
                    ref_edge = placements[-1]
                    t_ref = float(ref_edge["t_start"])
                    dst_id = str(ref_edge.get("dst", ""))
                    ref_label = node_labels.get(dst_id, _short_node(dst_id))
                else:
                    return None, None, clause, False
        else:
            placements = [
                e
                for e in entity_edges
                if e.get("relation") in {Relation.RESTING_ON.value, Relation.INSIDE.value}
            ]
            ref_edge = placements[-1] if placements else entity_edges[-1]
            t_ref = float(ref_edge["t_start"])
            dst_id = str(ref_edge.get("dst", ""))
            ref_label = node_labels.get(dst_id, _short_node(dst_id))

        prior_edges = [
            e for e in entity_edges if float(e["t_start"]) < t_ref - 1e-4
        ]
        if not prior_edges:
            return None, ref_edge, ref_label, True

        if is_action:
            action_cands = [e for e in prior_edges if e.get("relation") == Relation.ACTION.value]
            prior_edge = action_cands[-1] if action_cands else prior_edges[-1]
        else:
            loc_cands = [
                e
                for e in prior_edges
                if e.get("relation") in {Relation.RESTING_ON.value, Relation.INSIDE.value}
            ]
            prior_edge = loc_cands[-1] if loc_cands else prior_edges[-1]

        return prior_edge, ref_edge, ref_label, True

    def event_summary(self) -> str | None:
        """Summarize the recorded pick-up/placement sequence for this entity."""

        timeline = self.timeline()
        placements = [
            edge
            for edge in timeline
            if edge.get("relation") in {Relation.RESTING_ON.value, Relation.INSIDE.value}
        ]
        if not placements:
            # A short clip may contain an interaction without a confirmed
            # placement (for example the object is picked up but the clip ends
            # before it is put down).  Still report the interaction rather than
            # claiming that nothing happened.
            interaction = next(
                (
                    edge
                    for edge in timeline
                    if edge.get("relation") == Relation.ACTION.value
                ),
                None,
            )
            if interaction is None:
                interaction = next(
                    (
                        edge
                        for edge in timeline
                        if edge.get("relation") == Relation.STATE_TRANSITION.value
                        and edge.get("to_state") == "INTERACTED"
                    ),
                    None,
                )
            if interaction is None:
                return None
            interaction_t = float(interaction.get("t_start", 0.0))
            return (
                f"You interacted with the {self.label} at t={interaction_t:.1f}s; "
                "no placement was recorded."
            )

        labels = {
            str(node["node_id"]): str(node.get("label") or _short_node(str(node["node_id"])))
            for node in self.nodes
        }
        labels[self.node_id] = self.label

        def destination(edge: dict[str, Any]) -> str:
            return labels.get(str(edge.get("dst", "")), _short_node(str(edge.get("dst", ""))))

        def preposition(edge: dict[str, Any]) -> str:
            return "inside" if edge.get("relation") == Relation.INSIDE.value else "on"

        first = placements[0]
        last = placements[-1]
        first_label = destination(first)
        last_label = destination(last)
        first_t = float(first.get("t_start", 0.0))
        last_t = float(last.get("t_start", 0.0))
        pickup = next(
            (
                edge
                for edge in timeline
                if edge.get("relation") == Relation.STATE_TRANSITION.value
                and edge.get("to_state") == "HELD"
            ),
            None,
        )
        pickup_t = float(pickup.get("t_start", 0.0)) if pickup else None
        pronoun = "them" if is_plural(self.label) else "it"
        subject = "They" if is_plural(self.label) else "It"

        if first_label == last_label:
            if pickup_t is None:
                return (
                    f"The {self.label} stayed {preposition(last)} the {last_label}; "
                    f"no pick-up was recorded."
                )
            if self.state == "HELD":
                return (
                    f"You picked up the {self.label} at t={pickup_t:.1f}s "
                    f"and are still holding {pronoun}."
                )
            return (
                f"You picked up the {self.label} at t={pickup_t:.1f}s and put {pronoun} back "
                f"{preposition(last)} the {last_label} at t={last_t:.1f}s."
            )

        if pickup_t is None:
            return (
                f"The {self.label} moved from the {first_label} to the {last_label} "
                f"at t={last_t:.1f}s; no pick-up was recorded."
            )
        return (
            f"You picked up the {self.label} at t={pickup_t:.1f}s and placed {pronoun} "
            f"{preposition(last)} the {last_label} at t={last_t:.1f}s. "
            f"{subject} had been {preposition(first)} the {first_label} at t={first_t:.1f}s."
        )

    def fallback_answer(self) -> str:
        """Deterministic answer used when no local LLM is configured."""

        label = self.label
        query_text = (self.query or "").strip()
        if is_event_query(query_text):
            summary = self.event_summary()
            if summary:
                return summary
        field = attribute_query_field(query_text)
        if field:
            if not self.attributes:
                return (
                    f"I don't have appearance details for the {label} yet. "
                    "The perception pass has not recorded a usable crop."
                )
            description = describe_attribute(self.attributes, field)
            if not description:
                return f"I don't have that attribute recorded for the {label}."
            if field == "all":
                return (
                    f"The {label} {'look' if is_plural(label) else 'looks'} {description} "
                    f"(last observed at t={self.last_seen:.1f}s)."
                )
            return (
                f"The {label} {'are' if is_plural(label) else 'is'} {description} "
                f"(last observed at t={self.last_seen:.1f}s)."
            )
        if is_temporal_multihop_query(query_text):
            prior_edge, ref_edge, ref_label, ref_found = self.walk_timeline_before(query_text)
            if not ref_found:
                return "I don't have that in my memory."
            if prior_edge is None:
                target = ref_label or "that"
                return f"I don't have a record of where {label} was before the {target}."

            node_labels = {
                str(n["node_id"]): str(n.get("label") or _short_node(str(n["node_id"])))
                for n in self.nodes
            }
            dst_id = str(prior_edge.get("dst", ""))
            dst_label = node_labels.get(dst_id, _short_node(dst_id))
            t_start = float(prior_edge.get("t_start", 0.0))
            rel = str(prior_edge.get("relation", ""))
            zone = (
                f" in the {self.zone_label}"
                if self.zone_label and self.zone_label.lower() != dst_label.lower()
                else ""
            )
            subject = f"the {label}"
            past_be = "were" if is_plural(label) else "was"

            if _PREVIOUS_RE.search(query_text.lower()) and not _BEFORE_RE.search(query_text.lower()):
                prefix = "Previously"
            elif ref_edge is not None:
                ref_rel = str(ref_edge.get("relation", ""))
                if ref_rel == Relation.INSIDE.value:
                    prefix = f"Before being placed in the {ref_label}"
                elif ref_rel == Relation.RESTING_ON.value:
                    prefix = f"Before being placed on the {ref_label}"
                elif (
                    ref_rel == Relation.STATE_TRANSITION.value
                    and str(ref_edge.get("to_state")) == "HELD"
                ):
                    prefix = "Before being picked up"
                elif ref_rel == Relation.ACTION.value and str(ref_edge.get("action")) == "PICK_UP":
                    prefix = "Before being picked up"
                else:
                    prefix = f"Before the {ref_label}"
            else:
                prefix = f"Before {ref_label or 'that'}"

            if rel == Relation.RESTING_ON.value:
                return (
                    f"{prefix}, {subject} {past_be} resting on the {dst_label}{zone} "
                    f"at t={t_start:.1f}s."
                )
            if rel == Relation.INSIDE.value:
                return (
                    f"{prefix}, {subject} {past_be} inside the {dst_label}{zone} "
                    f"at t={t_start:.1f}s."
                )
            if rel == Relation.ACTION.value:
                act = str(prior_edge.get("action", "interacted")).lower().replace("_", " ")
                return f"{prefix}, {subject} {past_be} {act} at t={t_start:.1f}s."
            return f"{prefix}, {subject} {past_be} at {dst_label}{zone} at t={t_start:.1f}s."

        anchor = self.terminal_anchor
        location = "an unrecorded location"
        if anchor:
            anchor_label = anchor.get("label") or _anchor_ref(str(anchor.get("anchor_id", "")))
            zone = f" in the {self.zone_label}" if self.zone_label else ""
            location = f"the {anchor_label}{zone}"
            placed_at = float(anchor.get("t_start", self.last_seen))
        else:
            placed_at = None

        be = "are" if is_plural(label) else "is"
        if self.state == "HELD":
            answer = f"You are holding the {label} (state HELD since t={self.state_since:.1f}s)."
        elif self.state == "CONTAINED":
            container = self.container_label or "a container"
            answer = (
                f"The {label} {be} inside the {container}"
                f"{' in the ' + self.zone_label if self.zone_label else ''} "
                f"(state CONTAINED since t={self.state_since:.1f}s)."
            )
        elif self.state in {"VISIBLE_STATIC"}:
            answer = f"The {label} {be} resting on {location}."
            if placed_at is not None:
                answer += f" Last placed at t={placed_at:.1f}s."
        elif self.state in {"VISIBLE_MOVING", "INTERACTED"}:
            answer = (
                f"The {label} {be} currently {self.state.lower()} "
                f"(since t={self.state_since:.1f}s)."
            )
            if anchor and placed_at is not None:
                answer += f" It was last placed on {location} at t={placed_at:.1f}s."
        elif self.state in {"OCCLUDED", "OUT_OF_VIEW", "LOST"}:
            answer = (
                f"The {label} {be} not currently visible (state {self.state}, "
                f"last seen at t={self.last_seen:.1f}s)."
            )
            if anchor and placed_at is not None:
                answer += f" Its last known placement was {location} at t={placed_at:.1f}s."
        else:
            answer = f"The {label} {be} in state {self.state} as of t={self.last_seen:.1f}s."

        if self.last_interaction:
            answer += (
                f" Last interaction: {self.last_interaction.get('kind')} at "
                f"t={float(self.last_interaction.get('t', 0.0)):.1f}s."
            )
        return answer

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "entity_id": self.entity_id,
            "node_id": self.node_id,
            "label": self.label,
            "state": self.state,
            "state_since": round(self.state_since, 3),
            "last_seen": round(self.last_seen, 3),
            "match_score": round(self.match_score, 3),
            "confidence": round(self.confidence, 3),
            "match_reason": self.match_reason,
            "candidates": self.candidates,
            "last_interaction": self.last_interaction,
            "terminal_anchor": self.terminal_anchor,
            "container_id": self.container_id,
            "container_label": self.container_label,
            "zone_id": self.zone_id,
            "zone_label": self.zone_label,
            "attributes": self.attributes,
            "nodes": self.nodes,
            "edges": [
                {
                    **{k: v for k, v in edge.items() if k not in {"evidence", "embedding", "bbox"}},
                }
                for edge in self.edges
            ],
        }


def _short_node(node_id: str) -> str:
    for prefix in ("entity:", "anchor:", "zone:"):
        if node_id.startswith(prefix):
            return node_id[len(prefix) :]
    return node_id


def _anchor_ref(anchor_id: str) -> str:
    return _short_node(anchor_id)


class GraphQuery:
    """Entity retrieval + 2-hop spatiotemporal serialization."""

    def __init__(
        self,
        foreground: ForegroundGraph,
        background: BackgroundGraph,
        *,
        store: SQLiteStore | None = None,
        thresholds: Thresholds | None = None,
        bus: EventBus | None = None,
        min_match: float = 0.5,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.foreground = foreground
        self.background = background
        self.store = store
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.bus = bus or EventBus()
        self.min_match = min_match
        self.entity_aliases: dict[str, str] = (
            aliases if isinstance(aliases, dict) else dict(aliases or {})
        )

    def canonical_entity(self, entity_id: str) -> str:
        value = raw_entity_id(str(entity_id).strip())
        seen: set[str] = set()
        while value in self.entity_aliases and value not in seen:
            seen.add(value)
            value = self.entity_aliases[value]
        return value

    # ------------------------------------------------------------- resolution
    def rank(
        self,
        text: str,
        *,
        embedder: Embedder | None = None,
        min_score: float | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Rank entity candidates with aliases, visibility and confidence."""

        threshold = self.min_match if min_score is None else min_score
        query_embedding = None
        if embedder is not None:
            try:
                query_embedding = tuple(float(v) for v in embedder(text))
            except Exception as exc:  # pragma: no cover - embedder failure
                self.bus.query("text embedder failed", entity=text, error=repr(exc))

        candidates: list[dict[str, Any]] = []
        for attrs in self.foreground.entities():
            label = str(attrs.get("label", ""))
            score, reason = match_details(text, label)
            if query_embedding is not None and attrs.get("embedding"):
                similarity = cosine_similarity(query_embedding, attrs["embedding"])
                if similarity > score:
                    score = similarity
                    reason = "embedding similarity"
            if score <= 0.0:
                continue

            state = str(attrs.get("state") or "NEW")
            visible = state in _VISIBLE_STATES
            rank_score = score + (0.04 if visible else 0.0) + (0.01 if score >= 1.0 else 0.0)
            candidates.append(
                {
                    "entity_id": raw_entity_id(str(attrs["node_id"])),
                    "label": label,
                    "state": state,
                    "match_score": round(float(score), 4),
                    "reason": reason,
                    "last_seen": float(attrs.get("last_seen", 0.0) or 0.0),
                    "visible": visible,
                    "_rank": rank_score,
                }
            )

        candidates.sort(
            key=lambda item: (-item["_rank"], -item["last_seen"], item["entity_id"])
        )
        if not candidates or float(candidates[0]["match_score"]) < threshold:
            self.bus.query(
                "entity resolution miss",
                query=text,
                best=candidates[0]["entity_id"] if candidates else None,
                score=round(float(candidates[0]["match_score"]), 3) if candidates else 0.0,
                threshold=threshold,
            )
            return []

        best = candidates[0]
        second_score = next(
            (
                float(item["_rank"])
                for item in candidates[1:]
                if float(item["match_score"]) >= threshold
            ),
            0.0,
        )
        separation = (
            1.0
            if second_score <= 0.0
            else max(0.0, 1.0 - second_score / max(float(best["_rank"]), 1e-9))
        )
        confidence = min(1.0, float(best["match_score"]) * (0.70 + 0.30 * separation))
        best["confidence"] = round(confidence, 4)
        for item in candidates[1:]:
            item["confidence"] = round(min(1.0, float(item["match_score"]) * 0.70), 4)

        self.bus.query(
            "entity resolved",
            query=text,
            entity=best["entity_id"],
            score=round(float(best["match_score"]), 3),
            confidence=round(confidence, 3),
            reason=best["reason"],
            candidates=len(candidates),
        )
        public = [
            {key: value for key, value in item.items() if key != "_rank"}
            for item in candidates[: max(1, int(limit))]
        ]
        return public

    def resolve(
        self,
        text: str,
        *,
        embedder: Embedder | None = None,
        min_score: float | None = None,
    ) -> tuple[str | None, float]:
        ranking = self.rank(text, embedder=embedder, min_score=min_score)
        if not ranking:
            return None, 0.0
        best = ranking[0]
        return str(best["entity_id"]), float(best["match_score"])

    # ------------------------------------------------------------------ build
    def build(self, entity_id: str, query: str = "") -> QueryResult | None:
        entity_id = self.canonical_entity(entity_id)
        attrs = self.foreground.get_entity(entity_id)
        if attrs is None:
            return None
        node_id = entity_node(entity_id)

        nodes, edges = self.foreground.subgraph_nodes_edges(node_id, depth=2)
        nodes, edges = self._attach_background(nodes, edges)
        nodes = [self.node_public_attrs(node) for node in nodes]

        anchor_record = self._terminal_anchor(entity_id)
        anchor = self._anchor_payload(anchor_record, entity_id)
        zone_id = (anchor or {}).get("zone_id") or attrs.get("zone_id")
        if zone_id:
            zone_id = zone_id if zone_id.startswith("zone:") else f"zone:{zone_id}"
        zone_label = self.background.zone_label(zone_id)

        container_id = attrs.get("container_id")
        container_label = None
        if container_id:
            container_attrs = self.foreground.get_entity(str(container_id))
            if container_attrs:
                container_label = str(container_attrs.get("label"))

        return QueryResult(
            query=query,
            entity_id=entity_id,
            node_id=node_id,
            label=str(attrs.get("label", entity_id)),
            state=str(attrs.get("state", "NEW")),
            state_since=float(attrs.get("state_since", attrs.get("last_seen", 0.0))),
            last_seen=float(attrs.get("last_seen", 0.0)),
            match_score=1.0,
            last_interaction=self._last_interaction(entity_id),
            terminal_anchor=anchor,
            container_id=str(container_id) if container_id else None,
            container_label=container_label,
            zone_id=zone_id,
            zone_label=zone_label,
            attributes=public_attributes(attrs.get("attrs") or {}),
            nodes=nodes,
            edges=edges,
        )

    def query(
        self,
        text: str,
        *,
        embedder: Embedder | None = None,
    ) -> QueryResult | None:
        ranking = self.rank(text, embedder=embedder, limit=8)
        if not ranking:
            return None
        best = ranking[0]
        result = self.build(str(best["entity_id"]), query=text)
        if result is not None:
            result.match_score = float(best["match_score"])
            result.confidence = float(best.get("confidence", best["match_score"]))
            result.match_reason = str(best.get("reason", "direct lookup"))
            result.candidates = ranking
            self.bus.query(
                "subgraph serialized",
                entity=result.entity_id,
                nodes=len(result.nodes),
                edges=len(result.edges),
                confidence=round(result.confidence, 3),
            )
        return result

    # --------------------------------------------------------------- retrieval
    def _terminal_anchor(self, entity_id: str) -> AnchorRecord | None:
        if self.store is not None:
            record = self.store.terminal_anchor(entity_id)
            if record is not None:
                return record
        return None

    def _anchor_payload(self, record: AnchorRecord | None, entity_id: str) -> dict[str, Any] | None:
        if record is not None:
            anchor_id = record.anchor_id
            zone_id = self.background.zone_of(anchor_id)
            if not zone_id:
                # INSIDE anchors point at a foreground entity.  Its zone is the
                # only durable room context for that placement.
                container_attrs = self.foreground.get_entity(raw_entity_id(anchor_id)) or {}
                zone_id = container_attrs.get("zone_id")
            if zone_id:
                zone_id = str(zone_id)
                if not zone_id.startswith(ZONE_PREFIX):
                    zone_id = f"{ZONE_PREFIX}{zone_id}"
            return {
                "id": record.id,
                "entity_id": entity_id,
                "anchor_id": anchor_id,
                "anchor_ref": _anchor_ref(anchor_id),
                "label": self._anchor_label(anchor_id),
                "relation": record.relation,
                "t_start": record.t_start,
                "t_end": record.t_end,
                "is_terminal": record.is_terminal,
                "retained": record.retained,
                "confidence": record.confidence,
                "zone_id": zone_id,
                "zone_label": self.background.zone_label(zone_id),
            }

        placement = self.foreground.terminal_placement(entity_id, Relation.RESTING_ON)
        if placement is None:
            placement = self.foreground.terminal_placement(entity_id, Relation.INSIDE)
        if placement is None:
            return None
        anchor_id = str(placement["dst"])
        zone_value = self.background.zone_of(anchor_id)
        if not zone_value:
            entity_attrs = self.foreground.get_entity(entity_id) or {}
            zone_value = entity_attrs.get("zone_id")
        return {
            "id": None,
            "entity_id": entity_id,
            "anchor_id": anchor_id,
            "anchor_ref": _anchor_ref(anchor_id),
            "label": self._anchor_label(anchor_id),
            "relation": placement.get("relation"),
            "t_start": placement.get("t_start"),
            "t_end": placement.get("t_end"),
            "is_terminal": bool(placement.get("is_terminal")),
            "retained": True,
            "confidence": placement.get("confidence", 1.0),
            "zone_id": zone_value,
            "zone_label": self.background.zone_label(zone_value),
        }

    def _anchor_label(self, anchor_id: str) -> str:
        label = self.background.anchor_label(anchor_id)
        if label:
            return label
        if anchor_id.startswith("entity:"):
            container = self.foreground.get_entity(raw_entity_id(anchor_id))
            if container:
                return str(container.get("label", raw_entity_id(anchor_id)))
        return _anchor_ref(anchor_id)

    def _last_interaction(self, entity_id: str) -> dict[str, Any] | None:
        node_id = entity_node(entity_id)
        best: dict[str, Any] | None = None
        best_t = float("-inf")

        for _, _, data in self.foreground.graph.out_edges(node_id, data=True):
            if data.get("relation") != Relation.STATE_TRANSITION.value:
                continue
            to_state = str(data.get("to_state", ""))
            if to_state not in _INTERACTION_STATES:
                continue
            t_start = float(data.get("t_start") or 0.0)
            if t_start >= best_t:
                best_t = t_start
                best = {
                    "t": t_start,
                    "kind": to_state,
                    "from_state": data.get("from_state"),
                    "to_state": to_state,
                    "agent": "USER",
                    "source": "state_machine",
                    "reason": (data.get("evidence") or {}).get("reason"),
                }

        for _, _, data in self.foreground.graph.in_edges(node_id, data=True):
            if data.get("relation") != Relation.ACTION.value:
                continue
            t_start = float(data.get("t_start") or 0.0)
            if t_start >= best_t:
                best_t = t_start
                best = {
                    "t": t_start,
                    "kind": data.get("action", "ACTION"),
                    "agent": "USER",
                    "source": data.get("source"),
                    "confidence": data.get("confidence"),
                    "clip_ref": data.get("clip_ref"),
                }
        return best

    def _attach_background(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        by_id: dict[str, dict[str, Any]] = {str(n["node_id"]): n for n in nodes}
        extra_edges: list[dict[str, Any]] = []
        for node in list(by_id.values()):
            node_id = str(node["node_id"])
            # Anchor / zone nodes are external endpoints of foreground edges, so
            # NetworkX auto-created them without attributes. Hydrate from the
            # background graph before serializing the 2-hop context.
            if node_id.startswith(ANCHOR_PREFIX):
                background_attrs = self.background.node(node_id)
                if background_attrs:
                    node.update(background_attrs)
                node.setdefault("kind", "furniture")
            elif node_id.startswith(ZONE_PREFIX):
                background_attrs = self.background.node(node_id)
                if background_attrs:
                    node.update(background_attrs)
                node.setdefault("kind", "zone")

            kind = node.get("kind")
            anchor_id = node_id
            if kind == "furniture":
                zone_id = node.get("zone_id") or self.background.zone_of(anchor_id)
                if not zone_id:
                    continue
                zone_attrs = self.background.node(zone_id)
                if zone_attrs and zone_id not in by_id:
                    by_id[zone_id] = {
                        "node_id": zone_id,
                        "distance": int(node.get("distance", 0)) + 1,
                        "kind": "zone",
                        **zone_attrs,
                    }
                extra_edges.append(
                    {
                        "src": anchor_id,
                        "dst": zone_id,
                        "key": f"zone:{anchor_id}",
                        "relation": Relation.IN_ZONE.value,
                        "t_start": node.get("created_at"),
                        "t_end": None,
                        "is_terminal": False,
                    }
                )
            elif kind == "entity":
                # The entity's parent zone is one hop beyond the foreground
                # subgraph, so hydrate it explicitly.  This keeps CONTAINED
                # objects (whose terminal anchor is the container entity, not a
                # furniture anchor) from losing their room context.
                zone_id = node.get("zone_id")
                if not zone_id:
                    continue
                zone_id = str(zone_id)
                if not zone_id.startswith(ZONE_PREFIX):
                    zone_id = f"{ZONE_PREFIX}{zone_id}"
                zone_attrs = self.background.node(zone_id)
                if zone_attrs is None:
                    continue
                if zone_id not in by_id:
                    by_id[zone_id] = {
                        "node_id": zone_id,
                        "distance": int(node.get("distance", 0)) + 1,
                        "kind": "zone",
                        **zone_attrs,
                    }
                extra_edges.append(
                    {
                        "src": node_id,
                        "dst": zone_id,
                        "key": f"zone:{node_id}",
                        "relation": Relation.IN_ZONE.value,
                        "t_start": node.get("first_seen"),
                        "t_end": None,
                        "is_terminal": False,
                    }
                )
        edges = edges + extra_edges
        nodes = sorted(
            by_id.values(),
            key=lambda n: (int(n.get("distance", 0)), str(n["node_id"])),
        )
        return nodes, edges

    # ------------------------------------------------------------ convenience
    def summary(self) -> list[dict[str, Any]]:
        """Lightweight state table for the dashboard overview panel."""

        rows = []
        for attrs in self.foreground.entities():
            entity_id = raw_entity_id(str(attrs["node_id"]))
            anchor = self._anchor_payload(self._terminal_anchor(entity_id), entity_id)
            zone_id = (anchor or {}).get("zone_id") or attrs.get("zone_id")
            zone_label = (anchor or {}).get("zone_label") or self.background.zone_label(zone_id)
            rows.append(
                {
                    "entity_id": entity_id,
                    "label": attrs.get("label"),
                    "state": attrs.get("state"),
                    "state_since": attrs.get("state_since"),
                    "last_seen": attrs.get("last_seen"),
                    "anchor": (anchor or {}).get("label"),
                    "zone": zone_label or zone_id,
                    "anchor_t": (anchor or {}).get("t_start"),
                }
            )
        return sorted(rows, key=lambda r: -float(r.get("last_seen") or 0.0))

    @staticmethod
    def node_public_attrs(node: dict[str, Any]) -> dict[str, Any]:
        public = {k: v for k, v in node.items() if k not in {"embedding", "attrs"}}
        if isinstance(public.get("bbox"), tuple):
            public["bbox"] = list(public["bbox"])
        return public

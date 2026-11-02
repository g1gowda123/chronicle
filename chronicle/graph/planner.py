"""Deterministic event/temporal query planner over the episodic graph.

The retrieval prompt is intentionally *entity-centric*: it serialises one
entity's 2-hop neighbourhood.  Many natural memory questions are instead
*event-centric* ("when did I first pick up X?", "how many X did I see?",
"what did I do with X?").  Answering those with the latest entity state is
wrong even when retrieval is perfect: the newest state transition is often an
occlusion, and action edges live on the incoming side of a different node.

This module plans such questions before the generic GraphRAG path and answers
them directly from timestamped action/placement edges.  It is deterministic, so
it doubles as the template backend used in the paper's main table.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from chronicle.geometry import cosine_similarity
from chronicle.graph.ids import entity_node, raw_entity_id
from chronicle.graph.query import GraphQuery, focus_phrases
from chronicle.text import canonical_phrase, content_tokens, match_details
from chronicle.types import Relation

# Two detector tracks whose crop embeddings agree this closely are treated as
# two fragments of one physical instance when counting.  The threshold is
# deliberately high: it only collapses near-duplicate crops, not two genuinely
# different objects of the same category.
INSTANCE_SIMILARITY = 0.90

_COUNT_RE = re.compile(
    r"\bhow many\s+(.+?)\s*(?:\?|$)",
    re.IGNORECASE,
)
_GONE_RE = re.compile(
    r"\bwhere\s+(?:did|has|have)\b.+\b(?:go|went|gone|end up|ended up|disappear|disappeared)\b",
    re.IGNORECASE,
)
_FIRST_RE = re.compile(r"\bwhen\s+did\b", re.IGNORECASE)
_ACTIVITY_RE = re.compile(
    r"\b(?:what\s+did\b.*\bdo\s+with\b|what\s+happened\s+to\b|what\s+(?:is|are|was|were)\b.*\bdoing\s+with\b)",
    re.IGNORECASE,
)
_GERUND_RE = re.compile(
    r"\bwhat\s+(?:is|are|was|were)\b.*?\b"
    r"(slicing|cutting|stirring|pouring|holding|placing|putting|picking|"
    r"opening|closing|wiping|washing|using|moving|carrying|eating|drinking)\b",
    re.IGNORECASE,
)
_GENERIC_NOUNS = {
    "controller",
    "controllers",
    "object",
    "objects",
    "thing",
    "things",
    "item",
    "items",
    "instance",
    "instances",
}
_GERUND_TO_LABELS: dict[str, frozenset[str]] = {
    "slicing": frozenset({"SLICE", "SLICED", "CUT"}),
    "cutting": frozenset({"CUT", "SLICE", "SLICED"}),
    "stirring": frozenset({"STIR", "STIRRED"}),
    "pouring": frozenset({"POUR", "POURED"}),
    "holding": frozenset({"HOLD", "HELD", "PICK_UP"}),
    "placing": frozenset({"PLACE", "PLACED", "PUT_DOWN"}),
    "putting": frozenset({"PLACE", "PLACED", "PUT_DOWN"}),
    "picking": frozenset({"PICK_UP", "PICKED_UP", "GRAB"}),
    "opening": frozenset({"OPEN", "OPENED"}),
    "closing": frozenset({"CLOSE", "CLOSED"}),
    "wiping": frozenset({"WIPE", "WIPED"}),
    "washing": frozenset({"WASH", "WASHED"}),
    "using": frozenset({"USE", "USED", "PICK_UP"}),
    "moving": frozenset({"MOVE", "MOVED"}),
    "carrying": frozenset({"CARRY", "CARRIED", "HELD"}),
    "eating": frozenset({"EAT", "ATE", "EATING"}),
    "drinking": frozenset({"DRINK", "DRANK", "DRINKING"}),
}
_COUNT_SUFFIX_RE = re.compile(
    r"\b(?:are|is|were|was|do|did|does|have|has|there|in|on|at|inside|present|"
    r"currently|now)\b.*$",
    re.IGNORECASE,
)

_VERBS: tuple[tuple[str, frozenset[str]], ...] = (
    ("pick up", frozenset({"PICK_UP", "PICKED_UP", "GRAB", "GRABBED", "TAKE", "TAKEN"})),
    ("grab", frozenset({"PICK_UP", "PICKED_UP", "GRAB", "GRABBED", "TAKE", "TAKEN"})),
    ("touch", frozenset({"TOUCH", "TOUCHED", "PICK_UP", "INTERACTED"})),
    ("hold", frozenset({"HOLD", "HELD", "PICK_UP", "CARRIED_IN_HAND"})),
    ("place", frozenset({"PLACE", "PLACED", "PUT", "PUT_DOWN"})),
    ("put", frozenset({"PLACE", "PLACED", "PUT", "PUT_DOWN"})),
    ("insert", frozenset({"INSERT_INTO", "PLACE", "PLACED"})),
    ("remove", frozenset({"REMOVE_FROM", "REMOVE", "PICK_UP"})),
    ("withdraw", frozenset({"REMOVE_FROM", "REMOVE", "WITHDRAW"})),
    ("open", frozenset({"OPEN", "OPENED"})),
    ("close", frozenset({"CLOSE", "CLOSED"})),
    ("slice", frozenset({"SLICE", "SLICED", "CUT"})),
    ("cut", frozenset({"CUT", "SLICE", "SLICED"})),
    ("stir", frozenset({"STIR", "STIRRED"})),
    ("pour", frozenset({"POUR", "POURED"})),
    ("move", frozenset({"MOVE", "MOVED"})),
    ("align", frozenset({"ALIGN", "ALIGNED"})),
    ("slide", frozenset({"SLIDE", "SLID", "MOVE", "MOVED"})),
    ("eat", frozenset({"EAT", "ATE", "EATING"})),
    ("drink", frozenset({"DRINK", "DRANK", "DRINKING"})),
    ("use", frozenset({"USE", "USED", "PICK_UP"})),
)
_PAST = {
    "PICK_UP": "picked up",
    "PICKED_UP": "picked up",
    "GRAB": "grabbed",
    "GRABBED": "grabbed",
    "TOUCH": "touched",
    "TOUCHED": "touched",
    "INTERACTED": "interacted with",
    "HOLD": "held",
    "HELD": "held",
    "CARRIED_IN_HAND": "held",
    "PLACE": "placed",
    "PLACED": "placed",
    "PUT": "put down",
    "PUT_DOWN": "put down",
    "INSERT_INTO": "inserted",
    "REMOVE_FROM": "removed",
    "REMOVE": "removed",
    "WITHDRAW": "withdrew",
    "OPEN": "opened",
    "CLOSE": "closed",
    "SLICE": "sliced",
    "SLICED": "sliced",
    "CUT": "cut",
    "STIR": "stirred",
    "POUR": "poured",
    "MOVE": "moved",
    "MOVED": "moved",
    "ALIGN": "aligned",
    "SLIDE": "slid",
    "EAT": "ate",
    "DRINK": "drank",
    "USE": "used",
}
_INTERACTION_STATES = {"INTERACTED", "HELD", "CONTAINED"}


@dataclass(frozen=True)
class QueryPlan:
    """One deterministic interpretation of an event/temporal question."""

    kind: str
    object_phrase: str | None = None
    argument: str | None = None


@dataclass(frozen=True)
class PlanAnswer:
    """Answer produced by the event planner (empty when the plan misses)."""

    query: str
    answer: str
    citations: tuple[dict[str, Any], ...]
    grounded: bool
    method: str
    entity: str | None = None
    state: str | None = None
    zone: str | None = None
    last_seen: float | None = None


def _clean_phrase(phrase: str | None) -> str | None:
    if not phrase:
        return None
    value = re.sub(r"\s+", " ", phrase).strip(" ?!.,;:")
    value = _COUNT_SUFFIX_RE.sub("", value).strip()
    value = re.sub(r"^(?:the|a|an|my|your|our|their|his|her)\s+", "", value)
    words = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]*", value.lower())
        if token not in _GENERIC_NOUNS
    ]
    cleaned = " ".join(words).strip()
    return cleaned or value or None


def _object_phrase(query: str) -> str | None:
    """Return the object the question is about, ignoring the subject."""

    phrases = [
        phrase
        for phrase in focus_phrases(query)
        if not _is_person_phrase(phrase)
    ]
    if phrases:
        return _clean_phrase(phrases[-1])
    return None


def _is_person_phrase(phrase: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", phrase.lower()))
    return bool(tokens & {"person", "people", "man", "woman", "user", "chef", "someone", "i"})


def plan_query(query: str) -> QueryPlan | None:
    """Map a question to an event plan, or ``None`` for generic retrieval."""

    text = (query or "").strip()
    if not text:
        return None

    match = _COUNT_RE.search(text)
    if match:
        phrase = _clean_phrase(match.group(1))
        if phrase:
            return QueryPlan("count", object_phrase=phrase)

    if _GONE_RE.search(text):
        return QueryPlan("previous_location", object_phrase=_object_phrase(text))

    if _FIRST_RE.search(text) and re.search(r"\bfirst\b", text, re.IGNORECASE):
        verb = _verb_for_query(text)
        return QueryPlan(
            "first_action",
            object_phrase=_object_phrase(text),
            argument=verb[0] if verb else None,
        )

    if _ACTIVITY_RE.search(text):
        return QueryPlan("activity", object_phrase=_object_phrase(text))

    match = _GERUND_RE.search(text)
    if match:
        return QueryPlan("activity_object", argument=match.group(1).lower())

    return None


def _verb_for_query(query: str) -> tuple[str, frozenset[str]] | None:
    lower = query.lower()
    for phrase, labels in _VERBS:
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            return phrase, labels
    return None


def _past(action: str) -> str:
    value = str(action or "ACTION").upper()
    return _PAST.get(value, value.replace("_", " ").lower())


def _canonical(label: str) -> str:
    return canonical_phrase(str(label))


def _entity_attrs(engine: GraphQuery, entity_id: str) -> dict[str, Any]:
    return engine.foreground.get_entity(entity_id) or {}


def _zone_label(engine: GraphQuery, attrs: dict[str, Any]) -> str | None:
    zone_id = attrs.get("zone_id")
    if not zone_id:
        return None
    return engine.background.zone_label(str(zone_id))


def _entity_labels(engine: GraphQuery) -> list[dict[str, Any]]:
    return list(engine.foreground.entities())


def _label_matches(phrase: str, label: str) -> bool:
    score, _ = match_details(phrase, label)
    if score >= 0.5:
        return True
    phrase_tokens = content_tokens(phrase)
    label_tokens = content_tokens(label)
    return bool(label_tokens) and label_tokens.issubset(phrase_tokens)


def _matching_entities(engine: GraphQuery, phrase: str | None) -> list[dict[str, Any]]:
    if phrase:
        matches = [
            attrs
            for attrs in _entity_labels(engine)
            if _label_matches(phrase, str(attrs.get("label", "")))
        ]
        if matches:
            # Alias labels ("controller" for "joy-con") denote the same class
            # and must be counted together.
            keys = {_canonical(str(attrs.get("label", ""))) for attrs in matches}
            matched_ids = {str(attrs["node_id"]) for attrs in matches}
            for attrs in _entity_labels(engine):
                node_id = str(attrs["node_id"])
                if node_id in matched_ids:
                    continue
                if _canonical(str(attrs.get("label", ""))) in keys:
                    matches.append(attrs)
                    matched_ids.add(node_id)
            return matches
    return []


def _group_entities(engine: GraphQuery, label: str) -> list[str]:
    """All tracks sharing the canonical label (handles joy-con/controller)."""

    key = _canonical(label)
    return [
        raw_entity_id(str(attrs["node_id"]))
        for attrs in _entity_labels(engine)
        if _canonical(str(attrs.get("label", ""))) == key
    ]


def _resolve(
    engine: GraphQuery,
    query: str,
    phrase: str | None,
) -> dict[str, Any] | None:
    candidates: list[str] = []
    if phrase:
        candidates.append(phrase)
    for focus in reversed(focus_phrases(query)):
        if not _is_person_phrase(focus):
            candidates.append(focus)
    if not phrase:
        # Only fall back to the whole question when it did not name a concrete
        # object.  Otherwise an unresolved "cup" must stay unresolved instead
        # of silently resolving to the subject ("person").
        candidates.append(query)
    seen: set[str] = set()
    for text in candidates:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        ranking = engine.rank(text, limit=3)
        if ranking:
            return ranking[0]
    return None


def _unknown(query: str) -> PlanAnswer:
    return PlanAnswer(
        query=query,
        answer="I don't have that in my memory.",
        citations=(),
        grounded=False,
        method="event-planner-miss",
    )


def _action_rows(
    engine: GraphQuery,
    entity_ids: list[str],
    *,
    labels: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_id in entity_ids:
        attrs = _entity_attrs(engine, entity_id)
        node_id = entity_node(entity_id)
        for _, _, data in engine.foreground.graph.in_edges(node_id, data=True):
            if data.get("relation") != Relation.ACTION.value:
                continue
            action = str(data.get("action") or "ACTION").upper()
            if labels is not None and action not in labels:
                continue
            rows.append(
                {
                    "entity_id": entity_id,
                    "entity": str(attrs.get("label", entity_id)),
                    "action": action,
                    "t": float(data.get("t_start") or 0.0),
                    "confidence": float(data.get("confidence") or 0.0),
                }
            )
    rows.sort(key=lambda row: (row["t"], row["entity_id"]))
    return rows


def _interaction_rows(engine: GraphQuery, entity_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_id in entity_ids:
        attrs = _entity_attrs(engine, entity_id)
        node_id = entity_node(entity_id)
        for _, _, data in engine.foreground.graph.out_edges(node_id, data=True):
            if data.get("relation") != Relation.STATE_TRANSITION.value:
                continue
            to_state = str(data.get("to_state") or "").upper()
            if to_state not in _INTERACTION_STATES:
                continue
            rows.append(
                {
                    "entity_id": entity_id,
                    "entity": str(attrs.get("label", entity_id)),
                    "action": to_state,
                    "t": float(data.get("t_start") or 0.0),
                    "confidence": 1.0,
                }
            )
    rows.sort(key=lambda row: (row["t"], row["entity_id"]))
    return rows


def _citation(t: float, relation: str, zone: str | None) -> tuple[dict[str, Any], ...]:
    return (
        {
            "timestamp": float(t),
            "relation": relation,
            "zone": zone,
        },
    )


def _display_label(rows: list[dict[str, Any]]) -> str:
    counts = Counter(
        str(row.get("entity") or row.get("label") or "object") for row in rows
    )
    return counts.most_common(1)[0][0] if counts else "object"


def _plural(label: str, count: int) -> str:
    if count == 1 or label.endswith("s"):
        return label
    if label.endswith(("ch", "sh", "ss", "x", "z")):
        return f"{label}es"
    if label.endswith("y") and label[-2:-1] not in {"a", "e", "i", "o", "u"}:
        return f"{label[:-1]}ies"
    return f"{label}s"


def _cluster_instances(
    engine: GraphQuery,
    candidates: list[dict[str, Any]],
) -> list[list[str]]:
    """Collapse near-duplicate detector tracks for instance counting.

    ByteTrack can split one physical object into several ids after occlusion.
    Tracks that are never co-visible are already merged by the consolidator;
    the remaining failure mode is a duplicate track that *is* co-visible.  Crop
    embeddings separate those from genuinely different instances of the same
    label, so we union tracks whose cosine similarity is very high.
    """

    ids = [raw_entity_id(str(attrs["node_id"])) for attrs in candidates]
    parent = {entity_id: entity_id for entity_id in ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    embeddings: dict[str, tuple[float, ...]] = {}
    for entity_id in ids:
        embedding = _entity_attrs(engine, entity_id).get("embedding")
        if embedding:
            embeddings[entity_id] = tuple(float(value) for value in embedding)
    ordered = sorted(embeddings)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if cosine_similarity(embeddings[left], embeddings[right]) >= INSTANCE_SIMILARITY:
                union(left, right)

    clusters: dict[str, list[str]] = {}
    for entity_id in ids:
        clusters.setdefault(find(entity_id), []).append(entity_id)
    return [sorted(cluster) for _, cluster in sorted(clusters.items())]


def _count_answer(
    engine: GraphQuery,
    query: str,
    plan: QueryPlan,
) -> PlanAnswer | None:
    candidates = _matching_entities(engine, plan.object_phrase)
    if not candidates:
        return _unknown(query)
    clusters = _cluster_instances(engine, candidates)
    count = len(clusters)
    label = _display_label(candidates)
    newest = max(float(attrs.get("last_seen", 0.0) or 0.0) for attrs in candidates)
    attrs = max(candidates, key=lambda item: float(item.get("last_seen", 0.0) or 0.0))
    zone = _zone_label(engine, attrs)
    return PlanAnswer(
        query=query,
        answer=(
            f"There are {count} {_plural(label, count)} in my memory "
            f"(resolved from {len(candidates)} detector tracks)."
        ),
        citations=_citation(newest, "COUNT", zone),
        grounded=True,
        method="instance-count",
        entity=label,
        zone=zone,
        last_seen=newest,
    )


def _first_action_answer(
    engine: GraphQuery,
    query: str,
    plan: QueryPlan,
) -> PlanAnswer | None:
    target = _resolve(engine, query, plan.object_phrase)
    if target is None or _is_person_phrase(str(target.get("label", ""))):
        return _unknown(query)
    entity_id = str(target["entity_id"])
    label = str(target.get("label") or entity_id)
    group = _group_entities(engine, label)
    verb = _verb_for_query(query)
    labels = verb[1] if verb else None
    rows = _action_rows(engine, group, labels=labels)
    if not rows:
        rows = _interaction_rows(engine, group)
    if not rows:
        return _unknown(query)

    best = rows[0]
    attrs = _entity_attrs(engine, best["entity_id"])
    zone = _zone_label(engine, attrs)
    action = _past(best["action"])
    return PlanAnswer(
        query=query,
        answer=(
            f"The person first {action} the {label} at "
            f"t={best['t']:.1f} seconds."
        ),
        citations=_citation(best["t"], "ACTION", zone),
        grounded=True,
        method="event-first-action",
        entity=label,
        zone=zone,
        last_seen=best["t"],
    )


def _activity_answer(
    engine: GraphQuery,
    query: str,
    plan: QueryPlan,
) -> PlanAnswer | None:
    target = _resolve(engine, query, plan.object_phrase)
    if target is None or _is_person_phrase(str(target.get("label", ""))):
        return _unknown(query)
    entity_id = str(target["entity_id"])
    label = str(target.get("label") or entity_id)
    group = _group_entities(engine, label)
    rows = _action_rows(engine, group)
    if not rows:
        rows = _interaction_rows(engine, group)
    if not rows:
        return _unknown(query)

    # Prefer the action that recurs and was verified most confidently; ties go
    # to the earliest observation so the answer is stable across runs.
    scores: Counter[str] = Counter()
    for row in rows:
        scores[row["action"]] += 1.0 + max(0.0, row["confidence"])
    action = max(scores, key=lambda value: (scores[value], -min(r["t"] for r in rows if r["action"] == value)))
    chosen = [row for row in rows if row["action"] == action]
    best = max(chosen, key=lambda row: row["confidence"])
    attrs = _entity_attrs(engine, best["entity_id"])
    zone = _zone_label(engine, attrs)
    return PlanAnswer(
        query=query,
        answer=(
            f"The person {_past(action)} the {label} at "
            f"t={best['t']:.1f} seconds."
        ),
        citations=_citation(best["t"], "ACTION", zone),
        grounded=True,
        method="event-activity",
        entity=label,
        zone=zone,
        last_seen=best["t"],
    )


def _placement_rows(engine: GraphQuery, entity_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity_id in entity_ids:
        attrs = _entity_attrs(engine, entity_id)
        node_id = entity_node(entity_id)
        for _, dst, data in engine.foreground.graph.out_edges(node_id, data=True):
            relation = str(data.get("relation") or "")
            if relation not in {Relation.RESTING_ON.value, Relation.INSIDE.value}:
                continue
            target_id = str(dst)
            target_label = engine.background.anchor_label(target_id)
            if not target_label and target_id.startswith("entity:"):
                container = _entity_attrs(engine, raw_entity_id(target_id))
                target_label = str(container.get("label") or raw_entity_id(target_id))
            rows.append(
                {
                    "entity_id": entity_id,
                    "entity": str(attrs.get("label", entity_id)),
                    "relation": relation,
                    "target": target_label or raw_entity_id(target_id),
                    "t": float(data.get("t_start") or 0.0),
                }
            )
    rows.sort(key=lambda row: (row["t"], row["entity_id"]))
    return rows


def _previous_location_answer(
    engine: GraphQuery,
    query: str,
    plan: QueryPlan,
) -> PlanAnswer | None:
    target = _resolve(engine, query, plan.object_phrase)
    if target is None:
        return _unknown(query)
    entity_id = str(target["entity_id"])
    label = str(target.get("label") or entity_id)
    group = _group_entities(engine, label)
    rows = _placement_rows(engine, group)
    if not rows:
        return _unknown(query)

    last = rows[-1]
    attrs = _entity_attrs(engine, last["entity_id"])
    zone = _zone_label(engine, attrs)
    same_as_first = len({row["target"] for row in rows}) == 1
    if same_as_first and last["relation"] == Relation.RESTING_ON.value:
        answer = (
            f"The {label} ended up resting on the {last['target']} at "
            f"t={last['t']:.1f} seconds."
        )
    elif last["relation"] == Relation.INSIDE.value:
        answer = (
            f"The {label} ended up inside the {last['target']} at "
            f"t={last['t']:.1f} seconds."
        )
    else:
        answer = (
            f"The {label} ended up on the {last['target']} at "
            f"t={last['t']:.1f} seconds."
        )
    return PlanAnswer(
        query=query,
        answer=answer,
        citations=_citation(last["t"], last["relation"], zone),
        grounded=True,
        method="event-destination",
        entity=label,
        zone=zone,
        last_seen=last["t"],
    )


def _activity_object_answer(
    engine: GraphQuery,
    query: str,
    plan: QueryPlan,
) -> PlanAnswer | None:
    verb = str(plan.argument or "").lower()
    labels = _GERUND_TO_LABELS.get(verb)
    if not labels:
        return _unknown(query)
    rows: list[dict[str, Any]] = []
    for attrs in _entity_labels(engine):
        entity_id = raw_entity_id(str(attrs["node_id"]))
        rows.extend(_action_rows(engine, [entity_id], labels=labels))
    if not rows:
        return _unknown(query)
    rows.sort(key=lambda row: (row["t"], row["entity_id"]))
    best = rows[0]
    objects = Counter(row["entity"] for row in rows)
    object_label = objects.most_common(1)[0][0]
    attrs = _entity_attrs(engine, best["entity_id"])
    zone = _zone_label(engine, attrs)
    return PlanAnswer(
        query=query,
        answer=(
            f"The person was {plan.argument} the {object_label} at "
            f"t={best['t']:.1f} seconds."
        ),
        citations=_citation(best["t"], "ACTION", zone),
        grounded=True,
        method="event-activity-object",
        entity=object_label,
        zone=zone,
        last_seen=best["t"],
    )


def answer_query_plan(query: str, *, engine: GraphQuery) -> PlanAnswer | None:
    """Plan and answer an event/temporal question, or return ``None``."""

    plan = plan_query(query)
    if plan is None:
        return None
    if plan.kind == "count":
        return _count_answer(engine, query, plan)
    if plan.kind == "first_action":
        return _first_action_answer(engine, query, plan)
    if plan.kind == "activity":
        return _activity_answer(engine, query, plan)
    if plan.kind == "previous_location":
        return _previous_location_answer(engine, query, plan)
    if plan.kind == "activity_object":
        return _activity_object_answer(engine, query, plan)
    return None

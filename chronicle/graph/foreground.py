"""Foreground graph: episodic entities, placements and action multi-edges.

Every relation is a time-intervaled directed edge.  State evolution is modelled
as a self-loop on the entity whose ``t_start`` is the transition timestamp and
whose ``t_end`` is closed by the next transition, giving an exact history of how
long each state lasted.  Action multi-edges run ``USER -> entity`` exactly as in
the specification.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import networkx as nx

from chronicle.config import DEFAULT_THRESHOLDS, Thresholds
from chronicle.graph.ids import AGENT_NODE, entity_node, raw_entity_id
from chronicle.storage.records import EdgeRecord, EntityRecord, TransitionRecord
from chronicle.telemetry import EventBus
from chronicle.types import ObjectState, Relation, StateTransition


class ForegroundGraph:
    """``nx.MultiDiGraph`` of episodic entities and their relations."""

    def __init__(
        self,
        thresholds: Thresholds | None = None,
        *,
        bus: EventBus | None = None,
    ) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.bus = bus or EventBus()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.graph.add_node(AGENT_NODE, kind="agent", label="USER", ref_id="USER")
        self._seq = 0

    # ------------------------------------------------------------------ helpers
    def _next_key(self, kind: str) -> str:
        self._seq += 1
        return f"{kind}:{self._seq}"

    def node_id(self, entity_id: str) -> str:
        return entity_node(entity_id)

    def has_entity(self, entity_id: str) -> bool:
        return self.graph.has_node(entity_node(entity_id))

    # ----------------------------------------------------------------- entities
    def upsert_entity(
        self,
        entity_id: str,
        label: str,
        *,
        state: ObjectState | str = ObjectState.NEW,
        state_since: float,
        first_seen: float,
        last_seen: float,
        bbox: Sequence[float] | None = None,
        embedding: Sequence[float] | None = None,
        track_id: int | None = None,
        zone_id: str | None = None,
        container_id: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> str:
        node_id = entity_node(entity_id)
        state_value = state.value if isinstance(state, ObjectState) else str(state)
        if self.graph.has_node(node_id):
            data = self.graph.nodes[node_id]
            data.update(
                {
                    "label": label,
                    "state": state_value,
                    "state_since": float(state_since),
                    "last_seen": float(last_seen),
                }
            )
            if bbox is not None:
                data["bbox"] = tuple(float(v) for v in bbox)
            if embedding is not None:
                data["embedding"] = tuple(float(v) for v in embedding)
            if track_id is not None:
                data["track_id"] = int(track_id)
            if zone_id:
                data["zone_id"] = zone_id
            data["container_id"] = container_id
            if attrs:
                data.setdefault("attrs", {}).update(attrs)
        else:
            self.graph.add_node(
                node_id,
                kind="entity",
                ref_id=entity_id,
                label=label,
                state=state_value,
                state_since=float(state_since),
                first_seen=float(first_seen),
                last_seen=float(last_seen),
                bbox=tuple(float(v) for v in bbox) if bbox is not None else None,
                embedding=tuple(float(v) for v in embedding) if embedding is not None else None,
                track_id=track_id,
                zone_id=zone_id,
                container_id=container_id,
                attrs=dict(attrs or {}),
            )
        return node_id

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        node_id = entity_node(entity_id)
        if not self.graph.has_node(node_id):
            return None
        return dict(self.graph.nodes[node_id])

    def entity_ids(self) -> list[str]:
        return [
            raw_entity_id(node_id)
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("kind") == "entity"
        ]

    def entities(self) -> list[dict[str, Any]]:
        return [
            {**dict(attrs), "node_id": node_id}
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("kind") == "entity"
        ]

    def set_zone(self, entity_id: str, zone_id: str | None) -> None:
        node_id = entity_node(entity_id)
        if self.graph.has_node(node_id) and zone_id:
            self.graph.nodes[node_id]["zone_id"] = zone_id

    # -------------------------------------------------------------- transitions
    def record_transition(self, transition: StateTransition) -> str:
        node_id = entity_node(transition.entity_id)
        if not self.graph.has_node(node_id):
            self.upsert_entity(
                transition.entity_id,
                transition.label,
                state=transition.from_state,
                state_since=transition.t_event,
                first_seen=transition.t_event,
                last_seen=transition.t_event,
            )
        self.close_relation(
            transition.entity_id,
            Relation.STATE_TRANSITION,
            transition.t_event,
            clear_terminal=True,
        )
        key = self._next_key("state")
        self.graph.add_edge(
            node_id,
            node_id,
            key=key,
            relation=Relation.STATE_TRANSITION.value,
            from_state=transition.from_state.value,
            to_state=transition.to_state.value,
            t_start=float(transition.t_event),
            t_end=None,
            is_terminal=False,
            evidence=dict(transition.evidence),
            clip_ref=transition.clip_ref,
        )
        data = self.graph.nodes[node_id]
        data["state"] = transition.to_state.value
        data["state_since"] = float(transition.t_event)
        return key

    # ---------------------------------------------------------------- placements
    def open_placement(
        self,
        entity_id: str,
        target_node_id: str,
        relation: Relation,
        t: float,
        *,
        confidence: float = 1.0,
        evidence: dict[str, Any] | None = None,
    ) -> str:
        """Open a spatial edge, retiring the previous one but keeping history."""

        node_id = entity_node(entity_id)
        if not self.graph.has_node(node_id):
            raise KeyError(f"unknown entity {entity_id!r}")
        self.close_relation(entity_id, relation, t, clear_terminal=True)
        key = self._next_key(relation.value.lower())
        self.graph.add_edge(
            node_id,
            target_node_id,
            key=key,
            relation=relation.value,
            t_start=float(t),
            t_end=None,
            is_terminal=True,
            confidence=float(confidence),
            evidence=dict(evidence or {}),
        )
        return key

    def open_resting(
        self,
        entity_id: str,
        anchor_id: str,
        t: float,
        *,
        confidence: float = 1.0,
        evidence: dict[str, Any] | None = None,
    ) -> str:
        return self.open_placement(
            entity_id,
            anchor_id,
            Relation.RESTING_ON,
            t,
            confidence=confidence,
            evidence=evidence,
        )

    def open_inside(
        self,
        entity_id: str,
        container_id: str,
        t: float,
        *,
        confidence: float = 1.0,
        evidence: dict[str, Any] | None = None,
    ) -> str:
        return self.open_placement(
            entity_id,
            entity_node(container_id),
            Relation.INSIDE,
            t,
            confidence=confidence,
            evidence=evidence,
        )

    def close_relation(
        self,
        entity_id: str,
        relation: Relation,
        t: float,
        *,
        clear_terminal: bool = False,
    ) -> int:
        node_id = entity_node(entity_id)
        if not self.graph.has_node(node_id):
            return 0
        closed = 0
        for _, _, _key, data in list(self.graph.out_edges(node_id, keys=True, data=True)):
            if data.get("relation") != relation.value:
                continue
            if data.get("t_end") is None:
                data["t_end"] = float(t)
                closed += 1
            if clear_terminal:
                data["is_terminal"] = False
        return closed

    def open_placements(
        self, entity_id: str, relation: Relation | None = None
    ) -> list[dict[str, Any]]:
        node_id = entity_node(entity_id)
        if not self.graph.has_node(node_id):
            return []
        result = []
        for src, dst, key, data in self.graph.out_edges(node_id, keys=True, data=True):
            if data.get("t_end") is not None:
                continue
            if relation is not None and data.get("relation") != relation.value:
                continue
            if data.get("relation") in {
                Relation.RESTING_ON.value,
                Relation.INSIDE.value,
            }:
                result.append({"key": key, "src": src, "dst": dst, **dict(data)})
        return result

    def terminal_placement(self, entity_id: str, relation: Relation) -> dict[str, Any] | None:
        node_id = entity_node(entity_id)
        if not self.graph.has_node(node_id):
            return None
        best: dict[str, Any] | None = None
        best_t = float("-inf")
        for _, dst, key, data in self.graph.out_edges(node_id, keys=True, data=True):
            if data.get("relation") != relation.value or not data.get("is_terminal"):
                continue
            t_start = float(data.get("t_start") or 0.0)
            if t_start >= best_t:
                best_t = t_start
                best = {"key": key, "src": node_id, "dst": dst, **dict(data)}
        return best

    # ------------------------------------------------------------------ actions
    def add_action(
        self,
        entity_id: str,
        action: str,
        t: float,
        *,
        confidence: float = 1.0,
        source: str = "florence-2",
        clip_ref: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> str:
        node_id = entity_node(entity_id)
        if not self.graph.has_node(node_id):
            raise KeyError(f"unknown entity {entity_id!r}")
        key = self._next_key("action")
        self.graph.add_edge(
            AGENT_NODE,
            node_id,
            key=key,
            relation=Relation.ACTION.value,
            action=action,
            t_start=float(t),
            t_end=None,
            is_terminal=False,
            confidence=float(confidence),
            source=source,
            clip_ref=clip_ref,
            evidence=dict(evidence or {}),
        )
        self.bus.graph(
            "action edge",
            t,
            agent="USER",
            entity=entity_id,
            action=action,
            confidence=round(confidence, 3),
        )
        return key

    def recent_action(self, entity_id: str) -> dict[str, Any] | None:
        node_id = entity_node(entity_id)
        if not self.graph.has_node(node_id):
            return None
        best: dict[str, Any] | None = None
        best_t = float("-inf")
        for _, _, key, data in self.graph.in_edges(node_id, keys=True, data=True):
            if data.get("relation") != Relation.ACTION.value:
                continue
            t_start = float(data.get("t_start") or 0.0)
            if t_start >= best_t:
                best_t = t_start
                best = {"key": key, **dict(data)}
        return best

    # --------------------------------------------------------------- persistence
    def load_entities(self, records: Iterable[EntityRecord]) -> None:
        for record in records:
            self.upsert_entity(
                record.id,
                record.label,
                state=record.state,
                state_since=record.state_since,
                first_seen=record.first_seen,
                last_seen=record.last_seen,
                bbox=record.bbox,
                embedding=record.embedding,
                track_id=record.track_id,
                zone_id=record.zone_id,
                container_id=record.attrs.get("container_id"),
                attrs=record.attrs,
            )

    def load_transitions(self, records: Iterable[TransitionRecord]) -> None:
        for record in records:
            transition = StateTransition(
                entity_id=record.entity_id,
                label=self.graph.nodes.get(entity_node(record.entity_id), {}).get(
                    "label", record.entity_id
                ),
                from_state=ObjectState(record.from_state),
                to_state=ObjectState(record.to_state),
                t_event=record.t_event,
                evidence=record.evidence,
                clip_ref=record.clip_ref,
            )
            self.record_transition(transition)

    def add_edge_record(self, record: EdgeRecord) -> str:
        key = f"edge:{record.id}"
        self.graph.add_edge(
            record.src,
            record.dst,
            key=key,
            relation=record.relation,
            t_start=record.t_start,
            t_end=record.t_end,
            is_terminal=record.is_terminal,
            confidence=record.props.get("confidence", 1.0),
            evidence=record.props.get("evidence", {}),
            **{k: v for k, v in record.props.items() if k not in {"confidence", "evidence"}},
        )
        dst = record.dst
        if record.relation == Relation.RESTING_ON.value and self.graph.has_node(dst):
            zone_id = self.graph.nodes[dst].get("zone_id")
            node_id = record.src
            if zone_id and self.graph.has_node(node_id):
                self.graph.nodes[node_id]["zone_id"] = zone_id
        return key

    # ------------------------------------------------------------------ exports
    def subgraph_nodes_edges(
        self, node_id: str, depth: int = 2
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Nodes and edges within ``depth`` hops, edge endpoints included."""

        if not self.graph.has_node(node_id):
            return [], []
        undirected = self.graph.to_undirected(as_view=True)
        distances = nx.single_source_shortest_path_length(undirected, node_id, cutoff=depth)
        keep = set(distances)
        nodes = [
            {"node_id": n, "distance": distances[n], **dict(self.graph.nodes[n])}
            for n in sorted(keep, key=lambda n: (distances[n], n))
        ]
        edges: list[dict[str, Any]] = []
        for src, dst, key, data in self.graph.edges(keys=True, data=True):
            if src in keep and dst in keep:
                edges.append(
                    {
                        "src": src,
                        "dst": dst,
                        "key": key,
                        "from": src,
                        "to": dst,
                        **dict(data),
                    }
                )
        return nodes, edges

    def visualization_snapshot(
        self, include_state_edges: bool = True
    ) -> dict[str, list[dict[str, Any]]]:
        nodes: list[dict[str, Any]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            kind = attrs.get("kind", "entity")
            if kind == "agent":
                nodes.append(
                    {
                        "id": node_id,
                        "label": attrs.get("label", "USER"),
                        "group": "agent",
                        "title": "Egocentric wearer",
                    }
                )
                continue
            state = attrs.get("state", "NEW")
            nodes.append(
                {
                    "id": node_id,
                    "label": attrs.get("label", node_id),
                    "group": "entity",
                    "state": state,
                    "title": f"{attrs.get('label', node_id)}<br/>state: {state}",
                }
            )
        edges: list[dict[str, Any]] = []
        for src, dst, key, data in self.graph.edges(keys=True, data=True):
            relation = data.get("relation", "")
            if relation == Relation.STATE_TRANSITION.value and not include_state_edges:
                continue
            label = data.get("action") or relation
            title = f"{label} @ t={data.get('t_start')}s"
            edges.append(
                {
                    "id": key,
                    "from": src,
                    "to": dst,
                    "label": label,
                    "title": title,
                    "arrows": "to",
                    "dashes": data.get("t_end") is not None,
                }
            )
        return {"nodes": nodes, "edges": edges}

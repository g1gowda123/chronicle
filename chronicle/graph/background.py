"""Background graph: invariant topological anchors.

Zones are rooms / work areas; furniture are the fixed supports inside them.  A
camera pan that settles produces a keyframe CLIP embedding which is compared
against every known zone.  A cosine similarity above 0.85 relocalizes the camera
instead of minting a duplicate room, which is how the graph stays bounded over a
long recording.  Role C owns the CLIP encoder; this module owns the decision and
the topology.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import networkx as nx

from chronicle.config import DEFAULT_THRESHOLDS, Thresholds
from chronicle.geometry import cosine_similarity
from chronicle.graph.ids import (
    ZONE_PREFIX,
    anchor_node,
    raw_zone_id,
    zone_node,
)
from chronicle.storage.records import FurnitureRecord, ZoneRecord
from chronicle.telemetry import EventBus
from chronicle.text import normalize
from chronicle.types import Relation

_SLUG = re.compile(r"[^a-z0-9]+")


class BackgroundGraph:
    """``nx.DiGraph`` of zones and fixed furniture anchors."""

    def __init__(
        self,
        thresholds: Thresholds | None = None,
        *,
        bus: EventBus | None = None,
    ) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.bus = bus or EventBus()
        self.graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------- zones
    def ensure_zone(
        self,
        label: str,
        t: float,
        *,
        embedding: Iterable[float] | None = None,
        zone_id: str | None = None,
    ) -> str:
        """Return a zone node, creating it only when place recognition fails."""

        embedding_tuple = tuple(float(v) for v in embedding) if embedding is not None else None

        if zone_id:
            node_id = zone_node(zone_id)
            if self.graph.has_node(node_id):
                self._touch(node_id, t, embedding_tuple)
                return node_id
            return self._create_zone(node_id, label or raw_zone_id(node_id), t, embedding_tuple)

        match = self.relocalize(embedding_tuple, t, label=label)
        if match:
            return match
        return self._create_zone(self._unique_zone_id(label), label, t, embedding_tuple)

    def relocalize(
        self,
        embedding: Iterable[float] | None,
        t: float,
        *,
        label: str | None = None,
    ) -> str | None:
        """Visual place recognition against known zones.

        Returns the matching zone node id, or ``None`` when a new zone is needed.
        """

        embedding_tuple = tuple(float(v) for v in embedding) if embedding is not None else None
        zones = [
            (node_id, attrs)
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("kind") == "zone"
        ]
        if not zones:
            return None

        best_id: str | None = None
        best_score = 0.0
        for node_id, attrs in zones:
            if embedding_tuple is not None and attrs.get("embedding"):
                score = cosine_similarity(embedding_tuple, attrs["embedding"])
            elif label:
                score = 1.0 if normalize(attrs.get("label", "")) == normalize(label) else 0.0
            else:
                score = 0.0
            if score > best_score:
                best_id, best_score = node_id, score

        if best_id is not None and best_score >= self.thresholds.place_cosine:
            self._touch(best_id, t, embedding_tuple)
            self.bus.graph(
                "place recognised",
                t,
                zone=best_id,
                cosine=round(best_score, 4),
                threshold=self.thresholds.place_cosine,
            )
            return best_id
        if best_id is not None:
            self.bus.graph(
                "place mismatch",
                t,
                best_zone=best_id,
                cosine=round(best_score, 4),
                threshold=self.thresholds.place_cosine,
            )
        return None

    def _unique_zone_id(self, label: str) -> str:
        slug = _SLUG.sub("-", normalize(label)).strip("-") or "zone"
        base = f"{ZONE_PREFIX}{slug}"
        candidate = base
        index = 2
        while self.graph.has_node(candidate):
            candidate = f"{base}-{index}"
            index += 1
        return candidate

    def _create_zone(
        self,
        node_id: str,
        label: str,
        t: float,
        embedding: tuple[float, ...] | None,
    ) -> str:
        node_id = zone_node(node_id)
        self.graph.add_node(
            node_id,
            kind="zone",
            label=label or raw_zone_id(node_id),
            ref_id=raw_zone_id(node_id),
            embedding=embedding,
            created_at=t,
            last_seen=t,
            visit_count=1,
        )
        self.bus.graph("zone minted", t, zone=node_id, label=label)
        return node_id

    def _touch(self, node_id: str, t: float, embedding: tuple[float, ...] | None) -> None:
        attrs = self.graph.nodes[node_id]
        attrs["last_seen"] = t
        attrs["visit_count"] = int(attrs.get("visit_count", 1)) + 1
        if embedding is not None:
            attrs["embedding"] = embedding

    # --------------------------------------------------------------- furniture
    def ensure_furniture(
        self,
        anchor_id: str,
        label: str,
        zone_id: str | None,
        t: float,
    ) -> str:
        """Create or refresh a fixed support anchor and link it to its zone."""

        node_id = anchor_node(anchor_id)
        if self.graph.has_node(node_id):
            attrs = self.graph.nodes[node_id]
            attrs["label"] = label or attrs.get("label", "surface")
            attrs["last_seen"] = t
            if zone_id:
                self._link_zone(node_id, zone_id, t)
            return node_id

        self.graph.add_node(
            node_id,
            kind="furniture",
            label=label or "surface",
            ref_id=anchor_id,
            zone_id=zone_id,
            created_at=t,
            last_seen=t,
        )
        if zone_id:
            self._link_zone(node_id, zone_id, t)
        self.bus.graph("anchor minted", t, anchor=node_id, label=label, zone=zone_id)
        return node_id

    def _link_zone(self, anchor_id: str, zone_id: str, t: float) -> None:
        zone_id = zone_node(zone_id)
        if not self.graph.has_node(zone_id):
            self._create_zone(zone_id, raw_zone_id(zone_id), t, None)
        self.graph.nodes[anchor_id]["zone_id"] = zone_id
        if not self.graph.has_edge(anchor_id, zone_id):
            self.graph.add_edge(
                anchor_id,
                zone_id,
                relation=Relation.IN_ZONE.value,
                t_start=t,
                t_end=None,
            )

    # ------------------------------------------------------------------ lookups
    def zone_of(self, anchor_id: str) -> str | None:
        node_id = anchor_node(anchor_id)
        if not self.graph.has_node(node_id):
            return None
        zone_id = self.graph.nodes[node_id].get("zone_id")
        if zone_id:
            return zone_id
        for _, target in self.graph.out_edges(node_id):
            if self.graph.nodes.get(target, {}).get("kind") == "zone":
                return target
        return None

    def zone_label(self, zone_id: str | None) -> str | None:
        if not zone_id:
            return None
        node_id = zone_node(zone_id)
        if not self.graph.has_node(node_id):
            return None
        return self.graph.nodes[node_id].get("label")

    def anchor_label(self, anchor_id: str) -> str | None:
        node_id = anchor_node(anchor_id)
        if not self.graph.has_node(node_id):
            return None
        return self.graph.nodes[node_id].get("label")

    def node(self, node_id: str) -> dict[str, Any] | None:
        if not self.graph.has_node(node_id):
            return None
        return dict(self.graph.nodes[node_id])

    def has_node(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)

    def furniture_in(self, zone_id: str) -> list[str]:
        return [
            node_id
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("kind") == "furniture" and attrs.get("zone_id") == zone_node(zone_id)
        ]

    def zone_count(self) -> int:
        return sum(1 for _, a in self.graph.nodes(data=True) if a.get("kind") == "zone")

    # --------------------------------------------------------------- persistence
    def load(self, zones: Iterable[ZoneRecord], furniture: Iterable[FurnitureRecord]) -> None:
        for record in zones:
            node_id = zone_node(record.id)
            self.graph.add_node(
                node_id,
                kind="zone",
                label=record.label,
                ref_id=raw_zone_id(node_id),
                embedding=record.embedding,
                created_at=record.created_at,
                last_seen=record.last_seen,
                visit_count=record.visit_count,
            )
        for record in furniture:
            node_id = anchor_node(record.id)
            self.graph.add_node(
                node_id,
                kind="furniture",
                label=record.label,
                ref_id=record.id,
                zone_id=record.zone_id,
                created_at=record.created_at,
                last_seen=record.created_at,
            )
            if record.zone_id:
                zone_id = zone_node(record.zone_id)
                if not self.graph.has_node(zone_id):
                    self._create_zone(zone_id, raw_zone_id(zone_id), record.created_at, None)
                self.graph.add_edge(
                    node_id,
                    zone_id,
                    relation=Relation.IN_ZONE.value,
                    t_start=record.created_at,
                    t_end=None,
                )

    # ------------------------------------------------------------------ exports
    def visualization_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            kind = attrs.get("kind", "zone")
            nodes.append(
                {
                    "id": node_id,
                    "label": attrs.get("label", node_id),
                    "group": kind,
                    "title": f"{kind}: {attrs.get('label', node_id)}",
                }
            )
        edges = []
        for src, dst, attrs in self.graph.edges(data=True):
            edges.append(
                {
                    "from": src,
                    "to": dst,
                    "label": attrs.get("relation", "IN_ZONE"),
                    "arrows": "to",
                    "dashes": False,
                }
            )
        return {"nodes": nodes, "edges": edges}

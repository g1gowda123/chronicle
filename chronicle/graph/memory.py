"""Dual-graph facade: lifecycle engine + NetworkX graphs + SQLite projection.

``ChronicleMemory`` is the single object the ingestion loop talks to.  It is the
seam between Role A's perception frames and Roles C/D's semantic and UI layers:

    frame = FrameContext(t=..., objects=..., hands=..., supports=..., camera=...)
    transitions = memory.observe(frame, on_transition=queue_vlm_job)
    answer = memory.query("where are my keys?")

Everything is written to SQLite as it happens, and the in-memory graphs stay in
sync so a dashboard can render mutations with zero query latency.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from chronicle.config import (
    CONTAINER_LABELS,
    DEFAULT_THRESHOLDS,
    STATIC_FURNITURE_LABELS,
    FurnitureConfig,
    Thresholds,
)
from chronicle.geometry import bbox_to_polygon
from chronicle.graph.adapter import frame_context_from_result
from chronicle.graph.background import BackgroundGraph
from chronicle.graph.consolidation import GraphConsolidator, MergeReport
from chronicle.graph.foreground import ForegroundGraph
from chronicle.graph.furniture import FurnitureMatch, FurnitureRegistry
from chronicle.graph.ids import (
    AGENT_NODE,
    anchor_node,
    entity_node,
    raw_anchor_id,
    raw_entity_id,
    raw_zone_id,
    zone_node,
)
from chronicle.graph.query import GraphQuery, QueryResult
from chronicle.state_machine import LifecycleEngine
from chronicle.storage.records import AnchorRecord
from chronicle.storage.sqlite_store import SQLiteStore
from chronicle.telemetry import EventBus, TelemetryTag
from chronicle.types import (
    FrameContext,
    FrameResult,
    ObjectState,
    Relation,
    StateTransition,
    SupportSurface,
)

TransitionHook = Callable[[StateTransition, FrameContext], None]

_DISAPPEARED = {
    ObjectState.OCCLUDED,
    ObjectState.OUT_OF_VIEW,
    ObjectState.LOST,
}


class ChronicleMemory:
    """The whole Role B stack behind one thread-safe facade."""

    def __init__(
        self,
        db_path: str | Path = "data/chronicle.db",
        *,
        thresholds: Thresholds | None = None,
        bus: EventBus | None = None,
        container_labels: Iterable[str] = CONTAINER_LABELS,
        furniture_labels: Iterable[str] = STATIC_FURNITURE_LABELS,
        furniture_config: FurnitureConfig | None = None,
        strict_support: bool = False,
        lost_from_occluded: bool = False,
        observation_interval_s: float = 0.10,
        load: bool = True,
    ) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.bus = bus or EventBus()
        self.observation_interval_s = float(observation_interval_s)
        self.furniture = FurnitureRegistry(furniture_config, labels=furniture_labels)
        self.store = SQLiteStore(db_path, bus=self.bus)
        self.background = BackgroundGraph(self.thresholds, bus=self.bus)
        self.foreground = ForegroundGraph(self.thresholds, bus=self.bus)
        self.engine = LifecycleEngine(
            self.thresholds,
            container_labels=container_labels,
            strict_support=strict_support,
            lost_from_occluded=lost_from_occluded,
            bus=self.bus,
        )
        self._entity_aliases: dict[str, str] = {}
        self._furniture_aliases: dict[str, str] = {}
        self.query_engine = GraphQuery(
            self.foreground,
            self.background,
            store=self.store,
            thresholds=self.thresholds,
            bus=self.bus,
            aliases=self._entity_aliases,
        )
        self.consolidator = GraphConsolidator(
            self.foreground,
            self.background,
            self.furniture,
            self.store,
            engine=self.engine,
            bus=self.bus,
            config=self.furniture.config,
            entity_aliases=self._entity_aliases,
            furniture_aliases=self._furniture_aliases,
        )
        self._active_zone: str | None = None
        self._last_observation: dict[str, float] = {}
        self._last_purge: float | None = None
        self._furniture_persisted: dict[str, float] = {}
        if load:
            self._restore()
            self._load_furniture_registry()
            self._rebuild_aliases()
        self.bus.emit(
            TelemetryTag.SYSTEM,
            "chronicle memory online",
            db=self.store.path,
            zones=self.background.zone_count(),
        )

    # --------------------------------------------------------------- lifecycle
    def __enter__(self) -> ChronicleMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.store.close()
        self.bus.emit(TelemetryTag.SYSTEM, "chronicle memory closed", db=self.store.path)

    # ---------------------------------------------------------------- ingestion
    def observe(
        self,
        frame: FrameContext,
        *,
        on_transition: TransitionHook | None = None,
    ) -> list[StateTransition]:
        """Advance the whole stack by one presentation timestamp."""

        t = frame.t
        self._canonicalize_frame(frame)
        self._ensure_frame_context(frame)
        self._route_furniture(frame)
        self._ensure_entities(frame)

        transitions = self.engine.update(frame)

        for transition in transitions:
            self._apply_transition(transition, frame)

        seen: set[str] = set()
        for obj in frame.objects:
            seen.add(obj.entity_id)
            snapshot = self.engine.get(obj.entity_id)
            self._persist_entity(snapshot, obj)
        for transition in transitions:
            if transition.entity_id in seen:
                continue
            self._persist_entity(self.engine.get(transition.entity_id), None)

        self._record_observations(frame, transitions)
        self._maybe_purge(t)

        if on_transition is not None:
            for transition in transitions:
                try:
                    on_transition(transition, frame)
                except Exception as exc:  # pragma: no cover - isolate consumers
                    self.bus.emit(
                        TelemetryTag.VLM,
                        "transition hook failed",
                        t,
                        entity=transition.entity_id,
                        error=repr(exc),
                    )

        try:
            self.maybe_consolidate(t)
        except Exception as exc:  # pragma: no cover - cleanup must not break ingestion
            self.bus.graph("consolidation sweep failed", t, error=repr(exc))
        return transitions

    def observe_result(
        self,
        result: FrameResult,
        *,
        supports: Sequence[SupportSurface] = (),
        zone_id: str | None = None,
        zone_label: str | None = None,
        on_transition: TransitionHook | None = None,
    ) -> list[StateTransition]:
        """Advance the stack from a Role A ``FrameResult`` in one call.

        Role A does not own support-surface detection; pass the current
        ``supports`` from the background-graph/keyframe pass when available.
        """

        frame = frame_context_from_result(
            result,
            supports=supports,
            zone_id=zone_id,
            zone_label=zone_label,
        )
        return self.observe(frame, on_transition=on_transition)

    def record_action(
        self,
        entity_id: str,
        action: str,
        t: float | None = None,
        *,
        confidence: float = 1.0,
        source: str = "florence-2",
        clip_ref: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> str:
        """Attach a VLM-validated action primitive (Role C calls this)."""

        snapshot = self.engine.get(entity_id)
        stamped = float(t if t is not None else (snapshot.last_seen if snapshot else 0.0))
        key = self.foreground.add_action(
            entity_id,
            action,
            stamped,
            confidence=confidence,
            source=source,
            clip_ref=clip_ref,
            evidence=evidence,
        )
        self.store.add_edge(
            AGENT_NODE,
            entity_node(entity_id),
            Relation.ACTION.value,
            stamped,
            props={
                "action": action,
                "confidence": float(confidence),
                "source": source,
                "clip_ref": clip_ref,
                "evidence": dict(evidence or {}),
            },
        )
        return key

    # ------------------------------------------------------------------ zones
    @property
    def active_zone(self) -> str | None:
        return self._active_zone

    def enter_zone(
        self,
        label: str,
        t: float,
        *,
        embedding: Sequence[float] | None = None,
        zone_id: str | None = None,
    ) -> str:
        """Create or refresh a zone, then make it the active camera zone."""

        node_id = self.background.ensure_zone(label, t, embedding=embedding, zone_id=zone_id)
        attrs = self.background.node(node_id) or {}
        self.store.upsert_zone(
            node_id,
            attrs.get("label", label),
            t,
            embedding=embedding,
        )
        self._active_zone = node_id
        return node_id

    def relocalize_zone(
        self,
        embedding: Sequence[float],
        t: float,
        *,
        label: str | None = None,
    ) -> str | None:
        """Role C's VPR entry point (0.85 cosine threshold)."""

        zone_id = self.background.relocalize(embedding, t, label=label)
        if zone_id is None:
            return None
        attrs = self.background.node(zone_id) or {}
        self.store.upsert_zone(
            zone_id,
            attrs.get("label", zone_id),
            t,
            embedding=embedding,
        )
        self._active_zone = zone_id
        return zone_id

    # ------------------------------------------------------------------ query
    def query(
        self,
        text: str,
        *,
        embedder: Callable[[str], Sequence[float]] | None = None,
    ) -> QueryResult | None:
        result = self.query_engine.query(text, embedder=embedder)
        if result is not None:
            self.bus.query(
                "answer context ready",
                entity=result.entity_id,
                state=result.state,
                answer=result.fallback_answer(),
            )
        return result

    def query_entity(self, entity_id: str, question: str = "") -> QueryResult | None:
        return self.query_engine.build(self.resolve_entity_alias(entity_id), query=question)

    def state_table(self) -> list[dict[str, Any]]:
        return self.query_engine.summary()

    def terminal_anchor(self, entity_id: str) -> AnchorRecord | None:
        return self.store.terminal_anchor(self.resolve_entity_alias(entity_id))

    # ------------------------------------------------------------ consolidation
    def resolve_entity_alias(self, entity_id: str) -> str:
        """Map a merged-away entity id onto its surviving keeper."""

        value = str(entity_id).strip().removeprefix("entity:")
        seen: set[str] = set()
        while value in self._entity_aliases and value not in seen:
            seen.add(value)
            value = self._entity_aliases[value]
        return value

    def resolve_furniture_alias(self, anchor_id: str) -> str:
        """Map a merged-away furniture anchor onto its surviving keeper."""

        value = str(anchor_id).strip().removeprefix("anchor:")
        seen: set[str] = set()
        while value in self._furniture_aliases and value not in seen:
            seen.add(value)
            value = self._furniture_aliases[value]
        return value

    def merge_entities(
        self,
        keep: str,
        remove: str,
        *,
        dry_run: bool = False,
        t: float | None = None,
    ) -> MergeReport:
        """Fold one foreground entity into another (audited, terminal-safe)."""

        return self.consolidator.merge_entities(
            keep,
            remove,
            dry_run=dry_run,
            t=t,
            reason="manual",
        )

    def merge_furniture(
        self,
        keep: str,
        remove: str,
        *,
        dry_run: bool = False,
        t: float | None = None,
    ) -> MergeReport:
        """Fold one background furniture anchor into another."""

        return self.consolidator.merge_furniture(
            keep,
            remove,
            dry_run=dry_run,
            t=t,
            reason="manual",
        )

    def maybe_consolidate(self, t: float, *, dry_run: bool = False) -> list[MergeReport]:
        """Run a duplicate sweep if the configured interval has elapsed."""

        return self.consolidator.maybe_consolidate(t, dry_run=dry_run)

    # ---------------------------------------------------------------- internal
    def _restore(self) -> None:
        snapshot = self.store.load_snapshot()
        if not snapshot.entities and not snapshot.zones:
            return
        self.background.load(snapshot.zones, snapshot.furniture)
        self.foreground.load_entities(snapshot.entities)
        self.foreground.load_transitions(snapshot.transitions)
        for edge in snapshot.edges:
            self.foreground.add_edge_record(edge)
        for record in snapshot.entities:
            self.engine.seed(
                record.id,
                record.label,
                ObjectState(record.state),
                state_since=record.state_since,
                first_seen=record.first_seen,
                last_seen=record.last_seen,
                bbox=record.bbox,
                support_id=None,
                container_id=record.attrs.get("container_id"),
                zone_id=record.zone_id,
            )
        if snapshot.zones:
            self._active_zone = max(snapshot.zones, key=lambda z: z.last_seen).id
        self.bus.emit(
            TelemetryTag.SYSTEM,
            "memory restored from disk",
            entities=len(snapshot.entities),
            zones=len(snapshot.zones),
            transitions=len(snapshot.transitions),
            anchors=len(snapshot.anchors),
        )

    def _load_furniture_registry(self) -> None:
        rows = []
        for record in self.store.load_furniture():
            attrs = dict(record.attrs or {})
            attrs.setdefault("hits", 1)
            attrs.setdefault("first_seen", record.created_at)
            attrs.setdefault("last_seen", record.created_at)
            attrs.setdefault("confirmed", True)
            rows.append(
                (
                    raw_anchor_id(str(record.id)),
                    str(record.label or "surface"),
                    raw_zone_id(str(record.zone_id)) if record.zone_id else None,
                    attrs,
                )
            )
        self.furniture.load_attributes(rows)

    def _route_furniture(self, frame: FrameContext) -> None:
        """Move static furniture detections out of the foreground graph.

        Role A does not distinguish scene structure from manipulable objects.
        The registry associates each furniture track with a persistent anchor,
        then the anchor is offered to the lifecycle engine as a support surface.
        This is the key defence against duplicate sofa/table nodes.
        """

        if not frame.objects:
            return
        zone_node_id = self._active_zone or self._default_zone(frame.t)
        zone_id = raw_zone_id(zone_node_id)
        objects = []
        supports = list(frame.supports)
        support_ids = {surface.anchor_id for surface in supports}
        for obj in frame.objects:
            if not self.furniture.is_furniture(obj.label):
                objects.append(obj)
                continue
            match = self.furniture.observe(
                obj.label,
                obj.bbox,
                zone_id,
                frame.t,
                track_id=obj.track_id,
                embedding=obj.embedding,
                camera_dx=frame.camera.dx,
                camera_dy=frame.camera.dy,
                frame_size=frame.frame_size,
            )
            if match is None:
                objects.append(obj)
                continue
            self._ensure_furniture_match(match, zone_id, frame.t, obj)
            if match.confirmed and match.anchor_id not in support_ids:
                surface = SupportSurface(
                    anchor_id=match.anchor_id,
                    label=match.label,
                    polygon=bbox_to_polygon(match.bbox),
                    zone_id=zone_id,
                    attributes={
                        "source": "furniture-track",
                        "track_id": obj.track_id,
                        "confidence": obj.confidence,
                    },
                )
                supports.append(surface)
                support_ids.add(match.anchor_id)
                self._ensure_support(surface, zone_node_id, frame.t)
        frame.objects = tuple(objects)
        frame.supports = tuple(supports)

    def _ensure_furniture_match(
        self,
        match: FurnitureMatch,
        zone_id: str | None,
        t: float,
        obj,
    ) -> str:
        record = self.furniture.record(match.anchor_id)
        attrs = record.as_attributes() if record is not None else {}
        attrs.update(
            {
                "source": "role-a-track",
                "confirmed": bool(match.confirmed),
                "last_track_id": obj.track_id,
            }
        )
        node_id = self.background.ensure_furniture(
            match.anchor_id,
            match.label,
            zone_id,
            t,
            bbox=match.bbox,
            embedding=record.embedding if record is not None else None,
            attrs=attrs,
        )
        previous = self._furniture_persisted.get(node_id, float("-inf"))
        if match.is_new or t - previous >= 1.0:
            self.store.upsert_furniture(
                node_id,
                match.label,
                zone_node(zone_id) if zone_id else None,
                t,
                attrs=attrs,
            )
            self._furniture_persisted[node_id] = t
        return node_id

    def _canonicalize_frame(self, frame: FrameContext) -> None:
        """Rewrite merged-away ids before the engine can mint them again."""

        if self._entity_aliases:
            objects = []
            for obj in frame.objects:
                canonical = self.resolve_entity_alias(obj.entity_id)
                objects.append(
                    obj if canonical == obj.entity_id else replace(obj, entity_id=canonical)
                )
            frame.objects = tuple(objects)

        if self._furniture_aliases:
            supports = []
            for surface in frame.supports:
                canonical = self.resolve_furniture_alias(surface.anchor_id)
                supports.append(
                    surface
                    if canonical == surface.anchor_id
                    else replace(surface, anchor_id=canonical)
                )
            frame.supports = tuple(supports)

    def _rebuild_aliases(self) -> None:
        """Restore merged-away id mappings from persisted ``merged_from`` rows."""

        for record in self.store.load_entities():
            entity_id = raw_entity_id(str(record.id))
            for removed in record.attrs.get("merged_from", []):
                removed_id = raw_entity_id(str(removed))
                if removed_id and removed_id != entity_id:
                    GraphConsolidator.register_alias(
                        self._entity_aliases, entity_id, removed_id
                    )
        for record in self.store.load_furniture():
            anchor_id = raw_anchor_id(str(record.id))
            for removed in record.attrs.get("merged_from", []):
                removed_id = raw_anchor_id(str(removed))
                if removed_id and removed_id != anchor_id:
                    GraphConsolidator.register_alias(
                        self._furniture_aliases, anchor_id, removed_id
                    )

    def _ensure_frame_context(self, frame: FrameContext) -> None:
        t = frame.t
        if frame.zone_id:
            label = str(frame.attributes.get("zone_label", frame.zone_id))
            zone_id = self.background.ensure_zone(label, t, zone_id=frame.zone_id)
            if zone_id != self._active_zone:
                attrs = self.background.node(zone_id) or {}
                self.store.upsert_zone(zone_id, attrs.get("label", label), t)
                self._active_zone = zone_id

        zone_id = self._active_zone or self._default_zone(t)
        for surface in frame.supports:
            self._ensure_support(surface, zone_id, t)

    def _default_zone(self, t: float) -> str:
        if self._active_zone is not None:
            return self._active_zone
        node_id = self.background.ensure_zone("unmapped area", t, zone_id="unmapped")
        self.store.upsert_zone(node_id, "unmapped area", t)
        self._active_zone = node_id
        self.bus.graph("default zone created", t, zone=node_id)
        return node_id

    def _ensure_support(self, surface: SupportSurface, zone_id: str, t: float) -> str:
        zone = zone_id
        if surface.zone_id:
            zone = zone_node(surface.zone_id)
            if not self.background.has_node(zone):
                self.background.ensure_zone(surface.zone_id, t, zone_id=surface.zone_id)
                self.store.upsert_zone(zone, surface.zone_id, t)
        node_id = self.background.ensure_furniture(surface.anchor_id, surface.label, zone, t)
        self.store.upsert_furniture(
            node_id,
            surface.label,
            zone,
            t,
            attrs={"fixed": surface.is_fixed, **dict(surface.attributes)},
        )
        return node_id

    def _ensure_entities(self, frame: FrameContext) -> None:
        """Create durable rows before transitions (SQLite foreign keys)."""

        for obj in frame.objects:
            if self.foreground.has_entity(obj.entity_id):
                continue
            snapshot = self.engine.get(obj.entity_id)
            state = snapshot.state if snapshot else ObjectState.NEW
            since = snapshot.state_since if snapshot else frame.t
            zone_id = (snapshot.zone_id if snapshot else None) or self._active_zone
            self.foreground.upsert_entity(
                obj.entity_id,
                obj.label,
                state=state,
                state_since=since,
                first_seen=frame.t,
                last_seen=frame.t,
                bbox=obj.bbox,
                embedding=obj.embedding,
                track_id=obj.track_id,
                zone_id=zone_id,
                container_id=snapshot.container_id if snapshot else None,
            )
            self.store.upsert_entity(
                obj.entity_id,
                obj.label,
                state.value,
                since,
                frame.t,
                frame.t,
                bbox=obj.bbox,
                embedding=obj.embedding,
                track_id=obj.track_id,
                zone_id=zone_id,
                attrs={},
            )
            self.bus.graph(
                "foreground node minted",
                frame.t,
                entity=obj.entity_id,
                label=obj.label,
                state=state.value,
            )

    def _persist_entity(self, snapshot, obj) -> None:
        if snapshot is None:
            return
        existing = self.foreground.get_entity(snapshot.entity_id) or {}
        first_seen = float(existing.get("first_seen", snapshot.state_since))
        zone_id = existing.get("zone_id") or snapshot.zone_id
        attrs = {**dict(existing.get("attrs") or {}), "container_id": snapshot.container_id}
        self.foreground.upsert_entity(
            snapshot.entity_id,
            snapshot.label,
            state=snapshot.state,
            state_since=snapshot.state_since,
            first_seen=first_seen,
            last_seen=snapshot.last_seen,
            bbox=snapshot.bbox,
            embedding=obj.embedding if obj is not None else None,
            track_id=obj.track_id if obj is not None else None,
            zone_id=zone_id,
            container_id=snapshot.container_id,
        )
        self.store.upsert_entity(
            snapshot.entity_id,
            snapshot.label,
            snapshot.state.value,
            snapshot.state_since,
            first_seen,
            snapshot.last_seen,
            bbox=snapshot.bbox,
            embedding=obj.embedding if obj is not None else None,
            track_id=obj.track_id if obj is not None else None,
            zone_id=zone_id,
            attrs=attrs,
        )

    def _apply_transition(self, transition: StateTransition, frame: FrameContext) -> None:
        t = transition.t_event
        evidence = dict(transition.evidence)
        self.store.add_transition(
            transition.entity_id,
            transition.from_state.value,
            transition.to_state.value,
            t,
            evidence=evidence,
            clip_ref=transition.clip_ref,
        )
        self.foreground.record_transition(transition)

        entering = transition.to_state
        leaving = transition.from_state

        if leaving is ObjectState.VISIBLE_STATIC:
            self._close_placement(transition.entity_id, t, Relation.RESTING_ON)
        elif leaving is ObjectState.CONTAINED:
            self._close_placement(transition.entity_id, t, Relation.INSIDE)
        elif leaving is ObjectState.HELD and entering is ObjectState.CONTAINED:
            self._close_placement(transition.entity_id, t, Relation.RESTING_ON)

        if entering is ObjectState.VISIBLE_STATIC:
            support_id = evidence.get("support_id")
            if support_id:
                anchor = anchor_node(str(support_id))
                confidence = float(evidence.get("support_score", 1.0) or 1.0)
                self._close_placement(transition.entity_id, t, Relation.INSIDE)
                self.foreground.open_resting(
                    transition.entity_id,
                    anchor,
                    t,
                    confidence=confidence,
                    evidence=evidence,
                )
                self.store.add_edge(
                    entity_node(transition.entity_id),
                    anchor,
                    Relation.RESTING_ON.value,
                    t,
                    is_terminal=True,
                    props={"confidence": confidence, "evidence": evidence},
                )
                self.store.add_anchor(
                    transition.entity_id,
                    anchor,
                    Relation.RESTING_ON.value,
                    t,
                    confidence=confidence,
                    evidence=evidence,
                )
                zone = self.background.zone_of(anchor)
                self.foreground.set_zone(transition.entity_id, zone)
                self.bus.graph(
                    "RESTING_ON edge",
                    t,
                    entity=transition.entity_id,
                    anchor=anchor,
                    zone=zone,
                )
            else:
                self.bus.graph(
                    "static without support anchor",
                    t,
                    entity=transition.entity_id,
                    reason="no support surface geometry in frame",
                )

        elif entering is ObjectState.CONTAINED:
            container = evidence.get("container_id")
            if container:
                self._close_placement(transition.entity_id, t, Relation.RESTING_ON)
                confidence = float(evidence.get("container_ios", 1.0) or 1.0)
                self.foreground.open_inside(
                    transition.entity_id,
                    str(container),
                    t,
                    confidence=confidence,
                    evidence=evidence,
                )
                self.store.add_edge(
                    entity_node(transition.entity_id),
                    entity_node(str(container)),
                    Relation.INSIDE.value,
                    t,
                    is_terminal=True,
                    props={"confidence": confidence, "evidence": evidence},
                )
                self.store.add_anchor(
                    transition.entity_id,
                    entity_node(str(container)),
                    Relation.INSIDE.value,
                    t,
                    confidence=confidence,
                    evidence=evidence,
                )
                self.bus.graph(
                    "INSIDE edge",
                    t,
                    entity=transition.entity_id,
                    container=container,
                )

        elif entering in _DISAPPEARED:
            self._close_placement(transition.entity_id, t, Relation.RESTING_ON)
            self._close_placement(transition.entity_id, t, Relation.INSIDE)

        if transition.to_state in {ObjectState.INTERACTED, ObjectState.HELD}:
            self.bus.vlm(
                "candidate composite clip for verification",
                t,
                entity=transition.entity_id,
                to_state=transition.to_state.value,
                clip_ref=transition.clip_ref,
            )

    def _close_placement(self, entity_id: str, t: float, relation: Relation) -> None:
        """Close an open spatial interval; the anchor row itself is retained."""

        closed = self.foreground.close_relation(entity_id, relation, t, clear_terminal=False)
        if closed == 0:
            return
        self.store.close_open_edges(entity_node(entity_id), relation.value, t)
        self.store.close_open_anchors(entity_id, relation.value, t)
        self.bus.graph(
            "placement closed",
            t,
            entity=entity_id,
            relation=relation.value,
        )

    def _record_observations(
        self, frame: FrameContext, transitions: Sequence[StateTransition]
    ) -> None:
        changed = {tr.entity_id for tr in transitions}
        for obj in frame.objects:
            last = self._last_observation.get(obj.entity_id, float("-inf"))
            if obj.entity_id not in changed and frame.t - last < self.observation_interval_s:
                continue
            snapshot = self.engine.get(obj.entity_id)
            state = snapshot.state.value if snapshot else None
            self.store.add_observation(obj.entity_id, frame.t, obj.bbox, state)
            self._last_observation[obj.entity_id] = frame.t

    def _maybe_purge(self, t: float) -> None:
        if self._last_purge is not None and t - self._last_purge < self.thresholds.purge_interval_s:
            return
        self.store.purge_trajectories(t, self.thresholds.trajectory_retention_s)
        self._last_purge = t

    # ------------------------------------------------------------------ exports
    def visualization_snapshot(self) -> dict[str, Any]:
        """Merged Vis.js payload for Role D's dashboard."""

        background = self.background.visualization_snapshot()
        foreground = self.foreground.visualization_snapshot()
        nodes = {node["id"]: node for node in background["nodes"]}
        for node in foreground["nodes"]:
            nodes[node["id"]] = node
        edges = background["edges"] + foreground["edges"]
        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "stats": self.stats(),
        }

    def stats(self) -> dict[str, Any]:
        states: dict[str, int] = {}
        for snapshot in self.engine.snapshot():
            states[snapshot.state.value] = states.get(snapshot.state.value, 0) + 1
        merges = self.consolidator.stats()
        return {
            "entities": self.store.count("entities"),
            "zones": self.background.zone_count(),
            "anchors": self.store.count("anchors"),
            "edges": self.store.count("edges"),
            "observations": self.store.count("observations"),
            "active_zone": self._active_zone,
            "camera": self.engine.camera_state.value,
            "states": states,
            "merges": merges,
            "entities_merged": merges["entities"],
            "furniture_merged": merges["furniture"],
        }

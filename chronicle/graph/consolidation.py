"""Conservative duplicate consolidation for legacy graph rows.

Role A occasionally mints a second identity for the same physical object after a
track loss, and older builds routed furniture detections straight into the
foreground graph.  This module cleans those rows up without inventing new
evidence:

* labels must canonicalize to the same alias group;
* zones must agree (a missing zone is treated as unknown, not as a mismatch);
* two genuinely co-visible instances of the same label are never merged;
* spatial, temporal and appearance evidence must clear a configurable score.

All merges are available as a dry-run first.  The actual mutation re-points
SQLite history and graph edges at the keeper, retains every anchor row, and
leaves an audit event on the ``[GRAPH]`` telemetry channel.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from chronicle.config import DEFAULT_FURNITURE_CONFIG, FurnitureConfig
from chronicle.geometry import cosine_similarity
from chronicle.graph.background import BackgroundGraph
from chronicle.graph.foreground import ForegroundGraph
from chronicle.graph.furniture import FurnitureRegistry, bbox_center, bbox_iou
from chronicle.graph.ids import anchor_node, entity_node, raw_anchor_id, raw_entity_id
from chronicle.state_machine import LifecycleEngine
from chronicle.storage.records import EntityRecord
from chronicle.storage.sqlite_store import SQLiteStore
from chronicle.telemetry import EventBus
from chronicle.text import canonical_label
from chronicle.types import ObjectState

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class MergeCandidate:
    """One scored duplicate pair, before any mutation."""

    kind: str
    keep: str
    remove: str
    score: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "keep": self.keep,
            "remove": self.remove,
            "score": round(self.score, 4),
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass
class MergeReport:
    """Outcome of one manual or automatic merge attempt."""

    kind: str
    keep: str
    remove: str
    merged: bool
    dry_run: bool
    score: float = 0.0
    reason: str = "manual"
    evidence: dict[str, Any] = field(default_factory=dict)
    preserved: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "skipped"
        if self.dry_run:
            return "would_merge"
        return "merged" if self.merged else "skipped"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "keep": self.keep,
            "remove": self.remove,
            "merged": self.merged,
            "dry_run": self.dry_run,
            "status": self.status,
            "score": round(self.score, 4),
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "preserved": dict(self.preserved),
            "error": self.error,
        }


@dataclass(frozen=True)
class _EntityView:
    entity_id: str
    label: str
    state: str
    state_since: float
    first_seen: float
    last_seen: float
    bbox: BBox | None
    embedding: tuple[float, ...] | None
    track_id: int | None
    zone_id: str | None
    attrs: dict[str, Any]


@dataclass(frozen=True)
class _FurnitureView:
    anchor_id: str
    label: str
    zone_id: str | None
    bbox: BBox | None
    embedding: tuple[float, ...] | None
    created_at: float
    first_seen: float
    last_seen: float
    track_ids: frozenset[int]
    confirmed: bool
    hits: int


class GraphConsolidator:
    """Find and apply safe duplicate merges across the whole Role B stack."""

    def __init__(
        self,
        foreground: ForegroundGraph,
        background: BackgroundGraph,
        furniture: FurnitureRegistry,
        store: SQLiteStore | None = None,
        *,
        engine: LifecycleEngine | None = None,
        bus: EventBus | None = None,
        config: FurnitureConfig | None = None,
        entity_aliases: Mapping[str, str] | None = None,
        furniture_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.foreground = foreground
        self.background = background
        self.furniture = furniture
        self.store = store
        self.engine = engine
        self.bus = bus or EventBus()
        self.config = config or DEFAULT_FURNITURE_CONFIG

        # Keep the memory facade's mutable maps by reference so query/ingest can
        # canonicalize a merged-away id immediately.
        self.entity_aliases: dict[str, str] = (
            entity_aliases if isinstance(entity_aliases, dict) else dict(entity_aliases or {})
        )
        self.furniture_aliases: dict[str, str] = (
            furniture_aliases
            if isinstance(furniture_aliases, dict)
            else dict(furniture_aliases or {})
        )

        self.sweeps = 0
        self.entities_merged = 0
        self.furniture_merged = 0
        self.last_sweep: float | None = None
        self._obs_cache: dict[str, list[tuple[float, BBox | None]]] = {}

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        return {
            "sweeps": self.sweeps,
            "entities": self.entities_merged,
            "furniture": self.furniture_merged,
            "total": self.entities_merged + self.furniture_merged,
            "last_sweep": self.last_sweep,
            "aliased_entities": len(self.entity_aliases),
            "aliased_furniture": len(self.furniture_aliases),
        }

    # -------------------------------------------------------------- candidates
    def find_candidates(self, t: float | None = None) -> list[MergeCandidate]:
        """Return every safe merge candidate, highest score first."""

        self._obs_cache = {}
        try:
            candidates = [*self._entity_candidates(), *self._furniture_candidates(t)]
        finally:
            self._obs_cache = {}
        candidates.sort(key=lambda c: (-c.score, c.kind, c.keep, c.remove))
        return candidates

    def _entity_candidates(self) -> list[MergeCandidate]:
        views = self._entity_views()
        groups: dict[str, list[_EntityView]] = {}
        for view in views:
            groups.setdefault(canonical_label(view.label), []).append(view)

        candidates: list[MergeCandidate] = []
        for group in groups.values():
            for index, left in enumerate(group):
                for right in group[index + 1 :]:
                    candidate = self._entity_pair(left, right)
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    def _furniture_candidates(self, t: float | None) -> list[MergeCandidate]:
        views = self._furniture_views()
        groups: dict[str, list[_FurnitureView]] = {}
        for view in views:
            groups.setdefault(canonical_label(view.label), []).append(view)

        candidates: list[MergeCandidate] = []
        for group in groups.values():
            for index, left in enumerate(group):
                for right in group[index + 1 :]:
                    candidate = self._furniture_pair(left, right, t)
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    # -------------------------------------------------------------- merge APIs
    def merge_entities(
        self,
        keep: str,
        remove: str,
        *,
        dry_run: bool = False,
        t: float | None = None,
        reason: str = "manual",
        allow_covisible: bool = False,
    ) -> MergeReport:
        """Fold ``remove`` into ``keep`` while retaining the full timeline."""

        keep_id = self._canonical_entity(keep)
        remove_id = self._canonical_entity(remove)
        report = MergeReport(
            kind="entity",
            keep=keep_id,
            remove=remove_id,
            merged=False,
            dry_run=dry_run,
            reason=reason,
        )
        if keep_id == remove_id:
            report.error = "keep and remove resolve to the same entity"
            return report

        keep_view = self._entity_view(keep_id)
        remove_view = self._entity_view(remove_id)
        if keep_view is None or remove_view is None:
            missing = keep_id if keep_view is None else remove_id
            report.error = f"unknown foreground entity {missing!r}"
            return report
        if canonical_label(keep_view.label) != canonical_label(remove_view.label):
            report.error = "labels do not share an alias group"
            return report
        if not _zones_compatible(keep_view.zone_id, remove_view.zone_id):
            report.error = "entities belong to incompatible zones"
            return report
        if not allow_covisible and self._entities_co_visible(keep_view, remove_view):
            report.error = "co-visible instances of the same label are never merged"
            return report

        report.score = 1.0
        report.evidence = {
            "labels": [keep_view.label, remove_view.label],
            "zones": [keep_view.zone_id, remove_view.zone_id],
            "gap_s": round(remove_view.first_seen - keep_view.last_seen, 4),
            "track_ids": [keep_view.track_id, remove_view.track_id],
        }
        report.preserved = self._entity_preserved(keep_id)
        if dry_run:
            self.bus.graph(
                "entity merge planned",
                t,
                keep=keep_id,
                removed=remove_id,
                reason=reason,
            )
            return report

        record = self._merge_entity_store(keep_id, remove_id)
        self._merge_entity_graph(keep_id, remove_id, keep_view, remove_view, record)
        self.register_alias(self.entity_aliases, keep_id, remove_id)
        self.entities_merged += 1
        report.merged = True
        report.preserved = self._entity_preserved(keep_id)
        self.bus.graph(
            "entity merged",
            t if t is not None else max(keep_view.last_seen, remove_view.last_seen),
            keep=keep_id,
            removed=remove_id,
            reason=reason,
            transitions=report.preserved.get("transitions", 0),
            anchors=report.preserved.get("anchors", 0),
            observations=report.preserved.get("observations", 0),
            edges=report.preserved.get("edges", 0),
        )
        return report

    def merge_furniture(
        self,
        keep: str,
        remove: str,
        *,
        dry_run: bool = False,
        t: float | None = None,
        reason: str = "manual",
        allow_covisible: bool = False,
    ) -> MergeReport:
        """Fold one persistent furniture anchor into another."""

        keep_id = self._canonical_furniture(keep)
        remove_id = self._canonical_furniture(remove)
        report = MergeReport(
            kind="furniture",
            keep=keep_id,
            remove=remove_id,
            merged=False,
            dry_run=dry_run,
            reason=reason,
        )
        if keep_id == remove_id:
            report.error = "keep and remove resolve to the same anchor"
            return report

        keep_view = self._furniture_view(keep_id)
        remove_view = self._furniture_view(remove_id)
        if keep_view is None or remove_view is None:
            missing = keep_id if keep_view is None else remove_id
            report.error = f"unknown furniture anchor {missing!r}"
            return report
        if canonical_label(keep_view.label) != canonical_label(remove_view.label):
            report.error = "labels do not share an alias group"
            return report
        if not _zones_compatible(keep_view.zone_id, remove_view.zone_id):
            report.error = "anchors belong to incompatible zones"
            return report
        if not allow_covisible and self._furniture_covisible(keep_view, remove_view):
            report.error = "co-visible furniture instances of the same label are never merged"
            return report
        geometry = _geometry(keep_view.bbox, remove_view.bbox)
        if geometry is None or geometry < 0.35:
            report.error = "furniture geometry is not consistent with one instance"
            return report

        report.score = 1.0
        report.evidence = {
            "labels": [keep_view.label, remove_view.label],
            "zones": [keep_view.zone_id, remove_view.zone_id],
            "iou": round(bbox_iou(keep_view.bbox, remove_view.bbox), 4),
            "geometry": round(geometry, 4),
            "track_ids": [sorted(keep_view.track_ids), sorted(remove_view.track_ids)],
        }
        report.preserved = self._furniture_preserved(keep_id)
        if dry_run:
            self.bus.graph(
                "furniture merge planned",
                t,
                keep=keep_id,
                removed=remove_id,
                reason=reason,
            )
            return report

        if self.store is not None:
            self.store.merge_furniture(
                keep_id,
                remove_id,
                attrs={
                    "merged_into": keep_id,
                    "merged_reason": reason,
                },
            )
        bg_keep = anchor_node(keep_id)
        bg_remove = anchor_node(remove_id)
        if self.background.has_node(bg_remove) or self.background.has_node(bg_keep):
            self.background.merge_furniture(
                keep_id,
                remove_id,
                t if t is not None else max(keep_view.last_seen, remove_view.last_seen),
                audit={
                    "reason": reason,
                    "score": report.score,
                    "zone": keep_view.zone_id or remove_view.zone_id,
                },
            )
        self.furniture.merge(keep_id, remove_id)
        self.register_alias(self.furniture_aliases, keep_id, remove_id)
        self.furniture_merged += 1
        report.merged = True
        report.preserved = self._furniture_preserved(keep_id)
        return report

    # ------------------------------------------------------------ auto sweep
    def maybe_consolidate(
        self,
        t: float,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> list[MergeReport]:
        """Run the periodic sweep, respecting the furniture config gates."""

        if not force and not self.config.consolidation_enabled:
            return []
        if (
            not force
            and self.last_sweep is not None
            and t - self.last_sweep < self.config.consolidation_interval_s
        ):
            return []

        candidates = self.find_candidates(t)
        if dry_run:
            return [self._dry_report(candidate, t) for candidate in candidates]

        self.last_sweep = float(t)
        self.sweeps += 1
        reports: list[MergeReport] = []
        merged = 0
        blocked: set[tuple[str, str, str]] = set()
        while merged < max(1, self.config.consolidation_max_merges):
            remaining = [
                candidate
                for candidate in self.find_candidates(t)
                if (candidate.kind, candidate.keep, candidate.remove) not in blocked
            ]
            if not remaining:
                break
            candidate = remaining[0]
            report = self._apply_candidate(candidate, t)
            reports.append(report)
            if report.merged:
                merged += 1
            else:
                blocked.add((candidate.kind, candidate.keep, candidate.remove))

        if merged:
            entities = sum(1 for report in reports if report.kind == "entity" and report.merged)
            furniture = sum(
                1 for report in reports if report.kind == "furniture" and report.merged
            )
            self.bus.graph(
                "consolidation sweep",
                t,
                entities=entities,
                furniture=furniture,
                total=merged,
            )
        return reports

    def _apply_candidate(self, candidate: MergeCandidate, t: float) -> MergeReport:
        if candidate.kind == "entity":
            return self.merge_entities(
                candidate.keep,
                candidate.remove,
                t=t,
                reason="auto: " + candidate.reason,
            )
        return self.merge_furniture(
            candidate.keep,
            candidate.remove,
            t=t,
            reason="auto: " + candidate.reason,
        )

    def _dry_report(self, candidate: MergeCandidate, t: float) -> MergeReport:
        report = MergeReport(
            kind=candidate.kind,
            keep=candidate.keep,
            remove=candidate.remove,
            merged=False,
            dry_run=True,
            score=candidate.score,
            reason="auto: " + candidate.reason,
            evidence=dict(candidate.evidence),
        )
        if candidate.kind == "entity":
            report.preserved = self._entity_preserved(candidate.keep)
            self.bus.graph(
                "entity merge planned",
                t,
                keep=candidate.keep,
                removed=candidate.remove,
                score=round(candidate.score, 4),
            )
        else:
            report.preserved = self._furniture_preserved(candidate.keep)
            self.bus.graph(
                "furniture merge planned",
                t,
                keep=candidate.keep,
                removed=candidate.remove,
                score=round(candidate.score, 4),
            )
        return report

    # -------------------------------------------------------------- graph apply
    def _merge_entity_store(self, keep: str, remove: str) -> EntityRecord | None:
        if self.store is None:
            return None
        try:
            return self.store.merge_entities(
                keep,
                remove,
                attrs={"merged_into": keep},
            )
        except KeyError:
            # In-memory tests and partially restored graphs may not have both
            # rows.  The graph merge still runs; SQLite simply has nothing to
            # re-point.
            return None

    def _merge_entity_graph(
        self,
        keep: str,
        remove: str,
        keep_view: _EntityView,
        remove_view: _EntityView,
        record: EntityRecord | None,
    ) -> None:
        keep_node = entity_node(keep)
        remove_node = entity_node(remove)

        if record is not None:
            self.foreground.upsert_entity(
                keep,
                record.label,
                state=ObjectState(record.state),
                state_since=record.state_since,
                first_seen=record.first_seen,
                last_seen=record.last_seen,
                bbox=record.bbox,
                embedding=record.embedding,
                track_id=record.track_id,
                zone_id=record.zone_id,
                container_id=record.attrs.get("container_id"),
                attrs=dict(record.attrs),
            )
            node = self.foreground.graph.nodes[keep_node]
            node["first_seen"] = float(record.first_seen)
            node["attrs"] = dict(record.attrs)
            latest_state = record.state
            state_since = record.state_since
            first_seen = record.first_seen
            last_seen = record.last_seen
            bbox = record.bbox
            container_id = record.attrs.get("container_id")
            zone_id = record.zone_id
        else:
            merged_attrs = dict(remove_view.attrs)
            merged_attrs.update(keep_view.attrs)
            merged_attrs["merged_from"] = list(
                dict.fromkeys(
                    [
                        *list(merged_attrs.get("merged_from", [])),
                        *list(remove_view.attrs.get("merged_from", [])),
                        remove,
                    ]
                )
            )
            node = self.foreground.graph.nodes[keep_node]
            node["attrs"] = merged_attrs
            node["first_seen"] = min(keep_view.first_seen, remove_view.first_seen)
            latest_is_remove = remove_view.last_seen > keep_view.last_seen
            latest = remove_view if latest_is_remove else keep_view
            latest_state = latest.state
            state_since = latest.state_since
            first_seen = min(keep_view.first_seen, remove_view.first_seen)
            last_seen = max(keep_view.last_seen, remove_view.last_seen)
            bbox = latest.bbox
            container_id = latest.attrs.get("container_id")
            zone_id = latest.zone_id

        self.foreground.rewire_node(remove_node, keep_node)
        self.foreground.replace_container_id(remove, keep)
        self.foreground.normalize_intervals(keep)
        self.foreground.remove_entity(remove)

        if self.engine is not None:
            self.engine.forget(remove)
            support_id = None
            if self.store is not None:
                anchor = self.store.terminal_anchor(keep)
                if anchor is not None:
                    support_id = raw_anchor_id(str(anchor.anchor_id))
            self.engine.seed(
                keep,
                record.label if record is not None else keep_view.label,
                latest_state,
                state_since=state_since,
                first_seen=first_seen,
                last_seen=last_seen,
                bbox=bbox,
                support_id=support_id,
                container_id=container_id,
                zone_id=zone_id,
            )

    # --------------------------------------------------------------- views
    def _entity_views(self) -> list[_EntityView]:
        views: list[_EntityView] = []
        for attrs in self.foreground.entities():
            view = self._view_from_attrs(attrs)
            if view is not None:
                views.append(view)
        return views

    def _entity_view(self, entity_id: str) -> _EntityView | None:
        attrs = self.foreground.get_entity(entity_id)
        if attrs is None:
            return None
        return self._view_from_attrs({**attrs, "node_id": entity_node(entity_id)})

    @staticmethod
    def _view_from_attrs(attrs: dict[str, Any]) -> _EntityView | None:
        node_id = str(attrs.get("node_id", ""))
        if not node_id:
            return None
        entity_id = raw_entity_id(node_id)
        first_seen = float(attrs.get("first_seen", attrs.get("state_since", 0.0)) or 0.0)
        return _EntityView(
            entity_id=entity_id,
            label=str(attrs.get("label") or entity_id),
            state=str(attrs.get("state") or ObjectState.NEW.value),
            state_since=float(attrs.get("state_since", first_seen) or first_seen),
            first_seen=first_seen,
            last_seen=float(attrs.get("last_seen", first_seen) or first_seen),
            bbox=_bbox(attrs.get("bbox")),
            embedding=_embedding(attrs.get("embedding")),
            track_id=int(attrs["track_id"]) if attrs.get("track_id") is not None else None,
            zone_id=_raw_zone(attrs.get("zone_id")),
            attrs=dict(attrs.get("attrs") or {}),
        )

    def _furniture_views(self) -> list[_FurnitureView]:
        rows: dict[str, _FurnitureView] = {}
        for row in self.background.furniture():
            raw_id = raw_anchor_id(str(row.get("anchor_id", "")))
            if not raw_id:
                continue
            attrs = dict(row.get("attrs") or {})
            rows[raw_id] = _FurnitureView(
                anchor_id=raw_id,
                label=str(row.get("label") or raw_id),
                zone_id=_raw_zone(row.get("zone_id")),
                bbox=_bbox(row.get("bbox")),
                embedding=_embedding(row.get("embedding") or attrs.get("embedding")),
                created_at=float(row.get("created_at") or 0.0),
                first_seen=float(attrs.get("first_seen") or row.get("created_at") or 0.0),
                last_seen=float(
                    attrs.get("last_seen") or row.get("last_seen") or row.get("created_at") or 0.0
                ),
                track_ids=frozenset(_ints(attrs.get("track_ids", []))),
                confirmed=bool(attrs.get("confirmed", True)),
                hits=max(1, int(attrs.get("hits", 1) or 1)),
            )

        for record in self.furniture.records:
            previous = rows.get(record.anchor_id)
            rows[record.anchor_id] = _FurnitureView(
                anchor_id=record.anchor_id,
                label=record.label,
                zone_id=_raw_zone(record.zone_id),
                bbox=_bbox(record.bbox) or (previous.bbox if previous else None),
                embedding=record.embedding or (previous.embedding if previous else None),
                created_at=previous.created_at if previous else record.first_seen,
                first_seen=record.first_seen,
                last_seen=max(record.last_seen, previous.last_seen if previous else record.last_seen),
                track_ids=frozenset(record.track_ids),
                confirmed=record.confirmed,
                hits=record.hits,
            )
        return list(rows.values())

    def _furniture_view(self, anchor_id: str) -> _FurnitureView | None:
        wanted = raw_anchor_id(anchor_id)
        for view in self._furniture_views():
            if view.anchor_id == wanted:
                return view
        return None

    # -------------------------------------------------------------- pair scoring
    def _entity_pair(self, left: _EntityView, right: _EntityView) -> MergeCandidate | None:
        if left.entity_id == right.entity_id:
            return None
        if canonical_label(left.label) != canonical_label(right.label):
            return None
        if not _zones_compatible(left.zone_id, right.zone_id):
            return None
        if self._entities_co_visible(left, right):
            return None

        earlier, later = sorted(
            (left, right), key=lambda view: (view.first_seen, view.entity_id)
        )
        gap = later.first_seen - earlier.last_seen
        if gap < -1e-6 or gap > self.config.consolidation_max_gap_s:
            return None

        earlier_bbox = self._last_bbox(earlier.entity_id) or earlier.bbox
        later_bbox = self._first_bbox(later.entity_id) or later.bbox
        geometry = _geometry(earlier_bbox, later_bbox)
        if geometry is None or geometry < 0.35:
            return None

        embedding = _embedding_similarity(earlier.embedding, later.embedding)
        if (
            embedding is not None
            and embedding < self.config.consolidation_embedding_cosine
        ):
            return None

        temporal = _temporal_score(gap, self.config.consolidation_max_gap_s)
        if embedding is None:
            score = 0.60 * temporal + 0.40 * geometry
        else:
            score = 0.45 * temporal + 0.35 * geometry + 0.20 * max(0.0, embedding)
        if earlier.track_id is not None and earlier.track_id == later.track_id:
            score = min(1.0, score + 0.10)
        if score < self.config.consolidation_min_score:
            return None

        return MergeCandidate(
            kind="entity",
            keep=earlier.entity_id,
            remove=later.entity_id,
            score=round(score, 4),
            reason="spatial-temporal" if embedding is None else "spatial-temporal+embedding",
            evidence={
                "labels": [earlier.label, later.label],
                "gap_s": round(gap, 4),
                "geometry": round(geometry, 4),
                "embedding": None if embedding is None else round(embedding, 4),
            },
        )

    def _furniture_pair(
        self,
        left: _FurnitureView,
        right: _FurnitureView,
        t: float | None,
    ) -> MergeCandidate | None:
        if left.anchor_id == right.anchor_id:
            return None
        if canonical_label(left.label) != canonical_label(right.label):
            return None
        if not _zones_compatible(left.zone_id, right.zone_id):
            return None
        if self._furniture_covisible(left, right):
            return None

        geometry = _geometry(left.bbox, right.bbox)
        if geometry is None or geometry < 0.40:
            return None

        embedding = _embedding_similarity(left.embedding, right.embedding)
        if (
            embedding is not None
            and embedding < self.config.consolidation_embedding_cosine
        ):
            return None

        overlap = min(left.last_seen, right.last_seen) - max(left.first_seen, right.first_seen)
        if overlap > 0.0:
            temporal = 0.5
        else:
            gap = max(left.first_seen, right.first_seen) - min(left.last_seen, right.last_seen)
            temporal = _temporal_score(max(0.0, gap), self.config.consolidation_max_gap_s)
        if embedding is None:
            score = 0.65 * geometry + 0.35 * temporal
        else:
            score = 0.55 * geometry + 0.25 * temporal + 0.20 * max(0.0, embedding)
        if score < self.config.consolidation_min_score:
            return None

        earlier = min((left, right), key=lambda view: (view.created_at, view.anchor_id))
        later = right if earlier is left else left
        return MergeCandidate(
            kind="furniture",
            keep=earlier.anchor_id,
            remove=later.anchor_id,
            score=round(score, 4),
            reason="furniture geometry" if embedding is None else "furniture geometry+embedding",
            evidence={
                "labels": [earlier.label, later.label],
                "iou": round(bbox_iou(earlier.bbox, later.bbox), 4),
                "geometry": round(geometry, 4),
                "embedding": None if embedding is None else round(embedding, 4),
                "t": t,
            },
        )

    # ----------------------------------------------------------- co-visibility
    def _entities_co_visible(self, left: _EntityView, right: _EntityView) -> bool:
        left_obs = self._observations(left.entity_id)
        right_obs = self._observations(right.entity_id)
        if left_obs and right_obs:
            right_times = sorted(time for time, _ in right_obs)
            epsilon = self._covisibility_epsilon
            for time, _ in left_obs:
                index = bisect_left(right_times, time)
                for candidate in (index - 1, index):
                    if 0 <= candidate < len(right_times):
                        if abs(right_times[candidate] - time) <= epsilon:
                            return True
            return False
        return _spans_overlap(left.first_seen, left.last_seen, right.first_seen, right.last_seen)

    def _furniture_covisible(self, left: _FurnitureView, right: _FurnitureView) -> bool:
        gap = max(left.first_seen, right.first_seen) - min(left.last_seen, right.last_seen)
        if gap > self._covisibility_epsilon:
            return False
        # Co-visible detections can only be the same physical instance when the
        # boxes essentially coincide; two sofas on opposite walls always fail.
        return bbox_iou(left.bbox, right.bbox) < self.config.consolidation_covisible_iou

    @property
    def _covisibility_epsilon(self) -> float:
        return 0.05

    def _observations(self, entity_id: str) -> list[tuple[float, BBox | None]]:
        if entity_id not in self._obs_cache:
            if self.store is None:
                self._obs_cache[entity_id] = []
            else:
                self._obs_cache[entity_id] = [
                    (float(time), bbox)
                    for time, bbox, _ in self.store.observations_for(
                        entity_id, limit=1_000_000
                    )
                ]
        return self._obs_cache[entity_id]

    def _first_bbox(self, entity_id: str) -> BBox | None:
        for _, bbox in self._observations(entity_id):
            if bbox is not None:
                return bbox
        return None

    def _last_bbox(self, entity_id: str) -> BBox | None:
        for _, bbox in reversed(self._observations(entity_id)):
            if bbox is not None:
                return bbox
        return None

    # --------------------------------------------------------------- aliases
    @staticmethod
    def register_alias(aliases: dict[str, str], keep: str, remove: str) -> None:
        if keep == remove:
            return
        for key, value in list(aliases.items()):
            if value == remove:
                aliases[key] = keep
        aliases[remove] = keep
        for key in list(aliases):
            target = key
            seen: set[str] = set()
            while target in aliases and target not in seen:
                seen.add(target)
                target = aliases[target]
            aliases[key] = target

    def _canonical_entity(self, entity_id: str) -> str:
        value = raw_entity_id(str(entity_id).strip())
        seen: set[str] = set()
        while value in self.entity_aliases and value not in seen:
            seen.add(value)
            value = self.entity_aliases[value]
        return value

    def _canonical_furniture(self, anchor_id: str) -> str:
        value = raw_anchor_id(str(anchor_id).strip())
        seen: set[str] = set()
        while value in self.furniture_aliases and value not in seen:
            seen.add(value)
            value = self.furniture_aliases[value]
        return value

    # -------------------------------------------------------------- preservation
    def _entity_preserved(self, entity_id: str) -> dict[str, int]:
        if self.store is None:
            return {}
        node_id = entity_node(entity_id)
        edges = 0
        if self.foreground.graph.has_node(node_id):
            edges = len(
                {
                    (src, dst, key)
                    for src, dst, key in self.foreground.graph.edges(keys=True)
                    if node_id in {src, dst}
                }
            )
        return {
            "transitions": len(self.store.load_transitions(entity_id)),
            "anchors": len(self.store.anchors_for(entity_id)),
            "observations": len(self.store.observations_for(entity_id, limit=1_000_000)),
            "edges": edges,
        }

    def _furniture_preserved(self, anchor_id: str) -> dict[str, int]:
        if self.store is None:
            return {}
        node_id = anchor_node(anchor_id)
        edges = 0
        if self.background.graph.has_node(node_id):
            edges += self.background.graph.out_degree(node_id)
            edges += self.background.graph.in_degree(node_id)
        return {"edges": edges}


def _bbox(value: Any) -> BBox | None:
    if value is None or len(value) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in value)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _embedding(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    vector = tuple(float(v) for v in value)
    return vector or None


def _embedding_similarity(
    left: tuple[float, ...] | None,
    right: tuple[float, ...] | None,
) -> float | None:
    if left is None or right is None:
        return None
    return cosine_similarity(left, right)


def _geometry(left: BBox | None, right: BBox | None) -> float | None:
    if left is None or right is None:
        return None
    if _size_ratio(left, right) < 0.25:
        return None
    overlap = bbox_iou(left, right)
    left_center = bbox_center(left)
    right_center = bbox_center(right)
    distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
    diagonal = max(
        1.0,
        math.hypot(left[2] - left[0], left[3] - left[1]),
        math.hypot(right[2] - right[0], right[3] - right[1]),
    )
    center_ratio = distance / diagonal
    if overlap < 0.02 and center_ratio > 0.35:
        return None
    return max(overlap, max(0.0, 1.0 - center_ratio))


def _size_ratio(left: BBox, right: BBox) -> float:
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    if left_area <= 0.0 or right_area <= 0.0:
        return 0.0
    return min(left_area, right_area) / max(left_area, right_area)


def _temporal_score(gap: float, max_gap: float) -> float:
    if max_gap <= 0.0:
        return 1.0 if gap <= 0.0 else 0.0
    return max(0.0, 1.0 - max(0.0, gap) / max_gap)


def _spans_overlap(
    left_first: float,
    left_last: float,
    right_first: float,
    right_last: float,
) -> bool:
    gap = max(left_first, right_first) - min(left_last, right_last)
    return gap <= 0.05


def _zones_compatible(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return True
    return _raw_zone(left) == _raw_zone(right)


def _raw_zone(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.removeprefix("zone:")


def _ints(values: Any) -> set[int]:
    result: set[int] = set()
    for value in values or ():
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


__all__ = ["GraphConsolidator", "MergeCandidate", "MergeReport"]

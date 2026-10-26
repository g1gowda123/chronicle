"""Immutable row projections for graph rebuilds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ZoneRecord:
    id: str
    label: str
    embedding: tuple[float, ...] | None
    created_at: float
    last_seen: float
    visit_count: int
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FurnitureRecord:
    id: str
    label: str
    zone_id: str | None
    created_at: float
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityRecord:
    id: str
    label: str
    state: str
    state_since: float
    first_seen: float
    last_seen: float
    bbox: tuple[float, float, float, float] | None
    embedding: tuple[float, ...] | None
    track_id: int | None
    zone_id: str | None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionRecord:
    id: int
    entity_id: str
    from_state: str
    to_state: str
    t_event: float
    evidence: dict[str, Any]
    clip_ref: str | None = None


@dataclass(frozen=True)
class EdgeRecord:
    id: int
    src: str
    dst: str
    relation: str
    t_start: float | None
    t_end: float | None
    is_terminal: bool
    props: dict[str, Any] = field(default_factory=dict)

    @property
    def open(self) -> bool:
        return self.t_end is None


@dataclass(frozen=True)
class AnchorRecord:
    id: int
    entity_id: str
    anchor_id: str
    relation: str
    t_start: float
    t_end: float | None
    is_terminal: bool
    retained: bool
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def anchored(self) -> bool:
        return self.t_end is None


@dataclass(frozen=True)
class StorageSnapshot:
    """Everything required to rebuild the dual graph in memory."""

    zones: tuple[ZoneRecord, ...] = ()
    furniture: tuple[FurnitureRecord, ...] = ()
    entities: tuple[EntityRecord, ...] = ()
    transitions: tuple[TransitionRecord, ...] = ()
    edges: tuple[EdgeRecord, ...] = ()
    anchors: tuple[AnchorRecord, ...] = ()

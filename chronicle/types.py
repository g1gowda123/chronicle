"""Typed data contracts shared by every Chronicle role.

Role A (perception) builds :class:`FrameContext` objects; Role B consumes them
here and emits :class:`StateTransition` events; Roles C and D read the graph and
the telemetry stream.  Keeping these types dependency-light is deliberate: a
perception worker can import them without pulling in NetworkX or SQLite.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from shapely.geometry import MultiPoint, Polygon

# MediaPipe Hands landmark indices used by the interaction gate.
WRIST_INDEX = 0
PALM_CENTER_INDEX = 9
FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)


class ObjectState(str, Enum):
    """The nine-state resilient object lifecycle (spec section 4)."""

    NEW = "NEW"
    VISIBLE_STATIC = "VISIBLE_STATIC"
    INTERACTED = "INTERACTED"
    VISIBLE_MOVING = "VISIBLE_MOVING"
    HELD = "HELD"
    CONTAINED = "CONTAINED"
    OCCLUDED = "OCCLUDED"
    OUT_OF_VIEW = "OUT_OF_VIEW"
    LOST = "LOST"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class CameraState(str, Enum):
    """Result of the affine parallax gate."""

    STATIONARY = "STATIONARY"
    PANNING = "PANNING"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class Relation(str, Enum):
    """Edge labels used by the dual graph."""

    RESTING_ON = "RESTING_ON"
    INSIDE = "INSIDE"
    STATE_TRANSITION = "STATE_TRANSITION"
    ACTION = "ACTION"
    IN_ZONE = "IN_ZONE"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True)
class CameraMotion:
    """Rigid background motion for a single frame, in pixels.

    Produced by Role A's ``cv2.estimateAffinePartial2D`` fit.  ``dx``/``dy`` are
    the translation components and ``dtheta_deg`` the in-plane rotation.
    """

    dx: float = 0.0
    dy: float = 0.0
    dtheta_deg: float = 0.0

    @property
    def magnitude(self) -> float:
        """``||T||`` from the specification, in pixels per frame."""

        return math.hypot(self.dx, self.dy)

    @property
    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.dx, self.dy, self.dtheta_deg))


@dataclass
class ObjectObservation:
    """One tracked foreground entity seen in the current frame.

    ``entity_id`` must be stable across track loss: Role A's ByteTrack + CLIP
    re-ID reconciliation is responsible for re-binding a new track id to an
    existing entity.  The engine never invents a second identity for the same
    object, which is what keeps the episodic graph from exploding.
    """

    entity_id: str
    label: str
    bbox: tuple[float, float, float, float]
    polygon: Polygon | None = None
    confidence: float = 1.0
    track_id: int | None = None
    reid_similarity: float | None = None
    embedding: Sequence[float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must be a non-empty stable identifier")
        if not self.label:
            raise ValueError("label must be a non-empty string")
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        if x2 < x1 or y2 < y1:
            raise ValueError(f"bbox must be (x1, y1, x2, y2) with x2>=x1, y2>=y1: {self.bbox!r}")
        self.bbox = (x1, y1, x2, y2)
        self.label = self.label.strip().lower()
        self.confidence = float(self.confidence)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def hull(self) -> Polygon:
        """Mask hull when available, otherwise the axis-aligned bbox."""

        if self.polygon is not None and not self.polygon.is_empty:
            return self.polygon
        return Polygon.from_bounds(*self.bbox)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "track_id": self.track_id,
        }


@dataclass
class HandObservation:
    """MediaPipe Hands output for one hand in one frame."""

    hand_id: str = "hand:0"
    landmarks: Sequence[tuple[float, float]] = ()
    polygon: Polygon | None = None
    is_left: bool | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hand_id:
            raise ValueError("hand_id must be a non-empty identifier")
        self.landmarks = tuple((float(x), float(y)) for x, y in self.landmarks)

    @classmethod
    def from_landmarks(
        cls,
        landmarks: Sequence[tuple[float, float]],
        hand_id: str = "hand:0",
        is_left: bool | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> HandObservation:
        """Build a hand observation, deriving a convex-hull contact polygon."""

        points = tuple((float(x), float(y)) for x, y in landmarks)
        polygon = None
        if len(points) >= 3:
            polygon = MultiPoint(points).convex_hull
        return cls(
            hand_id=hand_id,
            landmarks=points,
            polygon=polygon,
            is_left=is_left,
            attributes=dict(attributes or {}),
        )

    def hull(self) -> Polygon | None:
        if self.polygon is not None and not self.polygon.is_empty:
            return self.polygon
        if len(self.landmarks) >= 3:
            return MultiPoint(self.landmarks).convex_hull
        return None

    @property
    def fingertips(self) -> tuple[tuple[float, float], ...]:
        if not self.landmarks:
            return ()
        return tuple(self.landmarks[i] for i in FINGERTIP_INDICES if i < len(self.landmarks))

    @property
    def centroid(self) -> tuple[float, float] | None:
        if not self.landmarks:
            return None
        if len(self.landmarks) > PALM_CENTER_INDEX:
            return self.landmarks[PALM_CENTER_INDEX]
        xs = [p[0] for p in self.landmarks]
        ys = [p[1] for p in self.landmarks]
        return sum(xs) / len(xs), sum(ys) / len(ys)


@dataclass
class SupportSurface:
    """A surface or fixed furniture anchor detected in the current frame.

    ``anchor_id`` must be stable across frames (Role A reuses the anchor id for a
    persistent surface).  When the id is new, :class:`ChronicleMemory` creates the
    matching background-graph node on first use.
    """

    anchor_id: str
    label: str
    polygon: Polygon
    zone_id: str | None = None
    is_fixed: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.anchor_id:
            raise ValueError("anchor_id must be a non-empty stable identifier")
        if self.polygon is None or self.polygon.is_empty:
            raise ValueError("support polygon must be non-empty")
        self.label = (self.label or "surface").strip().lower()


@dataclass
class FrameContext:
    """Everything Role B needs to advance one presentation timestamp."""

    t: float
    objects: Sequence[ObjectObservation] = ()
    hands: Sequence[HandObservation] = ()
    supports: Sequence[SupportSurface] = ()
    camera: CameraMotion = field(default_factory=CameraMotion)
    frame_size: tuple[int, int] | None = None
    zone_id: str | None = None
    frame_index: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = float(self.t)
        if not math.isfinite(self.t) or self.t < 0:
            raise ValueError(f"presentation timestamp must be finite and >= 0, got {self.t!r}")
        if not self.camera.is_finite:
            raise ValueError("camera motion must contain finite values")
        self.objects = tuple(self.objects)
        self.hands = tuple(self.hands)
        self.supports = tuple(self.supports)

    def object_by_id(self, entity_id: str) -> ObjectObservation | None:
        for obj in self.objects:
            if obj.entity_id == entity_id:
                return obj
        return None


@dataclass(frozen=True)
class StateTransition:
    """A gated lifecycle transition emitted with full geometric evidence.

    Roles C and D consume these: C schedules a Florence-2 verification for
    INTERACTED/CONTAINED edges, D renders ``[STATE]`` telemetry and graph deltas.
    """

    entity_id: str
    label: str
    from_state: ObjectState
    to_state: ObjectState
    t_event: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
    clip_ref: str | None = None

    @property
    def tag(self) -> str:
        return f"{self.from_state.value}->{self.to_state.value}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "t_event": round(self.t_event, 4),
            "evidence": dict(self.evidence),
            "clip_ref": self.clip_ref,
        }


@dataclass(frozen=True)
class EntitySnapshot:
    """Read-only view of one entity's runtime state."""

    entity_id: str
    label: str
    state: ObjectState
    state_since: float
    last_seen: float
    bbox: tuple[float, float, float, float] | None = None
    support_id: str | None = None
    container_id: str | None = None
    hand_ios: float = 0.0
    velocity_px_s: float = 0.0
    zone_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "state": self.state.value,
            "state_since": round(self.state_since, 4),
            "last_seen": round(self.last_seen, 4),
            "bbox": list(self.bbox) if self.bbox else None,
            "support_id": self.support_id,
            "container_id": self.container_id,
            "hand_ios": round(self.hand_ios, 4),
            "velocity_px_s": round(self.velocity_px_s, 2),
            "zone_id": self.zone_id,
        }

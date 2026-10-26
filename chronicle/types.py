"""Typed data contracts shared by every Chronicle role.

Role A (perception) emits :class:`FrameResult` records built from
:class:`BBox`, :class:`Detection`, :class:`HandObservation`,
:class:`MotionEstimate` and :class:`TrackedObject`.  Role B consumes
:class:`FrameContext` and emits :class:`StateTransition` events; Roles C and D
read the graph and the telemetry stream.

Keeping these types dependency-light and serializable is deliberate: the
perception worker and the graph engine exchange plain dataclasses, and the
dashboard can render them without importing NetworkX or SQLite.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "label": self.label,
            "zone_id": self.zone_id,
            "is_fixed": self.is_fixed,
            "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in self.polygon.exterior.coords],
            "attributes": _jsonable(self.attributes),
        }


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


class CameraState(str, Enum):
    """Camera motion class shared by perception and the lifecycle engine.

    Role A emits ``STATIONARY`` / ``MOVING`` / ``PANNING`` / ``UNKNOWN`` from
    its affine fit; Role B's compensated-velocity gate only distinguishes
    ``STATIONARY`` from ``PANNING``.
    """

    STATIONARY = "STATIONARY"
    MOVING = "MOVING"
    PANNING = "PANNING"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class EventCategory(str, Enum):
    PERCEPTION = "PERCEPTION"
    GEOM = "GEOM"
    TRACK = "TRACK"
    REID = "REID"
    CLIP = "CLIP"
    SYSTEM = "SYSTEM"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, BBox):
        return value.as_dict()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in image coordinates ``(x1, y1, x2, y2)``."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        # Reorder rather than reject, so detections coming from different
        # backends cannot break the tracker on a reversed corner pair.
        if self.x2 < self.x1:
            x1, x2 = self.x2, self.x1
            object.__setattr__(self, "x1", x1)
            object.__setattr__(self, "x2", x2)
        if self.y2 < self.y1:
            y1, y2 = self.y2, self.y1
            object.__setattr__(self, "y1", y1)
            object.__setattr__(self, "y2", y2)

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> BBox:
        x1, y1, x2, y2 = (float(v) for v in values)
        return cls(x1, y1, x2, y2)

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> BBox:
        return cls(float(x), float(y), float(x) + float(width), float(y) + float(height))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def ymax(self) -> float:
        return self.y2

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def int_xyxy(self) -> tuple[int, int, int, int]:
        return tuple(int(round(v)) for v in self.xyxy)  # type: ignore[return-value]

    def expanded(self, pad: float, bounds: tuple[int, int] | None = None) -> BBox:
        x1, y1, x2, y2 = self.x1 - pad, self.y1 - pad, self.x2 + pad, self.y2 + pad
        if bounds is not None:
            width, height = bounds
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
        return BBox(x1, y1, max(x1, x2), max(y1, y2))

    def clipped(self, width: int, height: int) -> BBox:
        return BBox(
            max(0.0, min(self.x1, float(width))),
            max(0.0, min(self.y1, float(height))),
            max(0.0, min(self.x2, float(width))),
            max(0.0, min(self.y2, float(height))),
        )

    def intersection_area(self, other: BBox) -> float:
        iw = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        ih = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        return iw * ih

    def iou(self, other: BBox) -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def ios(self, other: BBox) -> float:
        """Intersection over smaller area (the spec's IoS)."""
        inter = self.intersection_area(other)
        smaller = min(self.area, other.area)
        return inter / smaller if smaller > 0 else 0.0

    def contains_point(self, point: Sequence[float], pad: float = 0.0) -> bool:
        x, y = float(point[0]), float(point[1])
        return self.x1 - pad <= x <= self.x2 + pad and self.y1 - pad <= y <= self.y2 + pad

    def as_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


# ---------------------------------------------------------------------------
# Per-frame perception
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Detection:
    label: str
    bbox: BBox
    confidence: float = 1.0
    track_id: int | None = None
    embedding: np.ndarray | None = None
    source: str = "detector"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "label": self.label,
            "bbox": self.bbox.as_dict(),
            "confidence": round(float(self.confidence), 4),
            "track_id": self.track_id,
            "source": self.source,
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        if self.metadata:
            data["metadata"] = _jsonable(self.metadata)
        return data


@dataclass
class HandObservation:
    """One MediaPipe hand seen in the current frame.

    Role A constructs this from a ``(21, 2)`` landmark array (or from the
    color/skin fallback); Role B also accepts a plain sequence of points.  The
    geometry helpers below serve both roles without importing MediaPipe.
    """

    landmarks: Sequence[tuple[float, float]] | np.ndarray = ()
    hand_id: str = "hand:0"
    polygon: Polygon | None = None
    is_left: bool | None = None
    confidence: float = 1.0
    handedness: str = "Right"
    source: str = "mediapipe"
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hand_id:
            raise ValueError("hand_id must be a non-empty identifier")
        if isinstance(self.landmarks, np.ndarray):
            array = np.asarray(self.landmarks, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != 2:
                raise ValueError(f"hand landmarks must be (N, 2), got {array.shape}")
            self.landmarks = array
        else:
            self.landmarks = tuple((float(x), float(y)) for x, y in self.landmarks)
        self.confidence = float(self.confidence)
        self.handedness = str(self.handedness).title()

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
        polygon = MultiPoint(points).convex_hull if len(points) >= 3 else None
        return cls(
            landmarks=points,
            hand_id=hand_id,
            polygon=polygon,
            is_left=is_left,
            attributes=dict(attributes or {}),
        )

    def _points(self) -> np.ndarray:
        if isinstance(self.landmarks, np.ndarray):
            return self.landmarks
        if not self.landmarks:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(self.landmarks, dtype=np.float32)

    @property
    def bbox(self) -> BBox:
        points = self._points()
        if len(points) == 0:
            return BBox(0.0, 0.0, 0.0, 0.0)
        return BBox(
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )

    @property
    def wrist(self) -> np.ndarray:
        points = self._points()
        return points[WRIST_INDEX] if len(points) else np.zeros(2, dtype=np.float32)

    @property
    def palm_center(self) -> np.ndarray:
        points = self._points()
        if len(points) > PALM_CENTER_INDEX:
            return points[PALM_CENTER_INDEX]
        if len(points):
            return points.mean(axis=0)
        return np.zeros(2, dtype=np.float32)

    @property
    def index_tip(self) -> np.ndarray:
        points = self._points()
        return points[8] if len(points) > 8 else self.wrist

    @property
    def centroid(self) -> tuple[float, float] | None:
        points = self._points()
        if len(points) == 0:
            return None
        center = self.palm_center
        return float(center[0]), float(center[1])

    def fingertips(self, indices: Sequence[int] = FINGERTIP_INDICES) -> np.ndarray:
        points = self._points()
        valid = [index for index in indices if 0 <= index < len(points)]
        return points[valid]

    def hull(self) -> Polygon | None:
        if self.polygon is not None and not self.polygon.is_empty:
            return self.polygon
        points = self._points()
        if len(points) >= 3:
            return MultiPoint([tuple(map(float, point)) for point in points]).convex_hull
        return None

    def as_dict(self, *, include_landmarks: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "hand_id": self.hand_id,
            "handedness": self.handedness,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.as_dict(),
            "wrist": self.wrist.tolist(),
            "palm_center": self.palm_center.tolist(),
            "index_tip": self.index_tip.tolist(),
            "source": self.source,
        }
        if self.is_left is not None:
            data["is_left"] = bool(self.is_left)
        if include_landmarks:
            data["landmarks"] = self._points().tolist()
        return data


@dataclass(slots=True)
class MotionEstimate:
    dx: float = 0.0
    dy: float = 0.0
    theta: float = 0.0  # radians
    inlier_ratio: float = 0.0
    tracked_points: int = 0
    valid: bool = False
    state: CameraState = CameraState.UNKNOWN
    transition: str | None = None  # "pan_start" | "pan_end"

    @property
    def magnitude(self) -> float:
        """Translation magnitude in pixels for the processed frame pair."""
        return math.hypot(self.dx, self.dy)

    @property
    def rotation_degrees(self) -> float:
        return math.degrees(self.theta)

    @property
    def is_pan(self) -> bool:
        return self.state is CameraState.PANNING

    @property
    def is_still(self) -> bool:
        return self.state is CameraState.STATIONARY

    @property
    def keyframe(self) -> bool:
        """A settled frame after a pan is a good CLIP/VPR keyframe."""
        return self.transition == "pan_end"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dx": round(float(self.dx), 3),
            "dy": round(float(self.dy), 3),
            "theta": round(float(self.theta), 5),
            "magnitude": round(float(self.magnitude), 3),
            "rotation_degrees": round(float(self.rotation_degrees), 3),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "tracked_points": int(self.tracked_points),
            "valid": bool(self.valid),
            "state": self.state.value,
            "transition": self.transition,
            "keyframe": self.keyframe,
        }


@dataclass(slots=True)
class TrackedObject:
    track_id: int
    label: str
    bbox: BBox
    confidence: float
    first_seen: float
    last_seen: float
    hits: int
    age: int
    time_since_update: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s in image space
    relative_velocity: tuple[float, float] = (0.0, 0.0)  # px/s minus camera motion
    embedding: np.ndarray | None = None
    source: str = "bytetrack"
    metadata: dict[str, Any] = field(default_factory=dict)
    entity_id: str | None = None
    reid_score: float | None = None
    reid_source: str = ""

    @property
    def speed(self) -> float:
        return math.hypot(*self.velocity)

    @property
    def relative_speed(self) -> float:
        return math.hypot(*self.relative_velocity)

    @property
    def centroid(self) -> tuple[float, float]:
        return self.bbox.centroid

    def as_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "track_id": self.track_id,
            "entity_id": self.entity_id,
            "label": self.label,
            "bbox": self.bbox.as_dict(),
            "confidence": round(float(self.confidence), 4),
            "first_seen": round(float(self.first_seen), 3),
            "last_seen": round(float(self.last_seen), 3),
            "hits": int(self.hits),
            "age": int(self.age),
            "time_since_update": int(self.time_since_update),
            "velocity": [round(v, 2) for v in self.velocity],
            "relative_velocity": [round(v, 2) for v in self.relative_velocity],
            "speed": round(float(self.speed), 2),
            "relative_speed": round(float(self.relative_speed), 2),
            "has_embedding": self.embedding is not None,
            "source": self.source,
            "reid_score": self.reid_score,
            "reid_source": self.reid_source,
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        if self.metadata:
            data["metadata"] = _jsonable(self.metadata)
        return data


# ---------------------------------------------------------------------------
# Hand/object interaction + temporal clips
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InteractionObservation:
    track_id: int
    label: str
    hand: str
    contact: bool
    ios: float
    fingertips_inside: bool
    active_since: float
    duration: float
    dwell_met: bool
    bbox: BBox

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "hand": self.hand,
            "contact": bool(self.contact),
            "ios": round(float(self.ios), 4),
            "fingertips_inside": bool(self.fingertips_inside),
            "active_since": round(float(self.active_since), 3),
            "duration": round(float(self.duration), 3),
            "dwell_met": bool(self.dwell_met),
            "bbox": self.bbox.as_dict(),
        }


@dataclass(slots=True)
class InteractionEvent:
    kind: str  # "start" | "end"
    track_id: int
    label: str
    hand: str
    timestamp: float
    start_timestamp: float
    duration: float
    ios: float
    trigger: str
    bbox: BBox
    frame_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "track_id": self.track_id,
            "label": self.label,
            "hand": self.hand,
            "timestamp": round(float(self.timestamp), 3),
            "start_timestamp": round(float(self.start_timestamp), 3),
            "duration": round(float(self.duration), 3),
            "ios": round(float(self.ios), 4),
            "trigger": self.trigger,
            "bbox": self.bbox.as_dict(),
            "frame_index": int(self.frame_index),
        }


@dataclass(slots=True)
class ClipFrame:
    frame_index: int
    timestamp: float
    image: np.ndarray  # BGR, downscaled to the ring-buffer resolution
    target_box: BBox | None = None
    hand_points: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 3),
            "target_box": None if self.target_box is None else self.target_box.as_dict(),
            "has_hand": self.hand_points is not None,
        }


@dataclass
class InteractionClip:
    """A structured 30-frame interaction window for the VLM worker."""

    clip_id: str
    track_id: int
    label: str
    hand: str
    start_ts: float
    end_ts: float
    trigger: str
    frames: list[ClipFrame]
    contact_index: int
    max_ios: float = 0.0
    composite: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def contact_ts(self) -> float | None:
        if 0 <= self.contact_index < len(self.frames):
            return self.frames[self.contact_index].timestamp
        return None

    def build_composite(self) -> np.ndarray | None:
        """Stitch pre-contact / contact / post-contact tiles into a 1x3 strip."""
        if not self.frames:
            return None
        count = len(self.frames)
        if count == 1:
            indices = [0, 0, 0]
        elif count == 2:
            indices = [0, 1, 1]
        else:
            indices = [0, max(0, min(self.contact_index, count - 1)), count - 1]
            if len(set(indices)) < 3:
                indices = [0, count // 2, count - 1]

        import cv2

        tiles: list[np.ndarray] = []
        # A clip cut by the ring length while contact is still ongoing has no
        # true post-contact tile; label it honestly instead of calling the last
        # still-contacting frame "POST".
        tags = (
            ("PRE", "CONTACT", "POST")
            if self.metadata.get("ended", True)
            else ("PRE", "CONTACT", "LATEST")
        )
        for tag, index in zip(tags, indices, strict=False):
            frame = self.frames[index]
            tile = frame.image.copy()
            if frame.target_box is not None:
                cv2.rectangle(tile, frame.target_box.int_xyxy, (0, 255, 0), 2)
            if frame.hand_points is not None:
                for point in frame.hand_points:
                    cv2.circle(
                        tile,
                        (int(round(point[0])), int(round(point[1]))),
                        3,
                        (0, 165, 255),
                        -1,
                    )
            label = f"{tag}  t={frame.timestamp:.2f}s"
            cv2.putText(
                tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                tile,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)

        height = max(tile.shape[0] for tile in tiles)
        normalized = [
            tile
            if tile.shape[0] == height
            else cv2.resize(tile, (int(tile.shape[1] * height / tile.shape[0]), height))
            for tile in tiles
        ]
        self.composite = np.hstack(normalized)
        return self.composite

    def save(self, directory: str | Any, *, save_frames: bool = False) -> dict[str, str]:
        import json
        from pathlib import Path

        import cv2

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        composite = self.composite if self.composite is not None else self.build_composite()
        paths: dict[str, str] = {}
        if composite is not None:
            composite_path = out / f"{self.clip_id}.jpg"
            cv2.imwrite(str(composite_path), composite)
            paths["composite"] = str(composite_path)
        if save_frames:
            frame_dir = out / self.clip_id
            frame_dir.mkdir(exist_ok=True)
            for frame in self.frames:
                frame_path = frame_dir / f"{frame.frame_index:06d}.jpg"
                cv2.imwrite(str(frame_path), frame.image)
            paths["frames"] = str(frame_dir)
        paths["metadata"] = str(out / f"{self.clip_id}.json")
        self.metadata["saved"] = dict(paths)
        Path(paths["metadata"]).write_text(
            json.dumps(self.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        return paths

    def as_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "track_id": int(self.track_id),
            "label": self.label,
            "hand": self.hand,
            "start_ts": round(float(self.start_ts), 3),
            "end_ts": round(float(self.end_ts), 3),
            "contact_ts": None if self.contact_ts is None else round(self.contact_ts, 3),
            "trigger": self.trigger,
            "max_ios": round(float(self.max_ios), 4),
            "frame_count": len(self.frames),
            "frames": [frame.as_dict() for frame in self.frames],
            "metadata": _jsonable(self.metadata),
        }


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TelemetryEvent:
    category: EventCategory
    message: str
    timestamp: float = field(default_factory=time.time)
    presentation_time: float | None = None
    frame_index: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        prefix = f"[{self.category.value}]"
        if self.presentation_time is not None:
            return f"{prefix} t={self.presentation_time:6.2f}s {self.message}"
        return f"{prefix} {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "message": self.message,
            "timestamp": round(float(self.timestamp), 4),
            "presentation_time": None
            if self.presentation_time is None
            else round(float(self.presentation_time), 3),
            "frame_index": self.frame_index,
            "data": _jsonable(self.data),
        }


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    wall_time: float
    motion: MotionEstimate
    detections: list[Detection]
    hands: list[HandObservation]
    tracks: list[TrackedObject]
    interactions: list[InteractionObservation]
    interaction_events: list[InteractionEvent]
    clips: list[InteractionClip]
    telemetry: list[TelemetryEvent]
    image: np.ndarray | None = None
    supports: list[SupportSurface] = field(default_factory=list)

    @property
    def camera_state(self) -> CameraState:
        return self.motion.state

    def as_dict(
        self, *, include_landmarks: bool = False, include_embeddings: bool = False
    ) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 3),
            "wall_time": round(float(self.wall_time), 4),
            "motion": self.motion.as_dict(),
            "camera_state": self.camera_state.value,
            "detections": [
                d.as_dict(include_embedding=include_embeddings) for d in self.detections
            ],
            "hands": [h.as_dict(include_landmarks=include_landmarks) for h in self.hands],
            "tracks": [t.as_dict(include_embedding=include_embeddings) for t in self.tracks],
            "supports": [support.as_dict() for support in self.supports],
            "interactions": [i.as_dict() for i in self.interactions],
            "interaction_events": [e.as_dict() for e in self.interaction_events],
            "clips": [c.as_dict() for c in self.clips],
            "telemetry": [t.as_dict() for t in self.telemetry],
        }


@dataclass
class RunSummary:
    video: str
    frames: int
    duration: float
    elapsed: float
    detections: int
    tracks: int
    interactions: int
    clips: int
    camera_states: dict[str, int] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "frames": self.frames,
            "duration": round(self.duration, 3),
            "elapsed": round(self.elapsed, 3),
            "processing_fps": round(self.fps, 2),
            "detections": self.detections,
            "tracks": self.tracks,
            "interactions": self.interactions,
            "clips": self.clips,
            "camera_states": self.camera_states,
        }

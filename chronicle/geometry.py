"""Deterministic 2D geometry used by the lifecycle state machine.

All spatial reasoning in Chronicle is 2D projective geometry on mask hulls.  The
specification deliberately avoids metric 3D reconstruction; correctness comes
from *relative* containment between hand, object and support polygons.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from shapely.geometry import Point, Polygon

BBox = tuple[float, float, float, float]


def bbox_to_polygon(bbox: BBox) -> Polygon:
    """Axis-aligned rectangle for an ``(x1, y1, x2, y2)`` box."""

    x1, y1, x2, y2 = (float(v) for v in bbox)
    return Polygon.from_bounds(x1, y1, x2, y2)


def to_polygon(value: Polygon | BBox | Sequence[tuple[float, float]]) -> Polygon:
    """Coerce a polygon, an ``xyxy`` box, or a point ring into a Polygon."""

    if isinstance(value, Polygon):
        if value.is_empty:
            raise ValueError("polygon must be non-empty")
        return value
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):  # type: ignore[arg-type]
        return bbox_to_polygon(tuple(float(v) for v in value))  # type: ignore[arg-type]
    return Polygon([(float(x), float(y)) for x, y in value])  # type: ignore[misc]


def _safe_area(poly: Polygon) -> float:
    return float(poly.area) if not poly.is_empty else 0.0


def ios(a: Polygon, b: Polygon) -> float:
    """Intersection over Smaller Area.

    ``IoS(A, B) = area(A ∩ B) / min(area(A), area(B))``

    Unlike IoU this is 1.0 when the smaller polygon is fully enclosed by the
    larger one, which is exactly the hand-in-object and object-in-container
    semantics the specification needs.
    """

    area_a = _safe_area(a)
    area_b = _safe_area(b)
    smaller = min(area_a, area_b)
    if smaller <= 0.0:
        return 0.0
    if not a.intersects(b):
        return 0.0
    inter = a.intersection(b)
    if inter.is_empty:
        return 0.0
    return min(1.0, float(inter.area) / smaller)


def iou(a: Polygon, b: Polygon) -> float:
    """Classic intersection over union, provided for diagnostics only."""

    union = _safe_area(a) + _safe_area(b) - _safe_area(a.intersection(b))
    if union <= 0.0:
        return 0.0
    return _safe_area(a.intersection(b)) / union


def centroid(poly: Polygon) -> tuple[float, float]:
    c = poly.centroid
    return float(c.x), float(c.y)


def bottom_edge_samples(poly: Polygon, samples: int = 21) -> list[tuple[float, float]]:
    """Points sampled along the ``ymax`` edge of a polygon's bounding box."""

    minx, _, maxx, maxy = poly.bounds
    if maxx <= minx:
        return [(minx, maxy)]
    step = (maxx - minx) / (samples - 1)
    return [(minx + step * i, maxy) for i in range(samples)]


def bottom_edge_midpoint(poly: Polygon) -> tuple[float, float]:
    minx, _, maxx, maxy = poly.bounds
    return (minx + maxx) / 2.0, maxy


@dataclass(frozen=True)
class SupportContact:
    """Outcome of the support-surface heuristic for one object/surface pair."""

    anchor_id: str
    supported: bool
    edge_overlap: float
    gap_px: float
    ios: float
    score: float

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "anchor_id": self.anchor_id,
            "supported": self.supported,
            "edge_overlap": round(self.edge_overlap, 4),
            "gap_px": round(self.gap_px, 2),
            "ios": round(self.ios, 4),
            "score": round(self.score, 4),
        }


def support_contact(
    obj: Polygon,
    support: Polygon,
    support_id: str,
    tolerance_px: float = 12.0,
    min_edge_overlap: float = 0.60,
    min_ios: float = 0.50,
) -> SupportContact:
    """Verify that an object rests on a support surface.

    Two complementary tests, both used by the specification's ``ymax`` contact
    heuristic:

    1. *Edge contact* — enough of the object's bottom edge lies within
       ``tolerance_px`` of the support polygon (the normal case: a mug on a
       table whose detector box covers the whole tabletop).
    2. *Overlap* — the object's hull is mostly inside the support region (a box
       detection that fully encloses the object).
    """

    if obj.is_empty or support.is_empty:
        return SupportContact(support_id, False, 0.0, math.inf, 0.0, 0.0)

    buffered = support.buffer(tolerance_px, join_style=2) if tolerance_px > 0 else support
    samples = bottom_edge_samples(obj)
    covered = sum(1 for x, y in samples if buffered.covers(Point(x, y)))
    edge_overlap = covered / len(samples)

    gap = min(float(support.distance(Point(x, y))) for x, y in samples)
    overlap = ios(obj, support)

    supported = edge_overlap >= min_edge_overlap or overlap >= min_ios
    score = max(edge_overlap, overlap)
    return SupportContact(
        anchor_id=support_id,
        supported=supported,
        edge_overlap=edge_overlap,
        gap_px=gap,
        ios=overlap,
        score=score,
    )


def point_inside(poly: Polygon, point: tuple[float, float]) -> bool:
    return bool(poly.covers(Point(point[0], point[1])))


def any_point_inside(poly: Polygon, points: Iterable[tuple[float, float]]) -> bool:
    return any(point_inside(poly, p) for p in points)


def polyline_velocity(
    history: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Estimate ``(vx, vy, speed)`` in px/s from ``(t, x, y)`` samples."""

    if len(history) < 2:
        return 0.0, 0.0, 0.0
    t0, x0, y0 = history[0]
    t1, x1, y1 = history[-1]
    dt = t1 - t0
    if dt <= 1e-6:
        return 0.0, 0.0, 0.0
    vx = (x1 - x0) / dt
    vy = (y1 - y0) / dt
    return vx, vy, math.hypot(vx, vy)


def cosine_similarity(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Cosine similarity that tolerates missing, empty and zero vectors."""

    if a is None or b is None:
        return 0.0
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for va, vb in zip(a, b, strict=True):
        fa = float(va)
        fb = float(vb)
        dot += fa * fb
        norm_a += fa * fa
        norm_b += fb * fb
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (math.sqrt(norm_a) * math.sqrt(norm_b))))


def ema_time_aware(previous: float | None, value: float, dt: float, tau: float) -> float:
    """Exponential smoother whose response is independent of frame rate."""

    if previous is None or dt <= 0.0 or tau <= 0.0:
        return value
    alpha = 1.0 - math.exp(-dt / tau)
    return previous + alpha * (value - previous)


def center_outside_frame(poly: Polygon, frame_size: tuple[int, int], margin: float = 0.0) -> bool:
    """True when a polygon's centroid has left the visual frustum."""

    cx, cy = centroid(poly)
    width, height = frame_size
    return not (-margin <= cx <= width + margin and -margin <= cy <= height + margin)

"""Role A static-support candidate extraction.

Role A is responsible for the *candidate* signal, not the final support
decision.  A furniture track (sofa, table, counter, ...) is associated across
frames with camera-motion compensation, given a stable ``anchor_id``, and
emitted as a :class:`SupportSurface`.  Role B can then create the background
anchor and use it for ``RESTING_ON`` reasoning.

This is intentionally lighter than Role B's ``FurnitureRegistry``: it only
needs frame-to-frame identity while the detector sees the object.  Role B owns
long-term re-entry, cross-session persistence and duplicate consolidation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from chronicle.perception.config import SupportConfig
from chronicle.types import BBox, SupportSurface, TrackedObject

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(label: str) -> str:
    return _SLUG.sub("-", label.strip().lower()).strip("-") or "surface"


def _shift(box: BBox, dx: float, dy: float) -> BBox:
    return BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)


@dataclass
class SupportTrack:
    anchor_id: str
    label: str
    bbox: BBox
    first_seen: float
    last_seen: float
    hits: int = 1
    track_ids: set[int] = field(default_factory=set)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 1

    def update(self, box: BBox, t: float, track_id: int | None) -> None:
        self.bbox = box
        self.last_seen = max(self.last_seen, t)
        self.hits += 1
        if track_id is not None:
            self.track_ids.add(int(track_id))


class SupportTracker:
    """Associate furniture tracks to stable per-frame support candidates."""

    def __init__(self, config: SupportConfig | None = None) -> None:
        self.config = config or SupportConfig()
        self.labels = frozenset(label.strip().lower() for label in self.config.labels)
        self._tracks: dict[str, SupportTrack] = {}
        self._seq = 0

    def __len__(self) -> int:
        return len(self._tracks)

    @property
    def tracks(self) -> tuple[SupportTrack, ...]:
        return tuple(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._seq = 0

    def update(
        self,
        tracks: list[TrackedObject],
        t: float,
        *,
        camera_dx: float = 0.0,
        camera_dy: float = 0.0,
    ) -> list[SupportSurface]:
        if not self.config.enabled:
            return []

        supports: list[SupportSurface] = []
        used: set[str] = set()
        for track in tracks:
            label = track.label.strip().lower()
            if label not in self.labels:
                continue
            best: SupportTrack | None = None
            best_score = -1.0
            for candidate in self._tracks.values():
                if candidate.anchor_id in used or candidate.label != label:
                    continue
                if t - candidate.last_seen > self.config.max_missed_s:
                    continue
                predicted = (
                    _shift(candidate.bbox, camera_dx, camera_dy)
                    if self.config.camera_compensation
                    else candidate.bbox
                )
                iou = predicted.iou(track.bbox)
                distance = math.hypot(
                    predicted.centroid[0] - track.bbox.centroid[0],
                    predicted.centroid[1] - track.bbox.centroid[1],
                )
                diagonal = max(
                    1.0,
                    math.hypot(
                        max(predicted.x2, track.bbox.x2) - min(predicted.x1, track.bbox.x1),
                        max(predicted.y2, track.bbox.y2) - min(predicted.y1, track.bbox.y1),
                    ),
                )
                proximity = max(
                    0.0,
                    1.0 - distance / max(diagonal * self.config.center_distance_ratio, 1.0),
                )
                same_track = 1.0 if track.track_id in candidate.track_ids else 0.0
                score = 0.65 * iou + 0.25 * proximity + 0.10 * same_track
                if iou < self.config.iou_threshold and proximity <= 0.0:
                    continue
                if score > best_score:
                    best = candidate
                    best_score = score

            if best is not None and best_score >= self.config.match_threshold:
                best.update(track.bbox, t, track.track_id)
                record = best
                used.add(best.anchor_id)
            else:
                record = self._create(label, track, t)
                used.add(record.anchor_id)

            if record.hits >= self.config.min_hits:
                supports.append(self._surface(record, track))
        return supports

    def _create(self, label: str, track: TrackedObject, t: float) -> SupportTrack:
        self._seq += 1
        anchor_id = f"support-{_slug(label)}-{self._seq}"
        while anchor_id in self._tracks:
            self._seq += 1
            anchor_id = f"support-{_slug(label)}-{self._seq}"
        record = SupportTrack(
            anchor_id=anchor_id,
            label=label,
            bbox=track.bbox,
            first_seen=t,
            last_seen=t,
            hits=1,
            track_ids={int(track.track_id)} if track.track_id is not None else set(),
        )
        self._tracks[anchor_id] = record
        return record

    def _surface(self, record: SupportTrack, track: TrackedObject) -> SupportSurface:
        bbox = record.bbox
        return SupportSurface(
            anchor_id=record.anchor_id,
            label=record.label,
            polygon=Polygon.from_bounds(*bbox.xyxy),
            is_fixed=True,
            attributes={
                "source": "role-a-support-track",
                "track_id": track.track_id,
                "confidence": round(float(track.confidence), 4),
                "bbox": list(bbox.xyxy),
                "hits": record.hits,
                "first_seen": record.first_seen,
                "last_seen": record.last_seen,
                "confirmed": True,
                # Role A is an honest candidate producer: Role B owns the final
                # support-contact verification.
                "verified": False,
            },
        )


__all__ = ["SupportConfig", "SupportTrack", "SupportTracker"]

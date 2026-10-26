"""Persistent static-furniture association.

Role A's detector is open-vocabulary, so a sofa, table or TV is emitted as an
ordinary track.  If those tracks are fed to the foreground graph every time the
camera re-enters a room, the episodic graph fills with duplicate ``sofa``
nodes.  Furniture is *background*: it belongs in the invariant zone/anchor graph
and should be associated across track losses, camera pans and re-detections.

The registry implements the object-centric association step described in
ConceptGraphs/Hydra/OVSG-style scene graphs:

* a label + zone prior prunes the candidate set;
* motion-compensated IoU and centre distance associate the current detection
  with a persistent anchor;
* an appearance embedding (CLIP/colour) breaks ties when geometry is weak;
* a short confirmation window suppresses one-frame false positives;
* two co-visible instances of the same label stay separate because a record
  cannot be matched twice at the same timestamp unless the boxes overlap.

The registry itself is storage-agnostic.  :class:`ChronicleMemory` mirrors
records into the background graph and SQLite, and the graph consolidator uses
the same geometry to merge legacy duplicates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from chronicle.config import (
    DEFAULT_FURNITURE_CONFIG,
    STATIC_FURNITURE_LABELS,
    FurnitureConfig,
)
from chronicle.geometry import cosine_similarity
from chronicle.text import canonical_label, normalize

_SLUG = re.compile(r"[^a-z0-9]+")
BBox = tuple[float, float, float, float]


def _bbox(value: Sequence[float] | None) -> BBox | None:
    if value is None or len(value) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in value)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def bbox_iou(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """IoU for two ``xyxy`` boxes without importing shapely."""

    box_a = _bbox(a)
    box_b = _bbox(b)
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def bbox_center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def bbox_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _shift(box: BBox, dx: float, dy: float) -> BBox:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _size_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)


@dataclass
class FurnitureRecord:
    """In-memory association state for one persistent furniture anchor."""

    anchor_id: str
    label: str
    zone_id: str | None
    bbox: BBox
    embedding: tuple[float, ...] | None = None
    hits: int = 1
    misses: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    track_ids: set[int] = field(default_factory=set)
    confirmed: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_attributes(self) -> dict[str, Any]:
        attrs = {
            "bbox": list(self.bbox),
            "hits": self.hits,
            "misses": self.misses,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "track_ids": sorted(self.track_ids),
            "confirmed": self.confirmed,
            **self.attributes,
        }
        if self.embedding is not None:
            attrs["embedding"] = list(self.embedding)
        return attrs

    @classmethod
    def from_attributes(
        cls,
        anchor_id: str,
        label: str,
        zone_id: str | None,
        attrs: dict[str, Any],
        *,
        created_at: float = 0.0,
    ) -> FurnitureRecord | None:
        box = _bbox(attrs.get("bbox"))
        if box is None:
            return None
        embedding = attrs.get("embedding")
        return cls(
            anchor_id=anchor_id,
            label=normalize(label) or "surface",
            zone_id=zone_id,
            bbox=box,
            embedding=tuple(float(v) for v in embedding) if embedding else None,
            hits=max(1, int(attrs.get("hits", 1))),
            misses=max(0, int(attrs.get("misses", 0))),
            first_seen=float(attrs.get("first_seen", created_at)),
            last_seen=float(attrs.get("last_seen", created_at)),
            track_ids={int(v) for v in attrs.get("track_ids", [])},
            confirmed=bool(attrs.get("confirmed", True)),
            attributes={
                k: v
                for k, v in attrs.items()
                if k
                not in {
                    "bbox",
                    "embedding",
                    "hits",
                    "misses",
                    "first_seen",
                    "last_seen",
                    "track_ids",
                    "confirmed",
                }
            },
        )


@dataclass(frozen=True)
class FurnitureMatch:
    """Result of associating one static-object detection."""

    anchor_id: str
    label: str
    zone_id: str | None
    bbox: BBox
    score: float
    is_new: bool
    confirmed: bool
    track_id: int | None = None


class FurnitureRegistry:
    """Associate static-object tracks to stable background anchors."""

    def __init__(
        self,
        config: FurnitureConfig | None = None,
        *,
        labels: Iterable[str] = STATIC_FURNITURE_LABELS,
    ) -> None:
        self.config = config or DEFAULT_FURNITURE_CONFIG
        self.labels = frozenset(normalize(label) for label in labels if normalize(label))
        self._records: dict[str, FurnitureRecord] = {}
        self._seq = 0

    # ------------------------------------------------------------------ setup
    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, anchor_id: str) -> bool:
        return anchor_id in self._records

    @property
    def records(self) -> tuple[FurnitureRecord, ...]:
        return tuple(self._records.values())

    def record(self, anchor_id: str) -> FurnitureRecord | None:
        return self._records.get(anchor_id)

    def is_furniture(self, label: str) -> bool:
        return normalize(label) in self.labels

    def add_label(self, label: str) -> None:
        normalized = normalize(label)
        if normalized:
            self.labels = frozenset({*self.labels, normalized})

    def load(self, records: Iterable[FurnitureRecord]) -> None:
        for record in records:
            self._records[record.anchor_id] = record
            self._seq = max(self._seq, _numeric_suffix(record.anchor_id))

    def load_attributes(
        self,
        rows: Iterable[tuple[str, str, str | None, dict[str, Any]]],
    ) -> None:
        for anchor_id, label, zone_id, attrs in rows:
            record = FurnitureRecord.from_attributes(
                anchor_id,
                label,
                zone_id,
                attrs,
            )
            if record is not None:
                self._records[anchor_id] = record
                self._seq = max(self._seq, _numeric_suffix(anchor_id))

    # -------------------------------------------------------------- observation
    def observe(
        self,
        label: str,
        bbox: Sequence[float],
        zone_id: str | None,
        t: float,
        *,
        track_id: int | None = None,
        embedding: Sequence[float] | None = None,
        camera_dx: float = 0.0,
        camera_dy: float = 0.0,
        frame_size: tuple[int, int] | None = None,
    ) -> FurnitureMatch | None:
        """Associate one detection, creating a pending anchor when necessary."""

        label = normalize(label)
        if label not in self.labels:
            return None
        box = _bbox(bbox)
        if box is None:
            return None
        embedding_tuple = (
            tuple(float(v) for v in embedding) if embedding is not None else None
        )
        best: FurnitureRecord | None = None
        best_score = -1.0
        for record in self._records.values():
            if record.label != label or not _zones_compatible(record.zone_id, zone_id):
                continue
            if t < record.first_seen - 1e-6:
                continue
            if t - record.last_seen > self.config.max_missed_s:
                continue
            # Two distinct instances visible in the same frame must not share a
            # record.  A re-detection of the same instance has strong overlap.
            if (
                abs(record.last_seen - t) <= 1e-6
                and track_id is not None
                and record.track_ids
                and track_id not in record.track_ids
                and bbox_iou(record.bbox, box) < 0.5
            ):
                continue

            predicted = _shift(record.bbox, camera_dx, camera_dy)
            score = self._association_score(
                predicted,
                box,
                record.embedding,
                embedding_tuple,
                frame_size,
                track_id=track_id,
                record=record,
            )
            if score > best_score:
                best, best_score = record, score

        if best is not None and best_score >= self.config.match_threshold:
            self._update(best, box, t, track_id, embedding_tuple)
            return FurnitureMatch(
                anchor_id=best.anchor_id,
                label=best.label,
                zone_id=best.zone_id,
                bbox=best.bbox,
                score=best_score,
                is_new=False,
                confirmed=best.confirmed,
                track_id=track_id,
            )

        record = self._create(label, box, zone_id, t, track_id, embedding_tuple)
        return FurnitureMatch(
            anchor_id=record.anchor_id,
            label=record.label,
            zone_id=record.zone_id,
            bbox=record.bbox,
            score=0.0,
            is_new=True,
            confirmed=record.confirmed,
            track_id=track_id,
        )

    def _association_score(
        self,
        predicted: BBox,
        current: BBox,
        stored_embedding: Sequence[float] | None,
        current_embedding: Sequence[float] | None,
        frame_size: tuple[int, int] | None,
        *,
        track_id: int | None,
        record: FurnitureRecord,
    ) -> float:
        overlap = bbox_iou(predicted, current)
        pc = bbox_center(predicted)
        cc = bbox_center(current)
        distance = math.hypot(pc[0] - cc[0], pc[1] - cc[1])
        if frame_size:
            diagonal = math.hypot(float(frame_size[0]), float(frame_size[1]))
        else:
            diagonal = math.hypot(
                max(predicted[2], current[2]) - min(predicted[0], current[0]),
                max(predicted[3], current[3]) - min(predicted[1], current[1]),
            )
        distance_ratio = distance / max(diagonal, 1.0)
        appearance = cosine_similarity(stored_embedding, current_embedding)
        size = _size_ratio(predicted, current)

        # A track id continuity is a strong, cheap association cue.  It is not
        # sufficient on its own: the tracker may reuse an id for another object.
        same_track = 1.0 if track_id is not None and track_id in record.track_ids else 0.0
        spatial = max(0.0, 1.0 - distance_ratio / max(self.config.center_distance_ratio, 1e-6))
        if stored_embedding is not None and current_embedding is not None:
            score = 0.45 * overlap + 0.20 * spatial + 0.25 * max(0.0, appearance) + 0.10 * size
        else:
            score = 0.60 * overlap + 0.30 * spatial + 0.10 * size
        score += 0.25 * same_track

        # Hard gates: overlapping boxes are always acceptable; otherwise the
        # centre must be close and the size/appearance must agree.  This is what
        # keeps two sofas on opposite sides of a lounge separate.
        if overlap >= self.config.iou_threshold:
            return min(1.0, score + 0.20)
        if distance_ratio > self.config.center_distance_ratio:
            return 0.0
        if size < self.config.min_size_ratio:
            return 0.0
        if (
            stored_embedding is not None
            and current_embedding is not None
            and appearance < self.config.embedding_cosine
            and overlap < self.config.weak_iou_threshold
        ):
            return 0.0
        return score

    def _create(
        self,
        label: str,
        bbox: BBox,
        zone_id: str | None,
        t: float,
        track_id: int | None,
        embedding: tuple[float, ...] | None,
    ) -> FurnitureRecord:
        self._seq += 1
        slug = _SLUG.sub("-", label).strip("-") or "furniture"
        anchor_id = f"{slug}-{self._seq}"
        while anchor_id in self._records:
            self._seq += 1
            anchor_id = f"{slug}-{self._seq}"
        record = FurnitureRecord(
            anchor_id=anchor_id,
            label=label,
            zone_id=zone_id,
            bbox=bbox,
            embedding=embedding,
            hits=1,
            first_seen=t,
            last_seen=t,
            track_ids={int(track_id)} if track_id is not None else set(),
            confirmed=self.config.min_hits <= 1,
        )
        self._records[anchor_id] = record
        return record

    def _update(
        self,
        record: FurnitureRecord,
        bbox: BBox,
        t: float,
        track_id: int | None,
        embedding: tuple[float, ...] | None,
    ) -> None:
        # Keep a light exponential prototype so appearance survives gradual
        # lighting/pose changes without storing every crop.
        if embedding is not None:
            if record.embedding is None or len(record.embedding) != len(embedding):
                record.embedding = embedding
            else:
                alpha = 0.15
                record.embedding = tuple(
                    (1.0 - alpha) * old + alpha * new
                    for old, new in zip(record.embedding, embedding, strict=True)
                )
        record.bbox = bbox
        record.last_seen = max(record.last_seen, t)
        record.hits += 1
        record.misses = 0
        if track_id is not None:
            record.track_ids.add(int(track_id))
        if record.hits >= self.config.min_hits:
            record.confirmed = True

    def mark_missed(self, anchor_ids: Iterable[str], t: float) -> None:
        """Age records that were not observed; used by periodic maintenance."""

        seen = set(anchor_ids)
        for anchor_id, record in self._records.items():
            if anchor_id not in seen and t > record.last_seen:
                record.misses += 1

    def prune(self, t: float, *, max_age_s: float | None = None) -> list[str]:
        """Forget unconfirmed one-off detections, returning removed anchor ids."""

        max_age = self.config.max_missed_s if max_age_s is None else max_age_s
        removed: list[str] = []
        for anchor_id, record in list(self._records.items()):
            if not record.confirmed and t - record.last_seen > max_age:
                removed.append(anchor_id)
                del self._records[anchor_id]
        return removed

    def merge(self, keep: str, remove: str) -> bool:
        """Fold one duplicate anchor record into another.

        The caller is responsible for the graph/SQLite side of the merge.  This
        method only consolidates in-memory association state and keeps the union
        of track ids, time bounds and audit metadata.
        """

        keep_record = self._records.get(keep)
        remove_record = self._records.get(remove)
        if remove_record is None or keep == remove:
            return False
        if keep_record is not None and (
            canonical_label(keep_record.label) != canonical_label(remove_record.label)
        ):
            return False
        if keep_record is not None and not _zones_compatible(
            keep_record.zone_id, remove_record.zone_id
        ):
            return False
        if keep_record is None:
            # The keeper may be a legacy graph/store row that predates the
            # registry.  Adopt the removed record's association state so later
            # detections bind to the keeper instead of recreating the duplicate.
            keep_record = replace(remove_record, anchor_id=keep)
            self._records[keep] = keep_record

        remove_is_newer = remove_record.last_seen > keep_record.last_seen
        if remove_is_newer:
            keep_record.bbox = remove_record.bbox
            if remove_record.embedding is not None:
                keep_record.embedding = remove_record.embedding
        elif keep_record.embedding is None:
            keep_record.embedding = remove_record.embedding

        keep_record.first_seen = min(keep_record.first_seen, remove_record.first_seen)
        keep_record.last_seen = max(keep_record.last_seen, remove_record.last_seen)
        keep_record.hits = max(1, keep_record.hits + remove_record.hits)
        keep_record.misses = max(keep_record.misses, remove_record.misses)
        keep_record.track_ids.update(remove_record.track_ids)
        keep_record.confirmed = keep_record.confirmed or remove_record.confirmed
        if keep_record.zone_id is None:
            keep_record.zone_id = remove_record.zone_id

        merged_from = [
            str(value)
            for value in (
                *keep_record.attributes.get("merged_from", []),
                *remove_record.attributes.get("merged_from", []),
                remove,
            )
        ]
        keep_record.attributes.update(
            {
                **remove_record.attributes,
                **keep_record.attributes,
                "merged_from": list(dict.fromkeys(merged_from)),
            }
        )
        del self._records[remove]
        return True


def _zones_compatible(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return True
    return a == b or a.removeprefix("zone:") == b.removeprefix("zone:")


def _numeric_suffix(value: str) -> int:
    tail = value.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else 0


__all__ = [
    "FurnitureMatch",
    "FurnitureRecord",
    "FurnitureRegistry",
    "bbox_area",
    "bbox_center",
    "bbox_iou",
]

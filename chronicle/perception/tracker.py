"""ByteTrack-style tracking with CLIP appearance reconciliation.

Association is two-stage: high-confidence detections first, then low-confidence
detections. Detections that remain unmatched are compared against an inactive
track pool using cosine similarity (the spec's SC > 0.82 gate) plus spatial
continuity. A re-identified track keeps its original ID, preventing duplicate
graph nodes after occlusion.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chronicle.perception.config import TrackerConfig
from chronicle.perception.utils import cosine_similarity, normalize
from chronicle.types import BBox, Detection, TrackedObject

LOGGER = logging.getLogger(__name__)

try:  # scipy is small; fall back to greedy if the environment omits it
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class Track:
    track_id: int
    label: str
    bbox: BBox
    confidence: float
    embedding: np.ndarray | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_frame: int = 0
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    active: bool = True
    announced: bool = False
    velocity: tuple[float, float] = (0.0, 0.0)
    prev_centroid: tuple[float, float] | None = None
    history: deque[tuple[float, BBox]] = field(default_factory=lambda: deque(maxlen=30))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.history.append((self.last_seen, self.bbox))

    @property
    def centroid(self) -> tuple[float, float]:
        return self.bbox.centroid

    def update(
        self,
        detection: Detection,
        timestamp: float,
        frame_index: int,
        *,
        smooth: float = 0.65,
        reset_velocity: bool = False,
    ) -> None:
        old = self.centroid
        new_bbox = detection.bbox
        if not reset_velocity and self.bbox.area > 0:
            blend = max(0.0, min(1.0, float(smooth)))
            x1 = blend * new_bbox.x1 + (1.0 - blend) * self.bbox.x1
            y1 = blend * new_bbox.y1 + (1.0 - blend) * self.bbox.y1
            x2 = blend * new_bbox.x2 + (1.0 - blend) * self.bbox.x2
            y2 = blend * new_bbox.y2 + (1.0 - blend) * self.bbox.y2
            new_bbox = BBox(x1, y1, max(x1 + 1.0, x2), max(y1 + 1.0, y2))
        self.bbox = new_bbox
        new = self.centroid
        dt = max(1e-3, timestamp - self.last_seen) if self.last_seen else 1e-3
        self.velocity = (
            (0.0, 0.0)
            if reset_velocity
            else ((new[0] - old[0]) / dt, (new[1] - old[1]) / dt)
        )
        self.prev_centroid = old
        self.confidence = detection.confidence
        self.last_seen = timestamp
        self.last_frame = frame_index
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.active = True
        if detection.embedding is not None:
            vector = normalize(detection.embedding)
            if vector is not None and float(np.linalg.norm(vector)) > 0:
                self.embedding = (
                    vector
                    if self.embedding is None
                    else normalize(0.75 * self.embedding + 0.25 * vector)
                )
        self.history.append((timestamp, self.bbox))

    def mark_lost(self) -> None:
        self.active = False
        self.time_since_update += 1

    def as_object(self) -> TrackedObject:
        reid_score = self.metadata.get("reid_score")
        reid_similarity = self.metadata.get("reid_similarity")
        decision_score = (
            reid_similarity
            if self.metadata.get("reid_source") == "appearance" and reid_similarity is not None
            else reid_score
        )
        return TrackedObject(
            track_id=self.track_id,
            label=self.label,
            bbox=self.bbox,
            confidence=self.confidence,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            velocity=self.velocity,
            embedding=self.embedding,
            source="bytetrack",
            metadata=dict(self.metadata),
            entity_id=str(self.metadata.get("entity_id") or f"track_{self.track_id}"),
            reid_score=float(decision_score) if decision_score is not None else None,
            reid_source=str(self.metadata.get("reid_source") or ""),
        )


@dataclass
class TrackerResult:
    tracks: list[TrackedObject] = field(default_factory=list)
    new_tracks: list[TrackedObject] = field(default_factory=list)
    lost: list[TrackedObject] = field(default_factory=list)
    rebound: list[TrackedObject] = field(default_factory=list)


class ByteTrackLite:
    """Two-stage tracker with an inactive Re-ID memory."""

    def __init__(
        self, config: TrackerConfig | None = None, *, embedder: object | None = None
    ) -> None:
        self.config = config or TrackerConfig()
        self.embedder = embedder
        self._active: dict[int, Track] = {}
        self._inactive: dict[int, Track] = {}
        self._next_id = 1
        self._lock = threading.RLock()
        self.reid_events: list[dict[str, Any]] = []

    def reset(self) -> None:
        with self._lock:
            self._active.clear()
            self._inactive.clear()
            self._next_id = 1
            self.reid_events.clear()

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------
    @staticmethod
    def _center_distance(a: BBox, b: BBox) -> float:
        return math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])

    def _appearance(self, track: Track, detection: Detection) -> float:
        return cosine_similarity(track.embedding, detection.embedding)

    def _association_score(self, track: Track, detection: Detection) -> float:
        iou = track.bbox.iou(detection.bbox)
        appearance = self._appearance(track, detection)
        distance = self._center_distance(track.bbox, detection.bbox)
        proximity = max(0.0, 1.0 - distance / max(1.0, self.config.max_center_distance))
        return 0.55 * iou + 0.25 * appearance + 0.20 * proximity

    def _valid_pair(self, track: Track, detection: Detection) -> bool:
        if track.label != detection.label:
            return False
        iou = track.bbox.iou(detection.bbox)
        distance = self._center_distance(track.bbox, detection.bbox)
        appearance = self._appearance(track, detection)
        return (
            iou > self.config.iou_gate
            or distance < self.config.max_center_distance
            or appearance >= self.config.reid_threshold
        )

    def _assign(
        self, tracks: list[Track], detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[Track], list[Detection]]:
        if not tracks or not detections:
            return [], tracks, detections
        scores = np.full((len(tracks), len(detections)), -1e6, dtype=np.float32)
        for ti, track in enumerate(tracks):
            for di, detection in enumerate(detections):
                if not self._valid_pair(track, detection):
                    continue
                scores[ti, di] = self._association_score(track, detection)
        pairs: list[tuple[int, int]] = []
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(-scores)
            pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=False) if scores[r, c] > -1e5]
        else:  # pragma: no cover
            candidates = sorted(
                (
                    (float(scores[ti, di]), ti, di)
                    for ti in range(len(tracks))
                    for di in range(len(detections))
                    if scores[ti, di] > -1e5
                ),
                reverse=True,
            )
            used_tracks: set[int] = set()
            used_detections: set[int] = set()
            for _, ti, di in candidates:
                if ti in used_tracks or di in used_detections:
                    continue
                used_tracks.add(ti)
                used_detections.add(di)
                pairs.append((ti, di))
        matched_tracks = {ti for ti, _ in pairs}
        matched_detections = {di for _, di in pairs}
        return (
            pairs,
            [track for index, track in enumerate(tracks) if index not in matched_tracks],
            [detection for index, detection in enumerate(detections) if index not in matched_detections],
        )

    # ------------------------------------------------------------------
    # Re-ID and track creation
    # ------------------------------------------------------------------
    def _fill_embeddings(self, frame: np.ndarray, detections: list[Detection]) -> None:
        if self.embedder is None or frame is None:
            return
        missing = [detection for detection in detections if detection.embedding is None]
        if not missing:
            return
        try:
            vectors = self.embedder.embed_crops(frame, [detection.bbox for detection in missing])
        except Exception as exc:
            LOGGER.warning("Embedding failed; Re-ID will use spatial continuity only: %s", exc)
            return
        for detection, vector in zip(missing, vectors, strict=False):
            normalized = normalize(vector)
            if normalized is not None and float(np.linalg.norm(normalized)) > 0:
                detection.embedding = normalized

    def _try_rebind(
        self, detection: Detection, timestamp: float
    ) -> tuple[Track | None, float, float, str]:
        best: tuple[float, float, str, Track] | None = None
        for track in self._inactive.values():
            if track.label != detection.label:
                continue
            age = timestamp - track.last_seen
            if age > self.config.spatial_reid_window:
                continue
            distance = self._center_distance(track.bbox, detection.bbox)
            if distance > self.config.spatial_reid_distance:
                continue
            appearance = self._appearance(track, detection)
            proximity = max(0.0, 1.0 - distance / max(1.0, self.config.spatial_reid_distance))
            age_score = max(0.0, 1.0 - age / max(1e-3, self.config.spatial_reid_window))
            score = 0.55 * appearance + 0.30 * proximity + 0.15 * age_score
            source = "appearance" if appearance >= self.config.reid_threshold else "spatial"
            if appearance < self.config.reid_threshold and proximity < 0.55:
                continue
            if score < 0.30:
                continue
            if best is None or score > best[0]:
                best = (score, appearance, source, track)
        if best is None:
            return None, 0.0, 0.0, "spatial"
        score, appearance, source, track = best
        return track, score, appearance, source

    def _new_track(self, detection: Detection, timestamp: float, frame_index: int) -> Track:
        track = Track(
            track_id=self._next_id,
            label=detection.label,
            bbox=detection.bbox,
            confidence=detection.confidence,
            embedding=normalize(detection.embedding),
            first_seen=timestamp,
            last_seen=timestamp,
            last_frame=frame_index,
            metadata={"detector": detection.source, "entity_id": f"track_{self._next_id}"},
        )
        self._next_id += 1
        return track

    def _refresh_embeddings(self, frame: np.ndarray) -> None:
        cfg = self.config
        if not self._active or cfg.refresh_interval <= 0:
            return
        tracks = list(self._active.values())
        try:
            vectors = self.embedder.embed_crops(frame, [track.bbox for track in tracks])  # type: ignore[union-attr]
        except Exception:
            return
        for track, vector in zip(tracks, vectors, strict=False):
            normalized = normalize(vector)
            if normalized is None or float(np.linalg.norm(normalized)) <= 0:
                continue
            track.embedding = (
                normalized
                if track.embedding is None
                else normalize(0.75 * track.embedding + 0.25 * normalized)
            )

    # ------------------------------------------------------------------
    # Public update
    # ------------------------------------------------------------------
    def update(
        self,
        frame: np.ndarray | None,
        detections: list[Detection],
        timestamp: float,
        frame_index: int = 0,
    ) -> TrackerResult:
        with self._lock:
            detections = [
                detection
                for detection in detections
                if detection.confidence >= self.config.low_confidence
            ]
            high = [
                detection
                for detection in detections
                if detection.confidence >= self.config.high_confidence
            ]
            low = [detection for detection in detections if detection.confidence < self.config.high_confidence]

            for track_id, track in list(self._inactive.items()):
                if timestamp - track.last_seen > self.config.inactive_ttl:
                    del self._inactive[track_id]

            active_tracks = list(self._active.values())
            updated_ids: set[int] = set()

            # Stage 1: high-confidence detections.
            if frame is not None and self.embedder is not None:
                self._fill_embeddings(frame, high)
            pairs, remaining_tracks, remaining_high = self._assign(active_tracks, high)
            for ti, di in pairs:
                track = active_tracks[ti]
                track.update(high[di], timestamp, frame_index, smooth=self.config.smooth)
                updated_ids.add(track.track_id)

            # Stage 2: low-confidence detections extend existing tracks only.
            low_pairs: list[tuple[int, int]] = []
            if remaining_tracks and low:
                low_scores = np.full((len(remaining_tracks), len(low)), -1e6, dtype=np.float32)
                for ti, track in enumerate(remaining_tracks):
                    for di, detection in enumerate(low):
                        if track.label != detection.label:
                            continue
                        iou = track.bbox.iou(detection.bbox)
                        if iou > self.config.iou_low_gate:
                            low_scores[ti, di] = iou
                if linear_sum_assignment is not None:
                    rows, cols = linear_sum_assignment(-low_scores)
                    low_pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=False) if low_scores[r, c] > -1e5]
                else:  # pragma: no cover
                    for ti, di in zip(*np.where(low_scores > -1e5), strict=False):
                        low_pairs.append((int(ti), int(di)))
                matched_tracks = {ti for ti, _ in low_pairs}
                for ti, di in low_pairs:
                    track = remaining_tracks[ti]
                    track.update(low[di], timestamp, frame_index, smooth=self.config.smooth)
                    updated_ids.add(track.track_id)
                remaining_tracks = [
                    track for index, track in enumerate(remaining_tracks) if index not in matched_tracks
                ]

            # Stage 3: appearance / spatial Re-ID for unmatched detections.
            if frame is not None:
                self._fill_embeddings(frame, remaining_high)
            rebound: list[Track] = []
            used_detections: set[int] = set()
            for di, detection in enumerate(remaining_high):
                track, score, appearance, source = self._try_rebind(detection, timestamp)
                if track is None:
                    continue
                self._inactive.pop(track.track_id, None)
                track.update(detection, timestamp, frame_index, reset_velocity=True)
                sim_used = appearance if source == "appearance" else score
                track.metadata["reid_score"] = round(sim_used, 4)
                track.metadata["reid_similarity"] = round(appearance, 4)
                track.metadata["reid_source"] = source
                self._active[track.track_id] = track
                rebound.append(track)
                updated_ids.add(track.track_id)
                used_detections.add(di)
                self.reid_events.append(
                    {
                        "track_id": track.track_id,
                        "label": track.label,
                        "score": round(score, 4),
                        "similarity": round(appearance, 4),
                        "source": source,
                        "timestamp": timestamp,
                    }
                )

            # Stage 4: new tracks.
            for di, detection in enumerate(remaining_high):
                if di in used_detections:
                    continue
                track = self._new_track(detection, timestamp, frame_index)
                self._active[track.track_id] = track
                updated_ids.add(track.track_id)

            # Unmatched active tracks become inactive (only confirmed tracks
            # are kept; one-frame false positives are dropped).
            lost: list[Track] = []
            for track in active_tracks:
                if track.track_id in updated_ids:
                    continue
                track.mark_lost()
                self._active.pop(track.track_id, None)
                if track.hits >= self.config.min_hits:
                    self._inactive[track.track_id] = track
                    lost.append(track)

            if (
                self.embedder is not None
                and frame is not None
                and self.config.refresh_interval > 0
                and frame_index % self.config.refresh_interval == 0
            ):
                self._refresh_embeddings(frame)

            confirmed = [
                track for track in self._active.values() if track.hits >= self.config.min_hits
            ]
            new_tracks: list[Track] = []
            for track in confirmed:
                if not track.announced:
                    track.announced = True
                    new_tracks.append(track)

            return TrackerResult(
                tracks=[track.as_object() for track in confirmed],
                new_tracks=[track.as_object() for track in new_tracks],
                lost=[track.as_object() for track in lost],
                rebound=[track.as_object() for track in rebound],
            )

    def inactive_tracks(self) -> list[Track]:
        with self._lock:
            return list(self._inactive.values())

    def all_tracks(self) -> list[Track]:
        with self._lock:
            return list(self._active.values()) + list(self._inactive.values())

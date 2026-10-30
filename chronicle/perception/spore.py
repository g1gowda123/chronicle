"""SPORE: label-free perception with query-time semantic binding.

SPORE ("sparse perception, on-demand recognition engine") inverts the usual
video-understanding pipeline.  Instead of running an open-vocabulary detector on
every frame and hoping its fixed prompt list covers the world, ingest spends its
budget on *evidence*: hand contact, motion-compensated visual change, and
salient patches.  Those observations become compact :mod:`capsules
<chronicle.perception.capsules>` with crops, embeddings and timestamps, but no
semantic labels.

A query later supplies the vocabulary.  The query engine routes the phrase to a
small set of capsules, runs YOLO-World only on their best views, verifies each
candidate with CLIP, optionally reads a support surface ("on the table") from
the same detector call, and abstains when evidence is weak.

The research question is not "can a detector name this frame?" but "how little
perception is enough to answer questions later?"  Everything here is measured:
frames decoded, keyframes, patches embedded, detector calls, and wall time.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from chronicle.perception.capsules import (
    ActionCapsule,
    CapsuleStore,
    ContextFrame,
    ObjectCapsule,
    SceneCapsule,
    SporeView,
    _cosine,
)
from chronicle.perception.config import PerceptionConfig
from chronicle.perception.detector import BaseDetector, build_detector
from chronicle.perception.embedder import EmbeddingService, build_embedder
from chronicle.perception.hands import BaseHands, build_hands
from chronicle.perception.motion import AffineMotionEstimator
from chronicle.perception.sources import VideoSource
from chronicle.types import BBox, Detection, HandObservation, MotionEstimate

LOGGER = logging.getLogger(__name__)

# Words that carry no object identity in an egocentric question.  Keeping this
# list small is deliberate: query terms are supposed to come from the user, not
# from a closed vocabulary.
_STOPWORDS = frozenset(
    """
    a an the this that these those there here is are was were be been being am
    do does did doing done have has had having i you he she it we they me him
    her us them my your his its our their what which who whom whose when where
    why how many much more most some any all both each few other another such
    no nor not only own same so than too very can will just don should now
    of in on at to from by with about into over after before between under
    above below up down out off again further then once and or but if while
    person people user someone somebody thing things object objects
    """.split()
)

_VERBS = frozenset(
    """
    pick picked picking put putting place placed placing move moved moving
    take took taken taking hold holding grab grabbed grabbing use used using
    open opened opening close closed closing press pressed pressing turn turned
    turning insert inserted inserting remove removed removing cut cutting fold
    folding unfold write wrote writing read reading switch switched switching
    lift lifted lifting drop dropped dropping hand hands arm attach attached
    attaching replace replaced replacing rotate rotated rotating slide slid
    sliding handle handled handling
    """.split()
)

_SUPPORT_LABELS = (
    "table",
    "work table",
    "countertop",
    "desk",
    "shelf",
    "floor",
    "tray",
    "box",
    "basket",
)

_SUPPORT_ALIASES = {
    "desk": "work table",
    "work table": "desk",
    "countertop": "counter top",
    "table": "countertop",
}

# OpenCV hue is 0..179.  Ranges are deliberately coarse: the point is a
# verifiable attribute answer from retained pixels, not a colour taxonomy.
_COLOUR_RANGES = (
    ("red", 0, 8),
    ("red", 170, 180),
    ("orange", 8, 20),
    ("yellow", 20, 35),
    ("green", 35, 85),
    ("cyan", 85, 102),
    ("blue", 102, 130),
    ("purple", 130, 158),
    ("magenta", 158, 170),
)


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SporeConfig:
    """Knobs for the perceiver and the query engine.

    Defaults are tuned for a 640px-wide egocentric stream on a laptop GPU.  The
    ablation switches (``use_hands``/``use_motion``/``use_saliency``,
    ``change_gate``, ``late_binding``, ``propagate_labels``) exist so each claim
    can be measured rather than asserted.
    """

    # Sampling and gating.
    sample_fps: float = 4.0
    max_width: int = 640
    change_gate: bool = True
    change_threshold: float = 0.035
    keyframe_max_gap: float = 1.2
    keyframe_min_gap: float = 0.20
    hand_keyframe: bool = True
    hand_min_area: float = 0.004
    pan_end_keyframe: bool = True

    # Proposal sources.
    use_hands: bool = True
    hand_pad: float = 0.9
    hand_pad_scales: tuple[float, ...] = (1.0, 2.0, 3.5)
    hand_lattice_radius: float = 1.5
    hand_lattice_scale: float = 2.2
    hand_lattice_angles: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
    hand_slots: int = 5
    use_motion: bool = True
    motion_patch_percentile: float = 92.0
    motion_components: int = 3
    use_saliency: bool = True
    use_regions: bool = True
    use_grid: bool = True
    grid_cell: int = 48
    grid_min_saliency: float = 0.08
    region_percentile: float = 95.0
    region_kernel: int = 11
    region_max: int = 8
    saliency_peaks: int = 9
    proposal_scales: tuple[float, ...] = (0.05, 0.08, 0.12, 0.18, 0.28)
    proposal_aspects: tuple[float, ...] = (1.0, 1.5)
    candidate_limit: int = 260
    proposal_nms_iou: float = 0.38
    patches_per_keyframe: int = 8
    min_patch_px: int = 28

    # Cheap proposal ranking (no CLIP yet).
    w_detail: float = 0.30
    w_hand: float = 0.28
    w_motion: float = 0.22
    w_saturation: float = 0.10
    w_center: float = 0.16
    w_size: float = 0.06
    # Egocentric objects sit below the horizon; a small downward bias keeps
    # posters, windows and ceiling lights from eating the patch budget.
    center_prior_y: float = 0.62

    # Capsule identity.
    match_threshold: float = 0.90
    match_spatial_gate: float = 0.12
    match_max_gap: float = 2.5
    max_views: int = 5
    max_capsules: int = 300
    compact_interval: float = 1.0

    # Actions and scenes.
    action_min_duration: float = 0.15
    action_close_grace: float = 0.40
    action_merge_gap: float = 0.40
    action_scan: bool = True
    action_scan_limit: int = 12
    action_scan_min_displacement: float = 15.0
    action_scan_min_duration: float = 0.20
    temporal_scan_limit: int = 16
    scene_interval: float = 4.0
    scene_min_change: float = 0.08
    scene_max_width: int = 480
    scene_embedding: bool = True

    # Deferred wide-field evidence: a compressed keyframe lets the query-time
    # detector see context that a tight capsule crop may have cut away.
    store_context: bool = True
    context_quality: int = 82
    context_max_frames: int = 400
    context_max_gap: float = 1.5
    context_binding: bool = True
    context_fusion: bool = True
    context_fusion_iou: float = 0.20
    context_fusion_center: float = 0.06
    context_fusion_max_tracks: int = 3
    context_sweep: bool = True
    context_sweep_frames: int = 5
    context_sweep_window: float = 2.0
    context_hand_only: bool = False
    context_min_gap: float = 0.75
    context_adaptive_gap: bool = False
    context_active_gap: float = 0.25
    context_idle_gap: float = 1.50

    # Query-time binding.
    query_capsules: int = 8
    query_views: int = 2
    query_detector_confidence: float = 0.22
    query_clip_threshold: float = 0.18
    query_strong_confidence: float = 0.72
    query_abstain_confidence: float = 0.24
    query_diversity: float = 0.28
    query_min_relevance: float = 0.10
    count_contexts: int = 32
    count_min_confidence: float = 0.30
    count_single_confidence: float = 0.50
    count_peak_confidence: float = 0.45
    count_track_gap: float = 20.0
    count_cluster_distance: float = 0.12
    count_cluster_iou: float = 0.15
    colour_min_saturation: int = 45
    colour_min_value: int = 35
    colour_min_chromatic_fraction: float = 0.10
    support_labels: tuple[str, ...] = _SUPPORT_LABELS
    infer_support: bool = True
    rescan: bool = False
    rescan_window: float = 1.0
    rescan_fps: float = 12.0
    rescan_max_frames: int = 16
    detect_cache: bool = True
    detect_cache_entries: int = 256
    late_binding: bool = True
    binding_mode: str = "detector"  # detector | clip | hybrid
    clip_prompt_templates: tuple[str, ...] = (
        "{term}",
        "a photo of a {term}",
        "a close-up photo of a {term}",
    )
    clip_negative_prompts: tuple[str, ...] = (
        "a person's hand",
        "a wooden table",
        "a blank wall",
        "a blurry image",
        "the background",
    )
    clip_negative_weight: float = 0.35
    clip_margin_center: float = 0.16
    clip_margin_slope: float = 11.0
    clip_max_view_area: float = 0.30
    propagate_labels: bool = False
    propagate_similarity: float = 0.93
    propagate_max_gap: float = 20.0
    propagate_confidence: float = 0.72


@dataclass(slots=True)
class SporeStats:
    frames_decoded: int = 0
    frames_sampled: int = 0
    keyframes: int = 0
    patches_considered: int = 0
    patches_embedded: int = 0
    capsules_created: int = 0
    views_retained: int = 0
    scene_embeddings: int = 0
    embed_calls: int = 0
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_decoded": self.frames_decoded,
            "frames_sampled": self.frames_sampled,
            "keyframes": self.keyframes,
            "patches_considered": self.patches_considered,
            "patches_embedded": self.patches_embedded,
            "capsules_created": self.capsules_created,
            "views_retained": self.views_retained,
            "scene_embeddings": self.scene_embeddings,
            "embed_calls": self.embed_calls,
            "elapsed_s": round(float(self.elapsed_s), 3),
        }


@dataclass(slots=True)
class Binding:
    """One query-time label attached to a capsule when a query needs it."""

    term: str
    label: str
    capsule_id: str
    confidence: float
    timestamp: float
    bbox: BBox
    support: str | None = None
    detector_confidence: float = 0.0
    clip_similarity: float = 0.0
    views_agree: int = 0
    views_tried: int = 0
    provenance: str = "detector"

    def as_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "label": self.label,
            "capsule_id": self.capsule_id,
            "confidence": round(float(self.confidence), 4),
            "timestamp": round(float(self.timestamp), 3),
            "bbox": [round(float(v), 1) for v in self.bbox.xyxy],
            "support": self.support,
            "detector_confidence": round(float(self.detector_confidence), 4),
            "clip_similarity": round(float(self.clip_similarity), 4),
            "views_agree": int(self.views_agree),
            "views_tried": int(self.views_tried),
            "provenance": self.provenance,
        }


@dataclass(slots=True)
class _DetectionTrack:
    """A same-class detection followed across retained keyframes."""

    boxes: list[BBox] = field(default_factory=list)
    stamps: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    supports: list[str | None] = field(default_factory=list)

    def add(
        self,
        detection: Detection,
        timestamp: float,
        support: str | None,
    ) -> None:
        self.boxes.append(detection.bbox)
        self.stamps.append(float(timestamp))
        self.confidences.append(float(detection.confidence))
        self.detections.append(detection)
        self.supports.append(support)

    @property
    def last_box(self) -> BBox:
        return self.boxes[-1]

    @property
    def mean_confidence(self) -> float:
        return sum(self.confidences) / max(1, len(self.confidences))

    def mean_box(self) -> BBox:
        total = sum(self.confidences) or 1.0
        return BBox(
            sum(box.x1 * conf for box, conf in zip(self.boxes, self.confidences, strict=False)) / total,
            sum(box.y1 * conf for box, conf in zip(self.boxes, self.confidences, strict=False)) / total,
            sum(box.x2 * conf for box, conf in zip(self.boxes, self.confidences, strict=False)) / total,
            sum(box.y2 * conf for box, conf in zip(self.boxes, self.confidences, strict=False)) / total,
        )

    def best_support(self) -> str | None:
        if not self.detections:
            return None
        index = max(range(len(self.detections)), key=lambda item: self.confidences[item])
        return self.supports[index]

    def temporal_score(self, at_time: float) -> float:
        weights = [math.exp(-abs(stamp - at_time) / 1.5) for stamp in self.stamps]
        total = max(1e-6, sum(weights))
        log_sum = sum(
            weight * math.log(max(1e-6, conf))
            for conf, weight in zip(self.confidences, weights, strict=False)
        )
        return math.exp(log_sum / total)


@dataclass(slots=True)
class SporeAnswer:
    question: str
    terms: list[str]
    bindings: list[Binding] = field(default_factory=list)
    answer: str = ""
    abstained: bool = True
    count: int | None = None
    detector_calls: int = 0
    capsules_considered: int = 0
    evidence_bytes: int = 0
    elapsed_s: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Binding | None:
        return self.bindings[0] if self.bindings else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "terms": list(self.terms),
            "answer": self.answer,
            "abstained": self.abstained,
            "count": self.count,
            "bindings": [binding.as_dict() for binding in self.bindings],
            "detector_calls": self.detector_calls,
            "capsules_considered": self.capsules_considered,
            "evidence_bytes": self.evidence_bytes,
            "elapsed_s": round(float(self.elapsed_s), 3),
            "stats": self.stats,
        }


@dataclass(slots=True)
class _Candidate:
    bbox: BBox
    source: str
    score: float
    detail: float
    hand_overlap: float
    motion: float
    saturation: float
    novelty: float = 0.0
    descriptor: np.ndarray | None = None


@dataclass(slots=True)
class _OpenAction:
    capsule_id: str
    hand_id: str
    start: float
    start_center: tuple[float, float]
    last_center: tuple[float, float]
    last_contact: float
    path_length: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize_max_width(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if max_width <= 0 or width <= max_width:
        return frame, 1.0
    scale = max_width / float(width)
    resized = cv2.resize(
        frame,
        (max_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _center_prior(width: int, height: int, cy_ratio: float = 0.62) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, cy_ratio * height
    diagonal = math.hypot(cx, cy) + 1e-6
    distance = np.hypot(xs - cx, ys - cy) / diagonal
    return np.clip(1.0 - distance, 0.0, 1.0)


def _warp_to_current(
    previous_gray: np.ndarray,
    current_shape: tuple[int, int],
    motion: MotionEstimate,
) -> np.ndarray | None:
    """Warp the previous gray frame into the current camera pose."""
    if not motion.valid or previous_gray.shape[:2] != current_shape:
        return None
    height, width = current_shape
    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), -math.degrees(motion.theta), 1.0
    )
    matrix[0, 2] += float(motion.dx)
    matrix[1, 2] += float(motion.dy)
    return cv2.warpAffine(
        previous_gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _hsv_descriptor(image: np.ndarray, bins: tuple[int, int, int] = (8, 4, 4)) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros(int(np.prod(bins)), dtype=np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        list(bins),
        [0, 180, 0, 256, 0, 256],
    ).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    if norm > 0:
        histogram /= norm
    return histogram.astype(np.float32)



def _hand_priority(candidates: Sequence[_Candidate], limit: int) -> list[_Candidate]:
    """Choose hand crops that cover object directions, not just high scores.

    The palm box is small and a held object (knife, spatula, bottle) extends
    to one side.  A plain top-k over hand crops collapses onto the same side,
    so reserve one crop per cardinal direction plus the widest context crop.
    """
    hand_candidates = [candidate for candidate in candidates if candidate.source.startswith("hand")]
    priority: list[_Candidate] = []
    pads = [candidate for candidate in hand_candidates if candidate.source.startswith("hand:pad")]
    if pads:
        priority.append(max(pads, key=lambda candidate: candidate.bbox.area))
    for angle in (0, 90, 180, 270):
        options = [
            candidate for candidate in hand_candidates if candidate.source == f"hand:{angle}"
        ]
        if options:
            priority.append(max(options, key=lambda candidate: candidate.score))
    for candidate in sorted(hand_candidates, key=lambda item: item.score, reverse=True):
        if all(candidate is not existing for existing in priority):
            priority.append(candidate)
    return priority[:limit]


def _coverage_top(
    candidates: Sequence[_Candidate],
    limit: int,
    cell: int,
    min_distance: float | None = None,
    hand_slots: int = 5,
) -> list[_Candidate]:
    """Spread the patch budget over the frame.

    One candidate per grid cell is not enough: a strong horizontal edge can
    win several adjacent cells with near-identical boxes.  Enforce a minimum
    centre distance as well so eight patches cover eight *places*, not eight
    scales of the same texture.

    Hand boxes get reserved slots: a held object usually extends well beyond
    the palm box (a knife, a spatula), so the hand context is sampled at
    several scales instead of competing with crisp static regions.
    """
    selected: list[_Candidate] = []
    used: set[tuple[int, int]] = set()
    cell = max(1, cell)
    spacing = 1.8 * cell if min_distance is None else min_distance
    hand_priority = _hand_priority(candidates, hand_slots)
    selected_ids: set[int] = set()
    for candidate in hand_priority:
        if len(selected) >= limit:
            return selected
        if any(candidate.bbox.iou(other.bbox) >= 0.65 for other in selected):
            continue
        cx, cy = candidate.bbox.centroid
        used.add((int(cx) // cell, int(cy) // cell))
        selected.append(candidate)
        selected_ids.add(id(candidate))
    for candidate in candidates:
        if id(candidate) in selected_ids or candidate.source.startswith("hand"):
            continue
        if len(selected) >= limit:
            return selected
        cx, cy = candidate.bbox.centroid
        key = (int(cx) // cell, int(cy) // cell)
        if key in used:
            continue
        if any(
            math.hypot(cx - other.bbox.centroid[0], cy - other.bbox.centroid[1]) < spacing
            for other in selected
        ):
            continue
        # A repeated texture (brick line, window frame, tiled floor) produces
        # many high-detail boxes with identical appearance.  Embedding each
        # one wastes the budget on a single capsule.
        if candidate.descriptor is not None and any(
            other.descriptor is not None
            and _descriptor_similarity(candidate.descriptor, other.descriptor) >= 0.985
            for other in selected
        ):
            continue
        used.add(key)
        selected.append(candidate)
        if len(selected) >= limit:
            return selected
    if len(selected) < limit:
        for candidate in candidates:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break
    return selected


def _nms(candidates: list[_Candidate], threshold: float) -> list[_Candidate]:
    if not candidates:
        return []
    boxes = np.array(
        [
            [candidate.bbox.x1, candidate.bbox.y1, candidate.bbox.x2, candidate.bbox.y2]
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    scores = np.array([candidate.score for candidate in candidates], dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou < threshold]
    return [candidates[index] for index in kept]


def _norm_percentile(values: np.ndarray, percentile: float = 95.0) -> float:
    if values.size == 0:
        return 1.0
    return max(1e-6, float(np.percentile(values, percentile)))


def _mean_box(values: np.ndarray, bbox: BBox) -> float:
    height, width = values.shape[:2]
    x1, y1, x2, y2 = bbox.clipped(width, height).int_xyxy
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(values[y1:y2, x1:x2]))


def _box_hand_overlap(box: BBox, hands: Sequence[HandObservation]) -> float:
    best = 0.0
    for hand in hands:
        best = max(best, box.ios(hand.bbox))
    return best


def _descriptor_similarity(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return 0.0
    return _cosine(first, second)


def _stored_descriptor(capsule: ObjectCapsule) -> np.ndarray | None:
    payload = capsule.metadata.get("descriptor")
    if not payload:
        return None
    try:
        return np.asarray(payload, dtype=np.float32)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Perceiver
# ---------------------------------------------------------------------------


class SporePerceiver:
    """Build a label-free capsule store from a video source.

    Example::

        perceiver = SporePerceiver()
        store = perceiver.ingest("data/real/holoassist_switch_34s.mp4")
        print(perceiver.stats.as_dict())
    """

    def __init__(
        self,
        config: SporeConfig | None = None,
        perception: PerceptionConfig | None = None,
        *,
        embedder: EmbeddingService | None = None,
        hands: BaseHands | None = None,
        motion: AffineMotionEstimator | None = None,
    ) -> None:
        self.config = config or SporeConfig()
        self.perception = perception or PerceptionConfig()
        self._embedder = embedder
        self._hands = hands
        self._motion = motion or AffineMotionEstimator(self.perception.motion)
        self._owns_embedder = embedder is None
        self._owns_hands = hands is None
        self.store = CapsuleStore()
        self.stats = SporeStats()
        self._previous_gray: np.ndarray | None = None
        self._last_keyframe_t = -1e9
        self._last_interaction_t = -1e9
        self._last_scene_t = -1e9
        self._last_compact_t = -1e9
        self._open_actions: dict[str, _OpenAction] = {}
        self._frame_size = (640, 480)

    # -- lifecycle ---------------------------------------------------------

    def _ensure_backends(self, source_name: str) -> None:
        if self._embedder is None:
            self._embedder = build_embedder(self.perception, source_name=source_name)
        if self._hands is None and self.config.use_hands:
            self._hands = build_hands(self.perception, source_name=source_name)

    def close(self) -> None:
        if self._owns_embedder and self._embedder is not None:
            self._embedder.close()
        if self._owns_hands and self._hands is not None:
            self._hands.close()
        self._embedder = None
        self._hands = None

    def __enter__(self) -> SporePerceiver:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- ingest ------------------------------------------------------------

    def ingest(
        self,
        source: Any,
        *,
        max_frames: int | None = None,
        fps_override: float | None = None,
        source_name: str = "",
    ) -> CapsuleStore:
        """Consume ``source`` and return the capsule store.

        ``source`` accepts everything :class:`VideoSource` accepts: a path, a
        camera index, or an iterable of frames.
        """
        start_time = time.perf_counter()
        self._ensure_backends(source_name)
        stream = VideoSource(source, fps_override=fps_override)
        fps = float(stream.fps or 0.0)
        stride = max(1, int(round(fps / max(0.1, self.config.sample_fps)))) if fps else 1
        for frame_index, (frame, timestamp, _) in enumerate(stream):
            self.stats.frames_decoded += 1
            if max_frames is not None and self.stats.frames_decoded > max_frames:
                break
            if frame_index % stride:
                continue
            if os.environ.get("SPORE_PROGRESS") and self.stats.frames_decoded % 200 == 0:
                print(
                    f"[spore] decoded={self.stats.frames_decoded}"
                    f" sampled={self.stats.frames_sampled}"
                    f" keyframes={self.stats.keyframes}"
                    f" capsules={len(self.store.capsules)}"
                    f" t={float(timestamp):.1f}s",
                    flush=True,
                )
            self._consume_frame(frame, float(timestamp), frame_index)
        self._close_all_actions()
        self.store.consolidate_actions(self.config.action_merge_gap)
        # Compaction is a batch cleanup, not a per-frame operation.  Running
        # it inside the ingest loop made long clips quadratic in capsule count.
        self._compact(0.0, force=True)
        self.stats.elapsed_s = time.perf_counter() - start_time
        return self.store

    def _consume_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> None:
        frame, _ = _resize_max_width(frame, self.config.max_width)
        height, width = frame.shape[:2]
        self._frame_size = (width, height)
        self.stats.frames_sampled += 1

        hands: list[HandObservation] = []
        if self._hands is not None and self.config.use_hands:
            try:
                hands = self._hands.detect(frame, frame_index, timestamp)
            except Exception as exc:  # pragma: no cover - backend guard
                LOGGER.debug("hand detection failed: %s", exc)
                hands = []

        hand_boxes = [hand.bbox for hand in hands]
        motion = self._motion.estimate(frame, exclude=hand_boxes, timestamp=timestamp)
        gray = _gray(frame)
        residual, change = self._change_signal(gray, motion)

        keyframe = self._is_keyframe(timestamp, change, hands, motion, frame_index)
        if keyframe:
            self.stats.keyframes += 1
            self._last_keyframe_t = timestamp
            self._extract_frame(frame, gray, residual, motion, hands, timestamp, frame_index)

        self._update_actions(hands, timestamp)
        self._maybe_scene(frame, timestamp, frame_index, change)
        self._previous_gray = gray

    def _grid_boxes(
        self,
        detail: np.ndarray,
        saturation: np.ndarray,
        center: np.ndarray,
        bounds: tuple[int, int],
    ) -> list[BBox]:
        """One tight box per grid cell around its most object-like blob.

        Saliency peaks alone are biased towards the single most textured
        region (a window, a bright label).  A grid pass guarantees that every
        part of the frame is *considered*, which is what makes a handful of
        patches per keyframe cover a cluttered table.
        """
        width, height = bounds
        detail_norm = np.clip(detail / _norm_percentile(detail), 0.0, 2.0) / 2.0
        saliency = (0.65 * detail_norm + 0.35 * saturation) * (0.55 + 0.45 * center)
        cell = max(24, int(self.config.grid_cell))
        boxes: list[BBox] = []
        for y0 in range(0, height, cell):
            for x0 in range(0, width, cell):
                patch = saliency[y0 : min(height, y0 + cell), x0 : min(width, x0 + cell)]
                if patch.size == 0:
                    continue
                peak = float(patch.max())
                if peak < self.config.grid_min_saliency:
                    continue
                mask = patch >= peak * 0.62
                ys, xs = np.nonzero(mask)
                if len(xs) < 12:
                    continue
                bx1, bx2 = int(xs.min()) + x0, int(xs.max()) + x0 + 1
                by1, by2 = int(ys.min()) + y0, int(ys.max()) + y0 + 1
                pad = 0.10 * max(bx2 - bx1, by2 - by1) + 2
                box = BBox(bx1 - pad, by1 - pad, bx2 + pad, by2 + pad).clipped(width, height)
                if box.width < self.config.min_patch_px * 0.6 or box.height < self.config.min_patch_px * 0.6:
                    continue
                if box.area > 0.09 * width * height:
                    continue
                boxes.append(box)
        return boxes

    def _region_boxes(self, detail: np.ndarray, bounds: tuple[int, int]) -> list[BBox]:
        """Tight boxes around compact high-detail blobs.

        Saliency peaks find *where* to look but their scale is arbitrary.  A
        high-percentile edge map, closed into blobs, gives one box per compact
        object-like region and covers static objects that never move.
        """
        width, height = bounds
        if detail.size == 0:
            return []
        threshold = float(np.percentile(detail, self.config.region_percentile))
        mask = (detail >= threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.region_kernel, self.config.region_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        boxes: list[BBox] = []
        frame_area = float(width * height)
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if area < max(60, 0.0006 * frame_area):
                continue
            if area > 0.10 * frame_area or w < 10 or h < 10:
                continue
            aspect = w / max(1.0, h)
            if aspect < 0.22 or aspect > 4.5:
                continue
            pad = 0.16 * max(w, h)
            boxes.append(
                BBox.from_xywh(x - pad, y - pad, w + 2 * pad, h + 2 * pad).clipped(width, height)
            )
        boxes.sort(key=lambda box: box.area, reverse=True)
        return boxes[: self.config.region_max]

    def _change_signal(
        self, gray: np.ndarray, motion: MotionEstimate
    ) -> tuple[np.ndarray | None, float]:
        if self._previous_gray is None:
            return None, 0.0
        warped = _warp_to_current(self._previous_gray, gray.shape[:2], motion)
        reference = warped if warped is not None else self._previous_gray
        if reference.shape[:2] != gray.shape[:2]:
            reference = cv2.resize(reference, (gray.shape[1], gray.shape[0]))
        difference = cv2.absdiff(gray, reference)
        # Erode one pixel to suppress resampling sparkle along edges.
        difference = cv2.erode(difference, np.ones((3, 3), np.uint8))
        change = float(np.mean(difference)) / 255.0
        return difference, change

    def _is_keyframe(
        self,
        timestamp: float,
        change: float,
        hands: Sequence[HandObservation],
        motion: MotionEstimate,
        frame_index: int,
    ) -> bool:
        if frame_index == 0 or self._previous_gray is None:
            return True
        if timestamp - self._last_keyframe_t < self.config.keyframe_min_gap:
            return False
        if not self.config.change_gate:
            return True
        if change >= self.config.change_threshold:
            return True
        if timestamp - self._last_keyframe_t >= self.config.keyframe_max_gap:
            return True
        if (
            self.config.pan_end_keyframe
            and motion.transition == "pan_end"
            and timestamp - self._last_keyframe_t >= self.config.keyframe_min_gap
        ):
            return True
        if self.config.hand_keyframe and hands:
            height, width = self._frame_size[1], self._frame_size[0]
            area = max(hand.bbox.area for hand in hands) / max(1.0, width * height)
            if (
                area >= self.config.hand_min_area
                and timestamp - self._last_interaction_t >= self.config.keyframe_min_gap
            ):
                self._last_interaction_t = timestamp
                return True
        return False

    # -- proposals ---------------------------------------------------------

    def _extract_frame(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        residual: np.ndarray | None,
        motion: MotionEstimate,
        hands: Sequence[HandObservation],
        timestamp: float,
        frame_index: int,
    ) -> None:
        if self.config.store_context and (hands or not self.config.context_hand_only):
            last_context = (
                self.store.contexts[-1].timestamp if self.store.contexts else -1e9
            )
            if self.config.context_adaptive_gap:
                min_gap = (
                    self.config.context_active_gap
                    if self._open_actions
                    else self.config.context_idle_gap
                )
            else:
                min_gap = self.config.context_min_gap
            if timestamp - last_context >= min_gap:
                ok, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.context_quality)],
                )
                if ok:
                    height, width = frame.shape[:2]
                    self.store.add_context(
                        ContextFrame(
                            timestamp=timestamp,
                            frame_index=frame_index,
                            image_jpeg=buffer.tobytes(),
                            metadata={
                                "frame_size": [width, height],
                                "hands": len(hands),
                                "hand_boxes": [
                                    [
                                        int(hand.bbox.x1),
                                        int(hand.bbox.y1),
                                        int(hand.bbox.x2),
                                        int(hand.bbox.y2),
                                    ]
                                    for hand in hands
                                ],
                                "hand_points": [
                                    [
                                        [int(point[0]), int(point[1])]
                                        for point in hand.fingertips()
                                    ]
                                    for hand in hands
                                ],
                            },
                        ),
                        max_frames=self.config.context_max_frames,
                    )
        candidates = self._proposals(frame, gray, residual, hands)
        self.stats.patches_considered += len(candidates)
        if not candidates:
            return
        selected = _coverage_top(
            candidates,
            self.config.patches_per_keyframe,
            self.config.grid_cell,
            hand_slots=self.config.hand_slots,
        )
        boxes = [candidate.bbox for candidate in selected]
        assert self._embedder is not None
        embeddings = self._embedder.embed_crops(frame, boxes)
        self.stats.embed_calls += 1
        self.stats.patches_embedded += len(boxes)
        height, width = frame.shape[:2]
        diagonal = math.hypot(width, height)

        for candidate, embedding in zip(selected, embeddings, strict=False):
            if embedding is None or float(np.linalg.norm(embedding)) <= 1e-9:
                continue
            view = SporeView(
                timestamp=timestamp,
                frame_index=frame_index,
                bbox=candidate.bbox,
                score=candidate.score,
                embedding=np.asarray(embedding, dtype=np.float32),
                image=self._crop(frame, candidate.bbox),
                source=candidate.source,
                sharpness=candidate.detail,
                hand_overlap=candidate.hand_overlap,
                motion=candidate.motion,
                novelty=candidate.novelty,
                metadata={"frame_size": [width, height], "change": round(float(candidate.motion), 4)},
            )
            capsule = self.store.match(
                view.embedding,
                timestamp,
                view.bbox,
                threshold=self.config.match_threshold,
                spatial_gate=self.config.match_spatial_gate,
                max_gap=self.config.match_max_gap,
                frame_diagonal=diagonal,
            )
            if capsule is None:
                capsule = self.store.add_capsule(
                    timestamp, view, max_views=self.config.max_views
                )
                self.stats.capsules_created += 1
            else:
                retained = capsule.add_view(view)
                if retained:
                    self.stats.views_retained += 1
            self._update_descriptor(capsule, candidate.descriptor)

    def _proposals(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        residual: np.ndarray | None,
        hands: Sequence[HandObservation],
    ) -> list[_Candidate]:
        height, width = gray.shape[:2]
        bounds = (width, height)
        detail = self._detail_map(gray)
        saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
        detail_scale = _norm_percentile(detail)
        center = _center_prior(width, height, self.config.center_prior_y)
        boxes: list[tuple[BBox, str]] = []

        if self.config.use_hands:
            for hand in hands:
                extent = max(hand.bbox.width, hand.bbox.height)
                for scale in self.config.hand_pad_scales:
                    pad = self.config.hand_pad * scale * extent
                    boxes.append((hand.bbox.expanded(pad, bounds=bounds), f"hand:pad{scale:g}"))
                # A held object usually extends to one side of the palm (a
                # knife, a spatula, a bottle).  Sample a small polar lattice
                # around the hand so the object gets its own tight-ish crop
                # instead of hiding inside one enormous hand box.
                cx, cy = hand.bbox.centroid
                side = self.config.hand_lattice_scale * extent
                for angle in self.config.hand_lattice_angles:
                    radians = math.radians(angle)
                    offset = self.config.hand_lattice_radius * extent
                    shifted = BBox.from_xywh(
                        cx + offset * math.cos(radians) - side / 2.0,
                        cy + offset * math.sin(radians) - side / 2.0,
                        side,
                        side,
                    ).clipped(width, height)
                    boxes.append((shifted, f"hand:{int(angle)}"))

        if self.config.use_motion and residual is not None:
            for box in self._motion_boxes(residual, bounds):
                boxes.append((box, "motion"))

        if self.config.use_saliency:
            for box in self._saliency_boxes(gray, detail, saturation, center, bounds):
                boxes.append((box, "saliency"))

        if self.config.use_regions:
            for box in self._region_boxes(detail, bounds):
                boxes.append((box, "region"))

        if self.config.use_grid:
            for box in self._grid_boxes(detail, saturation, center, bounds):
                boxes.append((box, "grid"))

        candidates: list[_Candidate] = []
        for box, source in boxes:
            box = box.clipped(width, height)
            if box.width < self.config.min_patch_px or box.height < self.config.min_patch_px:
                continue
            detail_value = _mean_box(detail, box) / detail_scale
            saturation_value = _mean_box(saturation, box)
            motion_value = (
                _mean_box(residual, box) / 255.0 if residual is not None else 0.0
            )
            hand_value = _box_hand_overlap(box, hands)
            cx, cy = box.centroid
            center_value = max(
                0.0,
                1.0
                - math.hypot(
                    cx - width / 2.0,
                    cy - self.config.center_prior_y * height,
                )
                / (math.hypot(width, height) / 2.0),
            )
            size_value = min(1.0, box.area / max(1.0, 0.12 * width * height))
            score = (
                self.config.w_detail * min(2.0, detail_value)
                + self.config.w_hand * hand_value
                + self.config.w_motion * min(3.0, motion_value * 12.0)
                + self.config.w_saturation * saturation_value
                + self.config.w_center * center_value
                + self.config.w_size * size_value
            )
            candidate = _Candidate(
                bbox=box,
                source=source,
                score=float(score),
                detail=float(detail_value),
                hand_overlap=float(hand_value),
                motion=float(motion_value),
                saturation=float(saturation_value),
                descriptor=_hsv_descriptor(self._crop(frame, box)),
            )
            candidates.append(candidate)

        candidates = _nms(candidates, self.config.proposal_nms_iou)[: self.config.candidate_limit]
        self._annotate_novelty(candidates)
        # Novelty is a tie-breaker, not a reason to ignore contact.
        candidates.sort(key=lambda item: item.score + 0.10 * item.novelty, reverse=True)
        return candidates

    def _motion_boxes(self, residual: np.ndarray, bounds: tuple[int, int]) -> list[BBox]:
        width, height = bounds
        threshold = max(14.0, float(np.percentile(residual, self.config.motion_patch_percentile)))
        mask = (residual >= threshold).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ranked = sorted(contours, key=cv2.contourArea, reverse=True)
        boxes: list[BBox] = []
        for contour in ranked[: self.config.motion_components]:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < self.config.min_patch_px**2:
                continue
            pad = 0.18 * max(w, h)
            boxes.append(
                BBox.from_xywh(x - pad, y - pad, w + 2 * pad, h + 2 * pad).clipped(width, height)
            )
        return boxes

    def _saliency_boxes(
        self,
        gray: np.ndarray,
        detail: np.ndarray,
        saturation: np.ndarray,
        center: np.ndarray,
        bounds: tuple[int, int],
    ) -> list[BBox]:
        width, height = bounds
        detail_norm = np.clip(detail / _norm_percentile(detail), 0.0, 2.0) / 2.0
        saliency = (0.65 * detail_norm + 0.35 * saturation) * (0.55 + 0.45 * center)
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (5, 5), 0)
        peak_radius = max(6, int(round(min(width, height) * 0.07)))
        work = saliency.copy()
        boxes: list[BBox] = []
        for _ in range(self.config.saliency_peaks):
            _, value, _, location = cv2.minMaxLoc(work)
            if value <= 1e-6:
                break
            cx, cy = location
            for scale in self.config.proposal_scales:
                side = scale * min(width, height)
                for aspect in self.config.proposal_aspects:
                    box_width = side * aspect
                    box_height = side / max(1e-6, math.sqrt(aspect))
                    boxes.append(
                        BBox.from_xywh(
                            cx - box_width / 2.0,
                            cy - box_height / 2.0,
                            box_width,
                            box_height,
                        ).clipped(width, height)
                    )
            cv2.circle(work, (cx, cy), peak_radius, 0.0, -1)
        return boxes

    def _annotate_novelty(self, candidates: list[_Candidate]) -> None:
        if not candidates:
            return
        stored = [
            descriptor
            for capsule in self.store.candidates()
            if (descriptor := _stored_descriptor(capsule)) is not None
        ]
        if not stored:
            for candidate in candidates:
                candidate.novelty = 1.0
            return
        valid = [candidate for candidate in candidates if candidate.descriptor is not None]
        if not valid:
            return
        memory = np.stack(stored).astype(np.float32)
        query = np.stack(
            [np.asarray(candidate.descriptor, dtype=np.float32) for candidate in valid]
        )
        memory_norm = np.linalg.norm(memory, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        similarity = (query @ memory.T) / np.maximum(1e-12, query_norm * memory_norm.T)
        best = np.clip(similarity.max(axis=1), 0.0, 1.0)
        for candidate, value in zip(valid, best, strict=True):
            candidate.novelty = float(1.0 - value)

    def _update_descriptor(self, capsule: ObjectCapsule, descriptor: np.ndarray | None) -> None:
        if descriptor is None:
            return
        current = _stored_descriptor(capsule)
        if current is None:
            capsule.metadata["descriptor"] = [round(float(v), 5) for v in descriptor]
            return
        blended = 0.7 * current + 0.3 * descriptor
        norm = float(np.linalg.norm(blended))
        if norm > 0:
            blended /= norm
        capsule.metadata["descriptor"] = [round(float(v), 5) for v in blended]

    @staticmethod
    def _detail_map(gray: np.ndarray) -> np.ndarray:
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        return cv2.GaussianBlur(np.abs(laplacian), (5, 5), 0)

    @staticmethod
    def _crop(frame: np.ndarray, box: BBox) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box.clipped(width, height).int_xyxy
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop.copy() if crop.size else None

    # -- actions -----------------------------------------------------------

    def _find_contact_capsule(
        self, hand: HandObservation, timestamp: float
    ) -> ObjectCapsule | None:
        if not self.store.capsules:
            return None
        best: tuple[float, ObjectCapsule] | None = None
        palm = hand.palm_center
        for capsule in self.store.capsules.values():
            evidence = 0.0
            for view in capsule.best_views:
                if timestamp - view.timestamp > 2.0:
                    continue
                overlap = view.bbox.ios(hand.bbox)
                if overlap > 0.05:
                    evidence = max(evidence, 0.6 + 0.4 * min(1.0, overlap * 3.0))
                elif view.bbox.contains_point(palm, pad=0.15 * max(view.bbox.width, view.bbox.height)):
                    evidence = max(evidence, 0.55)
            if capsule.last_bbox is not None and timestamp - capsule.last_seen <= 1.0:
                overlap = capsule.last_bbox.ios(hand.bbox)
                if overlap > 0.05:
                    evidence = max(evidence, 0.5 + 0.4 * min(1.0, overlap * 3.0))
            if evidence > 0.0 and (best is None or evidence > best[0]):
                best = (evidence, capsule)
        return best[1] if best is not None else None

    def _update_actions(self, hands: Sequence[HandObservation], timestamp: float) -> None:
        touched: set[str] = set()
        for index, hand in enumerate(hands):
            hand_id = f"hand-{index}"
            capsule = self._find_contact_capsule(hand, timestamp)
            if capsule is None:
                continue
            touched.add(hand_id)
            capsule.add_contact(timestamp, merge_gap=0.6)
            center = hand.palm_center
            action = self._open_actions.get(hand_id)
            if action is not None and action.capsule_id == capsule.capsule_id:
                action.path_length += math.hypot(
                    center[0] - action.last_center[0], center[1] - action.last_center[1]
                )
                action.last_center = center
                action.last_contact = timestamp
            else:
                if action is not None:
                    self._close_action(hand_id, timestamp)
                self._open_actions[hand_id] = _OpenAction(
                    capsule_id=capsule.capsule_id,
                    hand_id=hand_id,
                    start=timestamp,
                    start_center=center,
                    last_center=center,
                    last_contact=timestamp,
                )
        for hand_id, action in list(self._open_actions.items()):
            if hand_id in touched:
                continue
            if timestamp - action.last_contact > self.config.action_close_grace:
                self._close_action(hand_id, action.last_contact)

    def _close_action(self, hand_id: str, timestamp: float) -> None:
        action = self._open_actions.pop(hand_id, None)
        if action is None:
            return
        duration = max(0.0, timestamp - action.start)
        if duration < self.config.action_min_duration:
            return
        displacement = (
            action.last_center[0] - action.start_center[0],
            action.last_center[1] - action.start_center[1],
        )
        capsule = ActionCapsule(
            action_id=self.store.new_action_id(),
            start=action.start,
            end=timestamp,
            hand_id=action.hand_id,
            capsule_ids=[action.capsule_id],
            displacement=displacement,
            hand_speed=action.path_length / max(1e-6, duration),
            metadata={"path_length": round(action.path_length, 2)},
        )
        self.store.add_action(capsule)

    def _close_all_actions(self) -> None:
        for hand_id, action in list(self._open_actions.items()):
            self._close_action(hand_id, action.last_contact)

    # -- scenes and compaction --------------------------------------------

    def _maybe_scene(
        self, frame: np.ndarray, timestamp: float, frame_index: int, change: float
    ) -> None:
        if timestamp - self._last_scene_t < self.config.scene_interval and change < self.config.scene_min_change:
            return
        if timestamp - self._last_scene_t < self.config.keyframe_min_gap:
            return
        self._last_scene_t = timestamp
        image, _ = _resize_max_width(frame, self.config.scene_max_width)
        embedding = None
        if self.config.scene_embedding and self._embedder is not None:
            try:
                embedding = self._embedder.embed_scene(image)
                self.stats.scene_embeddings += 1
            except Exception as exc:  # pragma: no cover - backend guard
                LOGGER.debug("scene embedding failed: %s", exc)
        self.store.add_scene(
            SceneCapsule(
                timestamp=timestamp,
                frame_index=frame_index,
                score=change,
                embedding=embedding,
                image=image,
                metadata={"change": round(float(change), 4)},
            )
        )

    def _compact(self, timestamp: float, *, force: bool = False) -> None:
        if len(self.store.capsules) <= self.config.max_capsules:
            return
        if not force and timestamp - self._last_compact_t < self.config.compact_interval:
            return
        self._last_compact_t = timestamp
        capsules = list(self.store.capsules.values())
        best_pair: tuple[float, ObjectCapsule, ObjectCapsule] | None = None
        for index, first in enumerate(capsules):
            for second in capsules[index + 1 :]:
                if first.last_bbox is None or second.last_bbox is None:
                    continue
                # Cheap spatial gate before the 512-D similarity scan.  On
                # long clips most pairs are disjoint, so this turns an
                # O(n^2) embedding comparison into an O(n^2) box test.
                if first.last_bbox.iou(second.last_bbox) < 0.2:
                    continue
                similarity = first.similarity(second.prototype)
                if similarity < self.config.propagate_similarity:
                    continue
                if best_pair is None or similarity > best_pair[0]:
                    best_pair = (similarity, first, second)
        if best_pair is not None:
            _, first, second = best_pair
            for view in second.best_views:
                first.add_view(view)
            for start, end in second.contact_windows:
                first.contact_windows.append((start, end))
            first.contact_windows.sort()
            first.observations += second.observations
            first.last_seen = max(first.last_seen, second.last_seen)
            if first.last_bbox is None:
                first.last_bbox = second.last_bbox
            del self.store.capsules[second.capsule_id]
        # Merging alone cannot bound a long clip: background patches rarely
        # overlap, so the store would stay above the cap.  Evict by retained
        # interaction evidence instead of silently growing without limit.
        self.store.evict_to(self.config.max_capsules)


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------


class SporeQueryEngine:
    """Route questions to capsules and bind labels only where needed."""

    def __init__(
        self,
        store: CapsuleStore,
        config: SporeConfig | None = None,
        perception: PerceptionConfig | None = None,
        *,
        embedder: EmbeddingService | None = None,
        detector: BaseDetector | None = None,
        source: str | None = None,
    ) -> None:
        self.store = store
        self.config = config or SporeConfig()
        self.perception = perception or PerceptionConfig()
        self._embedder = embedder
        self._detector = detector
        self.source = source
        self._owns_embedder = embedder is None
        self._owns_detector = detector is None
        self._text_cache: dict[str, np.ndarray | None] = {}
        self._detection_cache: dict[tuple[Any, tuple[str, ...]], list[Detection]] = {}
        self.detection_cache_hits = 0
        self.detector_calls = 0

    def close(self) -> None:
        if self._owns_embedder and self._embedder is not None:
            self._embedder.close()
        if self._owns_detector and self._detector is not None:
            self._detector.close()

    def __enter__(self) -> SporeQueryEngine:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _ensure_backends(self) -> None:
        if self._embedder is None:
            self._embedder = build_embedder(self.perception)
        if self._detector is None and self.config.late_binding:
            self._detector = build_detector(self.perception)

    def _detect_cached(
        self,
        frame: np.ndarray,
        vocabulary: Sequence[str],
        *,
        frame_index: int,
        timestamp: float,
        cache_tag: Any | None = None,
    ) -> list[Detection]:
        """Detect with a bounded cache over immutable keyframes.

        Repeated questions about the same moment (for example a location, a
        colour and an activity question at one timestamp) otherwise re-run the
        same detector pass.  Cache keys include the exact prompt set because a
        different vocabulary is a different computation.
        """
        assert self._detector is not None
        if not self.config.detect_cache:
            self._detector.set_vocabulary(list(vocabulary))
            self.detector_calls += 1
            return self._detector.detect(
                frame, frame_index=frame_index, timestamp=timestamp
            )
        tag = frame_index if cache_tag is None else cache_tag
        key = (tag, tuple(vocabulary))
        cached = self._detection_cache.get(key)
        if cached is not None:
            self.detection_cache_hits += 1
            self._detection_cache[key] = self._detection_cache.pop(key)
            return cached
        self._detector.set_vocabulary(list(vocabulary))
        detections = self._detector.detect(
            frame, frame_index=frame_index, timestamp=timestamp
        )
        self.detector_calls += 1
        self._detection_cache[key] = detections
        while len(self._detection_cache) > max(1, self.config.detect_cache_entries):
            del self._detection_cache[next(iter(self._detection_cache))]
        return detections

    # -- routing -----------------------------------------------------------

    def _embed_text(self, text: str) -> np.ndarray | None:
        if text in self._text_cache:
            return self._text_cache[text]
        self._ensure_backends()
        assert self._embedder is not None
        vector = self._embedder.embed_text(text)
        self._text_cache[text] = vector
        return vector

    def _capsule_relevance(
        self, capsule: ObjectCapsule, term_vectors: Sequence[np.ndarray | None]
    ) -> float:
        best = 0.0
        vectors = [view.embedding for view in capsule.best_views if view.embedding is not None]
        for term in term_vectors:
            if term is None:
                continue
            for vector in vectors:
                best = max(best, _cosine(term, vector))
        return best

    def route(
        self, terms: Sequence[str], *, budget: int | None = None, at_time: float | None = None
    ) -> list[ObjectCapsule]:
        """Pick capsules worth a detector call, with greedy diversity."""
        budget = budget or self.config.query_capsules
        if not self.store.capsules:
            return []
        term_vectors = [self._embed_text(term) for term in terms]
        end_time = max((capsule.last_seen for capsule in self.store.capsules.values()), default=0.0)
        target_time = at_time if at_time is not None else end_time
        scored: list[tuple[float, ObjectCapsule]] = []
        for capsule in self.store.capsules.values():
            relevance = self._capsule_relevance(capsule, term_vectors)
            prior = 0.0
            if capsule.contact_windows:
                if at_time is None:
                    prior += 0.10
                else:
                    gap = min(
                        max(start - at_time, 0.0, at_time - end)
                        for start, end in capsule.contact_windows
                    )
                    if gap <= 2.0:
                        prior += 0.14 * math.exp(-gap)
            nearest = min(
                (abs(view.timestamp - target_time) for view in capsule.best_views),
                default=float("inf"),
            )
            if math.isfinite(nearest):
                prior += 0.18 * math.exp(-nearest / 2.0)
            prior += 0.05 * math.exp(-max(0.0, end_time - capsule.last_seen) / 60.0)
            scored.append((relevance + prior, capsule))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return []
        threshold = self.config.query_min_relevance
        accepted = [item for item in scored if item[0] >= threshold]
        if len(accepted) < min(3, len(scored)):
            accepted = scored[: min(3, len(scored))]

        selected: list[ObjectCapsule] = []
        remaining = list(accepted)
        while remaining and len(selected) < budget:
            best: tuple[float, ObjectCapsule] | None = None
            for score, capsule in remaining:
                diversity = 0.0
                if selected:
                    diversity = 1.0 - max(
                        max(0.0, capsule.similarity(other.prototype)) for other in selected
                    )
                utility = score + self.config.query_diversity * diversity
                if best is None or utility > best[0]:
                    best = (utility, capsule)
            if best is None:
                break
            selected.append(best[1])
            remaining = [item for item in remaining if item[1].capsule_id != best[1].capsule_id]
        return selected

    # -- binding -----------------------------------------------------------

    def _select_views(
        self,
        capsule: ObjectCapsule,
        term_vector: np.ndarray | None,
        at_time: float | None = None,
    ) -> list[SporeView]:
        views = [view for view in capsule.best_views if view.image is not None]
        if term_vector is not None:
            views.sort(
                key=lambda view: view.score
                + 0.35 * _cosine(term_vector, view.embedding)
                + (
                    0.5 * math.exp(-abs(view.timestamp - at_time) / 1.5)
                    if at_time is not None
                    else 0.0
                ),
                reverse=True,
            )
        else:
            views.sort(
                key=lambda view: view.score
                + (
                    0.5 * math.exp(-abs(view.timestamp - at_time) / 1.5)
                    if at_time is not None
                    else 0.0
                ),
                reverse=True,
            )
        chosen: list[SporeView] = []
        for view in views:
            if all(view_iou_simple(view, other) < 0.6 for other in chosen):
                chosen.append(view)
            if len(chosen) >= self.config.query_views:
                break
        return chosen

    def _map_detection(self, view: SporeView, box: BBox) -> BBox:
        frame_size = view.metadata.get("frame_size") or [10000, 10000]
        width, height = int(frame_size[0]), int(frame_size[1])
        return BBox(
            view.bbox.x1 + box.x1,
            view.bbox.y1 + box.y1,
            view.bbox.x1 + box.x2,
            view.bbox.y1 + box.y2,
        ).clipped(width, height)

    def bind(
        self,
        term: str,
        *,
        budget: int | None = None,
        at_time: float | None = None,
    ) -> list[Binding]:
        """Attach ``term`` to the capsules that actually contain it."""
        term = term.strip().lower()
        if not term:
            return []
        self._ensure_backends()
        capsules = self.route([term], budget=budget, at_time=at_time)
        if not capsules:
            return []
        mode = self.config.binding_mode.lower()
        bindings: list[Binding] = []
        if (
            mode in {"detector", "hybrid"}
            and self._detector is not None
            and self.config.context_binding
            and self.store.contexts
        ):
            if at_time is not None and self.config.context_fusion:
                bindings = self._bind_context_fused(term, capsules, at_time=at_time)
            else:
                bindings = self._bind_context(term, capsules, at_time=at_time)
                if at_time is not None and self.config.context_sweep:
                    swept = self._bind_context_sweep(term, capsules, at_time=at_time)
                    bindings = self._merge_bindings(bindings, swept)
        if mode in {"detector", "hybrid"} and self._detector is not None and not bindings:
            bindings = self._bind_detector(term, capsules, at_time=at_time)
        if (
            mode in {"detector", "hybrid"}
            and self.config.rescan
            and at_time is not None
            and (not bindings or bindings[0].confidence < self.config.query_abstain_confidence)
        ):
            rescanned = self._rescan(term, at_time)
            if rescanned and (
                not bindings or rescanned[0].confidence > bindings[0].confidence
            ):
                bindings = rescanned
        if mode == "clip" or (
            mode == "hybrid"
            and (
                not bindings
                or bindings[0].confidence < self.config.query_abstain_confidence
            )
        ):
            clip_bindings = self._bind_clip(term, capsules, at_time=at_time)
            if clip_bindings and (
                not bindings or clip_bindings[0].confidence > bindings[0].confidence
            ):
                bindings = clip_bindings
        if self.config.infer_support:
            for binding in bindings:
                self._infer_support(binding)
        bindings.sort(key=lambda item: item.confidence, reverse=True)
        if self.config.propagate_labels and bindings:
            bindings.extend(
                self._propagate(bindings[0], {item.capsule_id for item in bindings})
            )
        return bindings

    def _merge_bindings(
        self, existing: Sequence[Binding], extra: Sequence[Binding]
    ) -> list[Binding]:
        merged = list(existing)
        for candidate in extra:
            duplicate = False
            for current in merged:
                if current.capsule_id == candidate.capsule_id:
                    if current.bbox.iou(candidate.bbox) >= 0.5:
                        duplicate = True
                    if candidate.confidence > current.confidence:
                        current.confidence = candidate.confidence
                        current.timestamp = candidate.timestamp
                        current.bbox = candidate.bbox
                        current.support = candidate.support or current.support
                        current.provenance = candidate.provenance
                        current.detector_confidence = candidate.detector_confidence
                    break
            if not duplicate:
                merged.append(candidate)
        return merged

    def _bind_context_sweep(
        self,
        term: str,
        capsules: Sequence[ObjectCapsule],
        at_time: float,
    ) -> list[Binding]:
        """Ask the frames closest to the question time, not only routed capsules.

        Route ranks capsules by appearance; a moved object can be represented by
        an older capsule while the *current* moment has a much better view.  A
        tiny temporal sweep over retained keyframes runs the detector at the
        question time and attaches the result to the nearest capsule.
        """
        if not self.store.contexts or not capsules:
            return []
        assert self._detector is not None
        vocabulary = [term]
        if self.config.support_labels:
            vocabulary.extend(self.config.support_labels)
        support_set = {label.lower() for label in self.config.support_labels}
        window = self.config.context_sweep_window
        nearby = [
            context
            for context in self.store.contexts
            if abs(context.timestamp - at_time) <= window
        ]
        nearby.sort(key=lambda context: abs(context.timestamp - at_time))
        bindings: list[Binding] = []
        for context in nearby[: self.config.context_sweep_frames]:
            frame = context.image()
            if frame is None:
                continue
            detections = self._detect_cached(
                frame,
                vocabulary,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                cache_tag=context.frame_index,
            )
            term_detections = [
                detection
                for detection in detections
                if _label_matches(detection.label, term)
                and detection.confidence >= self.config.query_detector_confidence
            ]
            if not term_detections:
                continue
            detection = max(term_detections, key=lambda item: item.confidence)
            capsule = max(
                capsules,
                key=lambda item: max(
                    view.score - 0.12 * abs(view.timestamp - context.timestamp)
                    for view in item.best_views
                ),
            )
            binding = Binding(
                term=term,
                label=detection.label.strip().lower(),
                capsule_id=capsule.capsule_id,
                confidence=float(detection.confidence) * 0.88,
                timestamp=context.timestamp,
                bbox=detection.bbox,
                support=self._support_for(detections, detection.bbox, support_set),
                detector_confidence=float(detection.confidence),
                clip_similarity=0.0,
                views_agree=1,
                views_tried=1,
                provenance="context_sweep",
            )
            bindings.append(binding)
            self._attach(binding)
            if binding.confidence >= self.config.query_strong_confidence:
                break
        return bindings

    def _bind_context_fused(
        self,
        term: str,
        capsules: Sequence[ObjectCapsule],
        at_time: float,
    ) -> list[Binding]:
        """Fuse the same detection across nearby keyframes before naming it.

        A single keyframe can contain several instances of one class, and
        detector confidence flips between them as the camera moves.  Tracking
        detections across a short temporal window and scoring the *track*
        instead of the frame removes those one-frame coin flips.  Tracks are
        then attached to the capsule whose view actually covers them.
        """
        assert self._detector is not None
        vocabulary = [term]
        if self.config.support_labels:
            vocabulary.extend(self.config.support_labels)
        support_set = {label.lower() for label in self.config.support_labels}

        contexts = [
            context
            for context in self.store.contexts
            if abs(context.timestamp - at_time) <= self.config.context_sweep_window
        ]
        contexts.sort(key=lambda context: abs(context.timestamp - at_time))
        contexts = contexts[: self.config.context_sweep_frames]
        if not contexts:
            nearest = self.store.context_at(
                at_time, max_gap=self.config.context_max_gap
            )
            contexts = [nearest] if nearest is not None else []
        contexts.sort(key=lambda context: context.timestamp)

        tracks: list[_DetectionTrack] = []
        for context in contexts:
            frame = context.image()
            if frame is None:
                continue
            detections = self._detect_cached(
                frame,
                vocabulary,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                cache_tag=context.frame_index,
            )
            frame_size = context.metadata.get("frame_size", [640, 480])
            frame_width, frame_height = int(frame_size[0]), int(frame_size[1])
            diagonal = math.hypot(frame_width, frame_height)
            for detection in detections:
                if not _label_matches(detection.label, term):
                    continue
                if detection.confidence < self.config.query_detector_confidence:
                    continue
                support = self._support_for(detections, detection.bbox, support_set)
                matched: _DetectionTrack | None = None
                matched_score = -1.0
                for track in tracks:
                    overlap = detection.bbox.iou(track.last_box)
                    dx = detection.bbox.centroid[0] - track.last_box.centroid[0]
                    dy = detection.bbox.centroid[1] - track.last_box.centroid[1]
                    distance = math.hypot(dx, dy) / max(1.0, diagonal)
                    area_ratio = detection.bbox.area / max(1.0, track.last_box.area)
                    close = (
                        distance <= self.config.context_fusion_center
                        and 0.4 <= area_ratio <= 2.5
                    )
                    if overlap < self.config.context_fusion_iou and not close:
                        continue
                    score = overlap + 0.2 * (1.0 - min(1.0, distance / 0.35))
                    if score > matched_score:
                        matched = track
                        matched_score = score
                if matched is None:
                    matched = _DetectionTrack()
                    tracks.append(matched)
                matched.add(detection, context.timestamp, support)

        if not tracks:
            return []

        scored: list[tuple[float, _DetectionTrack]] = []
        for track in tracks:
            persistence = 0.70 + 0.30 * min(1.0, len(track.stamps) / 2.0)
            score = track.temporal_score(at_time) * persistence
            scored.append((score, track))
        scored.sort(key=lambda item: item[0], reverse=True)

        bindings: list[Binding] = []
        for _score, track in scored[: self.config.context_fusion_max_tracks]:
            box = track.mean_box()
            capsule: ObjectCapsule | None = None
            capsule_score = -1.0
            capsule_view: SporeView | None = None
            for candidate in capsules:
                for view in candidate.best_views:
                    coverage = view.bbox.intersection_area(box) / max(1.0, box.area)
                    # A huge background capsule "covers" every detection.  Weight
                    # coverage by how tightly the view bounds the object so
                    # provenance points at the object capsule, not the scene.
                    tightness = min(1.0, box.area / max(1.0, view.bbox.area))
                    fit = min(1.0, coverage) * (0.35 + 0.65 * tightness)
                    dx = view.bbox.centroid[0] - box.centroid[0]
                    dy = view.bbox.centroid[1] - box.centroid[1]
                    diagonal = math.hypot(*view.metadata.get("frame_size", [640, 480]))
                    proximity = max(
                        0.0,
                        1.0 - math.hypot(dx, dy) / max(1.0, 0.35 * diagonal),
                    )
                    utility = (
                        0.55 * fit
                        + 0.30 * proximity
                        + 0.15 * min(1.0, view.score)
                    )
                    if utility > capsule_score:
                        capsule = candidate
                        capsule_score = utility
                        capsule_view = view
            if capsule is None:
                continue
            if any(binding.bbox.iou(box) >= 0.5 for binding in bindings):
                continue
            best_index = max(
                range(len(track.detections)),
                key=lambda item: track.confidences[item],
            )
            confidence = (
                track.temporal_score(at_time)
                * (0.70 + 0.30 * min(1.0, len(track.stamps) / 2.0))
                * (0.82 + 0.18 * (capsule_view.score if capsule_view else 0.0))
            )
            binding = Binding(
                term=term,
                label=track.detections[best_index].label.strip().lower(),
                capsule_id=capsule.capsule_id,
                confidence=float(confidence),
                timestamp=track.stamps[best_index],
                bbox=box,
                support=track.best_support(),
                detector_confidence=track.mean_confidence,
                clip_similarity=0.0,
                views_agree=len(track.stamps),
                views_tried=len(contexts),
                provenance="context_track",
            )
            bindings.append(binding)
            self._attach(binding)
        return bindings

    def _bind_context(
        self,
        term: str,
        capsules: Sequence[ObjectCapsule],
        at_time: float | None = None,
    ) -> list[Binding]:
        """Run the detector on a retained keyframe instead of a tight crop.

        A capsule crop is an attention record, not a detector input: held
        objects routinely extend past the crop boundary or are occluded by the
        hand.  The keyframe gives the detector the full scene at the same
        moment, while still spending zero detector calls during ingest.
        """
        assert self._detector is not None
        term_vector = self._embed_text(term)
        vocabulary = [term]
        if self.config.support_labels:
            vocabulary.extend(self.config.support_labels)
        support_set = {label.lower() for label in self.config.support_labels}
        binding_by_capsule: dict[str, Binding] = {}
        for capsule in capsules:
            best: Binding | None = None
            views_tried = 0
            views_agree = 0
            for view in self._select_views(capsule, term_vector, at_time=at_time):
                context = self.store.context_at(
                    view.timestamp, max_gap=self.config.context_max_gap
                )
                if context is None:
                    continue
                frame = context.image()
                if frame is None:
                    continue
                detections = self._detect_cached(
                    frame,
                    vocabulary,
                    frame_index=context.frame_index,
                    timestamp=context.timestamp,
                    cache_tag=context.frame_index,
                )
                views_tried += 1
                term_detections = [
                    detection
                    for detection in detections
                    if _label_matches(detection.label, term)
                    and detection.confidence >= self.config.query_detector_confidence
                ]
                if not term_detections:
                    continue
                views_agree += 1
                detection = max(term_detections, key=lambda item: item.confidence)
                support = self._support_for(detections, detection.bbox, support_set)
                confidence = float(detection.confidence) * (
                    0.78 + 0.22 * min(1.0, view.score)
                )
                binding = Binding(
                    term=term,
                    label=detection.label.strip().lower(),
                    capsule_id=capsule.capsule_id,
                    confidence=confidence,
                    timestamp=context.timestamp,
                    bbox=detection.bbox,
                    support=support,
                    detector_confidence=float(detection.confidence),
                    clip_similarity=0.0,
                    views_agree=1,
                    views_tried=views_tried,
                    provenance="context",
                )
                if best is None or binding.confidence > best.confidence:
                    best = binding
                if binding.confidence >= self.config.query_strong_confidence and views_tried >= 2:
                    break
            if best is None:
                continue
            best.views_agree = views_agree
            best.views_tried = views_tried
            best.confidence *= 0.85 + 0.15 * (views_agree / max(1, views_tried))
            previous = binding_by_capsule.get(best.capsule_id)
            if previous is None or best.confidence > previous.confidence:
                binding_by_capsule[best.capsule_id] = best
        for binding in binding_by_capsule.values():
            self._attach(binding)
        return list(binding_by_capsule.values())

    def _bind_detector(
        self,
        term: str,
        capsules: Sequence[ObjectCapsule],
        at_time: float | None = None,
    ) -> list[Binding]:
        assert self._detector is not None
        term_vector = self._embed_text(term)
        vocabulary = [term]
        if self.config.support_labels:
            vocabulary.extend(self.config.support_labels)
        support_set = {label.lower() for label in self.config.support_labels}
        bindings: list[Binding] = []
        for capsule in capsules:
            best: Binding | None = None
            views_tried = 0
            views_agree = 0
            for view in self._select_views(capsule, term_vector, at_time=at_time):
                if view.image is None:
                    continue
                views_tried += 1
                detections = self._detect_cached(
                    view.image,
                    vocabulary,
                    frame_index=view.frame_index,
                    timestamp=view.timestamp,
                    cache_tag=("view", id(view)),
                )
                term_detections = [
                    detection
                    for detection in detections
                    if _label_matches(detection.label, term)
                    and detection.confidence >= self.config.query_detector_confidence
                ]
                if not term_detections:
                    continue
                views_agree += 1
                detection = max(term_detections, key=lambda item: item.confidence)
                clip_similarity = self._clip_similarity(view, detection.bbox, term_vector)
                if (
                    term_vector is not None
                    and clip_similarity < self.config.query_clip_threshold
                ):
                    continue
                support = self._support_for(detections, detection.bbox, support_set)
                confidence = (
                    float(detection.confidence)
                    * (0.55 + 0.45 * max(0.0, clip_similarity))
                    * (0.70 + 0.30 * min(1.0, view.score))
                )
                binding = Binding(
                    term=term,
                    label=detection.label.strip().lower(),
                    capsule_id=capsule.capsule_id,
                    confidence=confidence,
                    timestamp=view.timestamp,
                    bbox=self._map_detection(view, detection.bbox),
                    support=support,
                    detector_confidence=float(detection.confidence),
                    clip_similarity=float(clip_similarity),
                    views_agree=1,
                    views_tried=views_tried,
                )
                if best is None or binding.confidence > best.confidence:
                    best = binding
                if binding.confidence >= self.config.query_strong_confidence and views_tried >= 2:
                    break
            if best is None:
                continue
            best.views_agree = views_agree
            best.views_tried = views_tried
            best.confidence *= 0.78 + 0.22 * (views_agree / max(1, views_tried))
            bindings.append(best)
            self._attach(best)
        return bindings

    def _bind_clip(
        self,
        term: str,
        capsules: Sequence[ObjectCapsule],
        at_time: float | None = None,
    ) -> list[Binding]:
        """Detector-free binding with a prompt ensemble and negative calibration.

        YOLO-World only knows concepts from its training distribution, so a
        query for "red hex nut" can fail even when the capsule contains a
        perfect crop.  CLIP was trained on far broader image-text data; scoring
        the same crop against several verbalisations of the query (and against
        distractor prompts) recovers many of those misses at zero detector
        calls.
        """
        assert self._embedder is not None
        if not capsules:
            return []
        prompt_vectors = [
            vector
            for vector in (
                self._embed_text(template.format(term=term))
                for template in self.config.clip_prompt_templates
            )
            if vector is not None
        ]
        if not prompt_vectors:
            return []
        negative_vectors = [
            vector
            for vector in (
                self._embed_text(prompt) for prompt in self.config.clip_negative_prompts
            )
            if vector is not None
        ]
        bindings: list[Binding] = []
        for capsule in capsules:
            scored: list[tuple[float, float, float, SporeView]] = []
            for view in capsule.best_views:
                if view.embedding is None:
                    continue
                frame_size = view.metadata.get("frame_size") or [0, 0]
                frame_area = float(frame_size[0]) * float(frame_size[1])
                if frame_area > 0 and view.bbox.area / frame_area > self.config.clip_max_view_area:
                    continue
                positive = max(_cosine(vector, view.embedding) for vector in prompt_vectors)
                negative = (
                    max((_cosine(vector, view.embedding) for vector in negative_vectors), default=0.0)
                    if negative_vectors
                    else 0.0
                )
                margin = positive - self.config.clip_negative_weight * max(0.0, negative)
                rank = margin
                if at_time is not None:
                    rank += 0.15 * math.exp(-abs(view.timestamp - at_time) / 1.5)
                scored.append((rank, margin, positive, view))
            if not scored:
                continue
            support = sum(
                1
                for _, candidate_margin, _, _ in scored
                if candidate_margin >= self.config.clip_margin_center
            )
            if support < min(2, len(scored)):
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            _, margin, positive, view = scored[0]
            confidence = 1.0 / (
                1.0
                + math.exp(
                    -self.config.clip_margin_slope
                    * (margin - self.config.clip_margin_center)
                )
            )
            confidence *= 0.55 + 0.45 * (support / len(scored))
            binding = Binding(
                term=term,
                label=term,
                capsule_id=capsule.capsule_id,
                confidence=float(confidence),
                timestamp=view.timestamp,
                bbox=view.bbox,
                support=None,
                detector_confidence=0.0,
                clip_similarity=float(positive),
                views_agree=support,
                views_tried=len(scored),
                provenance="clip",
            )
            bindings.append(binding)
            self._attach(binding)
        return bindings

    def _clip_similarity(
        self, view: SporeView, detection: BBox, term_vector: np.ndarray | None
    ) -> float:
        if term_vector is None or view.image is None or self._embedder is None:
            return 0.0
        height, width = view.image.shape[:2]
        x1, y1, x2, y2 = detection.clipped(width, height).int_xyxy
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = view.image[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        embedding = self._embedder.embed_image(crop)
        return _cosine(term_vector, embedding)

    def _rescan(self, term: str, at_time: float) -> list[Binding]:
        """Re-read the source video around a query time as a last evidence tier.

        Context frames are stored sparsely to keep memory bounded.  A short
        interaction can fall between two retained contexts, especially when the
        sampling rate changes the keyframe phase.  When the detector finds
        nothing on the retained evidence, a small foveal rescan of the original
        video recovers that interaction without paying to store every frame.
        """
        if (
            not self.config.rescan
            or not self.source
            or self._detector is None
            or at_time is None
        ):
            return []
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            return []
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            if not math.isfinite(fps) or fps <= 0:
                fps = 30.0
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            start = max(0.0, at_time - self.config.rescan_window)
            end = at_time + self.config.rescan_window
            start_frame = max(0, int(round(start * fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            step = max(1, int(round(fps / max(1.0, self.config.rescan_fps))))
            vocabulary = [term, *self.config.support_labels]
            support_set = {label.lower() for label in self.config.support_labels}
            tracks: list[_DetectionTrack] = []
            seen = 0
            detected = 0
            index = 0
            while detected < self.config.rescan_max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = start_frame + index
                timestamp = frame_index / fps
                index += 1
                if total and frame_index >= total:
                    break
                if timestamp > end:
                    break
                if (index - 1) % step != 0:
                    continue
                seen += 1
                detections = self._detect_cached(
                    frame,
                    vocabulary,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    cache_tag=("rescan", frame_index),
                )
                detected += 1
                height, width = frame.shape[:2]
                diagonal = math.hypot(width, height)
                for detection in detections:
                    if not _label_matches(detection.label, term):
                        continue
                    support = self._support_for(detections, detection.bbox, support_set)
                    match: _DetectionTrack | None = None
                    for track in tracks:
                        reference = track.mean_box()
                        overlap = reference.iou(detection.bbox)
                        dx = reference.centroid[0] - detection.bbox.centroid[0]
                        dy = reference.centroid[1] - detection.bbox.centroid[1]
                        distance = math.hypot(dx, dy)
                        area_ratio = reference.area / max(1.0, detection.bbox.area)
                        if (
                            overlap >= self.config.context_fusion_iou
                            or (
                                distance <= self.config.context_fusion_center * diagonal
                                and 0.4 <= area_ratio <= 2.5
                            )
                        ):
                            match = track
                            break
                    if match is None:
                        match = _DetectionTrack()
                        tracks.append(match)
                    match.add(detection, timestamp, support)
            if not tracks:
                return []
            best = max(
                tracks,
                key=lambda item: item.temporal_score(at_time)
                * (0.80 + 0.20 * min(1.0, len(item.stamps) / 2.0)),
            )
            box = best.mean_box()
            capsule_id = "rescan"
            best_fit = 0.0
            for capsule in self.store.capsules.values():
                for view in capsule.best_views:
                    coverage = view.bbox.intersection_area(box) / max(1.0, box.area)
                    tightness = min(1.0, box.area / max(1.0, view.bbox.area))
                    fit = min(1.0, coverage) * (0.35 + 0.65 * tightness)
                    if fit > best_fit:
                        best_fit = fit
                        capsule_id = capsule.capsule_id
            nearest = min(best.stamps, key=lambda stamp: abs(stamp - at_time))
            confidence = best.temporal_score(at_time) * (
                0.80 + 0.20 * min(1.0, len(best.stamps) / 2.0)
            )
            return [
                Binding(
                    term=term,
                    label=term,
                    capsule_id=capsule_id,
                    confidence=confidence,
                    timestamp=nearest,
                    bbox=box,
                    support=best.best_support(),
                    detector_confidence=best.mean_confidence,
                    views_agree=len(best.stamps),
                    views_tried=seen,
                    provenance="rescan",
                )
            ]
        finally:
            capture.release()

    def _infer_support(self, binding: Binding) -> None:
        """Fill in a support surface for CLIP-only bindings.

        Text-image bindings know *what* something is but not *where* it rests.
        One support-only detector pass over the retained keyframe supplies the
        missing relation without another full object-detection sweep.
        """
        if binding.support or not self.store.contexts:
            return
        context = self.store.context_at(
            binding.timestamp, max_gap=self.config.context_max_gap
        )
        if context is None:
            return
        frame = context.image()
        if frame is None or self._detector is None:
            return
        detections = self._detect_cached(
            frame,
            self.config.support_labels,
            frame_index=context.frame_index,
            timestamp=context.timestamp,
            cache_tag=context.frame_index,
        )
        binding.support = self._support_for(
            detections, binding.bbox, {label.lower() for label in self.config.support_labels}
        )

    def _dominant_colour(self, binding: Binding) -> str | None:
        """Read a colour attribute from the retained context around a binding.

        Attribute questions are evidence questions too.  Rather than captioning
        the whole frame, sample the central part of the bound box, ignore
        unsaturated pixels (hands, white table, shadows), and vote over coarse
        hue bins.  A low-chroma crop falls back to black/grey/white.
        """
        if not self.store.contexts:
            return None
        context = self.store.context_at(
            binding.timestamp, max_gap=self.config.context_max_gap
        )
        if context is None:
            return None
        frame = context.image()
        if frame is None:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = binding.bbox.clipped(width, height).int_xyxy
        if x2 <= x1 or y2 <= y1:
            return None
        inset_x = int((x2 - x1) * 0.15)
        inset_y = int((y2 - y1) * 0.15)
        crop = frame[y1 + inset_y : y2 - inset_y, x1 + inset_x : x2 - inset_x]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue = hsv[..., 0]
        saturation = hsv[..., 1]
        value = hsv[..., 2]
        chromatic = (saturation >= self.config.colour_min_saturation) & (
            value >= self.config.colour_min_value
        )
        total = float(chromatic.size)
        if total <= 0:
            return None
        if chromatic.mean() < self.config.colour_min_chromatic_fraction:
            grey_level = float(np.median(value))
            if grey_level >= 170.0:
                return "white"
            if grey_level <= 55.0:
                return "black"
            return "grey"
        votes: dict[str, int] = {}
        for name, low, high in _COLOUR_RANGES:
            if low < high:
                mask = chromatic & (hue >= low) & (hue < high)
            else:
                mask = chromatic & ((hue >= low) | (hue < high))
            votes[name] = votes.get(name, 0) + int(mask.sum())
        name, count = max(votes.items(), key=lambda item: item[1])
        if count < 0.30 * float(chromatic.sum()):
            return None
        return name

    @staticmethod
    def _support_for(
        detections: Sequence[Any], object_box: BBox, support_set: set[str]
    ) -> str | None:
        best: tuple[float, str] | None = None
        for detection in detections:
            label = detection.label.strip().lower()
            if label not in support_set:
                continue
            overlap = detection.bbox.ios(object_box)
            if overlap > 0.02:
                score = 1.0 + overlap
            else:
                ox, oy = object_box.centroid
                sx, sy = detection.bbox.centroid
                distance = math.hypot(ox - sx, oy - sy)
                reach = max(detection.bbox.width, detection.bbox.height, object_box.width)
                if distance > 2.5 * reach:
                    continue
                score = max(0.0, 1.0 - distance / (2.5 * reach))
            if score > 0 and (best is None or score > best[0]):
                best = (score, label)
        return best[1] if best is not None else None

    def _attach(self, binding: Binding) -> None:
        capsule = self.store.capsules.get(binding.capsule_id)
        if capsule is None:
            return
        row = binding.as_dict()
        history = capsule.metadata.setdefault("bindings", [])
        history.append(row)
        if len(history) > 12:
            del history[:-12]
        if binding.confidence >= capsule.label_confidence:
            capsule.label = binding.label
            capsule.label_confidence = binding.confidence
            capsule.label_views = binding.views_agree

    def _propagate(self, source: Binding, skip: set[str]) -> list[Binding]:
        origin = self.store.capsules.get(source.capsule_id)
        if origin is None:
            return []
        extra: list[Binding] = []
        for capsule in self.store.capsules.values():
            if capsule.capsule_id in skip:
                continue
            if abs(capsule.last_seen - source.timestamp) > self.config.propagate_max_gap:
                continue
            similarity = capsule.similarity(origin.prototype)
            if similarity < self.config.propagate_similarity:
                continue
            binding = Binding(
                term=source.term,
                label=source.label,
                capsule_id=capsule.capsule_id,
                confidence=source.confidence
                * self.config.propagate_confidence
                * max(0.0, similarity),
                timestamp=capsule.last_seen,
                bbox=capsule.last_bbox or source.bbox,
                support=source.support,
                detector_confidence=0.0,
                clip_similarity=similarity,
                provenance="propagation",
            )
            self._attach(binding)
            extra.append(binding)
        extra.sort(key=lambda item: item.confidence, reverse=True)
        return extra

    # -- question answering ------------------------------------------------

    def _scan_first(
        self,
        term: str,
        at_time: float | None,
    ) -> Binding | None:
        """Forward-scan retained keyframes for the first appearance of a term.

        A temporal question ("when did … first touch X") is not answered by the
        capsule nearest the question time: the event may be long past.  SPORE's
        compressed context frames form a sparse temporal index, so scanning
        them forward finds the first evidence at a fraction of dense-video cost.
        """
        if not self.config.action_scan or not self.store.contexts:
            return None
        contexts = sorted(self.store.contexts, key=lambda item: item.timestamp)
        if at_time is not None:
            contexts = [
                context
                for context in contexts
                if context.timestamp <= at_time + 0.5
            ]
        contexts = contexts[: self.config.temporal_scan_limit]
        if not contexts:
            return None
        assert self._detector is not None
        vocabulary = [term]
        if self.config.support_labels:
            vocabulary.extend(self.config.support_labels)
        support_set = {label.lower() for label in self.config.support_labels}
        for index, context in enumerate(contexts):
            frame = context.image()
            if frame is None:
                continue
            detections = self._detect_cached(
                frame,
                vocabulary,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                cache_tag=context.frame_index,
            )
            matches = [
                detection
                for detection in detections
                if _label_matches(detection.label, term)
                and detection.confidence >= self.config.query_detector_confidence
            ]
            if not matches:
                continue
            hands = [
                BBox.from_sequence(box)
                for box in (context.metadata.get("hand_boxes") or [])
            ]
            points = [
                point
                for group in (context.metadata.get("hand_points") or [])
                for point in group
            ]
            if points:
                # A palm-sized hand box can cover a whole shelf.  Fingertips are
                # the contact evidence: an object merely visible beside a hand
                # is not being touched.
                width, height = context.metadata.get("frame_size", [640, 480])
                pad = min(10.0, 0.02 * math.hypot(float(width), float(height)))
                matches = [
                    detection
                    for detection in matches
                    if any(detection.bbox.contains_point(point, pad=pad) for point in points)
                ]
                if not matches:
                    continue
            elif hands:
                # Seeing an object on a shelf is not touching it.  A first-touch
                # answer must have the hand and the detection at the same place.
                matches = [
                    detection
                    for detection in matches
                    if max((hand.ios(detection.bbox) for hand in hands), default=0.0)
                    >= 0.05
                ]
                if not matches:
                    continue
            detection = max(matches, key=lambda item: item.confidence)
            persistence = 0.0
            if index + 1 < len(contexts):
                following = contexts[index + 1]
                if following.timestamp - context.timestamp <= 1.5:
                    next_frame = following.image()
                    if next_frame is not None:
                        next_detections = self._detect_cached(
                            next_frame,
                            vocabulary,
                            frame_index=following.frame_index,
                            timestamp=following.timestamp,
                            cache_tag=following.frame_index,
                        )
                        if any(
                            _label_matches(item.label, term)
                            and item.confidence
                            >= self.config.query_detector_confidence
                            for item in next_detections
                        ):
                            persistence = 1.0
            confidence = float(detection.confidence) * (
                0.80 + 0.20 * persistence
            )
            capsule = max(
                self.store.capsules.values(),
                key=lambda item: max(
                    (view.bbox.iou(detection.bbox) for view in item.best_views),
                    default=0.0,
                ),
                default=None,
            )
            if capsule is None:
                return None
            binding = Binding(
                term=term,
                label=detection.label.strip().lower(),
                capsule_id=capsule.capsule_id,
                confidence=confidence,
                timestamp=context.timestamp,
                bbox=detection.bbox,
                support=self._support_for(detections, detection.bbox, support_set),
                detector_confidence=float(detection.confidence),
                clip_similarity=0.0,
                views_agree=1 + int(persistence),
                views_tried=index + 2,
                provenance="temporal_scan",
            )
            self._attach(binding)
            return binding
        return None

    def _count_contexts(self, at_time: float | None) -> list[ContextFrame]:
        """Choose contexts spread over the episode for lifetime counting.

        Counting must not depend on action segmentation: a better contact
        boundary should not change how many cups are reported.  Sample the
        retained contexts on a uniform temporal grid instead of action starts,
        then add the query-time keyframe.
        """
        limit = max(1, self.config.count_contexts)
        ordered = sorted(self.store.contexts, key=lambda item: item.timestamp)
        contexts: list[ContextFrame] = []
        if ordered:
            first, last = ordered[0].timestamp, ordered[-1].timestamp
            width = max(1e-6, last - first)
            for index in range(limit):
                target = first + (index + 0.5) * width / limit
                context = min(ordered, key=lambda item: abs(item.timestamp - target))
                if all(
                    abs(context.timestamp - found.timestamp) > 0.2
                    for found in contexts
                ):
                    contexts.append(context)
        if at_time is not None and self.store.contexts:
            nearest = self.store.context_at(at_time, max_gap=self.config.context_max_gap)
            if nearest is not None and all(
                abs(nearest.timestamp - found.timestamp) > 0.2 for found in contexts
            ):
                contexts.append(nearest)
        contexts.sort(key=lambda item: item.timestamp)
        return contexts

    def _cluster_tracks(
        self, tracks: Sequence[_DetectionTrack], frame_diagonal: float
    ) -> list[list[_DetectionTrack]]:
        """Group lifetime tracks that describe the same physical place.

        Sparse action-start contexts fragment one cup seen at 20 s and again at
        40 s into separate tracks.  Before counting, merge tracks whose mean
        boxes overlap or whose centres stay within a small fraction of the
        frame diagonal.  The highest-confidence track represents each cluster.
        """
        clusters: list[list[_DetectionTrack]] = []
        for track in sorted(tracks, key=lambda item: item.mean_confidence, reverse=True):
            box = track.mean_box()
            placed = False
            for cluster in clusters:
                representative = max(cluster, key=lambda item: item.mean_confidence)
                other = representative.mean_box()
                if box.iou(other) >= self.config.count_cluster_iou:
                    cluster.append(track)
                    placed = True
                    break
                dx = box.centroid[0] - other.centroid[0]
                dy = box.centroid[1] - other.centroid[1]
                if math.hypot(dx, dy) / max(1.0, frame_diagonal) <= self.config.count_cluster_distance:
                    cluster.append(track)
                    placed = True
                    break
            if not placed:
                clusters.append([track])
        return clusters

    def _count_term(
        self,
        term: str,
        at_time: float | None = None,
    ) -> tuple[int, list[Binding]]:
        """Count distinct detector-confirmed instances across the episode.

        Counting is not ``len(bindings)``: a single capsule can be matched by
        several views, and one physical object can produce several capsules.
        Detections from action-aligned keyframes are linked into tracks first;
        only persistent or high-confidence tracks are counted.
        """
        if self._detector is None:
            return 0, []
        vocabulary = [term]
        if self.config.support_labels:
            vocabulary.extend(self.config.support_labels)

        contexts = self._count_contexts(at_time)
        tracks: list[_DetectionTrack] = []
        peak = 0
        peak_hits: list[tuple[Any, ContextFrame]] = []
        frame_diagonal = math.hypot(640.0, 480.0)
        for context in contexts:
            frame = context.image()
            if frame is None:
                continue
            detections = self._detect_cached(
                frame,
                vocabulary,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                cache_tag=context.frame_index,
            )
            frame_size = context.metadata.get("frame_size", [640, 480])
            diagonal = math.hypot(float(frame_size[0]), float(frame_size[1]))
            frame_diagonal = diagonal
            frame_hits: list[Any] = []
            for detection in detections:
                if not _label_matches(detection.label, term):
                    continue
                if detection.confidence < self.config.count_min_confidence:
                    continue
                frame_hits.append(detection)
                matched: _DetectionTrack | None = None
                matched_score = -1.0
                for track in tracks:
                    if context.timestamp - track.stamps[-1] > self.config.count_track_gap:
                        continue
                    overlap = detection.bbox.iou(track.last_box)
                    dx = detection.bbox.centroid[0] - track.last_box.centroid[0]
                    dy = detection.bbox.centroid[1] - track.last_box.centroid[1]
                    distance = math.hypot(dx, dy) / max(1.0, diagonal)
                    area_ratio = detection.bbox.area / max(1.0, track.last_box.area)
                    close = (
                        distance <= self.config.context_fusion_center
                        and 0.4 <= area_ratio <= 2.5
                    )
                    if overlap < self.config.context_fusion_iou and not close:
                        continue
                    score = overlap + 0.2 * (1.0 - min(1.0, distance / 0.35))
                    if score > matched_score:
                        matched = track
                        matched_score = score
                if matched is None:
                    matched = _DetectionTrack()
                    tracks.append(matched)
                matched.add(detection, context.timestamp, None)
            if frame_hits:
                # Open-vocabulary detectors often emit overlapping boxes for
                # the same instance.  Deduplicate within the frame before using
                # peak occupancy as an instance count, and require reasonable
                # confidence so a single frame of low-confidence noise cannot
                # inflate the count.
                kept: list[Any] = []
                for detection in sorted(
                    frame_hits, key=lambda item: item.confidence, reverse=True
                ):
                    if detection.confidence < self.config.count_peak_confidence:
                        continue
                    if all(detection.bbox.iou(other.bbox) < 0.50 for other in kept):
                        kept.append(detection)
                frame_hits = kept
            if len(frame_hits) > peak:
                peak = len(frame_hits)
                peak_hits = [(detection, context) for detection in frame_hits]

        confirmed = [
            track
            for track in tracks
            if len(track.stamps) >= 2
            or max(track.confidences, default=0.0) >= self.config.count_single_confidence
        ]
        confirmed.sort(key=lambda track: track.mean_confidence, reverse=True)
        if peak >= 2:
            # When two or more instances are visible together, that is stronger
            # evidence than a lifetime track count: sparse action-start contexts
            # fragment one moving object into several tracks.  Report the
            # detections from the peak frame and use their count.
            peak_bindings: list[Binding] = []
            for detection, context in peak_hits:
                peak_bindings.append(
                    Binding(
                        term=term,
                        label=term,
                        capsule_id="",
                        confidence=min(0.95, 0.50 + 0.40 * detection.confidence),
                        timestamp=context.timestamp,
                        bbox=detection.bbox,
                        support=None,
                        detector_confidence=float(detection.confidence),
                        clip_similarity=None,
                        views_agree=1,
                        views_tried=len(contexts),
                        provenance="count_peak",
                    )
                )
            peak_bindings.sort(key=lambda item: item.confidence, reverse=True)
            return peak, peak_bindings
        bindings: list[Binding] = []
        clusters = self._cluster_tracks(confirmed, frame_diagonal)
        for cluster in clusters:
            track = max(cluster, key=lambda item: item.mean_confidence)
            box = track.mean_box()
            capsule_id = ""
            best_fit = 0.0
            for capsule in self.store.capsules.values():
                for view in capsule.best_views:
                    coverage = view.bbox.intersection_area(box) / max(1.0, box.area)
                    tightness = min(1.0, box.area / max(1.0, view.bbox.area))
                    fit = min(1.0, coverage) * (0.35 + 0.65 * tightness)
                    if fit > best_fit:
                        best_fit = fit
                        capsule_id = capsule.capsule_id
            confidence = min(
                0.95,
                0.45
                + 0.15 * min(4, len(track.stamps))
                + 0.20 * track.mean_confidence
                + 0.04 * min(3, len(cluster) - 1),
            )
            bindings.append(
                Binding(
                    term=term,
                    label=term,
                    capsule_id=capsule_id,
                    confidence=confidence,
                    timestamp=float(np.mean(track.stamps)),
                    bbox=box,
                    support=track.best_support(),
                    detector_confidence=track.mean_confidence,
                    clip_similarity=None,
                    views_agree=sum(len(item.stamps) for item in cluster),
                    views_tried=len(contexts),
                    provenance="count_track",
                )
            )
        return len(clusters), bindings

    def _scan_actions(
        self,
        term: str,
        question: str,
        at_time: float | None,
    ) -> Binding | None:
        """Use hand-object contact events as an index into retained keyframes.

        Temporal questions ("when did … first pick up X") and activity
        questions ("what did … do with X") are poorly served by the capsule
        closest to the question time: the object may already be in hand or out
        of frame.  Action capsules record *when* a hand touched something, so
        scanning their start frames finds the first interaction directly.
        """
        if not self.config.action_scan or not self.store.actions:
            return None
        text = question.lower()
        temporal = "when did" in text or "when was" in text
        activity = text.startswith("what did") or "what did the person" in text
        if not (temporal or activity):
            return None
        actions = [
            action
            for action in self.store.actions.values()
            if action.duration >= self.config.action_scan_min_duration
            and math.hypot(*action.displacement)
            >= self.config.action_scan_min_displacement
        ]
        if not actions:
            return None
        if temporal:
            actions.sort(key=lambda action: action.start)
        else:
            target = at_time if at_time is not None else actions[-1].end
            actions.sort(
                key=lambda action: abs(0.5 * (action.start + action.end) - target)
            )
        assert self._detector is not None
        vocabulary = [term]
        if self.config.support_labels:
            vocabulary.extend(self.config.support_labels)
        support_set = {label.lower() for label in self.config.support_labels}
        for attempts, action in enumerate(
            actions[: self.config.action_scan_limit], start=1
        ):
            timestamp = action.start
            context = self.store.context_at(timestamp, max_gap=1.0)
            if context is None:
                context = self.store.context_at(
                    0.5 * (action.start + action.end), max_gap=1.0
                )
            if context is None:
                continue
            frame = context.image()
            if frame is None:
                continue
            detections = self._detect_cached(
                frame,
                vocabulary,
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                cache_tag=context.frame_index,
            )
            matches = [
                detection
                for detection in detections
                if _label_matches(detection.label, term)
                and detection.confidence >= self.config.query_detector_confidence
            ]
            if not matches:
                continue
            detection = max(matches, key=lambda item: item.confidence)
            capsule_id = action.capsule_ids[0] if action.capsule_ids else ""
            capsule = self.store.capsules.get(capsule_id)
            if capsule is None:
                capsule = max(
                    self.store.capsules.values(),
                    key=lambda item: max(
                        (
                            view.bbox.iou(detection.bbox)
                            for view in item.best_views
                        ),
                        default=0.0,
                    ),
                )
            confidence = float(detection.confidence) * (
                0.70 + 0.30 * min(1.0, action.duration / 0.5)
            )
            binding = Binding(
                term=term,
                label=detection.label.strip().lower(),
                capsule_id=capsule.capsule_id,
                confidence=confidence,
                timestamp=action.start,
                bbox=detection.bbox,
                support=self._support_for(detections, detection.bbox, support_set),
                detector_confidence=float(detection.confidence),
                clip_similarity=0.0,
                views_agree=1,
                views_tried=attempts,
                provenance="action_scan",
            )
            self._attach(binding)
            return binding
        return None

    def _evidence_bytes(self, bindings: Sequence[Binding]) -> int:
        """Unique JPEG bytes attached to an answer's provenance chain.

        Counts the best capsule view and the retained context frame used by
        each binding, without double-counting evidence shared by bindings.
        This is the amount a downstream reader would need to fetch from the
        split evidence directory, not the whole store.
        """
        def blob_size(blob: Any) -> int:
            if blob is None:
                return 0
            if isinstance(blob, (bytes, bytearray)):
                return len(blob)
            if isinstance(blob, np.ndarray):
                ok, buffer = cv2.imencode(
                    ".jpg", blob, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )
                return int(buffer.nbytes) if ok else int(blob.nbytes)
            return int(getattr(blob, "nbytes", 0))

        seen: set[tuple[str, int, int]] = set()
        total = 0
        for binding in bindings:
            capsule = self.store.capsules.get(binding.capsule_id)
            if capsule is not None:
                view = capsule.best_view
                size = blob_size(view.image) if view is not None else 0
                if size:
                    key = ("view", id(capsule), int(view.timestamp * 1000))
                    if key not in seen:
                        seen.add(key)
                        total += size
            context = self.store.context_at(
                binding.timestamp, max_gap=self.config.context_max_gap
            )
            size = blob_size(context.image_jpeg) if context is not None else 0
            if size:
                key = ("context", 0, int(context.timestamp * 1000))
                if key not in seen:
                    seen.add(key)
                    total += size
        return total

    def answer(
        self,
        question: str,
        *,
        terms: Sequence[str] | None = None,
        budget: int | None = None,
        at_time: float | None = None,
    ) -> SporeAnswer:
        start = time.perf_counter()
        calls_before = self.detector_calls
        resolved = list(terms) if terms else extract_terms(question)
        answer = SporeAnswer(question=question, terms=resolved)
        if not resolved:
            answer.elapsed_s = time.perf_counter() - start
            return answer
        total_budget = budget or self.config.query_capsules
        shared = max(2, int(math.ceil(total_budget / max(1, len(resolved)))))
        lowered_question = question.lower()
        action_binding = None
        if "how many" in lowered_question and len(resolved) == 1:
            answer.count, answer.bindings = self._count_term(resolved[0], at_time)
        else:
            for term in resolved:
                answer.bindings.extend(self.bind(term, budget=shared, at_time=at_time))
            if len(resolved) == 1:
                if "when did" in lowered_question or "when was" in lowered_question:
                    action_binding = self._scan_first(resolved[0], at_time)
                    if action_binding is None:
                        action_binding = self._scan_actions(
                            resolved[0], question, at_time
                        )
                else:
                    action_binding = self._scan_actions(
                        resolved[0], question, at_time
                    )
        answer.bindings.sort(key=lambda item: item.confidence, reverse=True)
        if action_binding is not None:
            answer.bindings = [action_binding] + [
                item for item in answer.bindings if item is not action_binding
            ]
        answer.capsules_considered = len(
            self.route(resolved, budget=total_budget, at_time=at_time)
        )
        answer.detector_calls = self.detector_calls - calls_before
        answer.evidence_bytes = self._evidence_bytes(answer.bindings)
        answer.abstained = (
            (not answer.bindings and answer.count in (None, 0))
            or (
                answer.count is None
                and bool(answer.bindings)
                and answer.bindings[0].confidence < self.config.query_abstain_confidence
            )
        )
        colour = None
        if ("colour" in lowered_question or "color" in lowered_question) and answer.bindings:
            best_bound = next(
                (item for item in answer.bindings if item.bbox.area > 0.0), None
            )
            if best_bound is not None:
                colour = self._dominant_colour(best_bound)
        answer.answer = self._compose(
            question,
            answer.bindings,
            answer.abstained,
            count=answer.count,
            colour=colour,
        )
        answer.elapsed_s = time.perf_counter() - start
        return answer

    @staticmethod
    def _compose(
        question: str,
        bindings: Sequence[Binding],
        abstained: bool,
        *,
        count: int | None = None,
        colour: str | None = None,
    ) -> str:
        lowered = question.lower()
        if lowered.startswith("how many") and count is not None and bindings:
            return f"{count} {bindings[0].term}"
        if abstained or not bindings:
            return "not enough evidence"
        best = bindings[0]
        if colour and ("colour" in lowered or "color" in lowered):
            return f"{best.label} is {colour} at {best.timestamp:.1f}s"
        location = ""
        if best.support:
            alias = _SUPPORT_ALIASES.get(best.support)
            location = f" on the {best.support}"
            if alias and alias not in best.support:
                location += f" ({alias})"
        if lowered.startswith("where"):
            return f"{best.label}{location} at {best.timestamp:.1f}s"
        if lowered.startswith("how many"):
            return f"{len(bindings)} candidate(s) for {best.term} at {best.timestamp:.1f}s"
        if lowered.startswith("when did") or lowered.startswith("when was"):
            return f"first interaction with {best.label} at {best.timestamp:.1f}s"
        if lowered.startswith("what did"):
            return f"picked up or moved {best.label} at {best.timestamp:.1f}s"
        if lowered.startswith("what") and " do" in lowered:
            return f"{best.label} at {best.timestamp:.1f}s"
        return f"{best.label}{location} ({best.confidence:.2f}) at {best.timestamp:.1f}s"


def view_iou_simple(first: SporeView, second: SporeView) -> float:
    return first.bbox.iou(second.bbox)


def _label_matches(predicted: str, term: str) -> bool:
    predicted = predicted.strip().lower()
    term = term.strip().lower()
    if predicted == term:
        return True
    # Open-vocabulary models sometimes return the prompt with a qualifier
    # ("blue hex nut" for "hex nut"); accept that without accepting "cup" for
    # "cupboard" (require the term to be a meaningful substring).
    return len(term) >= 4 and term in predicted


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------


def extract_terms(question: str) -> list[str]:
    """Extract candidate object phrases from an egocentric question.

    This is intentionally a *parser*, not a vocabulary: it never maps a phrase
    onto a fixed label set.  It handles the patterns that dominate Chronicle's
    QA sets ("Where is the X?", "What did I do with the X?") and falls back to
    the last content nouns.
    """
    text = question.strip().lower()
    text = re.sub(r"[^a-z0-9\-\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []
    patterns = (
        r"\bwhere (?:is|are|was|were) (?:the |a |an )?(.+)$",
        r"\bwhere (?:did|does|do) (?:the |a |an )?(.+?) go\b",
        r"\bwhere did .*?\b(?:put|place|attach|insert|remove|open|press|replace|use|handle)\b (?:the |a |an )?(.+?)(?: at | in | on | into | after | before | with | from |$)",
        r"\bwhat did .*?\b(?:put|place|attach|insert|remove|open|press|replace|use|handle)\b (?:the |a |an )?(.+?)(?: to | into | on | after | before | at | with | from |$)",
        r"\bwhen did .*?(?:touch|pick up|grab|use|hold|open|close) (?:the |a |an )?(.+)$",
        r"\bhow many times did .*?\b(?:press|use|attach|insert|remove|open|close) (?:the |a |an )?(.+)$",
        r"\bhow many (.+?)(?: did| does| do| were| are| was| is| has| have)\b",
        r"\bhow many (.+?) (?:are|is|were|was)\b",
        r"\bwhat (?:colour|color) (?:is|are|was|were) (?:the |a |an )?(.+)$",
        r"\b(?:what|which) (?:did|do|does|is|are|was|were).*?\b(?:with|on|at|in|from|using) (?:the |a |an )?(.+)$",
        r"\b(?:pick up|picked up|put|placed|hold|holding|grab|grabbed|use|used) (?:the |a |an )?(.+)$",
        r"\b(?:find|locate|show me) (?:the |a |an )?(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            phrase = _clean_phrase(match.group(1))
            if phrase:
                return [phrase]
    tokens = [token for token in text.split() if token not in _STOPWORDS and token not in _VERBS]
    if not tokens:
        return []
    # Prefer the last two content tokens: in English questions the object tends
    # to be the final noun phrase ("blue hex nut", "sd card").
    phrase = " ".join(tokens[-2:])
    return [phrase]


def _clean_phrase(phrase: str) -> str:
    phrase = phrase.strip(" ?.!,'\"")
    phrase = re.split(
        r"\b(?:at|in|on|into|after|before|with|from|to|for|during)\b", phrase
    )[0]
    phrase = re.sub(r"\b(?:start|beginning|end|first|last|initially)$", "", phrase)
    phrase = re.sub(r"\b(?:the|a|an)$", "", phrase).strip()
    tokens = [token for token in phrase.split() if token not in _STOPWORDS]
    if not tokens:
        return ""
    # Drop leading determiners/pronouns that survived.
    while tokens and tokens[0] in {"it", "them", "that", "this"}:
        tokens.pop(0)
    return " ".join(tokens[:3]).strip()

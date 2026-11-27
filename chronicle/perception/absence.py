"""Fixed-label presence and absence verification over retained SPORE contexts.

The open-vocabulary query detector is asked for one term at a time.  That is
useful for coverage, but it also substitutes confusable categories: asked for
``mug`` it scores a drinking glass, asked for ``bowl`` it scores a plate of
chips.  The absence calibration in ``docs/research/ABSENCE.md`` shows that this
makes "not found" a poor negative test.

This module implements the complementary half: one pass of a *fixed-label*
detector over every retained context, cached, then per-term queries answered
from that cache.  On the classes it covers, a fixed-label detector has a
high-precision floor (look-alikes fall below it), so it can support negative
answers; the cost is that a user term outside its label space cannot be
checked at all.

Terminology is deliberate:

* ``present`` - at least one detection at or above ``threshold`` in the
  scanned contexts (and inside the time window, when one is asked for);
* ``not_observed`` - the term is covered by the detector but nothing reached
  the threshold.  This is **not** proof of absence: the object may be missed
  or occluded.  Callers must attach the measured false-negative rate;
* ``unsupported`` - the term has no class in this detector.  Never return a
  negative verdict for it.

The scan is exhaustive over the store, not a query-time sample, and is cached,
so ``check`` is free after ``scan``.  See ``scripts/spore_absence_verify.py``
for the evaluation harness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from chronicle.perception.capsules import CapsuleStore
from chronicle.types import BBox, Detection

# COCO class names for common query words whose wording differs.
COCO_ALIASES: dict[str, tuple[str, ...]] = {
    "television": ("tv",),
    "phone": ("cell phone",),
    "sofa": ("couch",),
    "table": ("dining table",),
    "glass": ("wine glass",),
}


class FixedLabelDetector(Protocol):
    """Minimal interface for a detector with a closed label set."""

    labels: set[str]

    def detect(
        self, frame: np.ndarray, *, timestamp: float = 0.0
    ) -> list[Detection]:  # pragma: no cover - protocol
        ...


class CocoYoloDetector:
    """COCO YOLOv8 detector, imported lazily so core installs stay model-free."""

    name = "coco-yolov8"

    def __init__(
        self,
        model: str = "yolov8s.pt",
        *,
        confidence: float = 0.05,
        imgsz: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "the fixed-label verifier needs ultralytics; install the "
                "perception requirements"
            ) from exc
        self.model_name = model
        self._model = YOLO(model)
        self.labels = {str(label) for label in self._model.names.values()}
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)

    def detect(
        self, frame: np.ndarray, *, timestamp: float = 0.0
    ) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        result = self._model(
            frame,
            conf=self.confidence,
            iou=0.45,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: list[Detection] = []
        for box in result[0].boxes:
            detections.append(
                Detection(
                    label=self._model.names[int(box.cls)],
                    bbox=BBox.from_sequence(box.xyxy[0].tolist()),
                    confidence=float(box.conf),
                    source=self.name,
                    metadata={"timestamp": float(timestamp)},
                )
            )
        return detections


@dataclass(slots=True)
class AbsenceVerdict:
    """One term query answered from the fixed-label scan."""

    term: str
    verdict: str  # present | not_observed | unsupported
    supported: bool
    labels: tuple[str, ...] = ()
    hits: int = 0
    contexts_scanned: int = 0
    threshold: float = 0.5
    confidence: float | None = None
    timestamp: float | None = None
    bbox: BBox | None = None
    window: tuple[float, float] | None = None
    provenance: str = "fixed_label"

    @property
    def observed(self) -> bool:
        return self.verdict == "present"

    def as_dict(self) -> dict:
        return {
            "term": self.term,
            "verdict": self.verdict,
            "supported": self.supported,
            "labels": list(self.labels),
            "hits": self.hits,
            "contexts_scanned": self.contexts_scanned,
            "threshold": self.threshold,
            "confidence": (
                round(float(self.confidence), 4)
                if self.confidence is not None
                else None
            ),
            "timestamp": (
                round(float(self.timestamp), 2)
                if self.timestamp is not None
                else None
            ),
            "bbox": self.bbox.as_dict() if self.bbox is not None else None,
            "window": list(self.window) if self.window is not None else None,
            "provenance": self.provenance,
        }


@dataclass(slots=True)
class _ScannedContext:
    timestamp: float
    detections: list[Detection] = field(default_factory=list)


class FixedLabelVerifier:
    """Exhaustive fixed-label scan of a capsule store's retained contexts."""

    def __init__(
        self,
        store: CapsuleStore,
        *,
        detector: FixedLabelDetector | None = None,
        model: str = "yolov8s.pt",
        threshold: float = 0.50,
        aliases: dict[str, tuple[str, ...]] | None = None,
        floor: float = 0.05,
    ) -> None:
        self.store = store
        self._detector = detector
        self.model = model
        self.threshold = float(threshold)
        self.aliases = dict(COCO_ALIASES if aliases is None else aliases)
        self.floor = float(floor)
        self._scanned: list[_ScannedContext] | None = None
        self.scan_calls = 0
        self.scan_seconds = 0.0

    # -- lifecycle ---------------------------------------------------------

    def _ensure_detector(self) -> FixedLabelDetector:
        if self._detector is None:
            self._detector = CocoYoloDetector(self.model, confidence=self.floor)
        return self._detector

    @property
    def labels(self) -> set[str]:
        return set(self._ensure_detector().labels)

    def scan(self) -> int:
        """Detect once per retained context.  Returns detector calls spent."""
        if self._scanned is not None:
            return 0
        detector = self._ensure_detector()
        scanned: list[_ScannedContext] = []
        started = time.perf_counter()
        calls = 0
        for context in self.store.contexts:
            frame = context.image()
            if frame is None:
                continue
            scanned.append(
                _ScannedContext(
                    timestamp=float(context.timestamp),
                    detections=detector.detect(
                        frame, timestamp=float(context.timestamp)
                    ),
                )
            )
            calls += 1
        self._scanned = scanned
        self.scan_calls = calls
        self.scan_seconds = time.perf_counter() - started
        return calls

    # -- queries -----------------------------------------------------------

    def wanted_labels(self, term: str) -> tuple[str, ...]:
        lowered = term.strip().lower()
        return self.aliases.get(lowered, (lowered,))

    def check(
        self,
        term: str,
        *,
        at_time: float | None = None,
        window: float = 2.5,
    ) -> AbsenceVerdict:
        """Answer a presence query, optionally inside a time window.

        ``not_observed`` means no detection reached ``threshold`` in the scanned
        contexts; it is not proof that the object was absent.  Terms outside the
        detector's label set return ``unsupported`` and are never negative for a
        different reason.
        """
        self.scan()
        assert self._scanned is not None
        wanted = self.wanted_labels(term)
        labels = self.labels
        supported = any(label in labels for label in wanted)
        window_bounds = (
            (max(0.0, at_time - window), at_time + window)
            if at_time is not None
            else None
        )
        considered = [
            context
            for context in self._scanned
            if window_bounds is None
            or window_bounds[0] <= context.timestamp <= window_bounds[1]
        ]
        if not supported:
            return AbsenceVerdict(
                term=term,
                verdict="unsupported",
                supported=False,
                labels=wanted,
                contexts_scanned=len(considered),
                threshold=self.threshold,
                window=window_bounds,
            )
        best: Detection | None = None
        best_time: float | None = None
        hits = 0
        wanted_set = set(wanted)
        for context in considered:
            for detection in context.detections:
                if detection.label not in wanted_set:
                    continue
                if detection.confidence < self.threshold:
                    continue
                hits += 1
                if best is None or detection.confidence > best.confidence:
                    best = detection
                    best_time = context.timestamp
        if best is None:
            return AbsenceVerdict(
                term=term,
                verdict="not_observed",
                supported=True,
                labels=wanted,
                hits=0,
                contexts_scanned=len(considered),
                threshold=self.threshold,
                window=window_bounds,
            )
        return AbsenceVerdict(
            term=term,
            verdict="present",
            supported=True,
            labels=wanted,
            hits=hits,
            contexts_scanned=len(considered),
            threshold=self.threshold,
            confidence=best.confidence,
            timestamp=best_time,
            bbox=best.bbox,
            window=window_bounds,
        )

    def check_many(
        self,
        terms: Sequence[str],
        *,
        at_time: float | None = None,
        window: float = 2.5,
    ) -> list[AbsenceVerdict]:
        return [self.check(term, at_time=at_time, window=window) for term in terms]

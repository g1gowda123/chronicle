"""Open-vocabulary detection with a palette fallback for the demo clip."""

from __future__ import annotations

import logging
import threading
from collections import Counter
from collections.abc import Iterable

import cv2
import numpy as np

from chronicle.perception.config import DetectorConfig, PerceptionConfig
from chronicle.perception.palette import MIN_AREA, PALETTE, hsv_ranges
from chronicle.types import BBox, Detection

LOGGER = logging.getLogger(__name__)


def filter_min_area(
    detections: Iterable[Detection], floors: dict[str, float] | None
) -> list[Detection]:
    if not floors:
        return list(detections)
    normalized = {str(key).lower(): float(value) for key, value in floors.items()}
    return [
        detection
        for detection in detections
        if normalized.get(detection.label.lower(), 0.0) <= 0.0
        or detection.bbox.area >= normalized[detection.label.lower()]
    ]


def deduplicate_detections(
    detections: Iterable[Detection],
    *,
    iou_threshold: float = 0.55,
    ios_threshold: float = 0.80,
    max_per_label: int = 3,
) -> list[Detection]:
    """Collapse duplicate / nested boxes from an open-vocabulary detector.

    YOLO-World commonly returns both an object and a crop inside it. IoU alone
    misses full containment, so IoS is used as the second gate. Different
    labels are never merged: a phone can legitimately sit on a table.
    """
    kept: list[Detection] = []
    counts: Counter[str] = Counter()
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if max_per_label > 0 and counts[detection.label] >= max_per_label:
            continue
        duplicate = False
        for other in kept:
            if other.label != detection.label:
                continue
            if (
                detection.bbox.iou(other.bbox) > iou_threshold
                or detection.bbox.ios(other.bbox) > ios_threshold
            ):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(detection)
        counts[detection.label] += 1
    return kept


class BaseDetector:
    name = "base"

    def detect(
        self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0
    ) -> list[Detection]:
        raise NotImplementedError

    def set_vocabulary(self, vocabulary: Iterable[str]) -> None:
        pass

    def close(self) -> None:
        pass


class ColorDetector(BaseDetector):
    """Zero-download fallback for the synthetic demo video.

    This only understands the flat colours in :mod:`chronicle.perception.palette`
    and is deliberately not used as a hidden production fallback for real
    footage.
    """

    name = "color-palette"

    def __init__(self, config: DetectorConfig, vocabulary: Iterable[str]) -> None:
        self.config = config
        self.vocabulary = [str(v).lower() for v in vocabulary]
        self.labels = [v for v in self.vocabulary if v in PALETTE]
        self._ranges = {
            label: hsv_ranges(PALETTE[label], hue_tolerance=4, saturation_tolerance=55, value_tolerance=55)
            for label in self.labels
        }
        self._kernel = np.ones((3, 3), np.uint8)

    def _mask(self, hsv: np.ndarray, label: str) -> np.ndarray:
        mask: np.ndarray | None = None
        for lower, upper in self._ranges[label]:
            part = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        if mask is None:
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

    def detect(
        self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0
    ) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        scale = min(1.0, 640.0 / max(1, width))
        small = (
            cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            if scale < 0.999
            else frame
        )
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        inverse = 1.0 / scale
        detections: list[Detection] = []
        for label in self.labels:
            mask = self._mask(hsv, label)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            minimum = MIN_AREA.get(label, 200) * scale * scale
            candidates: list[tuple[float, BBox, float]] = []
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < minimum:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw < 4 or bh < 4:
                    continue
                bbox = BBox(x * inverse, y * inverse, (x + bw) * inverse, (y + bh) * inverse)
                fill = area / max(1.0, float(bw * bh))
                confidence = float(np.clip(0.55 + 0.4 * fill, 0.0, 0.99))
                if confidence < self.config.confidence:
                    continue
                candidates.append((area, bbox, confidence))
            candidates.sort(key=lambda item: item[0], reverse=True)
            kept: list[Detection] = []
            for area, bbox, confidence in candidates:
                if any(bbox.iou(other.bbox) > 0.45 for other in kept):
                    continue
                kept.append(
                    Detection(
                        label=label,
                        bbox=bbox,
                        confidence=confidence,
                        source=self.name,
                        metadata={"area": area, "frame": frame_index},
                    )
                )
                if len(kept) >= 2:
                    break
            detections.extend(kept)
        detections.sort(key=lambda d: d.bbox.area, reverse=True)
        return deduplicate_detections(
            detections,
            iou_threshold=self.config.dedup_iou,
            ios_threshold=self.config.dedup_ios,
            max_per_label=self.config.max_per_label,
        )[: self.config.max_detections]


class YoloWorldDetector(BaseDetector):
    """Ultralytics YOLO-World open-vocabulary detector."""

    name = "yolo-world"

    def __init__(self, config: DetectorConfig, config_root: PerceptionConfig) -> None:
        self.config = config
        self._root = config_root
        self._model: object | None = None
        self._lock = threading.RLock()
        self._prompts: tuple[str, ...] = tuple(config.prompts or config_root.vocabulary)
        self._labels: set[str] = set(config_root.vocabulary)
        self._aliases = {str(k).lower(): str(v).lower() for k, v in config.label_aliases.items()}
        self._text_features: dict[str, object] = {}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOWorld
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YOLO-World is not installed. Install perception dependencies with "
                "`uv sync` or set detector.backend='color' for the synthetic demo."
            ) from exc
        model_path = self._root.resolve_model(self.config.model)
        LOGGER.info("Loading YOLO-World model %s", model_path)
        model = YOLOWorld(model_path)
        self._apply_classes(model, self._prompts)
        self._model = model

    def _apply_classes(self, model: object, labels: tuple[str, ...]) -> None:
        """Swap open-vocabulary prompts while reusing CLIP text features.

        ``YOLOWorld.set_classes`` rebuilds the CLIP text model on every call
        (~2 s per vocabulary switch on a laptop GPU).  We cache one text
        embedding per label, encode only new labels, update both the wrapper
        model and the predictor's ``AutoBackend`` copy, and keep the predictor
        alive.  A live predictor otherwise deep-copies the old ``txt_feats``
        and silently answers with the previous vocabulary.
        """
        import torch

        classes = list(labels)
        world = getattr(model, "model", None)
        setter = getattr(world, "set_classes", None)
        if setter is None:
            model.set_classes(classes)  # type: ignore[attr-defined]
            return
        try:
            missing = [label for label in classes if label not in self._text_features]
            if missing:
                encoded = world.get_text_pe(missing, cache_clip_model=True)  # type: ignore[attr-defined]
                for index, label in enumerate(missing):
                    self._text_features[label] = encoded[:, index : index + 1, :].detach()
            feats = torch.cat([self._text_features[label] for label in classes], dim=1)
        except Exception:  # pragma: no cover - compatibility fallback
            setter(classes)
            world.names = classes  # type: ignore[attr-defined]
            model.predictor = None  # type: ignore[attr-defined]
            return
        world.txt_feats = feats  # type: ignore[attr-defined]
        world.model[-1].nc = len(classes)  # type: ignore[attr-defined]
        world.names = classes  # type: ignore[attr-defined]
        predictor = getattr(model, "predictor", None)
        backend = getattr(predictor, "model", None)
        live_world = getattr(backend, "model", None)
        if live_world is not None:
            live_world.txt_feats = feats.to(live_world.txt_feats.device)
            live_world.model[-1].nc = len(classes)
            live_world.names = classes
            backend.names = classes  # type: ignore[attr-defined]

    def set_vocabulary(self, vocabulary: Iterable[str]) -> None:
        labels = tuple(dict.fromkeys(str(v).strip().lower() for v in vocabulary if str(v).strip()))
        if not labels or labels == tuple(self._prompts):
            return
        self._prompts = labels
        self._labels = set(labels)
        with self._lock:
            if self._model is not None:
                self._apply_classes(self._model, labels)

    def detect(
        self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0
    ) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        with self._lock:
            self._ensure_model()
            assert self._model is not None
            results = self._model.predict(  # type: ignore[attr-defined]
                frame,
                conf=float(self.config.confidence),
                iou=float(self.config.iou),
                imgsz=int(self.config.imgsz),
                max_det=int(self.config.max_detections),
                verbose=False,
                device=None if self._root.device == "auto" else self._root.device,
            )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        rows = boxes.data.detach().cpu().numpy()
        names = getattr(result, "names", {}) or {}
        detections: list[Detection] = []
        for row in rows:
            x1, y1, x2, y2, confidence, class_id = row[:6]
            class_index = int(class_id)
            if isinstance(names, dict):
                raw_label = str(names.get(class_index, class_index))
            else:
                raw_label = str(names[class_index]) if class_index < len(names) else str(class_index)
            raw_label = raw_label.lower()
            label = self._aliases.get(raw_label, raw_label)
            if self._labels and label not in self._labels:
                continue
            detections.append(
                Detection(
                    label=label,
                    bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(confidence),
                    source=self.name,
                    metadata={"frame": frame_index, "prompt": raw_label},
                )
            )
        detections = filter_min_area(detections, self.config.min_label_area)
        return deduplicate_detections(
            detections,
            iou_threshold=self.config.dedup_iou,
            ios_threshold=self.config.dedup_ios,
            max_per_label=self.config.max_per_label,
        )[: self.config.max_detections]

    def close(self) -> None:
        with self._lock:
            self._model = None


def build_detector(config: PerceptionConfig, *, source_name: str = "") -> BaseDetector:
    backend = config.detector.backend.lower()
    if backend == "auto":
        if config.is_demo_source(source_name):
            backend = "color"
        else:
            try:
                import ultralytics  # noqa: F401
            except Exception:
                LOGGER.warning(
                    "Ultralytics is unavailable; falling back to the flat-colour "
                    "detector. Real footage needs YOLO-World."
                )
                backend = "color"
            else:
                backend = "yolo-world"
    if backend == "color":
        return ColorDetector(config.detector, config.vocabulary)
    if backend == "yolo-world":
        return YoloWorldDetector(config.detector, config)
    raise ValueError(f"unknown detector backend: {config.detector.backend}")

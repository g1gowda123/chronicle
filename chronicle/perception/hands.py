"""Egocentric hand landmark extraction.

MediaPipe Tasks is the production path. A colour-skin fallback keeps the
synthetic demo runnable without downloading a model; it produces pseudo
landmarks in the same 21-point layout so downstream geometry is unchanged.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from chronicle.perception.config import HandConfig, PerceptionConfig
from chronicle.perception.palette import PALETTE, hsv_ranges
from chronicle.types import HandObservation

LOGGER = logging.getLogger(__name__)


class BaseHands:
    name = "base"

    def detect(
        self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0
    ) -> list[HandObservation]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MediaPipeHands(BaseHands):
    """MediaPipe Tasks ``HandLandmarker`` adapter."""

    name = "mediapipe"

    def __init__(self, config: HandConfig, model_path: str | Path) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "MediaPipe is not installed. Install perception dependencies with `uv sync`."
            ) from exc
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {model_path}. "
                "Run `python scripts/fetch_assets.py` from the repository root."
            )
        self._mp = mp
        self._vision = vision
        self.config = config
        self._upscale = max(1.0, float(config.upscale))
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max(1, int(config.max_hands)),
            min_hand_detection_confidence=float(config.confidence),
            min_hand_presence_confidence=float(config.confidence),
            min_tracking_confidence=float(config.confidence),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._last_timestamp_ms = -1

    def _timestamp_ms(self, timestamp: float) -> int:
        value = int(round(float(timestamp) * 1000.0))
        if value <= self._last_timestamp_ms:
            value = self._last_timestamp_ms + 1
        self._last_timestamp_ms = value
        return value

    def detect(
        self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0
    ) -> list[HandObservation]:
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        working = frame
        if self._upscale > 1.001:
            working = cv2.resize(
                frame,
                (
                    max(1, int(round(width * self._upscale))),
                    max(1, int(round(height * self._upscale))),
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        rgb = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        with self._lock:
            result = self._landmarker.detect_for_video(image, self._timestamp_ms(timestamp))
        hands: list[HandObservation] = []
        for index, landmarks in enumerate(result.hand_landmarks or []):
            points = np.array(
                [[point.x * width, point.y * height] for point in landmarks],
                dtype=np.float32,
            )
            handedness = "Right"
            confidence = 0.9
            try:
                categories = result.handedness[index]
                if categories:
                    category = categories[0]
                    handedness = (
                        getattr(category, "category_name", None)
                        or getattr(category, "display_name", None)
                        or handedness
                    )
                    confidence = float(getattr(category, "score", 0.9) or 0.9)
            except (IndexError, AttributeError):
                pass
            hands.append(
                HandObservation(
                    landmarks=points,
                    confidence=confidence,
                    handedness=handedness,
                    source=self.name,
                )
            )
        return hands

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


class ColorHands(BaseHands):
    """Skin-colour fallback for the synthetic demo.

    The 21 points are synthetic, but wrist / palm / index-tip positions are
    stable enough for hand-object containment and dwell gating.
    """

    name = "color-skin"

    def __init__(self, config: HandConfig, *, min_area: int = 400) -> None:
        self.config = config
        self.min_area = min_area
        self._ranges = hsv_ranges(
            PALETTE["hand"], hue_tolerance=10, saturation_tolerance=60, value_tolerance=60
        )
        self._kernel = np.ones((5, 5), np.uint8)

    def _mask(self, hsv: np.ndarray) -> np.ndarray:
        mask: np.ndarray | None = None
        for lower, upper in self._ranges:
            part = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        if mask is None:
            return np.zeros(hsv.shape[:2], dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

    @staticmethod
    def _pseudo_landmarks(contour: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Approximate the 21-point MediaPipe layout from a skin contour.

        The landmarks must span the real hand hull, because interaction geometry
        uses their convex hull for IoS. Sampling only the wrist-to-tip axis (as
        an earlier version did) made the reported hand far narrower than the
        pixels and suppressed genuine contacts.
        """
        x, y, width, height = bbox
        points = contour.reshape(-1, 2).astype(np.float32)
        fallback = np.array(
            [[x, y + height], [x + width / 2, y], [x + width, y + height]],
            dtype=np.float32,
        )
        if len(points) < 3:
            points = fallback
        hull = cv2.convexHull(points).reshape(-1, 2).astype(np.float32)
        if len(hull) < 3:
            hull = fallback
        wrist_index = int(np.argmax(hull[:, 1]))  # lowest point, egocentric wrist
        wrist = hull[wrist_index]
        hull = np.roll(hull, -wrist_index, axis=0)
        closed = np.vstack([hull, hull[:1]])
        edges = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        perimeter = float(edges.sum())
        landmarks = np.zeros((21, 2), dtype=np.float32)
        landmarks[0] = wrist
        if perimeter <= 1e-6:
            return landmarks
        cumulative = np.concatenate([[0.0], np.cumsum(edges)])
        # 20 points evenly spaced around the hull, excluding the wrist itself.
        for index, distance in enumerate(np.linspace(0.0, perimeter, 21, endpoint=False)[1:], start=1):
            edge = int(min(np.searchsorted(cumulative, distance, side="right") - 1, len(edges) - 1))
            fraction = (distance - cumulative[edge]) / max(1e-6, edges[edge])
            landmarks[index] = closed[edge] + fraction * (closed[edge + 1] - closed[edge])
        # MediaPipe's primary fingertip is index 8; pin it to the farthest
        # sampled point from the wrist.
        distances = np.linalg.norm(landmarks - wrist, axis=1)
        tip = int(np.argmax(distances[1:]) + 1)
        landmarks[[8, tip]] = landmarks[[tip, 8]]
        return landmarks

    def detect(
        self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0
    ) -> list[HandObservation]:
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
        contours, _ = cv2.findContours(self._mask(hsv), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        hands: list[HandObservation] = []
        inverse = 1.0 / scale
        minimum = self.min_area * scale * scale
        for contour in contours[: max(1, self.config.max_hands)]:
            area = float(cv2.contourArea(contour))
            if area < minimum:
                continue
            landmarks = self._pseudo_landmarks(contour, cv2.boundingRect(contour)) * inverse
            landmarks[:, 0] = np.clip(landmarks[:, 0], 0, width - 1)
            landmarks[:, 1] = np.clip(landmarks[:, 1], 0, height - 1)
            hands.append(
                HandObservation(
                    landmarks=landmarks,
                    confidence=float(np.clip(0.55 + area / 12000.0, 0.0, 0.99)),
                    handedness="Right",
                    source=self.name,
                )
            )
        return hands


class HybridHands(BaseHands):
    """MediaPipe first, colour fallback only when no hand is found."""

    name = "hybrid"

    def __init__(self, config: HandConfig, model_path: str | Path) -> None:
        self.primary = MediaPipeHands(config, model_path)
        self.fallback = ColorHands(config)

    def detect(
        self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0
    ) -> list[HandObservation]:
        primary = self.primary.detect(frame, frame_index, timestamp)
        if primary:
            return primary
        return self.fallback.detect(frame, frame_index, timestamp)

    def close(self) -> None:
        self.primary.close()


class NullHands(BaseHands):
    name = "off"

    def detect(
        self, frame: np.ndarray, frame_index: int = 0, timestamp: float = 0.0
    ) -> list[HandObservation]:
        return []


def build_hands(config: PerceptionConfig, *, source_name: str = "") -> BaseHands:
    backend = config.hands.backend.lower()
    if backend == "off":
        return NullHands()
    if backend == "auto":
        if config.is_demo_source(source_name):
            return ColorHands(config.hands)
        try:
            model = config.resolve_hand_model()
        except FileNotFoundError:
            LOGGER.warning(
                "MediaPipe hand model is missing; using the colour fallback. "
                "Run scripts/fetch_assets.py to enable real hand landmarks."
            )
            return ColorHands(config.hands)
        try:
            return HybridHands(config.hands, model)
        except Exception as exc:
            LOGGER.warning("MediaPipe could not start (%s); using colour hands.", exc)
            return ColorHands(config.hands)
    if backend == "mediapipe":
        return MediaPipeHands(config.hands, config.resolve_hand_model())
    if backend == "hybrid":
        return HybridHands(config.hands, config.resolve_hand_model())
    if backend == "color":
        return ColorHands(config.hands)
    raise ValueError(f"unknown hand backend: {config.hands.backend}")

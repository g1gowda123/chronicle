"""Lightweight OpenCV overlay for the CLI and demo recordings."""

from __future__ import annotations

import cv2
import numpy as np

from chronicle.perception.palette import PALETTE
from chronicle.types import FrameResult

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _label_color(label: str) -> tuple[int, int, int]:
    if label in PALETTE:
        return PALETTE[label]
    seed = sum(ord(char) for char in label) * 37
    hue = np.uint8(seed % 180)
    hsv = np.uint8([[[hue, 220, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_overlay(
    frame: np.ndarray,
    result: FrameResult,
    *,
    draw_hands: bool = True,
    draw_boxes: bool = True,
) -> np.ndarray:
    image = frame.copy()
    interacting = {
        observation.track_id
        for observation in result.interactions
        if observation.contact
    }

    if draw_boxes:
        for track in result.tracks:
            color = _label_color(track.label)
            thickness = 3 if track.track_id in interacting else 2
            cv2.rectangle(image, track.bbox.int_xyxy, color, thickness)
            label = f"#{track.track_id} {track.label} {track.confidence:.2f}"
            if track.metadata.get("reid_source"):
                label += f" reid:{track.metadata['reid_source']}"
            _text(image, label, (track.bbox.int_xyxy[0], max(14, track.bbox.int_xyxy[1] - 6)), color=color)

    if draw_hands:
        for hand in result.hands:
            points = hand.landmarks.astype(int)
            for start, end in HAND_CONNECTIONS:
                if start < len(points) and end < len(points):
                    cv2.line(image, tuple(points[start]), tuple(points[end]), (0, 200, 255), 1, cv2.LINE_AA)
            for point in points:
                cv2.circle(image, tuple(point), 2, (0, 120, 255), -1)

    motion = result.motion
    lines = [
        f"{motion.state.value}  d=({motion.dx:+.1f},{motion.dy:+.1f})px  "
        f"rot={motion.rotation_degrees:+.1f}deg",
        f"t={result.timestamp:.2f}s  tracks={len(result.tracks)}  "
        f"hands={len(result.hands)}  interactions={len(result.interactions)}",
    ]
    if motion.transition:
        lines.append(f"camera event: {motion.transition}")
    box_width = min(image.shape[1] - 16, 600)
    cv2.rectangle(image, (8, 8), (8 + box_width, 18 + 20 * len(lines)), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        _text(image, line, (16, 30 + 20 * index))
    return image

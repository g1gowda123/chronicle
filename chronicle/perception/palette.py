"""Flat-colour palette used by the offline demo detector and renderer."""

from __future__ import annotations

import cv2
import numpy as np

# BGR. Kept in sync with the synthetic demo clip; not a production detector.
PALETTE: dict[str, tuple[int, int, int]] = {
    "keys": (0, 220, 255),
    "phone": (255, 80, 0),
    "mug": (0, 0, 220),
    "cup": (255, 255, 0),
    "backpack": (0, 230, 0),
    "table": (30, 75, 220),
    "desk": (30, 75, 220),
    "sofa": (150, 110, 40),
    "box": (180, 40, 180),
    "bottle": (0, 255, 255),
    "book": (100, 40, 40),
    "laptop": (255, 0, 255),
    "hand": (120, 165, 220),
}

MIN_AREA: dict[str, int] = {
    "keys": 90,
    "phone": 420,
    "mug": 300,
    "cup": 300,
    "backpack": 900,
    "table": 12000,
    "desk": 12000,
    "sofa": 12000,
    "box": 650,
    "bottle": 240,
    "book": 420,
    "laptop": 900,
    "hand": 400,
}


def bgr_to_hsv(color: tuple[int, int, int]) -> tuple[int, int, int]:
    pixel = np.array([[color]], dtype=np.uint8)
    hue, saturation, value = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hue), int(saturation), int(value)


def hsv_ranges(
    color: tuple[int, int, int],
    *,
    hue_tolerance: int = 9,
    saturation_tolerance: int = 70,
    value_tolerance: int = 65,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    hue, saturation, value = bgr_to_hsv(color)
    low_hue, high_hue = hue - hue_tolerance, hue + hue_tolerance
    low_sat = max(0, saturation - saturation_tolerance)
    high_sat = min(255, saturation + saturation_tolerance)
    low_val = max(0, value - value_tolerance)
    high_val = min(255, value + value_tolerance)
    if low_hue < 0:
        return [
            ((low_hue + 180, low_sat, low_val), (179, high_sat, high_val)),
            ((0, low_sat, low_val), (high_hue, high_sat, high_val)),
        ]
    if high_hue > 179:
        return [
            ((low_hue, low_sat, low_val), (179, high_sat, high_val)),
            ((0, low_sat, low_val), (high_hue - 180, high_sat, high_val)),
        ]
    return [((low_hue, low_sat, low_val), (high_hue, high_sat, high_val))]

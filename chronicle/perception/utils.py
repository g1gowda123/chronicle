"""Small shared helpers for the perception package."""

from __future__ import annotations

from typing import Any

import numpy as np

from chronicle.types import BBox


def normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return np.zeros_like(array)
    return array / norm


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def crop_image(frame: np.ndarray, bbox: BBox, *, pad: int = 0) -> np.ndarray | None:
    if frame is None or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    box = bbox.expanded(pad, bounds=(width, height))
    x1, y1, x2, y2 = box.int_xyxy
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def resolve_device(device: str = "auto") -> str:
    if device and device.lower() != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def import_available(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False

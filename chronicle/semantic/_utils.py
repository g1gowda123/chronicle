"""Small dependency-light helpers shared by the semantic engine."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("chronicle.semantic")

_WARNED: set[str] = set()


def warn_once(key: str, message: str, *args: Any) -> None:
    """Log a warning only once per process (keeps demo output readable)."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args)


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return float32 L2-normalised rows; zero vectors are left as zeros."""
    array = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (normalising defensively)."""
    left = l2_normalize(np.asarray(a, dtype=np.float32).ravel())
    right = l2_normalize(np.asarray(b, dtype=np.float32).ravel())
    return float(np.dot(left, right))


def to_pil_image(image: Any, channel_order: str = "bgr") -> Image.Image:
    """Coerce a PIL image, path, or OpenCV-style array into an RGB PIL image.

    ``channel_order`` only applies to HxWx3/HxWx4 numpy arrays.  The default is
    ``bgr`` because the rest of the pipeline is OpenCV-based; pass ``rgb`` for
    arrays that already came from PIL/matplotlib.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return handle.convert("RGB")

    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape {array.shape!r}")

    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0) * 255.0 if array.max(initial=0) <= 1.0 else array
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[2] == 4:
        array = array[:, :, :3]

    if channel_order.lower() == "bgr":
        array = array[:, :, ::-1]

    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def slugify(text: str, fallback: str = "zone", max_length: int = 40) -> str:
    """Make a stable, filename-safe identifier fragment."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (slug[:max_length] or fallback).strip("_")


def shorten_label(text: str, max_length: int = 48) -> str:
    """Keep the useful first clause of a generated caption."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    first = re.split(r"[.;]", cleaned, maxsplit=1)[0].strip()
    if len(first) <= max_length:
        return first
    trimmed = first[:max_length].rsplit(" ", 1)[0].rstrip(", ")
    return trimmed + "…"

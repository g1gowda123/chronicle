"""Optional semantic-segmentation support head.

The open-vocabulary detector is good at naming the object a question asks
about, but it often cannot name the surface below that object: ``wall`` and
``floor`` are rarely detected as boxes.  A small ADE20K segmentation model
fills that gap with dense surface labels.  This module is deliberately
opt-in: it adds a model dependency and a query-time forward pass, so the core
perception path remains usable without it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .capsules import BBox

_WALL = frozenset({"wall", "windowpane", "painting", "blind", "door"})
_FLOOR = frozenset({"floor", "rug"})
_TABLE = frozenset({"table", "coffee table", "desk", "counter", "countertop"})
_SHELF = frozenset({"shelf", "cabinet", "bookcase"})
_ROAD = frozenset({"road", "sidewalk", "earth", "ground", "grass", "pavement", "parking lot"})
_GROUPS = {"wall": _WALL, "floor": _FLOOR, "table": _TABLE, "shelf": _SHELF}

_WALL_OBJECTS = frozenset(
    {"tv", "television", "monitor", "screen", "window", "painting", "picture", "mirror", "clock"}
)
_FURNITURE = frozenset({"sofa", "couch", "chair", "armchair", "bench", "stool"})
_TABLETOP_OBJECTS = frozenset({"plate", "cup", "mug", "bowl", "glass", "bottle", "knife", "fork", "spoon"})
_OUTDOOR_OBJECTS = frozenset(
    {"car", "truck", "bus", "bicycle", "motorcycle", "person", "people"}
)


def _group_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    return {
        group: sum(fractions.get(name, 0.0) for name in names)
        for group, names in _GROUPS.items()
    }


def _object_kind(label: str) -> tuple[bool, bool, bool]:
    label = label.strip().lower()
    wallish = any(token in label for token in _WALL_OBJECTS)
    furniture = any(token in label for token in _FURNITURE)
    tabletop = any(token in label for token in _TABLETOP_OBJECTS)
    return wallish, furniture, tabletop


def choose_support(
    label: str,
    current: str | None,
    fractions: Mapping[str, float],
) -> str | None:
    """Return a support override, or ``None`` to keep the detector's answer.

    ``fractions`` maps ADE20K class names to their fraction in the contact
    bands around the object.  Only background-ish supports (wall/floor) and a
    clearly better tabletop are overridden; a specific horizontal support is
    never replaced by a different horizontal support just because the
    segmenter disagrees, which avoids the kitchen ``countertop -> shelf``
    regression observed in the probe.
    """

    wallish, furniture, tabletop = _object_kind(label)
    outdoor = any(token in label.strip().lower() for token in _OUTDOOR_OBJECTS)
    current = (current or "").strip().lower()
    groups = _group_fractions(fractions)
    wall = max(
        fractions.get("wall", 0.0),
        fractions.get("windowpane", 0.0),
        fractions.get("painting", 0.0),
        fractions.get("blind", 0.0),
    )
    floor = groups["floor"]
    table = groups["table"]
    shelf = groups["shelf"]
    road = sum(fractions.get(name, 0.0) for name in _ROAD)

    # A wall-mounted object is the one case where a dense surface model beats
    # the box detector: the wall is behind the object, not below it.
    if wallish and wall >= 0.25:
        return "wall"
    if wallish and current == "wall":
        return None
    if current == "road":
        return None
    if outdoor and not current and road >= 0.35:
        return "road"

    if current in {"table", "countertop", "work table", "desk", "shelf"}:
        if furniture and current in {"desk", "shelf"} and floor >= 0.08:
            return "floor"
        if tabletop and current in {"desk", "shelf"} and table >= 0.35:
            return "table"
        return None
    if current == "wall":
        return None
    if furniture and floor >= 0.15:
        return "floor"
    if not current:
        if floor >= 0.35:
            return "floor"
        if table >= 0.35:
            return "table"
        if shelf >= 0.35:
            return "shelf"
    return None


class SurfaceSegmenter:
    """Lazy ADE20K surface segmenter used only when ``surface_support`` is on."""

    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._masks: dict[tuple[int, int], np.ndarray] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = (
            SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            .eval()
            .to(self._device)
        )
        self._processor = SegformerImageProcessor.from_pretrained(self.model_name)

    def _segment(self, image: np.ndarray, key: tuple[int, int] | None) -> np.ndarray:
        if key is not None and key in self._masks:
            return self._masks[key]
        self._load()
        assert self._model is not None and self._processor is not None
        import torch.nn.functional as functional
        from PIL import Image

        height, width = image.shape[:2]
        inputs = self._processor(
            images=Image.fromarray(image[:, :, ::-1]), return_tensors="pt"
        ).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        mask = (
            functional.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
            .argmax(1)[0]
            .cpu()
            .numpy()
            .astype(np.int16)
        )
        if key is not None:
            if len(self._masks) >= 24:
                self._masks.clear()
            self._masks[key] = mask
        return mask

    def _fractions(self, mask: np.ndarray, bbox: BBox) -> dict[str, float]:
        height, width = mask.shape[:2]
        x1, y1, x2, y2 = bbox.clipped(width, height).int_xyxy
        if x2 <= x1 or y2 <= y1:
            return {}
        edge_x = max(4, int(0.05 * width))
        edge_y = max(4, int(0.05 * height))
        bands = {
            "bottom": (max(0, y2 - 2), min(height, y2 + max(8, int(0.06 * height))), x1, x2),
            "below": (
                min(height, y2 + 2),
                min(height, y2 + max(24, int(0.10 * height))),
                max(0, x1 - edge_x),
                min(width, x2 + edge_x),
            ),
            "left": (y1, y2, max(0, x1 - edge_x), x1),
            "right": (y1, y2, x2, min(width, x2 + edge_x)),
            "above": (max(0, y1 - edge_y), y1, x1, x2),
        }
        counts: dict[str, int] = {}
        total = 0
        for top, bottom, left, right in bands.values():
            if right <= left or bottom <= top:
                continue
            values = mask[top:bottom, left:right].ravel()
            if values.size == 0:
                continue
            total += int(values.size)
            labels, counts_here = np.unique(values, return_counts=True)
            for label, count in zip(labels, counts_here, strict=False):
                name = str(self._model.config.id2label[int(label)])
                counts[name] = counts.get(name, 0) + int(count)
        if total <= 0:
            return {}
        return {name: count / float(total) for name, count in counts.items()}

    def support(
        self,
        image: np.ndarray,
        bbox: BBox,
        label: str,
        current: str | None = None,
        *,
        key: tuple[int, int] | None = None,
    ) -> str | None:
        mask = self._segment(image, key)
        fractions = self._fractions(mask, bbox)
        if not fractions:
            return None
        return choose_support(label, current, fractions)

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._masks.clear()

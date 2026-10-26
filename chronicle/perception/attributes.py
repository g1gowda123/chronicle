"""Cheap per-detection visual attributes for Role A.

The open-vocabulary detector returns a label and a box; a human immediately
wants to know *which* cup is meant.  This module extracts a small set of
deterministic appearance facts from the crop so the graph and dashboard can say
"the red mug on the left", rather than minting three indistinguishable ``mug``
nodes.

Everything here is intentionally classical vision (HSV + k-means + Laplacian
variance).  It runs in a few milliseconds per box on CPU, needs no model
download, and is good enough to disambiguate common objects.  It is not a
material classifier and does not claim to be one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from chronicle.attributes import classify_colour, merge_attributes, public_attributes
from chronicle.types import BBox

# Re-exported so callers can use one import for the attribute contract.
__all__ = [
    "AttributeExtractor",
    "AttributeConfig",
    "dominant_colours",
    "extract_object_attributes",
    "merge_attributes",
    "public_attributes",
]


class AttributeConfig:
    """Tunable knobs for the attribute extractor (duck-typed from config)."""

    __slots__ = ("colour_clusters", "enabled", "interval", "min_crop_px")

    def __init__(
        self,
        *,
        enabled: bool = True,
        interval: int = 3,
        min_crop_px: int = 8,
        colour_clusters: int = 3,
    ) -> None:
        self.enabled = bool(enabled)
        self.interval = max(1, int(interval))
        self.min_crop_px = max(1, int(min_crop_px))
        self.colour_clusters = max(1, min(6, int(colour_clusters)))


def _clip_bbox(bbox: BBox | tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    values = bbox.xyxy if isinstance(bbox, BBox) else bbox
    x1, y1, x2, y2 = (float(v) for v in values)
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def dominant_colours(
    crop: np.ndarray,
    *,
    clusters: int = 3,
) -> list[tuple[tuple[int, int, int], float]]:
    """Return ``[(bgr, weight), ...]`` from the central part of a crop.

    The crop border often contains support surfaces and neighbouring objects,
    so the sample is taken from the inner 80 % and weighted towards the middle.
    Weight is the cluster's share of that weighted sample.
    """

    if crop is None or crop.size == 0:
        return []
    height, width = crop.shape[:2]
    if width < 2 or height < 2:
        return []
    # Trim 10 % on every side; clamp so tiny boxes still have samples.
    mx = min(max(0, int(width * 0.10)), max(0, width // 2 - 1))
    my = min(max(0, int(height * 0.10)), max(0, height // 2 - 1))
    inner = crop[my : height - my, mx : width - mx]
    if inner.size == 0:
        inner = crop
    scale = min(1.0, 64.0 / max(1, inner.shape[1]), 64.0 / max(1, inner.shape[0]))
    if scale < 0.999:
        inner = cv2.resize(
            inner,
            (max(1, int(inner.shape[1] * scale)), max(1, int(inner.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    data = inner.reshape(-1, 3).astype(np.float32)
    k = max(1, min(int(clusters), len(data)))
    if k == 1:
        colour = tuple(int(v) for v in np.mean(data, axis=0))
        return [(colour, 1.0)]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 0.5)
    try:
        _, labels, centers = cv2.kmeans(
            data,
            k,
            None,
            criteria,
            2,
            cv2.KMEANS_PP_CENTERS,
        )
    except cv2.error:  # pragma: no cover - degenerate crop
        colour = tuple(int(v) for v in np.mean(data, axis=0))
        return [(colour, 1.0)]

    labels = labels.ravel()
    # Gaussian centre weight: a pixel in the middle of the box matters more.
    yy, xx = np.mgrid[0 : inner.shape[0], 0 : inner.shape[1]]
    cx = (inner.shape[1] - 1) / 2.0
    cy = (inner.shape[0] - 1) / 2.0
    weight = np.exp(-(((xx - cx) / max(1.0, cx * 1.15)) ** 2 + ((yy - cy) / max(1.0, cy * 1.15)) ** 2))
    # Suppress pixels that look like the crop border (table, wall, desk).  A
    # solid-colour object has a border identical to its body, so every sample
    # gets the same floor weight and the colour is unchanged.
    border = np.concatenate(
        [
            crop[0, :, :].reshape(-1, 3),
            crop[-1, :, :].reshape(-1, 3),
            crop[:, 0, :].reshape(-1, 3),
            crop[:, -1, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(data - background, axis=1)
    background_weight = np.clip(distance / 80.0, 0.15, 1.0)
    weight = (weight.ravel() * background_weight).astype(np.float32)
    totals = np.bincount(labels, weights=weight, minlength=len(centers))
    grand = float(totals.sum()) or 1.0
    order = list(np.argsort(totals)[::-1])

    # The centre of the box is almost always on the object itself.  If the
    # central patch is a substantial cluster, make it the primary colour even
    # when a larger background region (table, wall, neighbouring object) has a
    # higher global weight.
    label_grid = labels.reshape(inner.shape[0], inner.shape[1])
    window_y = max(1, int(inner.shape[0] * 0.22))
    window_x = max(1, int(inner.shape[1] * 0.22))
    centre_y = inner.shape[0] // 2
    centre_x = inner.shape[1] // 2
    central = label_grid[
        max(0, centre_y - window_y // 2) : centre_y + window_y // 2 + 1,
        max(0, centre_x - window_x // 2) : centre_x + window_x // 2 + 1,
    ].ravel()
    if central.size:
        counts = np.bincount(labels, minlength=len(centers))
        centre_label = int(np.bincount(central, minlength=len(centers)).argmax())
        if (
            centre_label in order
            and counts[centre_label] / max(1, labels.size) >= 0.12
        ):
            order.remove(centre_label)
            order.insert(0, centre_label)
    return [
        (tuple(int(round(v)) for v in centers[index]), float(totals[index] / grand))
        for index in order
        if totals[index] > 0
    ]


def _bgr_to_hex(colour: tuple[int, int, int]) -> str:
    blue, green, red = (max(0, min(255, int(v))) for v in colour)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _hsv(colour: tuple[int, int, int]) -> tuple[int, int, int]:
    pixel = np.array([[colour]], dtype=np.uint8)
    hue, saturation, value = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hue), int(saturation), int(value)


def _shape_from_aspect(aspect: float) -> str:
    if aspect <= 0.82:
        return "tall"
    if aspect >= 1.22:
        return "wide"
    return "square"


def _size_from_area(relative_area: float) -> str:
    if relative_area <= 0.012:
        return "small"
    if relative_area <= 0.075:
        return "medium"
    return "large"


def _brightness(value: float) -> str:
    if value < 70:
        return "dark"
    if value < 145:
        return "dim"
    return "bright"


def _saturation(value: float) -> str:
    if value < 55:
        return "muted"
    if value < 150:
        return "colourful"
    return "vivid"


def _texture(gray: np.ndarray) -> str:
    if gray.size == 0:
        return "plain"
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return "detailed" if laplacian >= 45.0 else "plain"


def _position(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / max(1.0, width)
    cy = (y1 + y2) / 2.0 / max(1.0, height)
    horizontal = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
    vertical = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "middle")
    return horizontal, vertical


def extract_object_attributes(
    frame: np.ndarray,
    bbox: BBox | tuple[float, float, float, float],
    *,
    min_crop_px: int = 8,
    clusters: int = 3,
) -> dict[str, Any]:
    """Extract simple appearance facts for one detection crop.

    Returns an empty dict when the crop is unusable, so callers can simply
    merge the result without special-casing missing attributes.
    """

    if frame is None or frame.size == 0:
        return {}
    height, width = frame.shape[:2]
    clipped = _clip_bbox(bbox, width, height)
    if clipped is None:
        return {}
    x1, y1, x2, y2 = clipped
    if x2 - x1 < min_crop_px or y2 - y1 < min_crop_px:
        return {}
    crop = frame[y1:y2, x1:x2]
    colours = dominant_colours(crop, clusters=clusters)
    if not colours:
        return {}

    (primary_bgr, primary_weight) = colours[0]
    hue, saturation, value = _hsv(primary_bgr)
    colour_name = classify_colour(hue, saturation, value)
    secondary_name = None
    for other_bgr, other_weight in colours[1:]:
        other_hue, other_saturation, other_value = _hsv(other_bgr)
        candidate = classify_colour(other_hue, other_saturation, other_value)
        if candidate != colour_name and other_weight >= 0.18:
            secondary_name = candidate
            break

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_gray = float(gray.mean()) if gray.size else 0.0
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_saturation = float(hsv_crop[..., 1].mean()) if hsv_crop.size else 0.0
    aspect = (x2 - x1) / max(1.0, float(y2 - y1))
    relative_area = ((x2 - x1) * (y2 - y1)) / max(1.0, float(width * height))
    horizontal, vertical = _position((x1, y1, x2, y2), width, height)

    attributes: dict[str, Any] = {
        "colour": colour_name,
        "colour_hex": _bgr_to_hex(primary_bgr),
        "colour_confidence": round(float(primary_weight), 3),
        "brightness": _brightness(mean_gray),
        "saturation": _saturation(mean_saturation),
        "shape": _shape_from_aspect(aspect),
        "aspect_ratio": round(float(aspect), 2),
        "size": _size_from_area(relative_area),
        "relative_area": round(float(relative_area), 4),
        "texture": _texture(gray),
        "position": horizontal,
        "vertical_band": vertical,
    }
    if secondary_name:
        attributes["secondary_colour"] = secondary_name
    return attributes


class AttributeExtractor:
    """Attach attributes to detections, throttled to every ``interval`` frames."""

    name = "crop-attributes"

    def __init__(self, config: AttributeConfig | Mapping[str, Any] | None = None) -> None:
        if isinstance(config, AttributeConfig):
            self.config = config
        elif isinstance(config, Mapping):
            self.config = AttributeConfig(**dict(config))
        elif config is not None and hasattr(config, "enabled"):
            self.config = AttributeConfig(
                enabled=bool(getattr(config, "enabled", True)),
                interval=int(getattr(config, "interval", 3)),
                min_crop_px=int(getattr(config, "min_crop_px", 8)),
                colour_clusters=int(getattr(config, "colour_clusters", 3)),
            )
        else:
            self.config = AttributeConfig()

    def annotate(self, frame: np.ndarray, detections: list[Any], frame_index: int = 0) -> int:
        """Mutate detections in place and return how many were annotated."""

        if not self.config.enabled:
            return 0
        if frame_index % self.config.interval != 0 and frame_index != 0:
            return 0
        annotated = 0
        for detection in detections:
            try:
                attributes = extract_object_attributes(
                    frame,
                    detection.bbox,
                    min_crop_px=self.config.min_crop_px,
                    clusters=self.config.colour_clusters,
                )
            except Exception:  # a crop failure must never stop ingestion
                continue
            if not attributes:
                continue
            metadata = dict(getattr(detection, "metadata", None) or {})
            metadata["attributes"] = merge_attributes(
                metadata.get("attributes"), attributes
            )
            detection.metadata = metadata
            annotated += 1
        return annotated

"""Small, dependency-light helpers for visual object attributes.

Role A extracts a handful of cheap appearance facts for every detection
(dominant colour, brightness, shape, relative size, texture, ...).  The graph
and dashboard only need to *read* those facts, so this module deliberately has
no OpenCV/numpy dependency: it is safe to import from Role B, Role C and the
UI on machines where the perception extras are not installed.

Attribute keys are part of the public snapshot contract.  Keep additions
backward compatible and use :func:`public_attributes` before exposing a raw
track metadata mapping, which also contains internal detector/Re-ID fields.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# Fields that are meaningful to a person reading the dashboard or asking a
# question.  Anything else in a track's metadata is internal bookkeeping.
PUBLIC_ATTRIBUTE_FIELDS: tuple[str, ...] = (
    "colour",
    "colour_hex",
    "colour_confidence",
    "secondary_colour",
    "brightness",
    "saturation",
    "shape",
    "aspect_ratio",
    "size",
    "relative_area",
    "texture",
    "position",
    "vertical_band",
)

# Canonical property names used by the attribute query parser.
ATTRIBUTE_PROPERTIES: tuple[str, ...] = (
    "colour",
    "shape",
    "size",
    "brightness",
    "saturation",
    "texture",
    "position",
)

COLOUR_WORDS: tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "grey",
    "gray",
)

_PROPERTY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "colour": ("colour", "color", "hue", "shade"),
    "shape": ("shape", "proportions", "aspect"),
    "size": ("size", "big", "large", "small", "how big"),
    "brightness": ("brightness", "bright", "dark", "lit"),
    "saturation": ("saturation", "vivid", "muted", "colourful", "colorful"),
    "texture": ("texture", "pattern", "plain", "smooth", "detailed"),
    "position": ("position", "side", "left", "right", "where in the frame"),
}
_APPEARANCE_WORDS = (
    "describe",
    "description",
    "appearance",
    "look like",
    "looks like",
    "what does",
    "attributes",
    "properties",
)


def _hsv_to_name(hue: int, saturation: int, value: int) -> str:
    """Map one OpenCV HSV triple to a human colour word.

    ``hue`` is in ``[0, 179]`` because OpenCV stores hue at half resolution.
    Achromatic pixels are classified before hue so shadows and highlights do
    not masquerade as saturated colours.
    """

    if value < 42:
        return "black"
    if saturation < 35:
        if value >= 190:
            return "white"
        if value <= 55:
            return "black"
        return "grey"

    hue = int(hue) % 180
    if 5 <= hue <= 25 and value < 115:
        return "brown"
    if hue < 8 or hue >= 170:
        return "red"
    if hue < 22:
        return "orange"
    if hue < 34:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 100:
        return "cyan"
    if hue < 130:
        return "blue"
    if hue < 155:
        return "purple"
    return "pink"


def classify_colour(hue: int, saturation: int, value: int) -> str:
    """Public alias for the canonical HSV colour classifier."""

    return _hsv_to_name(hue, saturation, value)


def _round(value: Any, digits: int = 3) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return round(number, digits)


def public_attributes(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only the user-facing simple properties.

    Track metadata nests extracted attributes under ``"attributes"``; older
    snapshots may store them at the top level.  Both shapes are accepted.
    """

    if not attrs:
        return {}
    merged: dict[str, Any] = dict(attrs)
    nested = attrs.get("attributes")
    if isinstance(nested, Mapping):
        merged.update(nested)
    result: dict[str, Any] = {}
    for key in PUBLIC_ATTRIBUTE_FIELDS:
        value = merged.get(key)
        if value is None or value == "":
            continue
        if key in {"colour_confidence", "aspect_ratio"}:
            value = _round(value, 2)
        elif key == "relative_area":
            value = _round(value, 4)
        result[key] = value
    return result


def merge_attributes(
    previous: Mapping[str, Any] | None,
    incoming: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fold a new observation into an existing attribute set.

    A single frame's dominant colour is noisy when an object is partly
    occluded.  Keep the established colour unless the new sample is clearly
    more confident; geometry/size facts are allowed to update immediately.
    """

    merged = dict(previous or {})
    if not incoming:
        return merged
    old_colour = merged.get("colour")
    new_colour = incoming.get("colour")
    if old_colour and new_colour and old_colour != new_colour:
        old_conf = _as_float(merged.get("colour_confidence"), 0.0)
        new_conf = _as_float(incoming.get("colour_confidence"), 0.0)
        if new_conf < old_conf + 0.12:
            # Keep the previous colour (and its swatch/secondary colour).
            incoming = {
                key: value
                for key, value in incoming.items()
                if not str(key).startswith(("colour", "secondary_colour"))
            }
    merged.update(dict(incoming))
    return merged


def attribute_value(attrs: Mapping[str, Any] | None, field: str) -> Any:
    """Read one public attribute value, accepting nested metadata."""

    return public_attributes(attrs).get(field)


def attribute_summary(attrs: Mapping[str, Any] | None, *, limit: int = 4) -> str:
    """Compact ``red · wide · large`` summary for tables and graph tooltips."""

    values = public_attributes(attrs)
    parts: list[str] = []
    colour = values.get("colour")
    if colour:
        parts.append(str(colour))
    for key in ("shape", "size", "texture", "brightness"):
        value = values.get(key)
        if value and str(value) not in parts:
            parts.append(str(value))
    return " · ".join(parts[: max(1, limit)])


def describe_attribute(attrs: Mapping[str, Any] | None, field: str) -> str | None:
    """Render one property as a short natural-language phrase."""

    values = public_attributes(attrs)
    if field == "colour":
        colour = values.get("colour")
        if not colour:
            return None
        phrase = str(colour)
        if values.get("secondary_colour"):
            phrase += f" with {values['secondary_colour']} tones"
        swatch = values.get("colour_hex")
        if swatch:
            phrase += f" ({swatch})"
        return phrase
    if field == "shape":
        shape = values.get("shape")
        if not shape:
            return None
        ratio = values.get("aspect_ratio")
        return f"{shape} (aspect ratio {ratio}:1)" if ratio else str(shape)
    if field == "size":
        size = values.get("size")
        if not size:
            return None
        area = values.get("relative_area")
        if area:
            return f"{size} (about {float(area) * 100:.1f}% of the frame)"
        return str(size)
    if field == "brightness":
        return values.get("brightness")
    if field == "saturation":
        return values.get("saturation")
    if field == "texture":
        return values.get("texture")
    if field == "position":
        horizontal = values.get("position")
        vertical = values.get("vertical_band")
        if horizontal and vertical:
            return f"{horizontal} of frame, {vertical}"
        return horizontal or vertical
    if field in {"all", "appearance"}:
        summary = attribute_summary(values, limit=8)
        return summary or None
    return None


def attribute_query_field(query: str) -> str | None:
    """Return the canonical property an appearance question asks about.

    Returns ``None`` for location/action questions and ``"all"`` for a broad
    "describe the cup" request.  Colour questions are matched first so
    "what colour is the red cup" resolves to ``colour``.
    """

    lower = (query or "").lower()
    if not lower.strip():
        return None
    if any(word in lower for word in _APPEARANCE_WORDS):
        return "all"
    for field in ATTRIBUTE_PROPERTIES:
        if any(re.search(rf"\b{re.escape(word)}\b", lower) for word in _PROPERTY_SYNONYMS[field]):
            return field
    return None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "ATTRIBUTE_PROPERTIES",
    "COLOUR_WORDS",
    "PUBLIC_ATTRIBUTE_FIELDS",
    "attribute_query_field",
    "attribute_summary",
    "attribute_value",
    "classify_colour",
    "describe_attribute",
    "merge_attributes",
    "public_attributes",
]

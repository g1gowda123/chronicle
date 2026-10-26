"""Canonical event contract for the Chronicle dashboard.

Role A/B/C only need to emit these small JSON objects (over HTTP, JSONL, or the
in-process bus).  The normaliser below is deliberately forgiving so that an
evolving pipeline can keep working while teammates iterate.

Canonical envelope::

    {
      "v": 1,
      "kind": "entity | state | edge | zone | interaction | telemetry | metrics | session",
      "t": 12.4,                 # presentation time in seconds
      "data": { ... }
    }

Docs: docs/EVENT_CONTRACT.md
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #

STATES: tuple[str, ...] = (
    "NEW",
    "VISIBLE_STATIC",
    "INTERACTED",
    "VISIBLE_MOVING",
    "HELD",
    "CONTAINED",
    "OCCLUDED",
    "OUT_OF_VIEW",
    "LOST",
)

STATE_COLORS: dict[str, str] = {
    "NEW": "#93a4b8",
    "VISIBLE_STATIC": "#35d07f",
    "INTERACTED": "#f5b942",
    "VISIBLE_MOVING": "#38bdf8",
    "HELD": "#c084fc",
    "CONTAINED": "#2dd4bf",
    "OCCLUDED": "#7c8ba1",
    "OUT_OF_VIEW": "#fb923c",
    "LOST": "#f26d6d",
}

#: States that mean "the user can no longer see this object right now".
OFFSCREEN_STATES = frozenset({"OCCLUDED", "OUT_OF_VIEW", "LOST"})

#: States which imply a live hand interaction.
ATTENTION_STATES = frozenset({"INTERACTED", "HELD", "CONTAINED"})

TERMINAL_STATES = frozenset({"LOST"})

STATE_ALIASES: dict[str, str] = {
    "NEW": "NEW",
    "DETECTED": "NEW",
    "CANDIDATE": "NEW",
    "VISIBLE_STATIC": "VISIBLE_STATIC",
    "STATIC": "VISIBLE_STATIC",
    "VISIBLE": "VISIBLE_STATIC",
    "STATIONARY": "VISIBLE_STATIC",
    "RESTING": "VISIBLE_STATIC",
    "VISIBLE_MOVING": "VISIBLE_MOVING",
    "MOVING": "VISIBLE_MOVING",
    "INTERACTED": "INTERACTED",
    "INTERACTING": "INTERACTED",
    "TOUCHED": "INTERACTED",
    "HELD": "HELD",
    "HOLDING": "HELD",
    "IN_HAND": "HELD",
    "GRASPED": "HELD",
    "CONTAINED": "CONTAINED",
    "CONTAINED_IN": "CONTAINED",
    "INSIDE": "CONTAINED",
    "OCCLUDED": "OCCLUDED",
    "OCCLUSION": "OCCLUDED",
    "OUT_OF_VIEW": "OUT_OF_VIEW",
    "OUT_OF_FRAME": "OUT_OF_VIEW",
    "OFFSCREEN": "OUT_OF_VIEW",
    "LOST": "LOST",
    "DORMANT": "LOST",
    "MISSING": "LOST",
}

CHANNEL_COLORS: dict[str, str] = {
    "GEOM": "#38bdf8",
    "TRACK": "#a78bfa",
    "STATE": "#fbbf24",
    "VLM": "#e879f9",
    "GRAPH": "#34d399",
    "QUERY": "#f472b6",
    "SYSTEM": "#94a3b8",
}

CHANNEL_ALIASES: dict[str, str] = {
    "GEOMETRY": "GEOM",
    "MOTION": "GEOM",
    "CAMERA": "GEOM",
    "PERCEPTION": "GEOM",
    "TRACKING": "TRACK",
    "REID": "TRACK",
    "DETECTION": "TRACK",
    "STATE_MACHINE": "STATE",
    "LIFECYCLE": "STATE",
    "SEMANTIC": "VLM",
    "FLORENCE": "VLM",
    "CLIP": "VLM",
    "KNOWLEDGE": "GRAPH",
    "SQLITE": "GRAPH",
    "NETWORKX": "GRAPH",
    "RAG": "QUERY",
    "QUERY_ENGINE": "QUERY",
}

KIND_ALIASES: dict[str, str] = {
    "session": "session",
    "hello": "session",
    "start": "session",
    "metadata": "session",
    "video": "session",
    "zone": "zone",
    "place": "zone",
    "zone_upsert": "zone",
    "room": "zone",
    "anchor": "zone",
    "entity": "entity",
    "entity_upsert": "entity",
    "detection": "entity",
    "track": "entity",
    "object": "entity",
    "state": "state",
    "state_change": "state",
    "state_transition": "state",
    "transition": "state",
    "edge": "edge",
    "relation": "edge",
    "edge_upsert": "edge",
    "interaction": "interaction",
    "action": "interaction",
    "vlm": "interaction",
    "metrics": "metrics",
    "metric": "metrics",
    "telemetry": "telemetry",
    "log": "telemetry",
    "log_entry": "telemetry",
    "trace": "telemetry",
}

_ENVELOPE_KEYS = {
    "v",
    "version",
    "kind",
    "type",
    "event",
    "event_type",
    "t",
    "ts",
    "timestamp",
    "time",
    "t_event",
    "data",
    "payload",
    "detail",
    "seq",
    "source",
    "origin",
}

_TAG_RE = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------- #
# Parsed event
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Event:
    """A validated, normalised pipeline event."""

    kind: str
    t: float
    data: dict[str, Any]
    seq: int = 0
    wall: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def entity_id(self) -> str | None:
        return _entity_ref(self.data)

    @property
    def channel(self) -> str:
        return normalise_channel(self.data.get("channel") or self.data.get("tag"))

    @property
    def message(self) -> str:
        for key in ("message", "msg", "text", "line"):
            value = self.data.get(key)
            if value:
                return str(value)
        return ""


def normalise_state(value: Any, default: str = "NEW") -> str:
    if value is None:
        return default
    key = re.sub(r"[\s\-]+", "_", str(value).strip()).upper()
    return STATE_ALIASES.get(key, default)


def normalise_channel(value: Any) -> str:
    if value is None:
        return "SYSTEM"
    key = re.sub(r"[\s\-\[\]]+", "_", str(value).strip()).upper().strip("_")
    if key in CHANNEL_COLORS:
        return key
    return CHANNEL_ALIASES.get(key, "SYSTEM")


def coerce_time(value: Any, default: float = 0.0) -> float:
    """Convert seconds / milliseconds / numeric strings to float seconds."""
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    # Unix epoch milliseconds / microseconds are common in logs.
    magnitude = abs(number)
    if magnitude > 1e15:
        return number / 1e6
    if magnitude > 1e10:
        return number / 1e3
    return number


def coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def coerce_bbox(
    value: Any,
    frame_width: float = 1280.0,
    frame_height: float = 720.0,
) -> tuple[float, float, float, float] | None:
    """Return a clamped, normalised ``(x1, y1, x2, y2)`` box.

    Accepts dicts (``x1/y1/x2/y2``, ``left/top/right/bottom`` or ``x/y/w/h``)
    and 4-item sequences in either normalised or pixel coordinates.
    """
    if not value:
        return None
    if isinstance(value, dict):
        lower = {str(k).lower(): v for k, v in value.items()}
        if {"x1", "y1", "x2", "y2"} <= lower.keys():
            box = [lower["x1"], lower["y1"], lower["x2"], lower["y2"]]
        elif {"xmin", "ymin", "xmax", "ymax"} <= lower.keys():
            box = [lower["xmin"], lower["ymin"], lower["xmax"], lower["ymax"]]
        elif {"left", "top", "right", "bottom"} <= lower.keys():
            box = [lower["left"], lower["top"], lower["right"], lower["bottom"]]
        elif {"x", "y", "w", "h"} <= lower.keys():
            box = [
                lower["x"],
                lower["y"],
                float(lower["x"]) + float(lower["w"]),
                float(lower["y"]) + float(lower["h"]),
            ]
        else:
            return None
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        box = list(value[:4])
    else:
        return None

    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    # Auto-detect pixel coordinates (any coordinate beyond the unit square).
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        x1 /= max(frame_width, 1.0)
        x2 /= max(frame_width, 1.0)
        y1 /= max(frame_height, 1.0)
        y2 /= max(frame_height, 1.0)

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = (min(max(v, 0.0), 1.0) for v in (x1, x2))
    y1, y2 = (min(max(v, 0.0), 1.0) for v in (y1, y2))
    if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
        return None
    return (x1, y1, x2, y2)


def clean_text(value: Any, limit: int = 400) -> str:
    """Strip tags/newlines from untrusted pipeline text before HTML rendering."""
    if value is None:
        return ""
    text = _TAG_RE.sub("", str(value)).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def pretty_ref(ref: Any) -> str:
    """Human label for an id such as ``keys_1`` -> ``keys``."""
    text = str(ref or "").strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+\d+$", "", text)
    return text.strip().title() or "Unknown"


def _entity_ref(data: dict[str, Any]) -> str | None:
    for key in ("entity_id", "entity", "object_id", "track_id", "item_id", "id"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    node = data.get("node")
    if isinstance(node, dict) and node.get("id") is not None:
        return str(node["id"])
    return None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def parse_event(raw: Any, seq: int = 0) -> Event | None:
    """Normalise one pipeline payload into an :class:`Event` (or ``None``)."""
    if not isinstance(raw, dict):
        return None

    kind_raw = raw.get("kind") or raw.get("type") or raw.get("event") or raw.get("event_type")
    data = raw.get("data") or raw.get("payload") or raw.get("detail")
    if not isinstance(data, dict):
        data = {}
    else:
        data = dict(data)

    # Allow flat payloads: everything outside the envelope is event data.
    for key, value in raw.items():
        if key not in _ENVELOPE_KEYS and key not in data:
            data[key] = value

    if kind_raw is None:
        if data.get("channel") is not None or raw.get("message") is not None:
            kind = "telemetry"
        elif _entity_ref(data) is not None and data.get("state") is not None:
            kind = "state"
        elif _entity_ref(data) is not None:
            kind = "entity"
        else:
            return None
    else:
        kind = KIND_ALIASES.get(str(kind_raw).strip().lower(), str(kind_raw).strip().lower())
        if kind not in KIND_ALIASES.values():
            return None

    raw_t = raw.get("t", raw.get("ts", raw.get("timestamp", raw.get("time", raw.get("t_event")))))
    if raw_t is None:
        raw_t = data.get("t_event", data.get("t", data.get("timestamp")))
    t = coerce_time(raw_t, default=0.0)

    if kind == "telemetry":
        data["channel"] = normalise_channel(data.get("channel") or data.get("tag") or raw.get("channel"))
        data["message"] = clean_text(
            data.get("message") or data.get("msg") or data.get("text") or data.get("line") or "",
            limit=800,
        )
        if not data["message"]:
            return None

    if kind == "entity" and _entity_ref(data) is None:
        return None
    if kind == "state":
        data["to"] = normalise_state(data.get("to") or data.get("state") or data.get("to_state"))
        if _entity_ref(data) is None:
            return None
    if kind == "edge":
        if not (data.get("source") or data.get("subject") or data.get("from_entity")):
            return None
        if not (data.get("target") or data.get("object") or data.get("to_entity")):
            return None
        if not (data.get("relation") or data.get("predicate") or data.get("type")):
            return None

    return Event(kind=kind, t=t, data=data, seq=seq, raw=raw)

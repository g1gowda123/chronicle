"""Thread-safe in-memory mirror of the Chronicle dual graph.

The dashboard keeps its own lightweight projection of the pipeline's SQLite /
NetworkX state so that rendering never blocks ingestion and so the UI can be
scrubbed through presentation time.
"""

from __future__ import annotations

import bisect
import html
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from .contracts import (
    ATTENTION_STATES,
    OFFSCREEN_STATES,
    STATE_COLORS,
    TERMINAL_STATES,
    Event,
    clean_text,
    coerce_bbox,
    coerce_float,
    normalise_state,
    pretty_ref,
)

# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class StateRecord:
    from_state: str
    to_state: str
    t: float
    reason: str = ""
    trigger: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_state,
            "to": self.to_state,
            "t": round(self.t, 3),
            "reason": self.reason,
            "trigger": self.trigger,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class BboxSample:
    t: float
    bbox: tuple[float, float, float, float]
    confidence: float | None = None


@dataclass(slots=True)
class InteractionRecord:
    t: float
    action: str
    caption: str = ""
    clip_sim: float | None = None
    strip_url: str | None = None
    crop_url: str | None = None
    model: str = ""
    latency_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "action": self.action,
            "caption": self.caption,
            "clip_sim": self.clip_sim,
            "strip_url": self.strip_url,
            "crop_url": self.crop_url,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "extra": self.extra,
        }


@dataclass(slots=True)
class Entity:
    id: str
    label: str
    cls: str = ""
    state: str = "NEW"
    confidence: float = 0.0
    clip_sim: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    crop_url: str | None = None
    zone_id: str | None = None
    track_id: str | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    state_since: float = 0.0
    is_stub: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)
    history: list[StateRecord] = field(default_factory=list)
    bbox_samples: list[BboxSample] = field(default_factory=list)
    interactions: list[InteractionRecord] = field(default_factory=list)

    # -- derived helpers -------------------------------------------------- #

    def state_at(self, t: float) -> str:
        if not self.history:
            return self.state
        idx = bisect.bisect_right([r.t for r in self.history], t + 1e-6) - 1
        return self.history[idx].to_state if idx >= 0 else self.history[0].from_state or "NEW"

    def bbox_at(self, t: float, tol: float = 1.2) -> tuple[float, float, float, float] | None:
        if not self.bbox_samples:
            return self.bbox
        times = [s.t for s in self.bbox_samples]
        idx = bisect.bisect_right(times, t + 0.08) - 1
        if idx < 0:
            if self.bbox_samples[0].t - t <= tol:
                return self.bbox_samples[0].bbox
            return None
        sample = self.bbox_samples[idx]
        if t - sample.t > tol:
            return None
        return sample.bbox

    def last_interaction_t(self) -> float | None:
        candidates: list[float] = []
        for record in self.history:
            if record.to_state in ATTENTION_STATES:
                candidates.append(record.t)
        for action in self.interactions:
            candidates.append(action.t)
        return max(candidates) if candidates else None

    def is_active_at(self, t: float) -> bool:
        if self.first_seen - 0.05 > t:
            return False
        state = self.state_at(t)
        if state in ATTENTION_STATES:
            return True
        return t - self.last_seen <= 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "cls": self.cls,
            "state": self.state,
            "confidence": self.confidence,
            "clip_sim": self.clip_sim,
            "bbox": self.bbox,
            "crop_url": self.crop_url,
            "zone_id": self.zone_id,
            "track_id": self.track_id,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "state_since": round(self.state_since, 3),
            "is_stub": self.is_stub,
            "attributes": self.attributes,
            "history": [r.as_dict() for r in self.history],
            "interactions": [i.as_dict() for i in self.interactions],
            "last_interaction_t": self.last_interaction_t(),
        }


@dataclass(slots=True)
class Edge:
    id: str
    source: str
    target: str
    relation: str
    t_start: float
    t_end: float | None = None
    clip_sim: float | None = None
    active: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def active_at(self, t: float) -> bool:
        if self.t_start > t + 1e-6:
            return False
        return self.t_end is None or t <= self.t_end + 1e-6

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "t_start": round(self.t_start, 3),
            "t_end": None if self.t_end is None else round(self.t_end, 3),
            "clip_sim": self.clip_sim,
            "active": self.active,
            "meta": self.meta,
        }


@dataclass(slots=True)
class Zone:
    id: str
    label: str
    clip_sim: float | None = None
    keyframe_url: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    polygon: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    is_furniture: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "clip_sim": self.clip_sim,
            "keyframe_url": self.keyframe_url,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "is_furniture": self.is_furniture,
            "attributes": self.attributes,
        }


@dataclass(slots=True)
class LogRecord:
    seq: int
    t: float
    channel: str
    level: str
    message: str
    entity_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    wall: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "t": self.t,
            "channel": self.channel,
            "level": self.level,
            "message": self.message,
            "entity_id": self.entity_id,
            "data": self.data,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    try:
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        r, g, b = 148, 163, 184
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


ACTION_RELATIONS = {
    "PICKED_UP",
    "PICK_UP",
    "PUT_DOWN",
    "PLACED_IN",
    "INSERT_INTO",
    "PLACED_ON",
    "TOUCHED",
    "OPENED",
    "CLOSED",
    "USED",
    "MOVED",
}
SPATIAL_RELATIONS = {"RESTING_ON", "INSIDE", "CONTAINED_IN", "SUPPORTS", "ON", "ATTACHED_TO"}


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class MemoryStore:
    """A bounded, time-addressable projection of the episodic memory graph."""

    def __init__(self, *, max_events: int = 6000, max_logs: int = 1600) -> None:
        self._lock = threading.RLock()
        self.entities: dict[str, Entity] = {}
        self.zones: dict[str, Zone] = {}
        self.edges: dict[str, Edge] = {}
        self.anchor_labels: dict[str, str] = {}
        self.session: dict[str, Any] = {
            "title": "Chronicle Episode",
            "video_url": None,
            "duration": 0.0,
            "fps": 30.0,
            "width": 1280,
            "height": 720,
            "source": "unknown",
        }
        self.metrics: dict[str, Any] = {}
        self.metric_history: deque[dict[str, Any]] = deque(maxlen=360)
        self.telemetry: deque[LogRecord] = deque(maxlen=max_logs)
        self.events: deque[Event] = deque(maxlen=max_events)
        self.interactions: deque[InteractionRecord] = deque(maxlen=600)
        self.active_zone_id: str | None = None

        self.revision = 0
        self.graph_revision = 0
        self.pose_revision = 0
        self.log_revision = 0
        self.metrics_revision = 0
        self.session_revision = 0
        self.epoch = 0

        self._log_seq = 0
        self._last_t = 0.0
        self._event_times: deque[float] = deque(maxlen=600)
        self._dropped = 0

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def clear(self, *, keep_session: bool = True) -> None:
        with self._lock:
            self.entities.clear()
            self.zones.clear()
            self.edges.clear()
            self.anchor_labels.clear()
            self.metrics.clear()
            self.metric_history.clear()
            self.telemetry.clear()
            self.events.clear()
            self.interactions.clear()
            self.active_zone_id = None
            self._event_times.clear()
            self._last_t = 0.0
            self._log_seq = 0
            if not keep_session:
                self.session.update(
                    {"title": "Chronicle Episode", "video_url": None, "duration": 0.0, "fps": 30.0}
                )
            self.epoch += 1
            self.revision += 1
            self.graph_revision += 1
            self.pose_revision += 1
            self.log_revision += 1
            self.metrics_revision += 1
            self.session_revision += 1

    def apply(self, event: Event) -> bool:
        """Apply one normalised event. Returns ``False`` if it was ignored."""
        with self._lock:
            handler = getattr(self, f"_apply_{event.kind}", None)
            if handler is None:
                self._dropped += 1
                return False
            try:
                handler(event)
            except Exception:  # a malformed producer must never kill the UI
                self._dropped += 1
                return False

            self.events.append(event)
            self._event_times.append(time.monotonic())
            self._last_t = max(self._last_t, event.t)
            self.revision += 1
            return True

    # -- entity --------------------------------------------------------- #

    def _get_entity(self, entity_id: str, t: float) -> Entity:
        entity = self.entities.get(entity_id)
        if entity is None:
            entity = Entity(
                id=entity_id,
                label=pretty_ref(entity_id),
                first_seen=t,
                last_seen=t,
                state_since=t,
                is_stub=True,
            )
            self.entities[entity_id] = entity
        return entity

    def _apply_entity(self, event: Event) -> None:
        data = event.data
        entity_id = str(
            data.get("entity_id") or data.get("id") or data.get("object_id") or data.get("track_id")
        )
        if not entity_id or entity_id in {"None", "nan"}:
            return
        entity = self._get_entity(entity_id, event.t)
        structural = entity.is_stub

        label = clean_text(data.get("label") or data.get("name") or data.get("class_name"), 60)
        if label:
            if entity.is_stub or label != entity.label:
                entity.label = label
                structural = True
        cls = clean_text(data.get("class") or data.get("category") or data.get("class_name"), 40)
        if cls:
            entity.cls = cls
        zone_id = data.get("zone_id") or data.get("zone") or data.get("anchor_id")
        if zone_id:
            if entity.zone_id != str(zone_id):
                entity.zone_id = str(zone_id)
                structural = True

        confidence = coerce_float(data.get("confidence", data.get("score")))
        if confidence is not None:
            entity.confidence = _clamp01(confidence)
        clip_sim = coerce_float(
            data.get("clip_sim", data.get("similarity", data.get("cosine", data.get("reid_sim"))))
        )
        if clip_sim is not None:
            entity.clip_sim = _clamp01(clip_sim)

        state_value = data.get("state")
        if state_value is not None:
            new_state = normalise_state(state_value, default=entity.state)
            if new_state != entity.state:
                self._append_state(
                    entity,
                    StateRecord(
                        from_state=entity.state,
                        to_state=new_state,
                        t=event.t,
                        reason=clean_text(data.get("reason"), 160) or "entity update",
                        trigger=dict(data.get("trigger") or {}),
                        confidence=entity.confidence or None,
                    ),
                )
                entity.state = new_state
                entity.state_since = event.t
                structural = True

        bbox = coerce_bbox(
            data.get("bbox_norm") or data.get("bbox") or data.get("box") or data.get("xyxy"),
            frame_width=float(self.session.get("width") or 1280),
            frame_height=float(self.session.get("height") or 720),
        )
        if bbox is not None:
            entity.bbox = bbox
            self._append_bbox(entity, BboxSample(event.t, bbox, entity.confidence or None))
            self.pose_revision += 1

        crop_url = data.get("crop_url") or data.get("crop") or data.get("thumbnail_url")
        if crop_url and crop_url != entity.crop_url:
            entity.crop_url = str(crop_url)
            structural = True

        track_id = data.get("track_id")
        if track_id is not None:
            entity.track_id = str(track_id)

        first_seen = coerce_float(data.get("first_seen"))
        if first_seen is not None:
            entity.first_seen = min(entity.first_seen, first_seen) if not entity.is_stub else first_seen
        else:
            entity.first_seen = min(entity.first_seen, event.t)

        last_seen = coerce_float(data.get("last_seen"))
        if last_seen is not None:
            entity.last_seen = max(entity.last_seen, last_seen)
        entity.last_seen = max(entity.last_seen, event.t)

        attributes = data.get("attributes") or data.get("meta")
        if isinstance(attributes, dict):
            entity.attributes.update(attributes)

        if entity.is_stub and (label or clip_sim is not None):
            entity.is_stub = False
            structural = True
        if structural:
            self.graph_revision += 1

    # -- state ---------------------------------------------------------- #

    def _apply_state(self, event: Event) -> None:
        data = event.data
        entity_id = str(
            data.get("entity_id") or data.get("id") or data.get("object_id") or data.get("track_id")
        )
        if not entity_id:
            return
        entity = self._get_entity(entity_id, event.t)
        to_state = normalise_state(data.get("to") or data.get("state") or data.get("to_state"))
        from_state = normalise_state(
            data.get("from") or data.get("from_state") or data.get("old_state"),
            default=entity.state,
        )
        if from_state != entity.state and not entity.history:
            from_state = entity.state
        record = StateRecord(
            from_state=from_state,
            to_state=to_state,
            t=event.t,
            reason=clean_text(data.get("reason") or data.get("trigger_reason"), 220),
            trigger=dict(data.get("trigger") or data.get("evidence") or {}),
            confidence=coerce_float(data.get("confidence")),
        )
        self._append_state(entity, record)
        entity.state = to_state
        entity.state_since = event.t
        entity.last_seen = max(entity.last_seen, event.t)
        if data.get("label"):
            entity.label = clean_text(data["label"], 60)
        if to_state == "VISIBLE_STATIC" and data.get("zone_id"):
            entity.zone_id = str(data["zone_id"])
        self._derive_log(
            event.t,
            "STATE",
            f"{entity.label.upper()} {from_state} → {to_state}"
            + (f" · {record.reason}" if record.reason else ""),
            entity.id,
            from_state=from_state,
            to_state=to_state,
            confidence=record.confidence,
        )
        self.graph_revision += 1

    # -- edges ---------------------------------------------------------- #

    def _apply_edge(self, event: Event) -> None:
        data = event.data
        source = str(data.get("source") or data.get("subject") or data.get("from_entity") or "")
        target = str(data.get("target") or data.get("object") or data.get("to_entity") or "")
        relation = str(data.get("relation") or data.get("predicate") or data.get("type") or "").upper()
        if not source or not target or not relation:
            return
        relation = relation.replace(" ", "_").replace("-", "_")

        t_start = coerce_float(data.get("t_start"))
        if t_start is None:
            t_start = event.t
        t_end = coerce_float(data.get("t_end"))
        active = data.get("active")
        if active is None:
            active = t_end is None
        if t_end is not None and t_end < t_start:
            t_end = t_start
        if not active:
            t_end = t_end if t_end is not None else event.t

        edge_id = data.get("id") or data.get("edge_id")
        if edge_id:
            edge = self.edges.get(str(edge_id))
        else:
            edge = None
            # Match an open, identical relation to avoid duplicate edges.
            for candidate in self.edges.values():
                if (
                    candidate.source == source
                    and candidate.target == target
                    and candidate.relation == relation
                    and candidate.t_end is None
                    and abs(candidate.t_start - t_start) < 0.5
                ):
                    edge = candidate
                    break
        if edge is None:
            edge = Edge(
                id=str(edge_id or f"{source}|{relation}|{target}|{t_start:.3f}"),
                source=source,
                target=target,
                relation=relation,
                t_start=t_start,
            )
        edge.t_start = min(edge.t_start, t_start)
        edge.t_end = t_end
        edge.active = bool(active)
        edge.clip_sim = coerce_float(data.get("clip_sim") or data.get("similarity")) or edge.clip_sim
        meta = data.get("meta") or data.get("attributes")
        if isinstance(meta, dict):
            edge.meta.update(meta)
        self.edges[edge.id] = edge

        # Keep readable labels for anchors that are not tracked entities.
        for ref_key, label_key in (("source", "source_label"), ("target", "target_label")):
            ref = source if ref_key == "source" else target
            label = data.get(label_key)
            if label and ref not in self.entities and ref not in self.zones:
                self.anchor_labels[ref] = clean_text(label, 60)
            elif ref not in self.entities and ref not in self.zones and ref.upper() != "USER":
                self.anchor_labels.setdefault(ref, pretty_ref(ref))

        if edge.relation in ACTION_RELATIONS:
            kind = "action"
        else:
            kind = "spatial"
        suffix = "" if active else " · released"
        self._derive_log(
            event.t,
            "GRAPH",
            f"{kind} edge {pretty_ref(source)} --[{edge.relation}]--> {pretty_ref(target)}"
            f" · t={edge.t_start:.2f}s{suffix}",
            None,
            relation=edge.relation,
            source=source,
            target=target,
        )
        self.graph_revision += 1

    # -- zones ---------------------------------------------------------- #

    def _apply_zone(self, event: Event) -> None:
        data = event.data
        zone_id = str(data.get("id") or data.get("zone_id") or data.get("place_id") or "")
        if not zone_id:
            return
        zone = self.zones.get(zone_id)
        if zone is None:
            zone = Zone(
                id=zone_id,
                label=clean_text(data.get("label") or data.get("name"), 60) or pretty_ref(zone_id),
                first_seen=event.t,
                last_seen=event.t,
            )
            self.zones[zone_id] = zone
        label = clean_text(data.get("label") or data.get("name"), 60)
        if label:
            zone.label = label
        zone.clip_sim = coerce_float(data.get("clip_sim") or data.get("similarity")) or zone.clip_sim
        zone.keyframe_url = str(data.get("keyframe_url") or data.get("keyframe") or zone.keyframe_url or "") or None
        bb = coerce_bbox(data.get("bbox") or data.get("polygon_bbox"))
        if bb:
            zone.bbox = bb
        polygon = data.get("polygon")
        if isinstance(polygon, (list, tuple)):
            points: list[tuple[float, float]] = []
            for point in polygon:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))
            if points:
                zone.polygon = points
        confidence = coerce_float(data.get("confidence"))
        if confidence is not None:
            zone.confidence = _clamp01(confidence)
        zone.is_furniture = bool(data.get("is_furniture") or data.get("furniture"))
        attributes = data.get("attributes") or data.get("meta")
        if isinstance(attributes, dict):
            zone.attributes.update(attributes)
        zone.last_seen = max(zone.last_seen, event.t)
        if data.get("active") or self.active_zone_id is None:
            self.active_zone_id = zone_id
        self.graph_revision += 1

    # -- interactions --------------------------------------------------- #

    def _apply_interaction(self, event: Event) -> None:
        data = event.data
        entity_id = str(
            data.get("entity_id") or data.get("id") or data.get("object_id") or data.get("track_id") or ""
        )
        if not entity_id:
            return
        entity = self._get_entity(entity_id, event.t)
        action = clean_text(
            data.get("action") or data.get("action_label") or data.get("verb") or "INTERACTION", 48
        ).upper().replace(" ", "_")
        record = InteractionRecord(
            t=event.t,
            action=action,
            caption=clean_text(data.get("caption") or data.get("description"), 500),
            clip_sim=coerce_float(data.get("clip_sim") or data.get("similarity") or data.get("score")),
            strip_url=str(data.get("strip_url") or "") or None,
            crop_url=str(data.get("crop_url") or "") or None,
            model=clean_text(data.get("model") or "", 40),
            latency_ms=coerce_float(data.get("latency_ms")),
            extra=dict(data.get("extra") or {}),
        )
        entity.interactions.append(record)
        entity.interactions.sort(key=lambda item: item.t)
        entity.interactions = entity.interactions[-40:]
        if record.crop_url:
            entity.crop_url = record.crop_url
        if data.get("label"):
            entity.label = clean_text(data["label"], 60)
        if data.get("state"):
            new_state = normalise_state(data["state"])
            if new_state != entity.state:
                self._append_state(
                    entity,
                    StateRecord(
                        from_state=entity.state,
                        to_state=new_state,
                        t=event.t,
                        reason=f"VLM action {action}",
                        confidence=record.clip_sim,
                    ),
                )
                entity.state = new_state
                entity.state_since = event.t
        self.interactions.append(record)
        self._derive_log(
            event.t,
            "VLM",
            f"{action} · {record.caption or 'no caption'}"
            + (f" · CLIP {record.clip_sim:.2f}" if record.clip_sim is not None else "")
            + (f" · {record.latency_ms:.0f}ms" if record.latency_ms is not None else ""),
            entity.id,
            action=action,
            clip_sim=record.clip_sim,
            model=record.model,
            strip_url=record.strip_url,
        )
        self.graph_revision += 1

    # -- metrics -------------------------------------------------------- #

    def _apply_metrics(self, event: Event) -> None:
        data = dict(event.data)
        data.pop("channel", None)
        camera = data.get("camera")
        if isinstance(camera, dict):
            self.metrics["camera"] = {**self.metrics.get("camera", {}), **camera}
            data.pop("camera", None)
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                self.metrics[key] = value
        self.metrics["t"] = event.t
        self.metric_history.append(
            {
                "t": event.t,
                "event_rate": self.event_rate(),
                "fps": self.metrics.get("fps"),
                "latency_ms": self.metrics.get("latency_ms"),
            }
        )
        self.metrics_revision += 1

    # -- session -------------------------------------------------------- #

    def _apply_session(self, event: Event) -> None:
        data = event.data
        allowed = {
            "title",
            "video_url",
            "duration",
            "fps",
            "width",
            "height",
            "source",
            "episode",
            "operator",
            "notes",
        }
        changed = False
        for key in allowed:
            if key in data and data[key] is not None:
                value = data[key]
                if key in {"duration", "fps", "width", "height"}:
                    number = coerce_float(value)
                    if number is None:
                        continue
                    value = number
                else:
                    value = clean_text(value, 200)
                if self.session.get(key) != value:
                    self.session[key] = value
                    changed = True
        if changed:
            self.session_revision += 1

    # -- telemetry ------------------------------------------------------ #

    def _apply_telemetry(self, event: Event) -> None:
        data = event.data
        self._log_seq += 1
        record = LogRecord(
            seq=self._log_seq,
            t=event.t,
            channel=str(data.get("channel") or "SYSTEM"),
            level=str(data.get("level") or "info").lower(),
            message=clean_text(data.get("message"), 800),
            entity_id=data.get("entity_id") or data.get("object_id"),
            data={k: v for k, v in data.items() if k not in {"message", "channel", "level"}},
        )
        self.telemetry.append(record)
        self.log_revision += 1

    # ------------------------------------------------------------------ #
    # Internal list helpers
    # ------------------------------------------------------------------ #

    def _derive_log(
        self,
        t: float,
        channel: str,
        message: str,
        entity_id: str | None = None,
        *,
        level: str = "info",
        **data: Any,
    ) -> None:
        """Mirror a semantic event into the telemetry stream.

        Producers are not required to emit a telemetry event for every state
        change; the dashboard synthesises a readable trace so the log stays
        useful even for a minimal integration.
        """
        self._log_seq += 1
        self.telemetry.append(
            LogRecord(
                seq=self._log_seq,
                t=t,
                channel=channel,
                level=level,
                message=clean_text(message, 800),
                entity_id=entity_id,
                data={**data, "derived": True},
            )
        )
        self.log_revision += 1

    @staticmethod
    def _append_state(entity: Entity, record: StateRecord) -> None:
        if (
            entity.history
            and entity.history[-1].to_state == record.to_state
            and abs(entity.history[-1].t - record.t) < 0.01
        ):
            entity.history[-1] = record
            return
        entity.history.append(record)
        if len(entity.history) > 1 and entity.history[-1].t < entity.history[-2].t:
            entity.history.sort(key=lambda item: item.t)
        if len(entity.history) > 120:
            del entity.history[:-120]

    @staticmethod
    def _append_bbox(entity: Entity, sample: BboxSample) -> None:
        if entity.bbox_samples:
            last = entity.bbox_samples[-1]
            if abs(last.t - sample.t) < 0.04:
                entity.bbox_samples[-1] = sample
                return
        entity.bbox_samples.append(sample)
        if len(entity.bbox_samples) > 900:
            del entity.bbox_samples[:100]

    # ------------------------------------------------------------------ #
    # Queries / projections
    # ------------------------------------------------------------------ #

    def event_rate(self, window: float = 5.0) -> float:
        now = time.monotonic()
        while self._event_times and now - self._event_times[0] > window:
            self._event_times.popleft()
        return len(self._event_times) / window if window else 0.0

    def latest_t(self) -> float:
        with self._lock:
            return self._last_t

    def note(
        self,
        channel: str,
        message: str,
        t: float,
        entity_id: str | None = None,
        **data: Any,
    ) -> None:
        """Append a dashboard-authored line to the telemetry stream."""
        with self._lock:
            self._derive_log(t, channel, message, entity_id, **data)

    def node_key(self, ref: str) -> str:
        if ref in self.entities:
            return ref
        if ref in self.zones:
            return f"zone:{ref}"
        if str(ref).upper() in {"USER", "ME", "PERSON", "OPERATOR"}:
            return "USER"
        return f"anchor:{ref}"

    def resolve_node(self, node_key: str) -> tuple[str, str] | None:
        """Return ``(kind, real_id)`` for a Vis node id."""
        if node_key == "USER":
            return ("user", "USER")
        if node_key.startswith("zone:"):
            return ("zone", node_key[5:])
        if node_key.startswith("anchor:"):
            return ("anchor", node_key[7:])
        if node_key in self.entities:
            return ("entity", node_key)
        return None

    def boxes_at(self, t: float) -> list[dict[str, Any]]:
        with self._lock:
            boxes: list[dict[str, Any]] = []
            for entity in self.entities.values():
                if entity.first_seen - 0.25 > t:
                    continue
                bbox = entity.bbox_at(t, tol=1.1)
                if bbox is None:
                    continue
                state = entity.state_at(t)
                boxes.append(
                    {
                        "id": entity.id,
                        "label": entity.label,
                        "state": state,
                        "color": STATE_COLORS.get(state, "#94a3b8"),
                        "bbox": [round(v, 5) for v in bbox],
                        "confidence": entity.confidence,
                        "clip_sim": entity.clip_sim,
                        "held": state in ATTENTION_STATES,
                    }
                )
            return boxes

    def entity_detail(self, entity_id: str, t: float) -> dict[str, Any] | None:
        with self._lock:
            entity = self.entities.get(entity_id)
            if entity is None:
                return None
            state = entity.state_at(t)
            spatial = [
                edge.as_dict()
                for edge in self.edges.values()
                if edge.source == entity_id or edge.target == entity_id
            ]
            spatial.sort(key=lambda item: item["t_start"], reverse=True)
            zone = self.zones.get(entity.zone_id or "")
            payload = entity.as_dict()
            payload.update(
                {
                    "state_at_playhead": state,
                    "state_color": STATE_COLORS.get(state, "#94a3b8"),
                    "zone": zone.as_dict() if zone else None,
                    "edges": spatial,
                    "action_edges": [e for e in spatial if e["relation"] in ACTION_RELATIONS],
                    "spatial_edges": [e for e in spatial if e["relation"] in SPATIAL_RELATIONS],
                    "last_interaction_t": entity.last_interaction_t(),
                }
            )
            return payload

    def zone_detail(self, zone_id: str) -> dict[str, Any] | None:
        with self._lock:
            zone = self.zones.get(zone_id)
            if zone is None:
                return None
            entities = [e.as_dict() for e in self.entities.values() if e.zone_id == zone_id]
            furniture = any(e.is_furniture for e in self.zones.values())
            return {**zone.as_dict(), "entities": entities, "has_furniture": furniture}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            entities = list(self.entities.values())
            states = Counter(e.state for e in entities)
            active = [e for e in entities if e.state not in OFFSCREEN_STATES]
            camera = self.metrics.get("camera") or {}
            gated = self.metrics.get("gated_frames")
            frames = self.metrics.get("frames")
            if isinstance(gated, (int, float)) and isinstance(frames, (int, float)) and frames:
                gating_ratio = min(max(float(gated) / float(frames), 0.0), 1.0)
            else:
                gating_ratio = None
            lost = [e for e in entities if e.state_at(self._last_t) == "LOST"]
            return {
                "entities": len(entities),
                "active_entities": len(active),
                "zones": len(self.zones),
                "edges": len(self.edges),
                "long_term_anchors": len([e for e in self.edges.values() if e.relation in SPATIAL_RELATIONS]),
                "states": dict(states),
                "lost": len(lost),
                "camera": camera,
                "gating_ratio": gating_ratio,
                "vlm_calls": int(self.metrics.get("vlm_calls") or len(self.interactions)),
                "fps": self.metrics.get("fps"),
                "latency_ms": self.metrics.get("latency_ms"),
                "event_rate": self.event_rate(),
                "last_latency_ms": self.metrics.get("last_latency_ms"),
                "track_count": self.metrics.get("track_count", len(entities)),
                "detections": self.metrics.get("detections"),
                "frames": frames,
                "gated_frames": gated,
                "active_zone": self.active_zone_id,
                "epoch": self.epoch,
            }

    # -- graph projection ---------------------------------------------- #

    def graph_snapshot(
        self,
        t: float,
        *,
        mode: str = "live",
        include_zones: bool = True,
        max_nodes: int = 220,
        max_edges: int = 360,
    ) -> dict[str, Any]:
        """Return a Vis.js-ready projection of the graph at presentation time *t*.

        ``mode='live'`` only shows entities/relations that exist at ``t``.
        ``mode='full'`` shows the whole episode, with future nodes dimmed.
        States are always resolved at ``t``.
        """
        with self._lock:
            full = mode == "full"
            nodes: dict[str, dict[str, Any]] = {}
            edge_payloads: list[dict[str, Any]] = []
            entity_nodes: dict[str, str] = {}

            # ---- entities -------------------------------------------------- #
            for entity in self.entities.values():
                future = entity.first_seen - 0.05 > t
                if future and not full:
                    continue
                state = "NEW" if future else entity.state_at(t)
                active = (not future) and entity.is_active_at(t)
                if full and not active and state not in ATTENTION_STATES and not future:
                    active = False
                color = STATE_COLORS.get(state, "#94a3b8")
                opacity = 0.32 if future else 1.0
                size = 15 + min(len(entity.interactions) * 2.5, 9)
                title = self._entity_tooltip(entity, state, t)
                node_id = entity.id
                entity_nodes[entity.id] = node_id
                nodes[node_id] = {
                    "id": node_id,
                    "label": entity.label,
                    "kind": "entity",
                    "state": state,
                    "shape": "dot",
                    "size": size,
                    "value": size,
                    "color": {
                        "background": _rgba(color, 0.92 if active else 0.55),
                        "border": color,
                        "highlight": {"background": color, "border": "#e2f5ff"},
                        "hover": {"background": color, "border": "#e2f5ff"},
                    },
                    "borderWidth": 3 if active else 1.5,
                    "opacity": opacity,
                    "font": {"color": "#dbe7f3" if active else "#8fa3b8", "size": 12, "face": "ui-monospace, monospace"},
                    "title": title,
                    "active": active,
                    "dim": future,
                    "entity_id": entity.id,
                    "clip_sim": entity.clip_sim,
                    "confidence": entity.confidence,
                    "future": future,
                }

            # ---- zones ----------------------------------------------------- #
            if include_zones:
                for zone in self.zones.values():
                    if not full and zone.first_seen - 0.05 > t:
                        continue
                    active = zone.id == self.active_zone_id
                    nodes[f"zone:{zone.id}"] = {
                        "id": f"zone:{zone.id}",
                        "label": zone.label.upper(),
                        "kind": "zone",
                        "state": "ANCHOR",
                        "shape": "box",
                        "size": 26,
                        "value": 26,
                        "color": {
                            "background": _rgba("#22d3ee", 0.13 if active else 0.06),
                            "border": _rgba("#22d3ee", 0.85 if active else 0.35),
                            "highlight": {"background": _rgba("#22d3ee", 0.22), "border": "#a5f3fc"},
                            "hover": {"background": _rgba("#22d3ee", 0.2), "border": "#a5f3fc"},
                        },
                        "borderWidth": 1,
                        "shapeProperties": {"borderDashes": [4, 3]},
                        "opacity": 1.0,
                        "font": {
                            "color": "#7dd3fc" if active else "#5c8ba0",
                            "size": 14,
                            "face": "ui-monospace, monospace",
                            "vadjust": -34,
                        },
                        "title": self._zone_tooltip(zone),
                        "active": active,
                        "dim": False,
                        "zone_id": zone.id,
                    }

            # ---- user / anchors ------------------------------------------- #
            user_present = False
            anchor_refs: set[str] = set()
            for edge in self.edges.values():
                if not full and edge.t_start - 0.05 > t:
                    continue
                for ref in (edge.source, edge.target):
                    key = self.node_key(ref)
                    if key == "USER":
                        user_present = True
                    elif key.startswith("anchor:"):
                        anchor_refs.add(ref)
            if user_present:
                nodes["USER"] = {
                    "id": "USER",
                    "label": "YOU",
                    "kind": "user",
                    "state": "AGENT",
                    "shape": "star",
                    "size": 22,
                    "value": 22,
                    "color": {
                        "background": _rgba("#f472b6", 0.9),
                        "border": "#fbcfe8",
                        "highlight": {"background": "#f472b6", "border": "#fff"},
                        "hover": {"background": "#f472b6", "border": "#fff"},
                    },
                    "borderWidth": 2,
                    "opacity": 1.0,
                    "font": {"color": "#fbcfe8", "size": 12, "face": "ui-monospace, monospace"},
                    "title": "<b>YOU</b><br><span style='color:#94a3b8'>egocentric agent</span>",
                    "active": True,
                    "dim": False,
                }
            for ref in anchor_refs:
                node_id = f"anchor:{ref}"
                label = self.anchor_labels.get(ref) or pretty_ref(ref)
                nodes[node_id] = {
                    "id": node_id,
                    "label": label,
                    "kind": "anchor",
                    "state": "ANCHOR",
                    "shape": "box",
                    "size": 18,
                    "value": 18,
                    "color": {
                        "background": _rgba("#64748b", 0.22),
                        "border": _rgba("#94a3b8", 0.6),
                        "highlight": {"background": _rgba("#94a3b8", 0.4), "border": "#e2e8f0"},
                    },
                    "borderWidth": 1,
                    "opacity": 1.0,
                    "font": {"color": "#94a3b8", "size": 11, "face": "ui-monospace, monospace"},
                    "title": f"<b>{html.escape(label)}</b><br><span style='color:#94a3b8'>anchor</span>",
                    "active": True,
                    "dim": False,
                }

            # ---- edges ----------------------------------------------------- #
            # For action multi-edges, keep the most recent started per triple.
            latest_action: dict[tuple[str, str, str], Edge] = {}
            spatial_edges: list[Edge] = []
            for edge in sorted(self.edges.values(), key=lambda e: e.t_start):
                if not full and edge.t_start - 0.05 > t:
                    continue
                if edge.source == edge.target:
                    continue
                active = edge.active_at(t)
                if not full and not active:
                    continue
                if edge.relation in ACTION_RELATIONS:
                    latest_action[(edge.source, edge.relation, edge.target)] = edge
                else:
                    spatial_edges.append(edge)

            chosen = spatial_edges + list(latest_action.values())
            chosen.sort(key=lambda e: e.t_start)
            for edge in chosen[-max_edges:]:
                source_key = self.node_key(edge.source)
                target_key = self.node_key(edge.target)
                for key, original in ((source_key, edge.source), (target_key, edge.target)):
                    if key not in nodes:
                        label = self.anchor_labels.get(original) or pretty_ref(original)
                        kind = "zone" if key.startswith("zone:") else "anchor"
                        nodes[key] = {
                            "id": key,
                            "label": label.upper() if kind == "zone" else label,
                            "kind": kind,
                            "state": "ANCHOR",
                            "shape": "box" if kind == "zone" else "database",
                            "size": 22 if kind == "zone" else 16,
                            "value": 22 if kind == "zone" else 16,
                            "color": {
                                "background": _rgba("#22d3ee" if kind == "zone" else "#64748b", 0.1),
                                "border": _rgba("#22d3ee" if kind == "zone" else "#94a3b8", 0.45),
                                "highlight": {"background": _rgba("#94a3b8", 0.3), "border": "#e2e8f0"},
                            },
                            "borderWidth": 1,
                            "opacity": 1.0,
                            "font": {"color": "#7dd3fc" if kind == "zone" else "#94a3b8", "size": 11},
                            "title": f"<b>{html.escape(label)}</b><br><span style='color:#94a3b8'>anchor</span>",
                            "active": True,
                            "dim": False,
                        }
                active = edge.active_at(t)
                future = edge.t_start - 0.05 > t
                is_action = edge.relation in ACTION_RELATIONS
                color = "#f5b942" if is_action else "#2dd4bf" if edge.relation in {"INSIDE", "CONTAINED_IN"} else "#35d07f"
                edge_id = edge.id
                edge_payloads.append(
                    {
                        "id": edge_id,
                        "from": source_key,
                        "to": target_key,
                        "label": edge.relation.replace("_", " ").lower(),
                        "relation": edge.relation,
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}} if is_action else {"to": {"enabled": True, "scaleFactor": 0.35}},
                        "dashes": not is_action,
                        "color": {"color": _rgba(color, 0.85 if active else 0.35), "highlight": color, "hover": color},
                        "width": 2.0 if active and is_action else 1.2,
                        "opacity": 1.0 if active else 0.45,
                        "font": {
                            "color": "#cbd5e1" if active else "#64748b",
                            "size": 10,
                            "face": "ui-monospace, monospace",
                            "strokeWidth": 3,
                            "strokeColor": "#070b12",
                            "align": "middle",
                        },
                        "title": self._edge_tooltip(edge),
                        "active": active,
                        "dim": future,
                        "t_start": round(edge.t_start, 3),
                        "t_end": None if edge.t_end is None else round(edge.t_end, 3),
                    }
                )

            # Keep the graph legible.
            if len(nodes) > max_nodes:
                keep = set()
                for edge_payload in edge_payloads:
                    keep.add(edge_payload["from"])
                    keep.add(edge_payload["to"])
                for node in nodes.values():
                    if len(keep) >= max_nodes:
                        break
                    if node["kind"] in {"entity", "user"}:
                        keep.add(node["id"])
                nodes = {key: value for key, value in nodes.items() if key in keep}
                edge_payloads = [
                    e for e in edge_payloads if e["from"] in nodes and e["to"] in nodes
                ][:max_edges]

            return {
                "t": round(t, 3),
                "mode": mode,
                "nodes": list(nodes.values()),
                "edges": edge_payloads,
                "revision": self.graph_revision,
                "epoch": self.epoch,
                "active_zone": self.active_zone_id,
            }

    # -- tooltips ------------------------------------------------------- #

    @staticmethod
    def _entity_tooltip(entity: Entity, state: str, t: float) -> str:
        color = STATE_COLORS.get(state, "#94a3b8")
        clip = f"{entity.clip_sim:.2f}" if entity.clip_sim is not None else "—"
        conf = f"{entity.confidence:.2f}" if entity.confidence else "—"
        last_seen = f"{entity.last_seen:.1f}s"
        return (
            f"<div style='font-family:ui-monospace,monospace;max-width:240px'>"
            f"<b style='font-size:13px'>{html.escape(entity.label.upper())}</b> "
            f"<span style='color:{color};font-weight:600'>{state}</span><br>"
            f"<span style='color:#94a3b8'>id</span> {html.escape(entity.id)}<br>"
            f"<span style='color:#94a3b8'>CLIP</span> {clip} &nbsp; "
            f"<span style='color:#94a3b8'>conf</span> {conf}<br>"
            f"<span style='color:#94a3b8'>last seen</span> {last_seen}<br>"
            f"<span style='color:#64748b'>click to inspect</span></div>"
        )

    @staticmethod
    def _zone_tooltip(zone: Zone) -> str:
        clip = f"{zone.clip_sim:.2f}" if zone.clip_sim is not None else "—"
        return (
            f"<div style='font-family:ui-monospace,monospace'>"
            f"<b style='color:#7dd3fc'>{html.escape(zone.label.upper())}</b><br>"
            f"<span style='color:#94a3b8'>topological anchor</span><br>"
            f"<span style='color:#94a3b8'>VPR CLIP</span> {clip}</div>"
        )

    @staticmethod
    def _edge_tooltip(edge: Edge) -> str:
        end = "active" if edge.t_end is None else f"{edge.t_end:.1f}s"
        return (
            f"<div style='font-family:ui-monospace,monospace;max-width:230px'>"
            f"<b>{html.escape(edge.relation)}</b><br>"
            f"{html.escape(edge.source)} → {html.escape(edge.target)}<br>"
            f"<span style='color:#94a3b8'>{edge.t_start:.1f}s → {end}</span></div>"
        )

    # -- timeline ------------------------------------------------------- #

    def log_since(self, last_seq: int, *, limit: int = 400) -> list[LogRecord]:
        with self._lock:
            records = [record for record in self.telemetry if record.seq > last_seq]
        return records[-limit:]

    def recent_logs(self, limit: int = 300) -> list[LogRecord]:
        with self._lock:
            return list(self.telemetry)[-limit:]

    def timeline_markers(self, limit: int = 160) -> list[dict[str, Any]]:
        with self._lock:
            markers: list[dict[str, Any]] = []
            for event in self.events:
                if event.kind not in {"state", "interaction", "edge", "zone", "session"}:
                    continue
                label = event.kind.upper()
                if event.kind == "state":
                    label = event.data.get("to") or "STATE"
                elif event.kind == "interaction":
                    label = event.data.get("action") or "VLM"
                elif event.kind == "edge":
                    label = event.data.get("relation") or "EDGE"
                elif event.kind == "zone":
                    label = event.data.get("label") or "ZONE"
                markers.append(
                    {
                        "t": round(event.t, 3),
                        "kind": event.kind,
                        "label": str(label),
                        "entity_id": event.entity_id,
                    }
                )
        return markers[-limit:]

    @property
    def dropped(self) -> int:
        return self._dropped

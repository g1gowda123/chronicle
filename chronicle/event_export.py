"""Export Role D's real-footage JSONL event stream from the runtime.

The dashboard consumes one JSON object per line (``kind`` =
``session|entity|state|interaction|edge|telemetry``).  This module produces that
contract directly from a ``ChronicleRuntime`` run, so the GPU/annotation owner
does not have to stitch Role A's per-frame JSONL together with a separate
snapshot by hand.

Example::

    python -m chronicle.event_export --source tabletop \
        --detector yolo-world --hands mediapipe --embedder openclip \
        --out data/real/holoassist_switch_34s.events.jsonl

The exporter emits **tracked** objects only, keeps their IDs stable between
entity/state/action events, and never invents detections.  It does not run a VLM
itself: interaction rows carry the action/caption/source that the runtime
actually recorded.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .integration import ChronicleRuntime
from .perception.config import PerceptionConfig
from .semantic.config import SemanticConfig
from .sources import ClipSource, source_for_key
from .telemetry import TelemetryEvent
from .types import FrameResult

EVENT_VERSION = 1
DEFAULT_HZ = 6.0
DEFAULT_MAX_EDGES = 2
DEFAULT_MAX_TELEMETRY = 200

_KIND_ORDER = {
    "session": 0,
    "entity": 1,
    "state": 2,
    "interaction": 3,
    "edge": 4,
    "telemetry": 5,
}

_STATE_REASONS = {
    "VISIBLE_STATIC": "resting / no independent motion",
    "VISIBLE_MOVING": "independent object motion",
    "INTERACTED": "hand contact",
    "HELD": "hand contact + object motion",
    "CONTAINED": "containment evidence",
    "OCCLUDED": "line of sight lost",
    "OUT_OF_VIEW": "camera pan / left frustum",
    "LOST": "archival timeout",
}


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return text.strip("_") or "entity"


def _trailing_number(value: str) -> str:
    match = re.search(r"(\d+)$", value)
    return match.group(1) if match else ""


def _unique_entity_id(label: str, raw_id: str, used: set[str]) -> str:
    base = _slug(label)
    suffix = _trailing_number(raw_id)
    candidate = f"{base}_{suffix}" if suffix else base
    stem = candidate
    index = 2
    while candidate in used:
        candidate = f"{stem}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _display_label(label: str) -> str:
    return label.replace("_", " ").replace("-", "-").title()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _clamp_box(
    bbox: Sequence[float], width: int, height: int
) -> list[int]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    ]


def _time_from_title(title: str) -> float:
    match = re.search(r"t=(-?\d+(?:\.\d+)?)s", str(title))
    return float(match.group(1)) if match else 0.0


def probe_video(path: str | Path) -> dict[str, Any]:
    """Read width/height/fps/frame count without decoding the whole clip."""

    try:
        import cv2
    except ImportError:  # pragma: no cover - core dependency is present
        return {"width": 0, "height": 0, "fps": 0.0, "frames": 0, "duration": 0.0}

    capture = cv2.VideoCapture(str(path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    duration = frames / fps if fps > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frames": frames,
        "duration": round(duration, 3),
    }


@dataclass(frozen=True)
class EventExport:
    """Result of one ``export_events`` run."""

    path: str
    events: int
    entities: int
    states: int
    interactions: int
    edges: int
    telemetry: int
    backends: Mapping[str, str]
    summary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "events": self.events,
            "entities": self.entities,
            "states": self.states,
            "interactions": self.interactions,
            "edges": self.edges,
            "telemetry": self.telemetry,
            "backends": dict(self.backends),
            "summary": dict(self.summary),
        }


class RoleDEventRecorder:
    """Collect runtime frames and snapshot facts into the dashboard contract."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float,
        duration: float,
        title: str,
        video_url: str,
        zone_id: str = "work_table",
        hz: float = DEFAULT_HZ,
        max_edges: int = DEFAULT_MAX_EDGES,
        max_telemetry: int = DEFAULT_MAX_TELEMETRY,
        notes: str = "",
    ) -> None:
        self.width = max(0, int(width))
        self.height = max(0, int(height))
        self.fps = float(fps)
        self.duration = float(duration)
        self.zone_id = zone_id
        self.min_interval = 1.0 / max(1e-6, float(hz))
        self.max_edges = max(0, int(max_edges))
        self.max_telemetry = max(0, int(max_telemetry))
        self._session = {
            "v": EVENT_VERSION,
            "kind": "session",
            "t": 0.0,
            "seq": 0,
            "data": {
                "title": title,
                "video_url": video_url,
                "duration": round(self.duration, 3),
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "source": "live",
                "notes": notes,
            },
        }
        self._entity_events: list[dict[str, Any]] = []
        self._last_emit: dict[str, float] = {}

    # ------------------------------------------------------------------ frames
    def on_frame(self, result: FrameResult) -> None:
        """Call from ``runtime.ingest_video(..., on_frame=...)``."""

        timestamp = float(result.timestamp)
        for track in result.tracks:
            raw_id = str(track.entity_id or f"track_{track.track_id}")
            last = self._last_emit.get(raw_id)
            if last is not None and timestamp - last < self.min_interval - 1e-9:
                continue
            self._last_emit[raw_id] = timestamp
            data: dict[str, Any] = {
                "id": raw_id,
                "label": _display_label(str(track.label or "object")),
                "class": str(track.label or "object").strip().lower(),
                "bbox": _clamp_box(track.bbox.xyxy, self.width, self.height),
                "confidence": round(float(track.confidence), 3),
                "track_id": str(track.track_id),
                "zone_id": self.zone_id,
                "attributes": {
                    "source": str(track.source or "detector"),
                    "tracker": "bytetrack-lite",
                },
            }
            if track.reid_score is not None:
                data["clip_sim"] = round(float(track.reid_score), 3)
                if track.reid_source:
                    data["attributes"]["reid_source"] = str(track.reid_source)
            self._entity_events.append(
                {
                    "v": EVENT_VERSION,
                    "kind": "entity",
                    "t": round(timestamp, 3),
                    "data": data,
                }
            )

    # ------------------------------------------------------------------ finish
    def finish(
        self,
        snapshot: Mapping[str, Any],
        telemetry: Iterable[TelemetryEvent] = (),
    ) -> list[dict[str, Any]]:
        """Merge per-frame observations with the completed graph snapshot."""

        mapping = self._build_mapping(snapshot)
        events: list[dict[str, Any]] = []
        for event in self._entity_events:
            data = dict(event["data"])
            data["id"] = mapping.get(str(data["id"]), str(data["id"]))
            events.append({**event, "data": data})

        events.extend(self._state_events(snapshot, mapping))
        events.extend(self._interaction_events(snapshot, mapping))
        events.extend(self._edge_events(snapshot, mapping))
        events.extend(self._telemetry_events(telemetry))

        ordered = [self._session, *events]
        ordered.sort(
            key=lambda event: (
                float(event.get("t") or 0.0),
                _KIND_ORDER.get(str(event.get("kind")), 9),
            )
        )
        for index, event in enumerate(ordered):
            event["seq"] = index
        return ordered

    def _build_mapping(self, snapshot: Mapping[str, Any]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        used: set[str] = set()
        for row in snapshot.get("entities") or []:
            raw_id = str(row.get("id") or "")
            if not raw_id:
                continue
            label = str(row.get("label") or raw_id)
            mapping[raw_id] = _unique_entity_id(label, raw_id, used)
        return mapping

    def _state_events(
        self, snapshot: Mapping[str, Any], mapping: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for row in snapshot.get("transitions") or []:
            raw_id = str(row.get("entity_id") or "")
            to_state = str(row.get("to_state") or "")
            data: dict[str, Any] = {
                "entity_id": mapping.get(raw_id, raw_id),
                "label": _display_label(str(row.get("label") or raw_id)),
                "from": str(row.get("from_state") or ""),
                "to": to_state,
                "reason": _STATE_REASONS.get(to_state, "state machine transition"),
                "confidence": 1.0,
            }
            clip_ref = row.get("clip_ref")
            if clip_ref:
                data["trigger"] = {"clip_ref": str(clip_ref)}
            events.append(
                {
                    "v": EVENT_VERSION,
                    "kind": "state",
                    "t": round(float(row.get("timestamp") or row.get("t_event") or 0.0), 3),
                    "data": data,
                }
            )
        return events

    def _interaction_events(
        self, snapshot: Mapping[str, Any], mapping: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for row in snapshot.get("actions") or []:
            raw_id = str(row.get("entity_id") or "")
            data: dict[str, Any] = {
                "entity_id": mapping.get(raw_id, raw_id),
                "action": str(row.get("action") or "ACTION"),
                "clip_sim": round(float(row.get("confidence") or 0.0), 3),
                "model": str(row.get("source") or "rule"),
                "strip_url": None,
                "crop_url": None,
            }
            caption = str(row.get("caption") or "").strip()
            if caption:
                data["caption"] = caption
            clip_ref = row.get("clip_ref")
            if clip_ref:
                data["clip_ref"] = str(clip_ref)
            events.append(
                {
                    "v": EVENT_VERSION,
                    "kind": "interaction",
                    "t": round(float(row.get("timestamp") or 0.0), 3),
                    "data": data,
                }
            )
        return events

    def _edge_events(
        self, snapshot: Mapping[str, Any], mapping: Mapping[str, str]
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for edge in snapshot.get("edges") or []:
            relation = str(edge.get("label") or "")
            if not relation or relation in {"ACTION", "STATE_TRANSITION"}:
                continue
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if src not in mapping or dst not in mapping:
                continue
            start = _time_from_title(str(edge.get("title") or ""))
            events.append(
                {
                    "v": EVENT_VERSION,
                    "kind": "edge",
                    "t": round(start, 3),
                    "data": {
                        "id": str(edge.get("id") or f"edge_{len(events)}"),
                        "source": mapping[src],
                        "target": mapping[dst],
                        "relation": relation,
                        "t_start": round(start, 3),
                        "active": not bool(edge.get("dashes")),
                    },
                }
            )
            if len(events) >= self.max_edges:
                break
        return events

    def _telemetry_events(
        self, telemetry: Iterable[TelemetryEvent]
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for event in telemetry:
            if event.t is None:
                continue
            data = _jsonable(event.data)
            if not isinstance(data, dict):
                data = {"value": data}
            data.update(
                {
                    "channel": event.tag.value,
                    "level": "info",
                    "message": event.message,
                }
            )
            events.append(
                {
                    "v": EVENT_VERSION,
                    "kind": "telemetry",
                    "t": round(float(event.t), 3),
                    "data": data,
                }
            )
        if self.max_telemetry and len(events) > self.max_telemetry:
            events = events[-self.max_telemetry :]
        return events


def write_events_jsonl(events: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write UTF-8 JSONL without a BOM and with a trailing newline."""

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(_jsonable(event), ensure_ascii=False, separators=(",", ":"))
        for event in events
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def export_events(
    source: ClipSource,
    data_dir: str | Path,
    out: str | Path,
    *,
    work_dir: str | Path = "runs/events",
    detector: str = "auto",
    hands: str = "auto",
    embedder: str = "auto",
    max_frames: int | None = None,
    hz: float = DEFAULT_HZ,
    title: str | None = None,
    video_url: str | None = None,
    zone_id: str | None = None,
    notes: str = "",
) -> EventExport:
    """Run one source through A -> B -> C and write the Role D event stream."""

    data_root = Path(data_dir).expanduser().resolve()
    video = source.video_path(data_root)
    if not video.is_file():
        raise FileNotFoundError(f"video not found for source {source.key!r}: {video}")
    metadata = probe_video(video)
    width = int(metadata["width"])
    height = int(metadata["height"])
    duration = float(metadata["duration"])
    fps = float(metadata["fps"])

    config = PerceptionConfig(vocabulary=source.vocabulary)
    config.detector.backend = detector
    config.hands.backend = hands
    config.embedder.backend = embedder

    work = Path(work_dir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / f"{source.key}.db"
    runtime = ChronicleRuntime(
        db_path,
        perception_config=config,
        semantic_config=SemanticConfig(llm_backend="template"),
        zone_label=source.zone_label,
        reset=True,
    )
    recorder = RoleDEventRecorder(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        title=title or source.label,
        video_url=video_url or f"/media/real/{video.name}",
        zone_id=zone_id or _slug(source.zone_label),
        hz=hz,
        notes=notes or f"{source.badge} · {source.description}",
    )
    try:
        summary = runtime.ingest_video(
            video,
            max_frames=max_frames,
            on_frame=recorder.on_frame,
        )
        snapshot = runtime.snapshot()
        events = recorder.finish(snapshot, runtime.bus.history())
        backends = runtime.backends()
        summary_payload = summary.as_dict()
    finally:
        runtime.close()

    path = write_events_jsonl(events, out)
    counts = {kind: sum(1 for event in events if event["kind"] == kind) for kind in _KIND_ORDER}
    return EventExport(
        path=str(path),
        events=len(events),
        entities=counts.get("entity", 0),
        states=counts.get("state", 0),
        interactions=counts.get("interaction", 0),
        edges=counts.get("edge", 0),
        telemetry=counts.get("telemetry", 0),
        backends=backends,
        summary=summary_payload,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tabletop", help="registered source key")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", required=True, help="output .events.jsonl path")
    parser.add_argument("--work-dir", default="runs/events")
    parser.add_argument("--detector", default="auto", help="auto|color|yolo-world")
    parser.add_argument("--hands", default="auto", help="auto|color|mediapipe")
    parser.add_argument("--embedder", default="auto", help="auto|color|openclip")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--title", default=None)
    parser.add_argument("--video-url", default=None)
    parser.add_argument("--zone-id", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--json", action="store_true", help="print the full result")
    args = parser.parse_args(argv)

    source = source_for_key(args.source, args.data_dir)
    result = export_events(
        source,
        args.data_dir,
        args.out,
        work_dir=args.work_dir,
        detector=args.detector,
        hands=args.hands,
        embedder=args.embedder,
        max_frames=args.max_frames,
        hz=args.hz,
        title=args.title,
        video_url=args.video_url,
        zone_id=args.zone_id,
        notes=args.notes,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 0 if result.events > 1 else 1


__all__ = [
    "DEFAULT_HZ",
    "EventExport",
    "RoleDEventRecorder",
    "export_events",
    "main",
    "probe_video",
    "write_events_jsonl",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

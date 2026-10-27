"""ZIP transfer for the zine dashboard: export/import graph timelines.

The dashboard already keeps the graph as an event timeline. This module packages
that timeline (plus a readable snapshot and CSVs) into one ZIP, and can import a
ZIP back without re-running perception or touching the source video. The VFM
artifacts are included when they exist, so an exported episode is self-contained.
"""

from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

MAX_IMPORT_BYTES = 256 * 1024 * 1024

#: The media cut is evidence, not a graph, so it stays out of the graph ZIP.
#: SQLite ``-wal``/``-shm`` sidecars are kept beside ``vault.db`` so the vault
#: opens with every committed transaction.
EVENT_CLIP_SKIP = frozenset({"clip.mp4"})


def _event_row(event: Any) -> dict[str, Any]:
    return {
        "v": 1,
        "kind": str(getattr(event, "kind", "state")),
        "t": round(float(getattr(event, "t", 0.0)), 3),
        "data": dict(getattr(event, "data", {}) or {}),
    }


def _events_jsonl(events: list[Any]) -> str:
    return "\n".join(
        json.dumps(_event_row(event), ensure_ascii=False, default=str) for event in events
    ) + ("\n" if events else "")


def _snapshot(store: Any, events: list[Any]) -> dict[str, Any]:
    entities = []
    for entity in store.entities.values():
        entities.append(
            {
                "id": entity.id,
                "label": entity.label,
                "state": entity.state,
                "zone_id": entity.zone_id,
                "first_seen": round(float(entity.first_seen or 0.0), 3),
                "last_seen": round(float(entity.last_seen or 0.0), 3),
                "cls": entity.cls,
                "confidence": entity.confidence,
            }
        )
    edges = [
        {
            "id": edge.id,
            "source": edge.source,
            "target": edge.target,
            "relation": edge.relation,
            "t_start": edge.t_start,
            "t_end": edge.t_end,
        }
        for edge in store.edges.values()
    ]
    zones = [{"id": zone.id, "label": zone.label} for zone in store.zones.values()]
    return {
        "version": 1,
        "exported_at": time.time(),
        "session": dict(store.session),
        "events": len(events),
        "entities": entities,
        "edges": edges,
        "zones": zones,
    }


def _csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _replay(events: list[Any]) -> Any:
    """Build the graph at the end of the full timeline, not at the playhead."""

    from .store import MemoryStore

    store = MemoryStore(max_events=max(len(events), 1))
    for event in events:
        store.apply(event)
    return store


def _write_event_clips(archive: zipfile.ZipFile, root: Path) -> None:
    """Add the per-event graphs/timelines produced by the ✂ events action."""

    base = (root / "event_clips").resolve()
    if not base.is_dir():
        return
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name in EVENT_CLIP_SKIP:
            continue
        try:
            relative = path.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        archive.write(path, arcname=relative.as_posix())


def build_export_zip(
    runtime: Any,
    data_dir: Path | str,
    *,
    vfm_dir: Path | str | None = None,
) -> bytes:
    """Return a ZIP with the full current graph, all timelines and VFM files."""

    root = Path(data_dir).expanduser().resolve()
    store = runtime.store
    events = list(getattr(runtime, "demo_events", None) or list(store.events))
    # ``runtime.store`` only contains events up to the playhead in demo mode,
    # while ``events`` is the whole timeline. Replay it so the snapshot and CSVs
    # describe the same graph the event count in the manifest promises.
    snapshot_store = _replay(events)
    snapshot = _snapshot(snapshot_store, events)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(
            {
                "version": 1,
                "created_at": time.time(),
                "title": snapshot_store.session.get("title") or "Chronicle episode",
                "events": len(events),
                "format": "chronicle-graph-zip",
            },
            indent=2,
        ))
        archive.writestr("current/events.jsonl", _events_jsonl(events))
        archive.writestr("current/snapshot.json", json.dumps(snapshot, indent=2, default=str))
        archive.writestr(
            "current/entities.csv",
            _csv(snapshot["entities"], ["id", "label", "state", "zone_id", "first_seen", "last_seen"]),
        )
        archive.writestr(
            "current/edges.csv",
            _csv(snapshot["edges"], ["id", "source", "target", "relation", "t_start", "t_end"]),
        )
        archive.writestr(
            "current/zones.csv",
            _csv(snapshot["zones"], ["id", "label"]),
        )
        # Every replay timeline the dashboard knows about.
        for folder in (root / "replays", root / "imports"):
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.events.jsonl")):
                archive.write(path, arcname=f"clips/{path.name}")
        # Per-event cuts: index, replay, graph snapshot, VLM record and vault.
        _write_event_clips(archive, root)
        # Optional VFM memory for the active clip.
        if vfm_dir is not None:
            vfm = Path(vfm_dir)
            if vfm.is_dir():
                for path in sorted(vfm.rglob("*")):
                    if not path.is_file():
                        continue
                    if path.name in {"source.db", "memory.zip"}:
                        continue
                    archive.write(path, arcname=f"vfm/{path.relative_to(vfm).as_posix()}")
    return buffer.getvalue()


def import_zip(data: bytes, data_dir: Path | str) -> dict[str, Any]:
    """Validate a graph ZIP and persist the first timeline to ``data/imports``."""

    if not data:
        raise ValueError("empty import")
    if len(data) > MAX_IMPORT_BYTES:
        raise ValueError("import is too large")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("not a ZIP file") from exc
    with archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        candidates = ["current/events.jsonl"] + sorted(
            name for name in names if name.startswith("clips/") and name.endswith(".events.jsonl")
        )
        chosen = next((name for name in candidates if name in names), None)
        if chosen is None:
            raise ValueError("ZIP has no current/events.jsonl or clips/*.events.jsonl")
        raw = archive.read(chosen)
    if not raw.strip():
        raise ValueError("timeline is empty")
    # Parse every line before writing anything into the dashboard data dir.
    count = 0
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid event JSON on line {count + 1}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"event {count + 1} is not an object")
        count += 1
    if count == 0:
        raise ValueError("timeline has no events")
    root = Path(data_dir).expanduser().resolve()
    target_dir = root / "imports"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"import-{time.strftime('%Y%m%d-%H%M%S')}.events.jsonl"
    target.write_bytes(raw)
    archive_names = chosen
    return {
        "status": "imported",
        "path": target,
        "relative_path": target.relative_to(root).as_posix(),
        "events": count,
        "source": archive_names,
    }

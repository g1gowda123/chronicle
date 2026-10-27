"""Export / import the mission-control memory as a portable ZIP bundle.

The bundle carries the full graph (live + episode projections), the entity /
action / transition tables, the current replay timeline and every cached replay
timeline, so a machine can be handed the whole memory and resume the demo.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

from .runtime import Runtime, load_demo_events

BUNDLE_FORMAT = 1
MAX_UPLOAD = 512 * 1024 * 1024
MAX_MEMBER = 256 * 1024 * 1024


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fields})
    return buffer.getvalue().encode("utf-8")


def export_bundle(runtime: Runtime, base: Path) -> bytes:
    """Build the ZIP bundle for the current episode."""
    store = runtime.store
    playhead = float(runtime.playhead)
    snapshot = runtime.snapshot(mode="full")
    graph_live = store.graph_snapshot(playhead, mode="live")
    graph_full = store.graph_snapshot(playhead, mode="full")

    entities = [entity.as_dict() for entity in store.entities.values()]
    actions = [record.as_dict() for record in store.interactions]
    transitions = [
        {"entity_id": entity.id, "label": entity.label, **record.as_dict()}
        for entity in store.entities.values()
        for record in entity.history
    ]
    edges = [edge.as_dict() for edge in store.edges.values()]

    timeline: list[dict[str, Any]] = []
    for event in runtime.demo_events:
        timeline.append(event.raw or {"kind": event.kind, "t": event.t, "data": event.data})

    manifest = {
        "format": BUNDLE_FORMAT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "title": store.session.get("title"),
        "duration": round(float(runtime.duration or 0.0), 3),
        "mode": runtime.mode,
        "playhead": round(playhead, 3),
        "counts": {
            "nodes": len(snapshot.get("nodes") or []),
            "edges": len(snapshot.get("edges") or []),
            "entities": len(entities),
            "actions": len(actions),
            "transitions": len(transitions),
            "events": len(timeline),
        },
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
        archive.writestr("snapshot.json", json.dumps(snapshot, indent=2, default=str))
        archive.writestr("graph-live.json", json.dumps(graph_live, indent=2, default=str))
        archive.writestr("graph-full.json", json.dumps(graph_full, indent=2, default=str))
        archive.writestr("entities.json", json.dumps(entities, indent=2, default=str))
        archive.writestr("actions.json", json.dumps(actions, indent=2, default=str))
        archive.writestr("transitions.json", json.dumps(transitions, indent=2, default=str))
        archive.writestr("edges.json", json.dumps(edges, indent=2, default=str))
        archive.writestr(
            "entities.csv",
            _csv_bytes(entities, ["id", "label", "cls", "state", "zone_id", "confidence", "first_seen", "last_seen"]),
        )
        archive.writestr(
            "actions.csv",
            _csv_bytes(actions, ["t", "action", "caption", "clip_sim", "model", "latency_ms"]),
        )
        archive.writestr(
            "transitions.csv",
            _csv_bytes(transitions, ["entity_id", "label", "t", "from", "to", "reason"]),
        )
        archive.writestr(
            "events.jsonl",
            "".join(json.dumps(event, separators=(",", ":"), default=str) + "\n" for event in timeline),
        )
        # every cached replay timeline travels with the bundle
        replays = base / "replays"
        if replays.is_dir():
            for path in sorted(replays.rglob("*.events.jsonl")):
                archive.write(path, f"replays/{path.relative_to(replays).as_posix()}")
        archive.writestr(
            "README.txt",
            "Chronicle memory bundle\n"
            "=======================\n\n"
            "manifest.json      summary + counts\n"
            "snapshot.json      runtime snapshot (graph, entities, tables, stats)\n"
            "graph-live.json    graph projection at the saved playhead\n"
            "graph-full.json    full-episode graph projection\n"
            "entities.json/.csv, actions.json/.csv, transitions.json/.csv, edges.json\n"
            "events.jsonl       replay timeline for the current episode\n"
            "replays/           cached replay timelines for the other clips\n\n"
            "Import this ZIP from the dashboard header to restore the replay.\n",
        )
    return buffer.getvalue()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return slug or "bundle"


def import_bundle(data: bytes, base: Path) -> dict[str, Any]:
    """Extract replay timelines from an exported ZIP and return the active one."""
    if not data:
        raise ValueError("empty upload")
    if len(data) > MAX_UPLOAD:
        raise ValueError("bundle too large")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("not a ZIP bundle") from exc

    with archive:
        members = []
        for info in archive.infolist():
            if not info.filename.lower().endswith(".jsonl"):
                continue
            if info.file_size > MAX_MEMBER:
                raise ValueError(f"{info.filename} is too large")
            members.append(info)
        if not members:
            raise ValueError("bundle contains no events.jsonl timelines")

        target_dir = base / "replays" / "imports"
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        written: list[Path] = []
        active: Path | None = None

        for info in members:
            raw = archive.read(info)
            first = next((line for line in raw.splitlines() if line.strip()), b"")
            try:
                payload = json.loads(first)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if payload.get("kind") != "session":
                continue
            name = Path(info.filename).name
            stem = _slug(Path(name).stem.replace(".events", ""))
            target = target_dir / f"{stem}-{stamp}.events.jsonl"
            counter = 1
            while target.exists():
                counter += 1
                target = target_dir / f"{stem}-{stamp}-{counter}.events.jsonl"
            target.write_bytes(raw)
            written.append(target)
            if info.filename in {"events.jsonl", "./events.jsonl"}:
                active = target

        if not written:
            raise ValueError("bundle has no valid replay timelines")
        active = active or written[0]
        events, session, duration = load_demo_events(active)
        if not events:
            raise ValueError("replay timeline is empty")
        return {
            "path": active.relative_to(base).as_posix(),
            "title": str(session.get("title") or "Imported replay"),
            "duration": round(float(duration or 0.0), 3),
            "timelines": len(written),
        }

"""Build a replay timeline from an existing Role B memory database.

Real clips already have a populated ``chronicle.db`` (entities, state
transitions, edges, zones). Rather than re-running perception, convert those
rows into the canonical ``.events.jsonl`` timeline the mission control replays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters import _SQLiteMirror
from .runtime import probe_duration


class _Collector:
    """Minimal Runtime stand-in so the SQLite mirror can emit raw events."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.ingest_errors: list[str] = []

    def ingest(self, raw: dict[str, Any], origin: str = "db") -> bool:  # noqa: ARG002
        self.events.append(raw)
        return True


def _count_rows(db_path: Path) -> int:
    import sqlite3

    try:
        con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=2.0)
        try:
            total = 0
            for table in ("entities", "state_transitions", "edges"):
                try:
                    total += con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                except sqlite3.Error:
                    continue
            return total
        finally:
            con.close()
    except sqlite3.Error:
        return 0


def events_from_db(
    db_path: str | Path,
    *,
    title: str,
    video_url: str,
    video_path: str | Path | None = None,
    fps: float = 30.0,
) -> list[dict[str, Any]]:
    db = Path(db_path)
    collector = _Collector()
    mirror = _SQLiteMirror(collector, db)  # type: ignore[arg-type]
    for _ in range(60):
        if mirror.poll() <= 0:
            break
    events = [event for event in collector.events if event.get("kind") != "session"]
    if not events:
        return []

    # The mirror timestamps entity rows at last_seen; rewind them to first_seen
    # so labels (person/cup/car/...) exist from the moment the track appears.
    for event in events:
        if event.get("kind") != "entity":
            continue
        data = event.get("data") or {}
        first = data.get("first_seen")
        if first is not None:
            try:
                event["t"] = max(0.0, float(first))
            except (TypeError, ValueError):
                pass

    duration = max((float(event.get("t") or 0.0) for event in events), default=0.0)
    if video_path is not None and Path(video_path).is_file():
        duration = max(duration, probe_duration(Path(video_path)))
    session = {
        "v": 1,
        "kind": "session",
        "t": 0.0,
        "data": {
            "title": title,
            "video_url": video_url,
            "duration": round(duration + 0.5, 3),
            "fps": fps,
            "width": 1280,
            "height": 720,
            "source": "db-replay",
        },
    }
    events.sort(key=lambda event: float(event.get("t") or 0.0))
    return [session] + events


def write_events(events: list[dict[str, Any]], out: str | Path) -> Path:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
    tmp.replace(target)
    return target


def has_memory(db_path: str | Path) -> bool:
    path = Path(db_path)
    return path.is_file() and _count_rows(path) > 0

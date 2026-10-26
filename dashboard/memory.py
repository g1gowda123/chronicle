"""Bridge between the zine dashboard's clip registry and the VFM memory service.

The dashboard knows which clip/video is on stage; the VFM package knows how to
compile that clip's graph DB into a video-free vault. This module resolves one
to the other and keeps a single background job manager per data directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronicle.sources import available_sources
from chronicle.vfm.service import VFMJobManager

_MANAGERS: dict[Path, VFMJobManager] = {}
_AUTO_ATTEMPTED: set[str] = set()


def enabled() -> bool:
    return os.getenv("CHRONICLE_VFM", "1").strip().lower() not in {"0", "false", "off", "no"}


def auto_enabled() -> bool:
    return os.getenv("CHRONICLE_VFM_AUTO", "1").strip().lower() not in {"0", "false", "off", "no"}


@dataclass(frozen=True)
class MemorySource:
    """Minimal source protocol expected by :class:`VFMJobManager`."""

    key: str
    label: str
    db: Path | None
    video: Path | None

    def db_path(self, data_dir: Path | str) -> Path:
        return self.db if self.db is not None else Path(data_dir) / "missing.db"

    def video_path(self, data_dir: Path | str) -> Path:
        return self.video if self.video is not None else Path(data_dir) / "missing.mp4"


def manager(data_dir: Path | str) -> VFMJobManager:
    root = Path(data_dir).expanduser().resolve()
    instance = _MANAGERS.get(root)
    if instance is None:
        instance = VFMJobManager(root, enabled=enabled())
        _MANAGERS[root] = instance
    return instance


def _events_session(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and payload.get("kind") == "session":
                    return dict(payload.get("data") or {})
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _registered_source(data_dir: Path, video: Path | None, key: str = "") -> Any:
    sources = available_sources(data_dir)
    if key:
        for source in sources:
            if source.key == key:
                return source
    if video is not None:
        try:
            target = video.resolve()
        except OSError:
            target = video
        for source in sources:
            try:
                if source.video_path(data_dir).resolve() == target:
                    return source
            except OSError:
                continue
    return None


def resolve(clip: str | None, data_dir: Path | str) -> MemorySource | None:
    """Map ``video:<rel>`` / ``events:<rel>`` to a DB + video pair."""

    root = Path(data_dir).expanduser().resolve()
    value = str(clip or "")
    kind, _, rel = value.partition(":")
    if kind not in {"video", "events"} or not rel:
        # Default to the synthetic demo when no clip is selected.
        kind, rel = "events", "demo_events.jsonl"

    if kind == "video":
        video = (root / rel).resolve()
        try:
            video.relative_to(root)
        except ValueError:
            return None
        if not video.is_file():
            return None
        source = _registered_source(root, video)
        if source is not None:
            db = source.db_path(root)
            return MemorySource(
                key=f"video:{source.key}",
                label=str(source.label),
                db=Path(db) if Path(db).is_file() else None,
                video=video,
            )
        # Uploaded clips: only compile if a graph DB already sits beside them.
        db = video.parent / "chronicle.db"
        return MemorySource(
            key="video:" + rel,
            label=video.stem,
            db=db if db.is_file() else None,
            video=video,
        )

    events = (root / rel).resolve()
    try:
        events.relative_to(root)
    except ValueError:
        return None
    if not events.is_file():
        return None
    session = _events_session(events)
    video: Path | None = None
    video_url = str(session.get("video_url") or "")
    if video_url.startswith("/media/"):
        candidate = (root / video_url.removeprefix("/media/")).resolve()
        if candidate.is_file():
            video = candidate

    stem_key = events.name.removesuffix(".events.jsonl")
    source = _registered_source(root, video, key=stem_key)
    if source is not None:
        db = source.db_path(root)
        return MemorySource(
            key=f"events:{source.key}",
            label=str(source.label),
            db=Path(db) if Path(db).is_file() else None,
            video=video or source.video_path(root),
        )

    if events.name == "demo_events.jsonl" or stem_key == "demo":
        db = root / "demo" / "chronicle.db"
        return MemorySource(
            key="events:demo",
            label=str(session.get("title") or "Synthetic demo"),
            db=db if db.is_file() else None,
            video=video or (root / "demo" / "chronicle_demo.mp4"),
        )

    db = events.parent / "chronicle.db"
    return MemorySource(
        key="events:" + rel,
        label=str(session.get("title") or stem_key),
        db=db if db.is_file() else None,
        video=video,
    )


def auto_compile(clip: str | None, data_dir: Path | str) -> dict[str, Any] | None:
    """Compile a clip's memory once, in the background, when it is first seen.

    Answers are deterministic and instant once the vault exists; this only pays
    the one-time VLM/evidence cost. A failed job is not retried automatically.
    """

    if not enabled() or not auto_enabled():
        return None
    root = Path(data_dir).expanduser().resolve()
    source = resolve(clip, root)
    if source is None or source.db is None or not source.db.is_file():
        return None
    instance = manager(root)
    if instance.has_memory(source):
        return None
    state = instance.status(source) or {}
    if state.get("status") in {"queued", "running"}:
        return None
    if source.key in _AUTO_ATTEMPTED:
        return None
    _AUTO_ATTEMPTED.add(source.key)
    job = instance.submit(source, source.video)
    return job.to_dict() if job is not None else None


def status(clip: str | None, data_dir: Path | str) -> dict[str, Any]:
    root = Path(data_dir).expanduser().resolve()
    source = resolve(clip, root)
    if source is None:
        return {"status": "unavailable", "message": "pick a clip first", "files": {}}
    if source.db is None:
        return {
            "status": "no-db",
            "message": "no graph DB for this clip; run the pipeline or upload an ingested run",
            "files": {},
        }
    return manager(root).status(source) or {
        "status": "not-compiled",
        "message": "memory not compiled yet",
        "files": {},
    }

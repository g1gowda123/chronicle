"""Dashboard runtime: event ingest, demo clock, time-travel, query dispatch.

One :class:`Runtime` instance is shared by every connected browser tab.  It is
the only writer to the :class:`MemoryStore`, so ingestion order is deterministic
and the UI can scrub the whole episode without touching pipeline processes.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .contracts import Event, parse_event
from .query import QueryEngine
from .store import MemoryStore

DEMO_MARKER_KINDS = {"state", "interaction", "edge", "zone", "session"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}


def probe_duration(path: Path) -> float:
    """Read a clip's duration with ffprobe (fast; falls back to 30 s)."""
    import subprocess

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return max(1.0, float(result.stdout.strip() or 30.0))
    except (OSError, ValueError, subprocess.SubprocessError):
        return 30.0


def clip_title(path: Path) -> str:
    """Pretty label for a clip filename (uploads use their original name)."""
    stem = path.stem
    if stem.lower() in {"source", "video", "clip", "upload"} and path.parent.name:
        parts = path.parent.name.split("-", 2)
        if len(parts) == 3 and parts[2]:
            stem = parts[2]
    stem = stem.replace("_", " ").strip()
    words = [word.upper() if word.lower().startswith("aria") else word.capitalize() for word in stem.split()]
    return " ".join(words) or path.name


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    configured = os.getenv("CHRONICLE_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else project_root() / "data"


def load_demo_events(path: Path) -> tuple[list[Event], dict[str, Any], float]:
    """Load a recorded demo timeline (JSONL) into normalised events."""
    if not path.exists():
        return [], {}, 0.0
    events: list[Event] = []
    session: dict[str, Any] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = parse_event(raw, seq=index)
        if event is None:
            continue
        events.append(event)
        if event.kind == "session" and event.t <= 0.01:
            session.update(
                {key: event.data[key] for key in ("title", "video_url", "duration", "fps", "width", "height") if key in event.data}
            )
    events.sort(key=lambda event: (event.t, event.seq))
    duration = float(session.get("duration") or (max((e.t for e in events), default=0.0) + 0.5))
    session.setdefault("duration", duration)
    return events, session, duration


class Runtime:
    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        demo_enabled: bool | None = None,
        start_t: float | None = None,
        video_override: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.store = MemoryStore()
        self.query = QueryEngine(self.store, data_dir=self.data_dir, video_path=self.video_file)
        self._lock = threading.RLock()
        self._seq = 0
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._started_at = time.time()

        if demo_enabled is None:
            demo_enabled = os.getenv("CHRONICLE_SOURCE", "demo").strip().lower() not in {"live", "pipeline"}
        replay_path = os.getenv("CHRONICLE_REPLAY") or ""
        demo_events_file = Path(replay_path).expanduser() if replay_path else self.data_dir / "demo_events.jsonl"
        self.demo_events, self.demo_session, self.demo_duration = load_demo_events(demo_events_file)
        if video_override is None:
            video_override = os.getenv("CHRONICLE_VIDEO") or None
        self.video_override = video_override
        self.mode = "demo" if demo_enabled and self.demo_events else "live"
        self.demo_available = bool(self.demo_events)
        self.duration = self.demo_duration if self.mode == "demo" else 0.0
        self.playhead = 0.0
        self.playing = bool(self.demo_events) and self.mode == "demo"
        self.rate = float(os.getenv("CHRONICLE_RATE", "1.0") or 1.0)
        self.demo_idx = 0
        self.live_events = 0
        self.rejected_events = 0
        self.last_ingest_at: float | None = None
        self.last_live_event_at: float | None = None
        self.last_live_origin: str | None = None
        self.ingest_errors: list[str] = []
        self._last_clock = time.monotonic()
        self._live_playhead_follow = True

        if self.mode == "demo":
            start = start_t
            if start is None:
                start = float(os.getenv("CHRONICLE_DEMO_START", "5.6") or 5.6)
            self._reset_demo_session()
            self.seek(max(0.0, min(start, self.duration - 0.1)), playing=self.playing)
        else:
            self.duration = 0.0
            self.playing = False
            self._reset_live_session()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _reset_demo_session(self) -> None:
        self.store.session.update(
            {
                "title": "Chronicle Episode",
                "video_url": None,
                "duration": self.demo_duration,
                "fps": 15.0,
                "width": 1280,
                "height": 720,
                **self.demo_session,
            }
        )
        self.duration = float(self.store.session.get("duration") or self.demo_duration)

    def _reset_live_session(self) -> None:
        self.store.session.update(
            {
                "title": "Live pipeline",
                "video_url": None,
                "duration": 0.0,
                "fps": 30.0,
                "width": 1280,
                "height": 720,
            }
        )
        self.duration = 0.0

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._task = asyncio.create_task(self._clock_loop(), name="chronicle-demo-clock")

        source = os.getenv("CHRONICLE_SOURCE", "").strip().lower()
        jsonl = os.getenv("CHRONICLE_EVENTS_JSONL")
        database = os.getenv("CHRONICLE_DB")
        if source in {"jsonl"} and not jsonl:
            jsonl = str(self.data_dir / "events.jsonl")
        if source in {"sqlite", "db"} and not database:
            database = str(self.data_dir / "chronicle.db")

        if jsonl:
            from .adapters import tail_jsonl

            asyncio.create_task(tail_jsonl(self, Path(jsonl)), name="chronicle-jsonl-tail")
        if database:
            from .adapters import poll_sqlite

            asyncio.create_task(poll_sqlite(self, Path(database)), name="chronicle-sqlite-poll")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    # ------------------------------------------------------------------ #
    # Clock / time travel
    # ------------------------------------------------------------------ #

    async def _clock_loop(self) -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.05)
            now = time.monotonic()
            dt = min(now - last, 0.5)
            last = now
            with self._lock:
                if self.mode != "demo" or not self.playing or not self.demo_events:
                    continue
                if self.duration <= 0:
                    continue
                self.playhead += dt * max(self.rate, 0.05)
                if self.playhead >= self.duration:
                    self.seek(0.0, playing=True)
                else:
                    self._dispatch_due()

    def _dispatch_due(self) -> int:
        count = 0
        while self.demo_idx < len(self.demo_events):
            event = self.demo_events[self.demo_idx]
            if event.t > self.playhead + 1e-6:
                break
            self.store.apply(event)
            self.demo_idx += 1
            count += 1
        return count

    def seek(self, t: float, *, playing: bool | None = None) -> None:
        """Scrub the episode. In demo mode the graph is deterministically rebuilt."""
        with self._lock:
            t = max(0.0, min(float(t), max(self.duration, 0.0)))
            if playing is not None:
                self.playing = playing
            self.playhead = t
            if self.mode == "demo" and self.demo_events:
                self._reset_demo_session()
                self.store.clear(keep_session=True)
                self.demo_idx = 0
                self._dispatch_due()
                self.playing = bool(playing) if playing is not None else self.playing

    def set_playing(self, playing: bool) -> None:
        with self._lock:
            self.playing = bool(playing)
            self._last_clock = time.monotonic()

    def toggle_playing(self) -> bool:
        self.set_playing(not self.playing)
        return self.playing

    def set_rate(self, rate: float) -> None:
        with self._lock:
            self.rate = max(0.1, min(float(rate), 4.0))

    def restart_demo(self, start: float = 0.0) -> None:
        with self._lock:
            if not self.demo_events:
                return
            self.mode = "demo"
            self.live_events = 0
            self.duration = self.demo_duration
            self._reset_demo_session()
            self.store.clear(keep_session=True)
            self.demo_idx = 0
            self.playhead = 0.0
            self.playing = True
            self.seek(max(0.0, min(start, self.duration - 0.05)))

    def load_replay(self, path: str | Path, start: float = 0.0) -> bool:
        """Switch the demo replay to another recorded ``.events.jsonl`` clip."""
        replay = Path(path)
        if not replay.is_absolute():
            replay = (self.data_dir / replay).resolve()
        events, session, duration = load_demo_events(replay)
        if not events:
            return False
        with self._lock:
            self.demo_events = events
            self.demo_session = session
            self.demo_duration = duration
            self.demo_available = True
            self.mode = "demo"
            self.live_events = 0
            self.duration = duration
            self._reset_demo_session()
            self.store.clear(keep_session=False)
            self.demo_idx = 0
            self.playhead = 0.0
            self.playing = True
            self.seek(max(0.0, min(start, self.duration - 0.05)))
        return True

    def play_video(
        self, video_url: str, title: str, duration: float = 30.0, *, fps: float = 30.0
    ) -> None:
        """Play a raw clip with no recorded memory events (no overlays yet)."""
        duration = max(1.0, float(duration or 30.0))
        session = {
            "title": title,
            "video_url": video_url,
            "duration": duration,
            "fps": fps,
            "width": 1280,
            "height": 720,
            "source": "video",
        }
        session_event = Event(kind="session", t=0.0, data=dict(session))
        with self._lock:
            self.demo_events = [session_event]
            self.demo_session = session
            self.demo_duration = duration
            self.demo_available = True
            self.mode = "demo"
            self.live_events = 0
            self.duration = duration
            self._reset_demo_session()
            self.store.clear(keep_session=False)
            self.demo_idx = 0
            self.playhead = 0.0
            self.playing = True
            self.seek(0.0)

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def ingest(self, raw: Any, origin: str = "http") -> bool:
        event = parse_event(raw, seq=self._seq + 1)
        self._seq += 1
        if event is None:
            self.rejected_events += 1
            if len(self.ingest_errors) < 40:
                self.ingest_errors.append(f"rejected {origin} payload: {str(raw)[:180]}")
            return False
        if origin != "demo":
            self._enter_live()
            self.live_events += 1
            self.last_live_event_at = time.time()
            self.last_live_origin = origin
            self.last_ingest_at = self.last_live_event_at
            with self._lock:
                self.store.apply(event)
                if self._live_playhead_follow:
                    self.playhead = max(self.playhead, event.t)
                    self.duration = max(self.duration, event.t)
                if event.kind == "session":
                    self._sync_live_session()
            return True
        with self._lock:
            return self.store.apply(event)

    def ingest_many(self, raws: Iterable[Any], origin: str = "http") -> int:
        accepted = 0
        for raw in raws:
            if self.ingest(raw, origin=origin):
                accepted += 1
        return accepted

    def _enter_live(self) -> None:
        with self._lock:
            if self.mode == "demo":
                self.mode = "live"
                self.playing = False
                self.playhead = 0.0
                self.duration = 0.0
                self.demo_idx = 0
                self._reset_live_session()
                self.store.clear(keep_session=False)
            self._live_playhead_follow = True

    def _sync_live_session(self) -> None:
        session = self.store.session
        self.duration = max(self.duration, float(session.get("duration") or 0.0))
        if session.get("video_url"):
            self.video_override = self.video_override or str(session["video_url"])

    # ------------------------------------------------------------------ #
    # Presentation helpers
    # ------------------------------------------------------------------ #

    @property
    def source_label(self) -> str:
        return "DEMO REPLAY" if self.mode == "demo" else "LIVE INGEST"

    @property
    def video_url(self) -> str | None:
        if self.video_override:
            return self.video_override
        url = self.store.session.get("video_url")
        if not url:
            return None
        url = str(url)
        if url.startswith(("http://", "https://", "/")):
            return url
        path = Path(url).expanduser()
        if not path.is_absolute():
            path = (self.data_dir / path).resolve()
        try:
            relative = path.resolve().relative_to(self.data_dir.resolve())
        except ValueError:
            return None
        return "/media/" + relative.as_posix()

    def video_file(self) -> Path | None:
        """Return the local video backing the current source, if one exists."""

        url = self.video_url
        if not url or url.startswith(("http://", "https://", "data:")):
            return None
        if url.startswith("/media/"):
            path = self.data_dir / url[len("/media/") :]
        else:
            path = Path(url).expanduser()
            if not path.is_absolute():
                path = self.data_dir / path
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return None
        return resolved if resolved.is_file() else None

    def snapshot(self, *, mode: str | None = None) -> dict[str, Any]:
        with self._lock:
            t = self.playhead
            graph_mode = mode or "live"
            stats = self.store.stats()
            return {
                "version": "1.0.0",
                "mode": self.mode,
                "source_label": self.source_label,
                "playing": self.playing,
                "rate": self.rate,
                "playhead": round(t, 3),
                "duration": round(self.duration, 3),
                "video_url": self.video_url,
                "title": self.store.session.get("title"),
                "fps": self.store.session.get("fps"),
                "width": self.store.session.get("width"),
                "height": self.store.session.get("height"),
                "graph": self.store.graph_snapshot(t, mode=graph_mode),
                "stats": stats,
                "epoch": self.store.epoch,
                "revision": self.store.revision,
                "demo_available": self.demo_available,
                "live_events": self.live_events,
                "rejected_events": self.rejected_events,
                "query_provider": self.query.provider_label(),
            }

    def timeline(self, *, include_future: bool = True) -> list[dict[str, Any]]:
        if self.mode == "demo" and self.demo_events:
            markers: list[dict[str, Any]] = []
            for event in self.demo_events:
                if event.kind not in DEMO_MARKER_KINDS:
                    continue
                if not include_future and event.t > self.playhead + 0.05:
                    continue
                label: Any = event.kind.upper()
                if event.kind == "state":
                    label = event.data.get("to") or "STATE"
                elif event.kind == "interaction":
                    label = event.data.get("action") or "VLM"
                elif event.kind == "edge":
                    label = event.data.get("relation") or "EDGE"
                elif event.kind == "zone":
                    label = event.data.get("label") or "ZONE"
                elif event.kind == "session":
                    label = "SESSION"
                markers.append(
                    {
                        "t": round(event.t, 3),
                        "kind": event.kind,
                        "label": str(label),
                        "entity_id": event.entity_id,
                        "future": event.t > self.playhead + 0.05,
                    }
                )
            return markers[-220:]
        return [
            {**marker, "future": False}
            for marker in self.store.timeline_markers()
        ]

    def stats(self) -> dict[str, Any]:
        data = self.store.stats()
        data.update(
            {
                "mode": self.mode,
                "source_label": self.source_label,
                "playing": self.playing,
                "playhead": round(self.playhead, 3),
                "duration": round(self.duration, 3),
                "live_events": self.live_events,
                "rejected_events": self.rejected_events,
                "dropped": self.store.dropped,
                "uptime_s": round(time.time() - self._started_at, 1),
                "demo_available": self.demo_available,
            }
        )
        return data


#: Shared runtime used by the NiceGUI app (tests build their own instances).
runtime = Runtime()

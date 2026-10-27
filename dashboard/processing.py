"""Background perception for clips uploaded from the zine dashboard.

``POST /api/upload`` used to only drop the file in ``data/uploads``: the video
played on stage but there was no graph, no object tags and nothing for the VFM
memory to compile.  This module finishes the job in the background so the
upload button behaves like a first-class source:

    data/uploads/<run-id>/source.mp4          the clip
    data/uploads/<run-id>/source.json         upload manifest
    data/uploads/<run-id>/chronicle.db        perception + graph memory
    data/uploads/<run-id>/source.events.jsonl dashboard replay (graph + tags)
    data/uploads/<run-id>/vfm/                video-free memory

Jobs run one at a time on a daemon thread; progress is written to
``ingest.json`` so the UI (and a server restart) can follow along.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronicle.perception.config import CS_LAB_VOCABULARY
from chronicle.ui.uploads import (
    VIDEO_SUFFIXES,
    UploadError,
    new_upload_target,
    probe_video,
    register_upload,
)

#: Prompt set used for uploads when the caller does not pass ``vocab=``.
#: CS-lab/desk clips are the demo default; ``vocab=a,b,c`` overrides it.
DEFAULT_UPLOAD_VOCABULARY: tuple[str, ...] = CS_LAB_VOCABULARY

RECORD_NAME = "ingest.json"


def parse_vocabulary(raw: str | None) -> tuple[str, ...]:
    """Parse a comma/semicolon separated vocabulary, falling back to the default."""

    if raw:
        items = tuple(
            dict.fromkeys(
                token.strip().lower()
                for token in raw.replace(";", ",").split(",")
                if token.strip()
            )
        )
        if items:
            return items
    return DEFAULT_UPLOAD_VOCABULARY


def _now() -> float:
    return time.time()


@dataclass
class UploadJob:
    key: str
    folder: Path
    video: Path
    title: str
    source_key: str
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = "waiting for the perception worker"
    error: str | None = None
    summary: dict[str, Any] | None = None
    events: str | None = None
    frames: int = 0
    detections: int = 0
    started_at: float = field(default_factory=_now)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "upload": self.key,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "summary": self.summary,
            "events": self.events,
            "title": self.title,
            "video": self.video.name,
            "frames": self.frames,
            "detections": self.detections,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def update(self, **changes: Any) -> None:
        for name, value in changes.items():
            setattr(self, name, value)

    def write(self) -> None:
        payload = self.to_dict()
        target = self.folder / RECORD_NAME
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            tmp.replace(target)
        except OSError:
            # A status file must never take the pipeline down.
            pass


def _make_playback_proxy(video: Path, duration: float) -> None:
    """Build a 480p, fast-start playback proxy for long clips.

    The dashboard keeps the original for perception/VLM. A lighter file keeps
    browser decoding and seeking cheap on 10+ minute uploads.
    """
    if duration < 300.0:
        return
    proxy = video.with_name(video.stem + ".proxy.mp4")
    if proxy.is_file():
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return
    tmp = video.with_name(video.stem + ".proxy.tmp.mp4")
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        "scale=-2:480",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            timeout=1800,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp.replace(proxy)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass


class UploadProcessor:
    """Serialise upload ingestion on one background thread."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self._lock = threading.RLock()
        self._jobs: dict[str, UploadJob] = {}
        self._queue: queue.Queue[UploadJob | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    # -- public API -----------------------------------------------------
    def submit(self, folder: Path, video: Path, source: Any) -> UploadJob:
        folder = Path(folder).expanduser().resolve()
        video = Path(video).expanduser().resolve()
        job = UploadJob(
            key=folder.name,
            folder=folder,
            video=video,
            title=str(getattr(source, "label", video.stem)),
            source_key=str(getattr(source, "key", folder.name)),
            events=f"uploads/{folder.name}/{video.stem}.events.jsonl",
        )
        with self._lock:
            self._jobs[job.key] = job
        job.write()
        self._ensure_worker()
        self._queue.put(job)
        return job

    def status(self, key: str) -> dict[str, Any] | None:
        key = Path(str(key or "")).name
        if not key or key in {".", ".."}:
            return None
        with self._lock:
            job = self._jobs.get(key)
            if job is not None:
                return job.to_dict()
        # Jobs survive a restart as a status file; an interrupted one is not
        # silently left "running" forever.
        record = self.data_dir / "uploads" / key / RECORD_NAME
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("status") in {"queued", "running"}:
            payload.update(
                status="error",
                stage="interrupted",
                message="processing was interrupted by a server restart",
                error="server restart; upload the clip again",
            )
        return payload

    def stop(self) -> None:
        self._queue.put(None)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    # -- worker ---------------------------------------------------------
    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="chronicle-upload-processor",
                daemon=True,
            )
            self._thread.start()

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._process(job)
            except Exception as error:  # keep the worker alive for the next clip
                with self._lock:
                    job.update(
                        status="error",
                        stage="error",
                        message="perception failed",
                        error=f"{type(error).__name__}: {error}",
                        finished_at=_now(),
                    )
                    job.write()
                traceback.print_exc()

    def _process(self, job: UploadJob) -> None:
        from chronicle.integration import ChronicleRuntime
        from chronicle.perception.config import PerceptionConfig
        from chronicle.semantic.config import SemanticConfig
        from chronicle.ui.uploads import load_upload_sources
        from dashboard.replay_db import events_from_db, write_events
        from dashboard.runtime import probe_duration

        source = next(
            (item for item in load_upload_sources(self.data_dir) if item.key == job.source_key),
            None,
        )
        if source is None:
            raise RuntimeError("upload manifest disappeared before processing")

        duration = probe_duration(job.video)
        with self._lock:
            job.update(
                status="running",
                stage="perception",
                progress=1,
                message="loading perception models",
            )
            job.write()

        config = PerceptionConfig()
        config.vocabulary = tuple(source.vocabulary)
        runtime = ChronicleRuntime(
            source.db_path(self.data_dir),
            perception_config=config,
            semantic_config=SemanticConfig.from_env(),
            zone_label=source.zone_label,
            reset=True,
        )

        def on_progress(state: dict[str, Any]) -> None:
            timestamp = float(state.get("timestamp") or 0.0)
            ratio = min(1.0, timestamp / duration) if duration > 0 else 0.0
            frames = int(state.get("frames") or 0)
            detections = int(state.get("detections") or 0)
            with self._lock:
                job.update(
                    progress=max(2, min(94, int(2 + ratio * 92))),
                    message=(
                        f"perceiving · t={timestamp:.1f}s · {frames} frames · "
                        f"{detections} detections"
                    ),
                    frames=frames,
                    detections=detections,
                )
                job.write()

        try:
            summary = runtime.ingest_video(job.video, on_progress=on_progress)
        finally:
            runtime.close()

        with self._lock:
            job.update(
                stage="replay",
                progress=96,
                message="building the dashboard replay",
                summary=summary.as_dict(),
                frames=summary.frames,
                detections=summary.detections,
            )
            job.write()

        video_rel = job.video.relative_to(self.data_dir).as_posix()
        events = events_from_db(
            source.db_path(self.data_dir),
            title=source.label,
            video_url="/media/" + video_rel,
            video_path=job.video,
        )
        if not events:
            raise RuntimeError("perception finished but produced no events")
        write_events(events, job.folder / f"{job.video.stem}.events.jsonl")

        with self._lock:
            job.update(
                status="done",
                stage="done",
                progress=100,
                message=f"clip ready · {summary.entities} tracks · {summary.detections} detections",
                finished_at=_now(),
            )
            job.write()

        # Long clips get a lighter playback file for the browser. This is
        # best-effort and never blocks the upload job.
        if duration >= 300.0:
            threading.Thread(
                target=_make_playback_proxy,
                args=(job.video, duration),
                name=f"proxy-{job.key}",
                daemon=True,
            ).start()

        # Important events deserve their own clips, 3-frame VLM pass, snapshot
        # and per-event vault. This is independent of the 20 s caption window.
        if os.getenv("CHRONICLE_EVENT_CLIPS_AUTO", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }:
            try:
                from dashboard.event_clips import build_event_clips

                with self._lock:
                    job.update(message="cutting + verifying event memories")
                    job.write()
                index = build_event_clips(
                    "events:" + str(job.events),
                    self.data_dir,
                    limit=4,
                    verify=True,
                    verify_limit=2,
                )
                with self._lock:
                    job.update(
                        message=(
                            f"clip ready · {index.get('verified', 0)} event memories "
                            f"verified · {len(index.get('clips') or [])} event clips"
                        )
                    )
                    job.write()
            except Exception:
                traceback.print_exc()

        # Compile the whole-clip video-free memory server-side as well, so it is
        # ready even if the browser tab that started the upload is closed.
        try:
            from dashboard import memory

            # `force=True` is important: a browser may have opened this clip
            # while perception was still running, producing an empty "thin"
            # vault. The DB is populated now, so rebuild it.
            memory.auto_compile("events:" + str(job.events), self.data_dir, force=True)
        except Exception:
            traceback.print_exc()


_PROCESSORS: dict[Path, UploadProcessor] = {}
_PROCESSORS_LOCK = threading.Lock()


def processor(data_dir: Path | str) -> UploadProcessor:
    root = Path(data_dir).expanduser().resolve()
    with _PROCESSORS_LOCK:
        instance = _PROCESSORS.get(root)
        if instance is None:
            instance = UploadProcessor(root)
            _PROCESSORS[root] = instance
        return instance


__all__ = [
    "DEFAULT_UPLOAD_VOCABULARY",
    "VIDEO_SUFFIXES",
    "UploadError",
    "UploadJob",
    "UploadProcessor",
    "new_upload_target",
    "parse_vocabulary",
    "processor",
    "probe_video",
    "register_upload",
]

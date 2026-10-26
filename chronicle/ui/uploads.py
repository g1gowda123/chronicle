"""Helpers for videos uploaded from the dashboard.

Uploads live under ``<data-dir>/uploads/<run-id>/`` and carry a small
``source.json`` manifest.  That gives the clip its own SQLite memory (exactly
like the bundled sources) and keeps it selectable after a server restart.
"""

from __future__ import annotations

import json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any

from chronicle.perception.config import PerceptionConfig
from chronicle.sources import ClipSource

UPLOAD_DIRNAME = "uploads"
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".m4v", ".mov", ".webm", ".ogv", ".ogg", ".avi", ".mkv"}
)
MAX_UPLOAD_BYTES = 4 * 1024**3


class UploadError(ValueError):
    """Raised when a selected file is not a supported video."""


def upload_root(data_dir: Path | str) -> Path:
    return Path(data_dir).expanduser().resolve() / UPLOAD_DIRNAME


def safe_upload_folder(name: str) -> str:
    """Reduce an uploaded filename to a path-safe folder suffix."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(name)).stem).strip("._-")
    return (cleaned or "video")[:48]


def new_upload_target(data_dir: Path | str, original_name: str) -> Path:
    """Create a unique upload folder and return its video path."""

    suffix = Path(str(original_name)).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        supported = ", ".join(sorted(VIDEO_SUFFIXES))
        raise UploadError(
            f"unsupported video type '{suffix or original_name}' (expected {supported})"
        )
    folder = upload_root(data_dir) / (
        f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}-"
        f"{safe_upload_folder(original_name)}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"source{suffix}"


def media_type(path: Path | str) -> str:
    guessed, _ = mimetypes.guess_type(Path(path).name)
    if guessed and guessed.startswith("video/"):
        return guessed
    return "video/mp4"


def probe_video(path: Path | str) -> tuple[int, int, float] | None:
    """Return ``(width, height, fps)`` when OpenCV can decode the first frame."""

    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return width, height, fps
    finally:
        capture.release()


def register_upload(
    data_dir: Path | str,
    video: Path | str,
    *,
    original_name: str,
    vocabulary: tuple[str, ...],
) -> ClipSource:
    """Persist the manifest for a freshly uploaded video and return its source."""

    data_root = Path(data_dir).expanduser().resolve()
    video_path = Path(video).expanduser().resolve()
    folder = video_path.parent
    original = Path(str(original_name))
    label = (original.stem or video_path.stem).strip() or "Uploaded video"
    source = ClipSource(
        key=f"upload-{folder.name}",
        label=label[:80],
        badge="UPLOAD",
        caption=f"{original.name} · live upload",
        description=(
            "Video selected in the dashboard. It is ingested immediately through "
            "the A → B → C pipeline."
        ),
        relative_path=video_path.relative_to(data_root).as_posix(),
        relative_db=f"{UPLOAD_DIRNAME}/{folder.name}/chronicle.db",
        zone_label="Uploaded Scene",
        vocabulary=tuple(vocabulary),
        questions=("what did I see?",),
    )
    manifest = {
        "key": source.key,
        "label": source.label,
        "caption": source.caption,
        "description": source.description,
        "video": video_path.name,
        "original_name": original.name,
        "zone_label": source.zone_label,
        "vocabulary": list(source.vocabulary),
        "questions": list(source.questions),
        "uploaded_at": time.time(),
    }
    (folder / "source.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return source


def _source_from_manifest(
    folder: Path, data_root: Path, manifest: dict[str, Any]
) -> ClipSource | None:
    video_name = Path(str(manifest.get("video") or "")).name
    if not video_name:
        return None
    video = folder / video_name
    if not video.is_file():
        return None
    try:
        relative_path = video.relative_to(data_root).as_posix()
    except ValueError:
        return None

    raw_vocabulary = manifest.get("vocabulary")
    vocabulary = (
        tuple(str(item).strip().lower() for item in raw_vocabulary if str(item).strip())
        if isinstance(raw_vocabulary, list)
        else ()
    )
    raw_questions = manifest.get("questions")
    questions = (
        tuple(str(item).strip() for item in raw_questions if str(item).strip())
        if isinstance(raw_questions, list)
        else ()
    )
    # Never let a hand-edited manifest shadow a bundled source key.
    key = str(manifest.get("key") or f"upload-{folder.name}")
    if not key.startswith("upload-"):
        key = f"upload-{folder.name}"
    return ClipSource(
        key=key,
        label=str(manifest.get("label") or video.stem),
        badge="UPLOAD",
        caption=str(manifest.get("caption") or f"{video.name} · live upload"),
        description=str(
            manifest.get("description")
            or "Video selected in the dashboard and processed live through the pipeline."
        ),
        relative_path=relative_path,
        relative_db=f"{UPLOAD_DIRNAME}/{folder.name}/chronicle.db",
        zone_label=str(manifest.get("zone_label") or "Uploaded Scene"),
        vocabulary=vocabulary or PerceptionConfig().vocabulary,
        questions=questions or ("what did I see?",),
    )


def load_upload_sources(data_dir: Path | str) -> list[ClipSource]:
    """Discover videos uploaded from the dashboard, newest first."""

    data_root = Path(data_dir).expanduser().resolve()
    root = upload_root(data_root)
    if not root.is_dir():
        return []
    folders = sorted(
        (item for item in root.iterdir() if item.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    sources: list[ClipSource] = []
    seen: set[str] = set()
    for folder in folders:
        try:
            payload = json.loads((folder / "source.json").read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        source = _source_from_manifest(folder, data_root, payload)
        if source is not None and source.key not in seen:
            seen.add(source.key)
            sources.append(source)
    return sources


__all__ = [
    "MAX_UPLOAD_BYTES",
    "UPLOAD_DIRNAME",
    "VIDEO_SUFFIXES",
    "UploadError",
    "load_upload_sources",
    "media_type",
    "new_upload_target",
    "probe_video",
    "register_upload",
    "safe_upload_folder",
    "upload_root",
]

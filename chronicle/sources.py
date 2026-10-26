"""Local demo sources shared by the CLI and the NiceGUI dashboard.

The team repo ships a synthetic fixture and one real HoloLens clip.  Additional
Aria Gen 2 clips are used when they have been fetched into ``data/real`` (the
same layout as the prototype).  Each source owns a separate SQLite database so
switching clips never mixes two episodes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from chronicle.perception.config import DEFAULT_VOCABULARY

DEFAULT_LIVE_STREAM = "0"


@dataclass(frozen=True)
class ClipSource:
    key: str
    label: str
    badge: str
    caption: str
    description: str
    relative_path: str
    relative_db: str
    zone_label: str
    vocabulary: tuple[str, ...]
    questions: tuple[str, ...]
    real: bool = True
    live: bool = False

    def video_path(self, data_dir: Path | str) -> Path:
        base = Path(data_dir).expanduser().resolve()
        primary = base / self.relative_path
        if primary.is_file():
            return primary
        fallback = base / "real" / Path(self.relative_path).name
        if fallback.is_file():
            return fallback
        return primary

    def db_path(self, data_dir: Path | str) -> Path:
        return Path(data_dir).expanduser().resolve() / self.relative_db

    def available(self, data_dir: Path | str) -> bool:
        if self.live:
            return True
        return self.video_path(data_dir).is_file()


def resolve_live_stream(value: str | None = None) -> str | int:
    """Resolve the live source: camera index, RTSP/HTTP URL, or video path.

    Precedence is ``value`` -> ``CHRONICLE_LIVE_STREAM`` -> camera ``0``.
    Numeric strings become OpenCV camera indices; everything else is passed
    through so ``rtsp://``, ``http://`` and local file sources all work.
    """
    text = value if value is not None else os.environ.get("CHRONICLE_LIVE_STREAM")
    text = (text or DEFAULT_LIVE_STREAM).strip()
    if not text:
        return 0
    return int(text) if text.isdigit() else text


SOURCES: tuple[ClipSource, ...] = (
    ClipSource(
        key="synthetic",
        label="Synthetic fixture",
        badge="SYNTHETIC",
        caption="Synthetic · deterministic test fixture",
        description=(
            "Generated 2D scene used as a zero-download smoke test. The colour "
            "detector, deterministic embeddings and template GraphRAG run fully "
            "offline."
        ),
        relative_path="demo/chronicle_demo.mp4",
        relative_db="demo/chronicle.db",
        zone_label="Synthetic Scene",
        vocabulary=DEFAULT_VOCABULARY,
        questions=("where are the keys?", "what did I see?"),
        real=False,
    ),
    ClipSource(
        key="live",
        label="Live camera stream",
        badge="LIVE",
        caption="Live camera stream · egocentric capture",
        description=(
            "Live webcam / RTSP / HTTP capture. Use --live-stream INDEX|URL|PATH "
            "to pick the device (default: camera 0), or set CHRONICLE_LIVE_STREAM."
        ),
        relative_path="live/stream.mp4",
        relative_db="live/chronicle.db",
        zone_label="Live Workspace",
        vocabulary=DEFAULT_VOCABULARY,
        questions=("what is on the table?", "what did I see?"),
        live=True,
        real=True,
    ),
    ClipSource(
        key="tabletop",
        label="HoloLens tabletop",
        badge="HOLOLENS 2",
        caption="HoloLens 2 · tabletop assembly",
        description=(
            "Genuine HoloLens 2 first-person tabletop footage: assembling "
            "Joy-Con controllers onto a Switch comfort grip."
        ),
        relative_path="real/holoassist_switch_34s.mp4",
        relative_db="real/hololens/chronicle.db",
        zone_label="Work Table",
        vocabulary=("joy-con", "switch console", "table", "cup", "mug", "keys"),
        questions=(
            "where is the joy-con?",
            "what did I do with the joy-con?",
            "what did I see?",
        ),
    ),
    ClipSource(
        key="walking",
        label="Aria walk · outdoor",
        badge="ARIA GEN 2",
        caption="Aria Gen 2 · outdoor walking",
        description="Genuine Aria Gen 2 first-person outdoor walking footage.",
        relative_path="real/walking/aria_walk_60s.mp4",
        relative_db="real/walking/chronicle.db",
        zone_label="Outdoor Path",
        vocabulary=("person", "cup", "backpack", "car", "table"),
        questions=("where was the cup last found?", "what did I see?"),
    ),
    ClipSource(
        key="cook",
        label="Aria cook · kitchen",
        badge="ARIA GEN 2",
        caption="Aria Gen 2 · indoor kitchen",
        description="Genuine Aria Gen 2 first-person kitchen footage.",
        relative_path="real/cook/aria_cook_45s.mp4",
        relative_db="real/cook/chronicle.db",
        zone_label="Kitchen",
        vocabulary=(
            "person",
            "cup",
            "bowl",
            "plate",
            "bottle",
            "knife",
            "cutting board",
            "sink",
            "stove",
            "countertop",
        ),
        questions=("where is the cup?", "what did I see?"),
    ),
    ClipSource(
        key="play",
        label="Aria play · living room",
        badge="ARIA GEN 2",
        caption="Aria Gen 2 · indoor living room",
        description=(
            "Genuine Aria Gen 2 first-person living-room footage with several "
            "people, a sofa, armchairs and a coffee table."
        ),
        relative_path="real/play/aria_play_45s.mp4",
        relative_db="real/play/chronicle.db",
        zone_label="Living Room",
        vocabulary=(
            "person",
            "cup",
            "bottle",
            "sofa",
            "chair",
            "table",
            "tv",
            "backpack",
        ),
        questions=(
            "where is the backpack?",
            "where is the cup?",
            "what did I see?",
        ),
    ),
)


def source_for_key(key: str, data_dir: Path | str) -> ClipSource:
    for source in SOURCES:
        if source.key == key:
            return source
    raise KeyError(f"unknown source {key!r}")


def source_for_video(video: Path | str, data_dir: Path | str) -> ClipSource | None:
    path = Path(video).expanduser().resolve()
    for source in SOURCES:
        if source.video_path(data_dir) == path:
            return source
    return None


def available_sources(data_dir: Path | str) -> list[ClipSource]:
    return [source for source in SOURCES if source.available(data_dir)]


def default_source(data_dir: Path | str) -> ClipSource:
    """Prefer a recorded real clip when one exists, otherwise the synthetic fixture.

    The live feed is opt-in (``--source live`` / ``CHRONICLE_LIVE_STREAM``);
    opening a camera by default would surprise the dashboard and smoke tests.
    """

    for source in SOURCES:
        if source.key in {"synthetic", "live"}:
            continue
        if source.available(data_dir):
            return source
    return source_for_key("synthetic", data_dir)


__all__ = [
    "ClipSource",
    "DEFAULT_LIVE_STREAM",
    "SOURCES",
    "available_sources",
    "default_source",
    "resolve_live_stream",
    "source_for_key",
    "source_for_video",
]

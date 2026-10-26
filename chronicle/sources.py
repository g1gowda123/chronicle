"""Local demo sources shared by the CLI and the NiceGUI dashboard.

The team repo ships a synthetic fixture and one real HoloLens clip.  Additional
Aria Gen 2 clips are used when they have been fetched into ``data/real`` (the
same layout as the prototype).  Each source owns a separate SQLite database so
switching clips never mixes two episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chronicle.perception.config import DEFAULT_VOCABULARY


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
        return self.video_path(data_dir).is_file()


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
    """Prefer a real clip when one exists, otherwise the synthetic fixture."""

    for source in SOURCES:
        if source.key != "synthetic" and source.available(data_dir):
            return source
    return source_for_key("synthetic", data_dir)


__all__ = [
    "ClipSource",
    "SOURCES",
    "available_sources",
    "default_source",
    "source_for_key",
    "source_for_video",
]

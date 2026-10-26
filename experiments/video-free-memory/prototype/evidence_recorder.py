"""Video-free evidence snapshots.

The video itself is disposable: the memory keeps the graph plus a small set of
JPEG keyframes captured at major events (state transitions, interactions, zone
changes and periodic scene anchors).  Snapshots are written next to the episode
database so an episode can be reviewed without retaining the source footage.

Layout::

    <episode>/evidence/index.jsonl
    <episode>/evidence/ev_00001_00012.40_transition.jpg

The index is append-only JSONL and the recorder keeps an in-memory copy for
cheap lookups.  Files are 640 px wide JPEGs (~30-80 KB each), so a 10-minute
episode typically costs a few megabytes instead of hundreds.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_WIDTH = 640
JPEG_QUALITY = 82
DEFAULT_MAX_ITEMS = 240


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    id: str
    t: float
    kind: str
    label: str
    file: str
    rel_path: str
    entity_id: str | None = None
    zone: str | None = None
    width: int = 0
    height: int = 0
    bytes: int = 0
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "t": round(float(self.t), 3),
            "kind": self.kind,
            "label": self.label,
            "file": self.file,
            "rel_path": self.rel_path,
            "entity_id": self.entity_id,
            "zone": self.zone,
            "width": self.width,
            "height": self.height,
            "bytes": self.bytes,
        }
        if self.meta:
            payload["meta"] = self.meta
        return payload


class EvidenceRecorder:
    """Thread-safe JPEG evidence writer with a JSONL index."""

    def __init__(
        self,
        episode_dir: str | Path,
        *,
        public_root: str | Path | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
        max_width: int = MAX_WIDTH,
        quality: int = JPEG_QUALITY,
    ) -> None:
        self.episode_dir = Path(episode_dir).expanduser().resolve()
        self.public_root = (
            Path(public_root).expanduser().resolve() if public_root is not None else None
        )
        self.root = self.episode_dir / "evidence"
        self.index_path = self.root / "index.jsonl"
        self.max_items = max(1, int(max_items))
        self.max_width = max(64, int(max_width))
        self.quality = max(30, min(100, int(quality)))
        self._lock = threading.RLock()
        self._entries: list[EvidenceEntry] = []
        self._load()

    # ------------------------------------------------------------------ reading
    def _load(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.index_path.is_file():
            return
        rows: list[EvidenceEntry] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                rows.append(
                    EvidenceEntry(
                        id=str(raw["id"]),
                        t=float(raw["t"]),
                        kind=str(raw["kind"]),
                        label=str(raw.get("label") or ""),
                        file=str(raw["file"]),
                        rel_path=str(raw.get("rel_path") or raw["file"]),
                        entity_id=raw.get("entity_id"),
                        zone=raw.get("zone"),
                        width=int(raw.get("width") or 0),
                        height=int(raw.get("height") or 0),
                        bytes=int(raw.get("bytes") or 0),
                        meta=raw.get("meta") if isinstance(raw.get("meta"), dict) else None,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        self._entries = rows

    def entries(self, *, limit: int | None = None, kind: str | None = None) -> list[EvidenceEntry]:
        with self._lock:
            rows = list(self._entries)
        if kind is not None:
            rows = [row for row in rows if row.kind == kind]
        if limit is not None:
            rows = rows[-max(0, int(limit)) :]
        return rows

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    def total_bytes(self) -> int:
        with self._lock:
            return sum(entry.bytes for entry in self._entries)

    def nearest(self, t: float, *, max_distance: float | None = None) -> EvidenceEntry | None:
        with self._lock:
            rows = list(self._entries)
        if not rows:
            return None
        target = float(t)
        best = min(rows, key=lambda entry: abs(entry.t - target))
        if max_distance is not None and abs(best.t - target) > float(max_distance):
            return None
        return best

    # ------------------------------------------------------------------ writing
    def capture(
        self,
        frame: Any,
        t: float,
        kind: str,
        label: str,
        *,
        entity_id: str | None = None,
        zone: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> EvidenceEntry | None:
        """Write one keyframe when the frame is usable.

        Returns ``None`` once the per-episode cap is reached, or when the frame
        cannot be encoded.  Never raises into the ingestion loop.
        """

        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        with self._lock:
            if len(self._entries) >= self.max_items:
                return None
            index = len(self._entries) + 1
            stamp = f"{float(t):08.2f}"
            safe_kind = "".join(ch for ch in kind if ch.isalnum() or ch in "-_") or "event"
            filename = f"ev_{index:05d}_{stamp}_{safe_kind}.jpg"
            target = self.root / filename
            try:
                width, height = self._write_jpeg(frame, target)
            except Exception:
                return None
            if width <= 0 or height <= 0:
                return None
            rel_path = filename
            if self.public_root is not None:
                try:
                    rel_path = target.relative_to(self.public_root).as_posix()
                except ValueError:
                    rel_path = target.as_posix()
            entry = EvidenceEntry(
                id=f"ev_{index:05d}",
                t=float(t),
                kind=kind,
                label=str(label or kind),
                file=filename,
                rel_path=rel_path,
                entity_id=entity_id,
                zone=zone,
                width=width,
                height=height,
                bytes=target.stat().st_size,
                meta=dict(meta or {}),
            )
            with self.index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")
            self._entries.append(entry)
            return entry

    def _write_jpeg(self, frame: Any, target: Path) -> tuple[int, int]:
        import cv2

        height, width = frame.shape[:2]
        scale = min(1.0, self.max_width / max(1, width))
        if scale < 0.999:
            frame = cv2.resize(
                frame,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = frame.shape[:2]
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        if not ok:
            return 0, 0
        target.write_bytes(encoded.tobytes())
        return int(width), int(height)

    # ------------------------------------------------------------------ helpers
    def prune_files(self, keep: int | None = None) -> int:
        """Remove files no longer referenced by the index (or beyond *keep*)."""

        with self._lock:
            limit = self.max_items if keep is None else max(0, int(keep))
            keep_names = {entry.file for entry in self._entries[-limit:]} if limit else set()
            removed = 0
            for path in self.root.glob("ev_*.jpg"):
                if path.name not in keep_names:
                    try:
                        path.unlink()
                        removed += 1
                    except OSError:
                        pass
            return removed

    def gallery(self, *, limit: int = 12) -> list[dict[str, Any]]:
        """Return the newest entries in a stable, UI-ready shape."""

        return [entry.to_dict() for entry in self.entries(limit=limit)]


def snapshot_kinds(entries: Iterable[EvidenceEntry]) -> dict[str, int]:
    """Small helper for summaries and tests."""

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.kind] = counts.get(entry.kind, 0) + 1
    return counts

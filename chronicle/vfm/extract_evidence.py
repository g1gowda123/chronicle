#!/usr/bin/env python3
"""Extract event snapshots from a video into a VFM vault, then optionally delete it.

This is the only step that needs the source video. It seeks to salient events,
writes 640 px JPEGs, records them in the vault, and can unlink the video so the
remaining memory is video-free.

Usage:
    python prototype/extract_evidence.py \
      --vault /tmp/cook.db \
      --video /home/g1/projects/chronicle/data/real/aria_cook_45s.mp4 \
      --out /tmp/evidence-cook \
      --data-root /tmp \
      --delete-video
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from .query import EventStore, load_store

MAJOR_MARKERS = ("INTERACTED", "HELD", "CONTAINED", "LOST", "NEW", "PICK", "PLACE", "INSERT")


def select_events(store: EventStore, *, min_gap: float = 2.0, max_items: int = 40) -> list[dict[str, Any]]:
    """Pick a small set of visually meaningful events."""

    selected: list[dict[str, Any]] = []
    last_t = -1e9
    for event in sorted(store.events, key=lambda row: float(row["t"])):
        kind = str(event["kind"])
        relation = f"{event.get('relation', '')} {event.get('object', '')}".upper()
        major = kind == "keyframe" or kind == "anchor" or any(
            marker in relation for marker in MAJOR_MARKERS
        )
        if not major:
            continue
        t = float(event["t"])
        if t - last_t < min_gap:
            continue
        selected.append(event)
        last_t = t
        if len(selected) >= max_items:
            break
    if len(selected) < 6:
        # Fall back to an even sample so a quiet episode is not blank.
        ordered = sorted(store.events, key=lambda row: float(row["t"]))
        step = max(1, len(ordered) // 8)
        for event in ordered[::step][:8]:
            if event not in selected:
                selected.append(event)
    return selected


def _read_frame(video: Path, t: float):
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if frames > 0 else None
        target = max(0.0, float(t))
        if duration is not None:
            target = min(target, max(0.0, duration - 1.0 / fps))
        capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(target * fps))
            ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def extract(
    vault: Path,
    video: Path,
    out_dir: Path,
    *,
    data_root: Path | None = None,
    delete_video: bool = False,
    max_items: int = 40,
    width: int = 640,
    quality: int = 82,
    image_format: str = "jpg",
    min_gap: float = 2.0,
) -> dict[str, Any]:
    import cv2

    image_format = "webp" if image_format.lower() == "webp" else "jpg"
    extension = "." + image_format
    encode_param = (
        [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
        if image_format == "webp"
        else [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )

    con = sqlite3.connect(str(vault))
    con.row_factory = sqlite3.Row
    con.execute(
        """
        create table if not exists evidence (
            id TEXT PRIMARY KEY, t REAL NOT NULL, kind TEXT NOT NULL, label TEXT,
            file TEXT NOT NULL, rel_path TEXT NOT NULL, bytes INTEGER NOT NULL DEFAULT 0,
            width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
            meta TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    con.commit()
    store = load_store(con)
    events = select_events(store, min_gap=min_gap, max_items=max_items)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    total_bytes = 0
    for index, event in enumerate(events, start=1):
        frame = _read_frame(video, float(event["t"]))
        if frame is None:
            continue
        height, width_px = frame.shape[:2]
        if width_px > width:
            scale = width / width_px
            frame = cv2.resize(
                frame,
                (width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            height, width_px = frame.shape[:2]
        filename = (
            f"ev_{index:04d}_{float(event['t']):08.2f}_{str(event['kind'])}{extension}"
        )
        path = out_dir / filename
        ok, encoded = cv2.imencode(extension, frame, encode_param)
        if not ok:
            continue
        path.write_bytes(encoded.tobytes())
        size = path.stat().st_size
        total_bytes += size
        rel_path = path.as_posix()
        if data_root is not None:
            try:
                rel_path = path.resolve().relative_to(data_root.resolve()).as_posix()
            except ValueError:
                rel_path = path.resolve().as_posix()
        con.execute(
            """
            insert or replace into evidence
                (id, t, kind, label, file, rel_path, bytes, width, height, meta)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"ev_{index:04d}",
                float(event["t"]),
                str(event["kind"]),
                str(event.get("subject") or ""),
                filename,
                rel_path,
                size,
                width_px,
                height,
                json.dumps(
                    {
                        "caption": str(event.get("object") or ""),
                        "format": image_format,
                        "quality": int(quality),
                    }
                ),
            ),
        )
        written += 1
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()

    deleted = False
    if delete_video and video.is_file():
        video.unlink()
        deleted = True

    return {
        "vault": str(vault),
        "video": str(video),
        "snapshots": written,
        "snapshot_bytes": total_bytes,
        "avg_bytes": round(total_bytes / written, 1) if written else 0,
        "width": width,
        "quality": quality,
        "format": image_format,
        "min_gap": min_gap,
        "video_deleted": deleted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--delete-video", action="store_true")
    parser.add_argument("--max-items", type=int, default=40)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--quality", type=int, default=82)
    parser.add_argument("--format", choices=("jpg", "webp"), default="jpg")
    parser.add_argument("--min-gap", type=float, default=2.0)
    args = parser.parse_args(argv)
    summary = extract(
        args.vault.expanduser().resolve(),
        args.video.expanduser().resolve(),
        args.out.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve() if args.data_root else None,
        delete_video=args.delete_video,
        max_items=max(1, args.max_items),
        width=max(160, args.width),
        quality=max(30, min(100, args.quality)),
        image_format=args.format,
        min_gap=max(0.0, args.min_gap),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

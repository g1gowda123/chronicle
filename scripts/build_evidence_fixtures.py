#!/usr/bin/env python3
"""Build the stored evidence images used by grounded query answers.

Each episode database can have an ``evidence/`` folder beside it::

    data/real/hololens/chronicle.db
    data/real/hololens/evidence/index.jsonl
    data/real/hololens/evidence/ev_0001_pick_up.jpg

The runtime reads ``index.jsonl`` first and only falls back to extracting a
video frame when no stored image fits the answer.  This script regenerates the
bundled indexes from the synthetic event stream, the HoloAssist interaction
strips and sampled Aria keyframes.

Usage::

    .venv/bin/python scripts/build_evidence_fixtures.py
    .venv/bin/python scripts/build_evidence_fixtures.py --source tabletop
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronicle.semantic.actions import coerce_action  # noqa: E402
from chronicle.sources import (  # noqa: E402
    SOURCES,
    ClipSource,
    source_for_key,
)

MAX_WIDTH = 640
JPEG_QUALITY = 82


def _write_jpeg(source: Path, target: Path, *, max_width: int = MAX_WIDTH) -> bool:
    import cv2

    image = cv2.imread(str(source))
    if image is None or image.size == 0:
        return False
    height, width = image.shape[:2]
    scale = min(1.0, max_width / max(1, width))
    if scale < 0.999:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return bool(
        cv2.imwrite(
            str(target),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
    )


def _extract_jpeg(video: Path, t: float, target: Path) -> bool:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        return False
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            return False
    finally:
        capture.release()
    height, width = frame.shape[:2]
    scale = min(1.0, MAX_WIDTH / max(1, width))
    if scale < 0.999:
        frame = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return bool(
        cv2.imwrite(
            str(target),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
    )


def _video_duration(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    return frames / fps if fps > 0 and frames > 0 else 0.0


def _action_from_event(event: str) -> str | None:
    word = event.strip().split(" ", 1)[0]
    try:
        return coerce_action(word).value
    except ValueError:
        return None


def _synthetic_rows(source: ClipSource, data_dir: Path, evidence_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    events_path = data_dir / "demo_events.jsonl"
    labels: dict[str, str] = {}
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            data = event.get("data") or {}
            if event.get("kind") == "entity" and data.get("id"):
                labels[str(data["id"])] = str(data.get("label") or "")
        for event in _iter_events(events_path):
            data = event.get("data") or {}
            if event.get("kind") != "interaction":
                continue
            url = data.get("strip_url") or data.get("crop_url")
            if not url:
                continue
            kind = "interaction_strip" if data.get("strip_url") else "interaction_crop"
            source_path = data_dir / str(url).removeprefix("/media/")
            if not source_path.is_file():
                continue
            name = f"ev_{len(rows) + 1:04d}_{kind}.jpg"
            if _write_jpeg(source_path, evidence_dir / name):
                entity_id = str(data.get("entity_id") or "")
                _append_existing(
                    rows,
                    name=name,
                    t=float(event.get("t") or 0.0),
                    kind=kind,
                    label=labels.get(entity_id, ""),
                    entity_id=entity_id or None,
                    action=str(data.get("action") or "") or None,
                    caption=str(data.get("caption") or ""),
                    clip_sim=data.get("clip_sim"),
                )
    for zone in sorted((data_dir / "zones").glob("*.png")):
        name = f"ev_{len(rows) + 1:04d}_zone_keyframe.jpg"
        if _write_jpeg(zone, evidence_dir / name):
            _append_existing(
                rows,
                name=name,
                t=0.0,
                kind="zone_keyframe",
                label=zone.stem.replace("_", " ").title(),
                zone_id=zone.stem,
            )
    return rows


def _iter_events(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def _tabletop_rows(
    source: ClipSource,
    data_dir: Path,
    evidence_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    annotation = data_dir / "real" / "holoassist_switch_34s.json"
    timeline: list[tuple[float, str]] = []
    if annotation.is_file():
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        for item in payload.get("timeline", []):
            if isinstance(item, dict) and item.get("event") is not None:
                timeline.append((float(item.get("t") or 0.0), str(item["event"])))

    clips = sorted((ROOT / "runs" / "real_clips").glob("*.json"))
    for index, clip_path in enumerate(clips, start=1):
        try:
            clip = json.loads(clip_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        contact = float(clip.get("contact_ts") or 0.0)
        action = _nearest_action(contact, timeline)
        label = str(clip.get("label") or "joy-con")
        name = f"ev_{len(rows) + 1:04d}_interaction_strip.jpg"
        image = clip_path.with_suffix(".jpg")
        if image.is_file() and _write_jpeg(image, evidence_dir / name):
            _append_existing(
                rows,
                name=name,
                t=contact,
                kind="interaction_strip",
                label=label,
                entity_id=label,
                zone_id=source.zone_label,
                action=action,
                caption=f"Interaction strip {index} at t={contact:.2f}s",
                clip_sim=None,
            )

    if not rows:
        # Fallback for a fresh clone without runs/real_clips: sample the clip
        # at the published action times.
        video = source.video_path(data_dir)
        for t, event in timeline:
            action = _action_from_event(event)
            if action is None:
                continue
            name = f"ev_{len(rows) + 1:04d}_interaction_strip.jpg"
            if _extract_jpeg(video, t, evidence_dir / name):
                _append_existing(
                    rows,
                    name=name,
                    t=t,
                    kind="interaction_strip",
                    label="joy-con",
                    entity_id="joy-con",
                    zone_id=source.zone_label,
                    action=action,
                    caption=event,
                )
    return rows


def _nearest_action(t: float, timeline: list[tuple[float, str]]) -> str | None:
    if not timeline:
        return None
    nearest = min(timeline, key=lambda item: abs(item[0] - t))
    if abs(nearest[0] - t) > 3.0:
        return None
    return _action_from_event(nearest[1])


def _sample_rows(
    source: ClipSource,
    data_dir: Path,
    evidence_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    video = source.video_path(data_dir)
    duration = _video_duration(video)
    if duration <= 0:
        return rows
    for fraction in (0.1, 0.5, 0.9):
        t = duration * fraction
        name = f"ev_{len(rows) + 1:04d}_zone_keyframe.jpg"
        if _extract_jpeg(video, t, evidence_dir / name):
            _append_existing(
                rows,
                name=name,
                t=t,
                kind="zone_keyframe",
                label=source.zone_label,
                zone_id=source.key,
                caption=f"{source.label} keyframe at t={t:.1f}s",
            )
    return rows


def _append_existing(
    rows: list[dict[str, Any]],
    *,
    name: str,
    t: float,
    kind: str,
    label: str = "",
    entity_id: str | None = None,
    zone_id: str | None = None,
    action: str | None = None,
    caption: str = "",
    clip_sim: Any = None,
) -> None:
    rows.append(
        {
            "file": name,
            "rel_path": f"evidence/{name}",
            "t": round(float(t), 3),
            "kind": kind,
            "label": label,
            "entity_id": entity_id,
            "zone_id": zone_id,
            "action": action,
            "caption": caption,
            "clip_sim": float(clip_sim) if clip_sim is not None else None,
        }
    )


def build_source(source: ClipSource, data_dir: Path) -> int:
    episode_dir = source.db_path(data_dir).parent
    evidence_dir = episode_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    # Drop only files this script owns, never user-provided evidence.
    for old in evidence_dir.glob("ev_*.jpg"):
        old.unlink(missing_ok=True)
    if source.key == "synthetic":
        rows = _synthetic_rows(source, data_dir, evidence_dir)
    elif source.key == "tabletop":
        rows = _tabletop_rows(source, data_dir, evidence_dir)
    else:
        rows = _sample_rows(source, data_dir, evidence_dir)
    print(f"{source.key}: {len(rows)} evidence rows -> {evidence_dir}")
    if not rows:
        return len(rows)
    index = evidence_dir / "index.jsonl"
    index.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--source", action="append", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.source:
        sources = [source_for_key(key, data_dir) for key in args.source]
    else:
        sources = [source for source in SOURCES if source.video_path(data_dir).is_file()]
    if not sources:
        print("no sources available", file=sys.stderr)
        return 1
    total = 0
    for source in sources:
        total += build_source(source, data_dir)
    print(f"total evidence rows: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

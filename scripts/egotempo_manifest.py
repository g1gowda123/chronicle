"""Build a clip manifest for the EgoTempo open-ended QA benchmark.

EgoTempo clip IDs follow ``{video_uid}_{start}_{end}``.  The same source video
can contribute several clips, so the manifest groups clips by ``video_uid`` and
tracks which questions belong to each clip.  This is the file the downloader
and the evaluation runner consume.

Example
-------
    python scripts/egotempo_manifest.py \
        --qa data/benchmarks/egotempo/egotempo_openQA.json \
        --out data/benchmarks/egotempo/manifest.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def parse_clip_id(clip_id: str) -> tuple[str, float, float]:
    value = clip_id[:-4] if clip_id.endswith(".mp4") else clip_id
    parts = value.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"clip_id must be {{video_uid}}_{{start}}_{{end}}: {clip_id!r}")
    video_uid, start_raw, end_raw = parts
    if not video_uid:
        raise ValueError(f"clip_id has an empty video_uid: {clip_id!r}")
    start, end = float(start_raw), float(end_raw)
    if end <= start:
        raise ValueError(f"clip has non-positive duration: {clip_id!r}")
    return video_uid, start, end


def build_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    annotations = payload.get("annotations") or []
    if not annotations:
        raise ValueError("QA file has no annotations")

    clips: dict[str, dict[str, Any]] = {}
    video_uids: set[str] = set()
    category_counts: Counter[str] = Counter()
    for row in annotations:
        clip_id = str(row["clip_id"])
        video_uid, start, end = parse_clip_id(clip_id)
        video_uids.add(video_uid)
        category = str(row.get("question_type", "unknown"))
        category_counts[category] += 1
        entry = clips.setdefault(
            clip_id,
            {
                "clip_id": clip_id,
                "video_uid": video_uid,
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "questions": [],
            },
        )
        entry["questions"].append(
            {
                "question_id": str(row["question_id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "question_type": category,
            }
        )

    ordered_clips = sorted(
        clips.values(), key=lambda item: (item["video_uid"], item["start"], item["end"])
    )
    by_video: dict[str, list[str]] = {}
    for clip in ordered_clips:
        by_video.setdefault(clip["video_uid"], []).append(clip["clip_id"])

    return {
        "source": "EgoTempo (CVPR 2025)",
        "license": "CC-BY-4.0 for the QA annotations; Ego4D license for the videos",
        "questions": len(annotations),
        "clip_count": len(ordered_clips),
        "videos": len(video_uids),
        "total_clip_seconds": round(sum(c["duration"] for c in ordered_clips), 1),
        "category_counts": dict(sorted(category_counts.items())),
        "video_uids": sorted(video_uids),
        "clips": ordered_clips,
        "clips_by_video": by_video,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("data/benchmarks/egotempo/egotempo_openQA.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmarks/egotempo/manifest.json"),
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.qa.read_text(encoding="utf-8"))
    manifest = build_manifest(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"{manifest['questions']} questions, {manifest['clip_count']} clips, "
        f"{manifest['videos']} Ego4D videos, "
        f"{manifest['total_clip_seconds'] / 60:.1f} clip-minutes -> {args.out}"
    )
    for category, count in manifest["category_counts"].items():
        print(f"  {category:<28} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download and trim the EgoTempo clips.

EgoTempo questions refer to clips cropped from Ego4D videos.  The official
notebook downloads the ``video_540ss`` version of each required Ego4D video and
trims the question window with ffmpeg.  This script does the same, in batches,
and writes a ``clips.json`` index used by the evaluation runner.

Prerequisites
-------------
1. An approved Ego4D license and AWS credentials in ``~/.aws/credentials``
   (https://ego4d-data.org/docs/start-here/).
2. The Ego4D CLI::

       pip install git+https://github.com/facebookresearch/Ego4d.git#egg=ego4d

Example
-------
    # Pilot: first 10 source videos only
    python scripts/prepare_egotempo.py --limit-videos 10

    # Full benchmark (downloads 222 videos, trims 365 clips)
    python scripts/prepare_egotempo.py

    # Everything already downloaded, only trim missing clips
    python scripts/prepare_egotempo.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _find_video(root: Path, video_uid: str) -> Path | None:
    direct = root / f"{video_uid}.mp4"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(f"{video_uid}.mp4"))
    return matches[0] if matches else None


def _trim(source: Path, destination: Path, start: float, end: float, *, dry_run: bool) -> None:
    start = max(0.0, float(start))
    end = max(start + 0.05, float(end))
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(source),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(destination),
    ]
    if dry_run:
        print("  " + " ".join(command))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _download(video_uids: Sequence[str], output_dir: Path, *, dry_run: bool) -> None:
    if not video_uids:
        return
    command = [
        "ego4d",
        "--video_uids",
        *video_uids,
        "-o",
        str(output_dir),
        "--datasets",
        "video_540ss",
        "--yes",
    ]
    print(f"Downloading {len(video_uids)} Ego4D videos into {output_dir}")
    if dry_run:
        print("  " + " ".join(command))
        return
    if shutil.which("ego4d") is None:
        raise RuntimeError(
            "ego4d CLI not found. Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/Ego4d.git#egg=ego4d\n"
            "and configure AWS credentials from your approved Ego4D license."
        )
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/benchmarks/egotempo/manifest.json"),
    )
    parser.add_argument(
        "--ego4d-dir",
        type=Path,
        default=Path("data/benchmarks/egotempo/ego4d"),
        help="where Ego4D video_540ss files are stored",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("data/benchmarks/egotempo/clips"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/benchmarks/egotempo/clips.json"),
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="pilot mode: only the first N source videos",
    )
    parser.add_argument(
        "--video-uid",
        action="append",
        default=None,
        help="restrict to one or more source video UIDs",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips = list(manifest.get("clips") or ())
    if not clips:
        raise SystemExit(f"no clips in {args.manifest}")

    selected_uids = list(args.video_uid or manifest.get("video_uids") or ())
    if args.limit_videos is not None:
        selected_uids = selected_uids[: max(0, args.limit_videos)]
    selected = set(selected_uids)
    clips = [clip for clip in clips if clip["video_uid"] in selected]
    missing = [uid for uid in selected_uids if _find_video(args.ego4d_dir, uid) is None]

    print(f"Selected {len(clips)} clips from {len(selected_uids)} source videos")
    print(f"Already downloaded: {len(selected_uids) - len(missing)}; missing: {len(missing)}")

    if not args.skip_download and missing:
        _download(missing, args.ego4d_dir, dry_run=args.dry_run)

    index: dict[str, str] = {}
    trimmed = 0
    skipped = 0
    for clip in clips:
        destination = args.clips_dir / f"{clip['clip_id']}.mp4"
        if destination.is_file() and destination.stat().st_size > 0:
            index[clip["clip_id"]] = str(destination)
            skipped += 1
            continue
        source = _find_video(args.ego4d_dir, clip["video_uid"])
        if source is None:
            print(f"  missing source for {clip['clip_id']} ({clip['video_uid']})", file=sys.stderr)
            continue
        print(f"  trim {clip['clip_id']} [{clip['start']:.1f}, {clip['end']:.1f}]")
        _trim(source, destination, clip["start"], clip["end"], dry_run=args.dry_run)
        index[clip["clip_id"]] = str(destination)
        trimmed += 1

    if not args.dry_run:
        args.index.parent.mkdir(parents=True, exist_ok=True)
        args.index.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Trimmed {trimmed}; already present {skipped}; index -> {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

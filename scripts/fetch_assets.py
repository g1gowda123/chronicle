#!/usr/bin/env python3
"""Download the optional model assets for Chronicle Role A.

Usage::

    python scripts/fetch_assets.py
    python scripts/fetch_assets.py --force

Assets land in ``models/`` (git-ignored) and are detected automatically by the
CLI and :class:`chronicle.perception.config.PerceptionConfig`.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ASSETS: dict[str, str] = {
    "yolov8s-worldv2.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
        "yolov8s-worldv2.pt"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def download(url: str, destination: Path, *, chunk_size: int = 1 << 20) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "chronicle-assets/0.1"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:6.1f} / {total / 1e6:6.1f} MB", end="")
        if total and written != total:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"incomplete download for {destination.name}: {written} of {total} bytes"
            )
    temporary.replace(destination)
    print(f"\r  saved {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="models", help="destination directory")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument(
        "--only",
        choices=("all", "yolo", "hands"),
        default="all",
        help="restrict downloads to one backend's asset",
    )
    args = parser.parse_args(argv)

    models = Path(args.models_dir).expanduser()
    models.mkdir(parents=True, exist_ok=True)
    selected = {
        "all": list(ASSETS),
        "yolo": ["yolov8s-worldv2.pt"],
        "hands": ["hand_landmarker.task"],
    }[args.only]

    failures = 0
    for name in selected:
        destination = models / name
        if destination.exists() and destination.stat().st_size > 0 and not args.force:
            print(f"skip {destination} (already present)")
            continue
        print(f"fetching {name} -> {destination}")
        try:
            download(ASSETS[name], destination)
        except Exception as exc:
            failures += 1
            print(f"  failed: {exc}", file=sys.stderr)

    if failures:
        print(
            "Some assets could not be downloaded. The pipeline still supports "
            "--detector color --hands color --embedder color for the offline demo.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

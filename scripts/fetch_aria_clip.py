#!/usr/bin/env python3
"""Fetch and trim Aria Gen 2 Pilot preview clips.

The Dataset Explorer exposes short-lived signed URLs for full-length preview
videos.  Those files are several hundred MB each, and their ``moov`` atom is at
the end, so remote ``ffmpeg -ss`` seeking is unreliable.  This helper downloads
the preview once into a local cache and trims the requested window locally.

Only the small manifest in ``data/aria_gen2pilot.json`` is committed; downloaded
videos stay under ``data/footage/`` (ignored by git) or another local path.

Examples::

    python scripts/fetch_aria_clip.py --sequence eat_0 --dry-run
    python scripts/fetch_aria_clip.py --sequence eat_0
    python scripts/fetch_aria_clip.py --all
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/aria_gen2pilot.json"
DEFAULT_API = "https://explorer.projectaria.com/data/gen2pilot/download_links"


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("sequences"), list):
        raise ValueError(f"{path}: expected a 'sequences' list")
    return payload


def _lookup(payload: dict, name: str) -> dict:
    for item in payload.get("sequences", []):
        if item.get("name") == name or item.get("sequence") == name:
            return item
    available = ", ".join(str(item.get("name")) for item in payload.get("sequences", []))
    raise KeyError(f"unknown sequence {name!r}; available: {available}")


def resolve_main_rgb(api: str, sequence: str) -> dict:
    with urllib.request.urlopen(api, timeout=60) as response:
        payload = json.load(response)
    try:
        entry = payload["sequences"][sequence]["video_main_rgb"]
    except KeyError as exc:
        raise KeyError(f"{sequence}: no video_main_rgb entry in {api}") from exc
    url = entry.get("download_url")
    if not url:
        raise ValueError(f"{sequence}: video_main_rgb has no download_url")
    return {
        "url": url,
        "filename": entry.get("filename") or f"{sequence}_video_main_rgb.mp4",
        "bytes": int(entry.get("file_size_bytes") or 0),
        "sha1": entry.get("sha1sum"),
    }


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download(url: str, destination: Path, *, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"[cache] {destination} ({destination.stat().st_size} bytes)", flush=True)
        return
    if shutil.which("curl"):
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ]
    else:
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys,urllib.request;"
                "urllib.request.urlretrieve(sys.argv[1], sys.argv[2])"
            ),
            url,
            str(destination),
        ]
    _run(cmd)


def trim(source: Path, destination: Path, start_s: float, duration_s: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration_s:.3f}",
            "-vf",
            "scale=640:-2",
            "-r",
            "10",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def fetch_one(
    item: dict,
    *,
    api: str,
    cache_dir: Path,
    force: bool = False,
    dry_run: bool = False,
    remove_full: bool = False,
) -> Path:
    sequence = str(item["sequence"])
    clip = (ROOT / item["clip"]).resolve()
    if dry_run:
        print(f"[plan] {item['name']}: {item['domain']} {item['start_s']:.1f}s+{item['duration_s']:.1f}s")
        print(f"       API      {api}")
        print(f"       cache    {cache_dir / (sequence + '_main_rgb.mp4')}")
        print(f"       clip     {clip}")
        return clip

    info = resolve_main_rgb(api, sequence)
    full = cache_dir / f"{sequence}_main_rgb.mp4"
    print(
        f"[fetch] {sequence}: {info['bytes'] / 1e6:.1f} MB -> {full}",
        flush=True,
    )
    download(info["url"], full, force=force)
    if clip.exists() and not force:
        print(f"[clip] keeping existing {clip}", flush=True)
    else:
        trim(full, clip, float(item["start_s"]), float(item["duration_s"]))
    if remove_full:
        full.unlink(missing_ok=True)
        print(f"[cache] removed {full}", flush=True)
    print(f"[done] {clip}", flush=True)
    return clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/chronicle/aria_gen2pilot")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remove-full", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    names = [item["name"] for item in manifest["sequences"]]
    requested = args.sequence or (names if args.all else [])
    if not requested:
        parser.error("pass --sequence NAME (repeatable) or --all")

    for name in requested:
        fetch_one(
            _lookup(manifest, name),
            api=args.api,
            cache_dir=args.cache_dir,
            force=args.force,
            dry_run=args.dry_run,
            remove_full=args.remove_full,
        )


if __name__ == "__main__":
    main()

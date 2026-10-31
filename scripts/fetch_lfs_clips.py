#!/usr/bin/env python3
"""Fetch Git LFS-tracked research clips without requiring git-lfs.

A plain checkout of this repository contains small Git LFS pointer files for a
few demo MP4s.  GitHub's media endpoint serves the real LFS bytes, so this
helper can restore them using only ``git`` and ``curl`` (or urllib).  It
verifies the pointer's SHA-256 and byte size before replacing the pointer.

Examples::

    python scripts/fetch_lfs_clips.py --dry-run
    python scripts/fetch_lfs_clips.py
    python scripts/fetch_lfs_clips.py data/footage/aria_cook_45s.mp4
    python scripts/fetch_lfs_clips.py --all --ref origin/research/spore-perception

Only files that are actually LFS pointers are touched.  Downloaded media stays
ignored by git; it is not added to a commit.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"
MEDIA_HOST = "https://media.githubusercontent.com/media"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_pointer(path: Path) -> dict[str, str | int] | None:
    """Return LFS pointer metadata, or None if *path* is not a pointer."""
    raw = path.read_bytes()
    if len(raw) > 1024 or not raw.startswith(POINTER_PREFIX.encode()):
        return None
    fields: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        key, _, value = line.partition(" ")
        if key and value:
            fields[key] = value
    oid = fields.get("oid", "")
    if oid.startswith("sha256:"):
        oid = oid.removeprefix("sha256:")
    size_text = fields.get("size", "")
    if not _SHA256_RE.match(oid):
        raise ValueError(f"{path}: invalid LFS oid {oid!r}")
    if not size_text.isdigit() or int(size_text) <= 0:
        raise ValueError(f"{path}: invalid LFS size {size_text!r}")
    return {"oid": oid, "size": int(size_text)}


def remote_slug(remote: str) -> str:
    """Convert a GitHub remote URL to ``owner/repo``."""
    remote = remote.strip()
    match = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?$", remote)
    if not match:
        raise ValueError(f"not a GitHub remote: {remote!r}")
    return f"{match.group(1)}/{match.group(2)}"


def media_url(slug: str, ref: str, path: str) -> str:
    quoted_ref = urllib.parse.quote(ref, safe="/")
    quoted_path = urllib.parse.quote(path, safe="/")
    return f"{MEDIA_HOST}/{slug}/{quoted_ref}/{quoted_path}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
    ) as handle:
        partial = Path(handle.name)
    try:
        if shutil.which("curl"):
            cmd = [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--output",
                str(partial),
                url,
            ]
            subprocess.run(cmd, check=True)
        else:
            urllib.request.urlretrieve(url, partial)
        if destination.exists():
            destination.unlink()
        partial.replace(destination)
        destination.chmod(0o644)
    finally:
        if partial.exists():
            partial.unlink()


def fetch(path: Path, *, slug: str, ref: str, dry_run: bool = False) -> bool:
    """Fetch one LFS pointer.  Return True when media was downloaded."""
    relative = path.relative_to(ROOT).as_posix()
    pointer = parse_pointer(path)
    if pointer is None:
        print(f"[skip] {relative} is not an LFS pointer", flush=True)
        return False
    url = media_url(slug, ref, relative)
    if dry_run:
        print(f"[dry-run] {relative} <- {url} ({pointer['size']} bytes)", flush=True)
        return False
    _download(url, path)
    actual_size = path.stat().st_size
    if actual_size != pointer["size"]:
        raise RuntimeError(
            f"{relative}: size mismatch: expected {pointer['size']}, got {actual_size}"
        )
    actual_oid = sha256_file(path)
    if actual_oid != pointer["oid"]:
        raise RuntimeError(
            f"{relative}: sha256 mismatch: expected {pointer['oid']}, got {actual_oid}"
        )
    print(f"[ok] {relative} ({actual_size} bytes)", flush=True)
    return True


def pointer_paths() -> list[Path]:
    raw = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.mp4", "*.webm", "*.vrs"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [ROOT / name for name in raw.decode().split("\0") if name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="LFS-tracked files to fetch")
    parser.add_argument(
        "--all",
        action="store_true",
        help="fetch every tracked LFS pointer under data/footage",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="git ref whose LFS content to download (default: HEAD)",
    )
    parser.add_argument("--repo", help="owner/repo override")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.paths:
        paths = [(ROOT / item).resolve() for item in args.paths]
    elif args.all:
        paths = pointer_paths()
    else:
        parser.error("pass one or more paths, or --all")

    slug = args.repo or remote_slug(_git("remote", "get-url", "origin"))
    fetched = 0
    for path in paths:
        if not path.exists():
            print(f"[missing] {path}", file=sys.stderr, flush=True)
            continue
        try:
            fetched += int(fetch(path, slug=slug, ref=args.ref, dry_run=args.dry_run))
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"[error] {path}: {exc}", file=sys.stderr, flush=True)
    print(f"[done] fetched {fetched} file(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

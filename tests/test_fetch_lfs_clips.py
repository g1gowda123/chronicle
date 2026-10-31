"""Tests for the Git LFS demo-clip fetcher."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fetch_lfs_clips import (  # noqa: E402
    POINTER_PREFIX,
    media_url,
    parse_pointer,
    remote_slug,
    sha256_file,
)


def _pointer(tmp_path: Path, payload: bytes) -> Path:
    oid = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "clip.mp4"
    path.write_text(
        f"{POINTER_PREFIX}\n"
        f"oid sha256:{oid}\n"
        f"size {len(payload)}\n"
    )
    return path


def test_parse_pointer_returns_oid_and_size(tmp_path: Path) -> None:
    payload = b"not really a video, but enough for a hash"
    path = _pointer(tmp_path, payload)
    pointer = parse_pointer(path)
    assert pointer == {"oid": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def test_parse_pointer_rejects_media(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    assert parse_pointer(path) is None


def test_remote_slug_supports_https_and_ssh() -> None:
    assert remote_slug("https://github.com/g1gowda123/chronicle.git") == "g1gowda123/chronicle"
    assert remote_slug("git@github.com:g1gowda123/chronicle.git") == "g1gowda123/chronicle"


def test_media_url_keeps_ref_and_path_readable() -> None:
    url = media_url("owner/repo", "research/spore-perception", "data/footage/a b.mp4")
    assert url == (
        "https://media.githubusercontent.com/media/owner/repo/"
        "research/spore-perception/data/footage/a%20b.mp4"
    )


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"abc" * 100_000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()

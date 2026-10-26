"""Video upload helpers for the dashboard (no NiceGUI required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronicle.ui.uploads import (
    UploadError,
    load_upload_sources,
    media_type,
    new_upload_target,
    probe_video,
    register_upload,
    safe_upload_folder,
    upload_root,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo" / "chronicle_demo.mp4"


def test_safe_upload_folder_strips_paths() -> None:
    assert safe_upload_folder("../../Living Room.mp4") == "Living_Room"
    assert safe_upload_folder("...") == "video"


def test_new_upload_target_rejects_unknown_suffix(tmp_path: Path) -> None:
    with pytest.raises(UploadError):
        new_upload_target(tmp_path, "notes.txt")
    assert not upload_root(tmp_path).exists()


def test_new_upload_target_creates_unique_folder(tmp_path: Path) -> None:
    first = new_upload_target(tmp_path, "Living Room.webm")
    second = new_upload_target(tmp_path, "Living Room.webm")
    assert first.suffix == ".webm"
    assert second.suffix == ".webm"
    assert first != second
    assert first.parent.parent == upload_root(tmp_path)
    assert first.parent.is_dir()
    assert second.parent.is_dir()


def test_register_upload_roundtrip(tmp_path: Path) -> None:
    target = new_upload_target(tmp_path, "Living Room.mp4")
    target.write_bytes(b"video-bytes")

    source = register_upload(
        tmp_path,
        target,
        original_name="Living Room.mp4",
        vocabulary=("cup", "keys"),
    )

    assert source.key == f"upload-{target.parent.name}"
    assert source.label == "Living Room"
    assert source.badge == "UPLOAD"
    assert source.video_path(tmp_path) == target
    assert source.db_path(tmp_path) == target.parent / "chronicle.db"
    assert source.vocabulary == ("cup", "keys")
    assert load_upload_sources(tmp_path) == [source]

    manifest = json.loads((target.parent / "source.json").read_text(encoding="utf-8"))
    assert manifest["video"] == target.name
    assert manifest["original_name"] == "Living Room.mp4"


def test_load_upload_sources_ignores_invalid_manifests(tmp_path: Path) -> None:
    root = upload_root(tmp_path)
    broken = root / "broken"
    broken.mkdir(parents=True)
    (broken / "source.json").write_text("{not json", encoding="utf-8")
    missing = root / "missing-video"
    missing.mkdir()
    (missing / "source.json").write_text(
        json.dumps({"video": "gone.mp4", "label": "gone"}),
        encoding="utf-8",
    )
    assert load_upload_sources(tmp_path) == []


def test_load_upload_sources_cannot_shadow_bundled_keys(tmp_path: Path) -> None:
    target = new_upload_target(tmp_path, "clip.mp4")
    target.write_bytes(b"video")
    manifest_path = target.parent / "source.json"
    manifest_path.write_text(
        json.dumps({"video": target.name, "key": "synthetic", "label": "fake"}),
        encoding="utf-8",
    )
    sources = load_upload_sources(tmp_path)
    assert [source.key for source in sources] == [f"upload-{target.parent.name}"]


def test_media_type_uses_video_suffix(tmp_path: Path) -> None:
    assert media_type(tmp_path / "clip.webm") == "video/webm"
    assert media_type(tmp_path / "clip.unknown") == "video/mp4"


@pytest.mark.skipif(not DEMO.is_file(), reason="demo clip is not present")
def test_probe_video_reads_the_demo_clip() -> None:
    probe = probe_video(DEMO)
    assert probe is not None
    width, height, fps = probe
    assert width == 640
    assert height == 360
    assert fps > 0

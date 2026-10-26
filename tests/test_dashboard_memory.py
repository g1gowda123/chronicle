"""VFM bridge for the zine dashboard (clip → DB/video resolution + API)."""

from __future__ import annotations

import json
from pathlib import Path

import importlib

from fastapi.testclient import TestClient

from dashboard import memory
import dashboard.app as app_module


def _fake_data(root: Path) -> None:
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "chronicle_demo.mp4").write_bytes(b"x" * 2048)
    (root / "demo" / "chronicle.db").write_bytes(b"")
    (root / "demo_events.jsonl").write_text(
        json.dumps(
            {
                "v": 1,
                "kind": "session",
                "t": 0.0,
                "data": {
                    "title": "Synthetic demo",
                    "video_url": "/media/demo/chronicle_demo.mp4",
                    "duration": 12.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "real" / "cook").mkdir(parents=True)
    (root / "real" / "cook" / "aria_cook_45s.mp4").write_bytes(b"x" * 2048)
    (root / "real" / "cook" / "chronicle.db").write_bytes(b"")


def test_resolve_demo_clip_to_db_and_video(tmp_path: Path) -> None:
    _fake_data(tmp_path)
    source = memory.resolve("events:demo_events.jsonl", tmp_path)
    assert source is not None
    assert source.key.startswith("events:")
    assert source.db is not None and source.db.is_file()
    assert source.video is not None and source.video.is_file()


def test_resolve_real_clip_to_registered_db(tmp_path: Path) -> None:
    _fake_data(tmp_path)
    source = memory.resolve("video:real/cook/aria_cook_45s.mp4", tmp_path)
    assert source is not None
    assert source.db is not None and source.db.name == "chronicle.db"
    assert source.video is not None and source.video.is_file()


def test_unknown_upload_has_no_db(tmp_path: Path) -> None:
    video = tmp_path / "uploads" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not a real video")
    source = memory.resolve("video:uploads/clip.mp4", tmp_path)
    assert source is not None
    assert source.db is None


def test_status_endpoint_answers_for_demo_clip() -> None:
    module = importlib.reload(app_module)
    client = TestClient(module.app)
    response = client.get("/api/vfm/status", params={"clip": "events:demo_events.jsonl"})
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert isinstance(body.get("files", {}), dict)


def test_compile_endpoint_rejects_unknown_clip() -> None:
    module = importlib.reload(app_module)
    client = TestClient(module.app)
    response = client.post("/api/vfm/compile", params={"clip": "video:nope/missing.mp4"})
    assert response.status_code == 400

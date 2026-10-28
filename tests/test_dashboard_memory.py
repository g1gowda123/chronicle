"""VFM bridge for the zine dashboard (clip → DB/video resolution + API)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

import dashboard.app as app_module
from dashboard import memory


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


def test_auto_compile_rebuilds_thin_vault_after_ingest(monkeypatch, tmp_path: Path) -> None:
    _fake_data(tmp_path)
    clip = "events:demo_events.jsonl"
    vault = tmp_path / "thin" / "vault.db"
    vault.parent.mkdir()
    vault.write_bytes(b"")

    calls: list[bool] = []

    class _Thin:
        def has_memory(self, source):
            return True

        def artifact(self, source, name):
            return vault

        def submit(self, source, video=None, *, force=False):
            calls.append(force)
            return type("Job", (), {"to_dict": lambda self: {"status": "queued"}})()

    monkeypatch.setattr(memory, "manager", lambda root: _Thin())
    monkeypatch.setattr(memory, "_db_has_memory", lambda db: True)
    monkeypatch.setattr(memory, "_AUTO_ATTEMPTED", set())

    assert memory.auto_compile(clip, tmp_path) == {"status": "queued"}
    assert calls == [True]


def test_auto_compile_skips_cached_clip(monkeypatch, tmp_path: Path) -> None:
    _fake_data(tmp_path)
    clip = "events:demo_events.jsonl"

    class _Cached:
        def has_memory(self, source):
            return True

        def submit(self, *args, **kwargs):
            raise AssertionError("cached clip must not be recompiled")

    monkeypatch.setattr(memory, "manager", lambda root: _Cached())
    assert memory.auto_compile(clip, tmp_path) is None

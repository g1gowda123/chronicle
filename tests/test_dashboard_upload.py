"""Uploads from the zine UI run perception instead of staying video-only."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import dashboard.app as app_module
from dashboard import processing
from dashboard.ui import page


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, object]] = []

    def submit(self, folder: Path, video: Path, source: object) -> SimpleNamespace:
        self.calls.append((folder, video, source))
        return SimpleNamespace(
            key=folder.name,
            status="queued",
            events=f"uploads/{folder.name}/{video.stem}.events.jsonl",
        )

    def status(self, key: str) -> dict[str, object] | None:
        if key == "known-run":
            return {"upload": key, "status": "done", "progress": 100, "events": "x"}
        return None


def _video_bytes() -> bytes:
    return b"\x00" * 2048


def test_upload_starts_perception_and_registers_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    fake = _FakeProcessor()
    monkeypatch.setattr(app_module.processing, "processor", lambda root: fake)
    monkeypatch.setattr(app_module.processing, "probe_video", lambda path: (640, 480, 30.0))
    monkeypatch.setattr(app_module, "probe_duration", lambda path: 12.5)

    client = TestClient(app_module.app)
    response = client.post(
        "/api/upload",
        params={"name": "Desk Clip.mp4"},
        content=_video_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["upload"]
    assert body["events"].endswith("source.events.jsonl")
    assert body["path"].startswith("uploads/")
    assert body["path"].endswith("/source.mp4")
    assert body["vocabulary"]

    folder = tmp_path / body["path"]
    assert folder.parent.parent == tmp_path / "uploads"
    assert (folder.parent / "source.json").is_file()
    assert len(fake.calls) == 1
    called_folder, called_video, source = fake.calls[0]
    assert called_folder == folder.parent
    assert called_video == folder
    assert source.key == "upload-" + folder.parent.name


def test_upload_status_reports_worker_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    fake = _FakeProcessor()
    monkeypatch.setattr(app_module.processing, "processor", lambda root: fake)
    client = TestClient(app_module.app)

    done = client.get("/api/upload/status", params={"key": "known-run"})
    assert done.status_code == 200
    assert done.json()["status"] == "done"

    missing = client.get("/api/upload/status", params={"key": "missing"})
    assert missing.status_code == 404


def test_upload_rejects_undecodable_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    fake = _FakeProcessor()
    monkeypatch.setattr(app_module.processing, "processor", lambda root: fake)
    monkeypatch.setattr(app_module.processing, "probe_video", lambda path: None)
    client = TestClient(app_module.app)

    response = client.post(
        "/api/upload",
        params={"name": "notes.mp4"},
        content=_video_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 400
    assert fake.calls == []
    assert not (tmp_path / "uploads").exists() or not any((tmp_path / "uploads").iterdir())


def test_clip_registry_lists_every_uploaded_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(page, "default_data_dir", lambda: tmp_path)
    for run in ("20260101-000000-aaaaaa-first", "20260101-000001-bbbbbb-second"):
        folder = tmp_path / "uploads" / run
        folder.mkdir(parents=True)
        video = folder / "source.mp4"
        video.write_bytes(_video_bytes())
        session = {
            "v": 1,
            "kind": "session",
            "t": 0.0,
            "data": {"title": run.rsplit("-", 1)[-1], "video_url": "/media/x"},
        }
        (folder / "source.events.jsonl").write_text(
            json.dumps(session) + "\n", encoding="utf-8"
        )

    values = [option["value"] for option in page._clip_options()]
    assert sum(value.startswith("events:uploads/") for value in values) == 2


def test_worker_builds_replay_and_queues_memory(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "uploads" / "run" / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(_video_bytes())
    source = processing.register_upload(
        tmp_path, video, original_name="clip.mp4", vocabulary=("cup",)
    )

    calls: dict[str, object] = {}

    class _Runtime:
        def __init__(self, *args, **kwargs) -> None:
            calls["db"] = kwargs.get("db_path", args[0] if args else None)

        def ingest_video(self, video, on_progress=None):
            assert on_progress is not None
            on_progress({"timestamp": 1.0, "frames": 10, "detections": 20})
            return SimpleNamespace(
                as_dict=lambda: {"frames": 10, "entities": 3},
                frames=10,
                detections=20,
                entities=3,
            )

        def close(self) -> None:
            calls["closed"] = True

    def _events_from_db(db_path, **kwargs):
        return [{"v": 1, "kind": "session", "t": 0.0, "data": {"title": "clip"}}]

    def _write_events(events, out):
        Path(out).write_text(json.dumps(events[0]) + "\n", encoding="utf-8")
        calls["events"] = Path(out)
        return Path(out)

    monkeypatch.setattr("chronicle.integration.ChronicleRuntime", _Runtime)
    monkeypatch.setattr("dashboard.runtime.probe_duration", lambda path: 2.0)
    monkeypatch.setattr("dashboard.replay_db.events_from_db", _events_from_db)
    monkeypatch.setattr("dashboard.replay_db.write_events", _write_events)
    monkeypatch.setattr("dashboard.memory.auto_compile", lambda clip, root: calls.update(memory=clip))

    processor = processing.UploadProcessor(tmp_path)
    try:
        job = processor.submit(video.parent, video, source)
        deadline = time.monotonic() + 5.0
        state = None
        while time.monotonic() < deadline:
            state = processor.status(job.key)
            if state and state["status"] in {"done", "error"}:
                break
            time.sleep(0.02)
        assert state is not None, "worker never reported a state"
        assert state["status"] == "done", state
        assert state["events"] == "uploads/run/source.events.jsonl"
        assert (video.parent / "source.events.jsonl").is_file()
        assert calls.get("memory") == "events:uploads/run/source.events.jsonl"
        assert calls.get("closed") is True
    finally:
        processor.stop()

"""Graph ZIP export/import for the zine dashboard."""

from __future__ import annotations

import importlib
import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

import dashboard.app as app_module
from dashboard import transfer
from dashboard.runtime import load_demo_events
from dashboard.store import MemoryStore
from dashboard.ui import theme

ROOT = Path(__file__).resolve().parents[1]


class _FakeRuntime:
    def __init__(self) -> None:
        self.store = MemoryStore()
        self.demo_events, _, _ = load_demo_events(ROOT / "data" / "replays" / "walking.events.jsonl")
        for event in self.demo_events:
            self.store.apply(event)


def test_export_zip_contains_graphs_and_all_replays(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    payload = transfer.build_export_zip(runtime, ROOT / "data")
    archive = zipfile.ZipFile(io.BytesIO(payload))
    names = set(archive.namelist())
    assert "manifest.json" in names
    assert "current/events.jsonl" in names
    assert "current/snapshot.json" in names
    assert "current/entities.csv" in names
    assert any(name.startswith("clips/") and name.endswith(".events.jsonl") for name in names)
    snapshot = archive.read("current/snapshot.json").decode("utf-8")
    assert '"entities"' in snapshot and '"edges"' in snapshot


def test_export_snapshot_uses_the_whole_timeline_not_the_playhead() -> None:
    runtime = _FakeRuntime()
    # A demo dashboard only replays events up to the playhead, so its live store
    # is a prefix of the timeline. Export must still describe the whole clip.
    runtime.store = MemoryStore()
    runtime.store.apply(runtime.demo_events[0])
    archive = zipfile.ZipFile(io.BytesIO(transfer.build_export_zip(runtime, ROOT / "data")))
    snapshot = json.loads(archive.read("current/snapshot.json"))
    assert snapshot["events"] == len(runtime.demo_events)
    assert snapshot["entities"], "snapshot should include the whole graph, not the playhead"


def test_export_includes_event_clip_graphs(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    folder = tmp_path / "event_clips" / "demo" / "01_held-keys_t00001000"
    folder.mkdir(parents=True)
    (folder.parent / "index.json").write_text("{}", encoding="utf-8")
    (folder / "events.jsonl").write_text("{}\n", encoding="utf-8")
    (folder / "graph.json").write_text("{}", encoding="utf-8")
    (folder / "vault.db").write_bytes(b"vault")
    (folder / "clip.mp4").write_bytes(b"video")
    (folder / "vault.db-wal").write_bytes(b"wal")

    archive = zipfile.ZipFile(io.BytesIO(transfer.build_export_zip(runtime, tmp_path)))
    names = set(archive.namelist())
    assert "event_clips/demo/index.json" in names
    assert "event_clips/demo/01_held-keys_t00001000/graph.json" in names
    assert "event_clips/demo/01_held-keys_t00001000/events.jsonl" in names
    assert "event_clips/demo/01_held-keys_t00001000/vault.db" in names
    assert "event_clips/demo/01_held-keys_t00001000/clip.mp4" not in names
    assert "event_clips/demo/01_held-keys_t00001000/vault.db-wal" in names


def test_export_includes_active_vfm_artifacts(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    vfm = tmp_path / "vfm"
    (vfm / "evidence").mkdir(parents=True)
    (vfm / "vault.db").write_bytes(b"vault")
    (vfm / "memory.html").write_text("<html></html>", encoding="utf-8")
    (vfm / "evidence" / "frame.webp").write_bytes(b"webp")
    (vfm / "source.db").write_bytes(b"source")
    (vfm / "memory.zip").write_bytes(b"zip")

    archive = zipfile.ZipFile(
        io.BytesIO(transfer.build_export_zip(runtime, tmp_path, vfm_dir=vfm))
    )
    names = set(archive.namelist())
    assert "vfm/vault.db" in names
    assert "vfm/memory.html" in names
    assert "vfm/evidence/frame.webp" in names
    assert "vfm/source.db" not in names
    assert "vfm/memory.zip" not in names


def test_import_zip_writes_timeline_and_counts_events(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    payload = transfer.build_export_zip(runtime, ROOT / "data")
    result = transfer.import_zip(payload, tmp_path)
    assert result["status"] == "imported"
    assert result["events"] > 0
    target = Path(result["path"])
    assert target.is_file() and target.parent.name == "imports"
    events, _, _ = load_demo_events(target)
    assert len(events) == result["events"]


def test_import_rejects_non_zip() -> None:
    try:
        transfer.import_zip(b"not a zip", "/tmp")
    except ValueError as error:
        assert "ZIP" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_toolbar_has_export_import_controls() -> None:
    html = theme.video_html({"video_url": "/media/demo.mp4", "title": "Demo"})
    assert "Chronicle.exportGraphs()" in html
    assert "Chronicle.importGraphs(this)" in html
    # Export points out/up; import points in/down (they used to be swapped).
    assert "⤒ export" in html
    assert "⤓ import" in html
    js = (ROOT / "dashboard" / "static" / "chronicle.js").read_text(encoding="utf-8")
    assert "exportGraphs()" in js
    assert "importGraphs(input)" in js
    assert "/api/export/graphs.zip" in js
    assert '"?clip=" + encodeURIComponent(value)' in js
    assert "/api/import/graphs" in js
    assert 'link.download = "chronicle_graphs.zip"' in js


def test_export_endpoint_returns_zip() -> None:
    module = importlib.reload(app_module)
    client = TestClient(module.app)
    response = client.get("/api/export/graphs.zip")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert zipfile.is_zipfile(io.BytesIO(response.content))

"""Headless tests for the VFM service and dashboard integration.

No Ollama, no network and no bundled clips: a tiny generated video and a small
SQLite graph exercise the whole compile -> pack -> ask -> download path.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from chronicle.sources import ClipSource
from chronicle.vfm import query
from chronicle.vfm.service import (
    VFMJobManager,
    artifacts,
    compile_memory,
    load_status,
    memory_job_dir,
)


def _make_graph_db(path: Path) -> Path:
    # Use the production schema so the dashboard runtime can open the same file
    # as the VFM compiler.
    from chronicle.storage.sqlite_store import SQLiteStore

    store = SQLiteStore(str(path))
    store.close()
    con = sqlite3.connect(str(path))
    try:
        con.execute("insert into zones values ('z1', 'Kitchen', null, 0, 4, 1, '{}')")
        con.execute("insert into furniture values ('a1', 'countertop', 'z1', 0, '{}')")
        con.execute(
            "insert into entities values ('e1', 'cup', 'VISIBLE_STATIC', 0.5, 0.5, 3.5,"
            " '[0,0,1,1]', null, 1, 'z1', '{}')"
        )
        con.execute(
            "insert into entities values ('e2', 'plate', 'VISIBLE_STATIC', 1.5, 1.5, 3.5,"
            " '[0,0,1,1]', null, 2, 'z1', '{}')"
        )
        con.execute(
            "insert into entities values ('e3', 'cup', 'VISIBLE_STATIC', 2.5, 2.5, 3.0,"
            " '[0,0,1,1]', null, 3, 'z1', '{}')"
        )
        con.execute("insert into observations values (1, 'e1', 4.0, '[0,0,1,1]', 'VISIBLE_STATIC')")
        con.execute("insert into observations values (2, 'e3', 4.0, '[0,0,1,1]', 'VISIBLE_STATIC')")
        con.execute(
            "insert into state_transitions values (1, 'e1', 'VISIBLE_STATIC', 'INTERACTED',"
            " 1.0, '{\"reason\": \"hand_contact\"}', null)"
        )
        con.execute(
            "insert into state_transitions values (2, 'e1', 'INTERACTED', 'VISIBLE_STATIC',"
            " 2.0, '{\"reason\": \"resting_on_support\", \"support_id\": \"a1\"}', null)"
        )
        con.execute(
            "insert into anchors (id, entity_id, anchor_id, relation, t_start, is_terminal,"
            " confidence) values (1, 'e1', 'a1', 'ON', 2.0, 1, 0.8)"
        )
        con.commit()
    finally:
        con.close()
    return path


def _make_video(path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    try:
        for index in range(40):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:, :] = (20, 30, 40)
            x = 10 + index * 2
            cv2.rectangle(frame, (x, 40), (x + 30, 90), (40, 200, 180), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _fake_captioner(video: Path, out: Path, **kwargs) -> dict:
    ledger = {
        "version": 1,
        "model": "fake-vlm",
        "mode": "test",
        "window_s": float(kwargs.get("window_s") or 4.0),
        "video": Path(video).name,
        "duration_s": 4.0,
        "generated_at": 0.0,
        "windows": [
            {
                "t_start": 0.0,
                "t_end": 4.0,
                "caption": "A person moves a cup onto the countertop.",
                "objects": ["cup", "countertop"],
                "actions": ["move"],
                "zone": "Kitchen",
                "tags": ["test"],
                "frames": [
                    {
                        "t": 2.0,
                        "caption": "hands move a cup",
                        "objects": ["cup"],
                        "actions": ["move"],
                        "zone": "Kitchen",
                    }
                ],
            }
        ],
    }
    Path(out).write_text(json.dumps(ledger), encoding="utf-8")
    return {"windows": 1, "bytes": Path(out).stat().st_size, "mode": "test"}


def _source(tmp_path: Path) -> ClipSource:
    return ClipSource(
        key="vfm-test",
        label="VFM test",
        badge="TEST",
        caption="generated fixture",
        description="headless VFM test",
        relative_path="clip.mp4",
        relative_db="chronicle.db",
        zone_label="Kitchen",
        vocabulary=("cup", "plate"),
        questions=("where is the cup?",),
    )


def test_compile_memory_produces_queryable_video_free_artifacts(tmp_path: Path) -> None:
    db = _make_graph_db(tmp_path / "chronicle.db")
    video = _make_video(tmp_path / "clip.mp4")
    job_dir = tmp_path / "job"

    result = compile_memory(
        video,
        db,
        job_dir,
        title="fixture",
        episode_id="fixture",
        captioner=_fake_captioner,
    )

    assert result["status"] == "done"
    assert result["duration_s"] == pytest.approx(4.0)
    assert result["video_retained"] is True
    assert result["video_deleted"] is False
    assert result["pack"]["windows"] == 1
    assert result["pack"]["chars"] > 200
    assert result["evidence"]["snapshots"] >= 1

    files = artifacts(job_dir)
    for name in ("vault.db", "memory.html", "memory_pack.md", "memory_pack.json",
                 "ledger.json", "events.csv", "windows.csv", "memory.zip"):
        assert files[name].is_file(), name
        assert files[name].stat().st_size > 0, name

    with zipfile.ZipFile(files["memory.zip"]) as archive:
        names = set(archive.namelist())
    assert {"vault.db", "memory.html", "memory_pack.md", "README.txt"} <= names
    assert any(name.startswith("evidence/") for name in names)

    # The vault answers without the video and without a VLM.
    con = sqlite3.connect(str(files["vault.db"]))
    con.row_factory = sqlite3.Row
    try:
        store = query.load_store(con)
        answer = query.answer(store, "where is the cup?")
    finally:
        con.close()
    assert "cup" in answer.answer.lower()
    assert answer.method.startswith("vault")

    # Playback is preserved by default.
    assert video.is_file()


def test_manager_compiles_in_background_and_persists_status(tmp_path: Path) -> None:
    _make_graph_db(tmp_path / "chronicle.db")
    _make_video(tmp_path / "clip.mp4")
    source = _source(tmp_path)
    manager = VFMJobManager(
        tmp_path,
        captioner=_fake_captioner,
        qwen_model="unused",
    )
    try:
        job = manager.submit(source, tmp_path / "clip.mp4")
        assert job is not None
        status = manager.wait(source, timeout=60.0)
        assert status is not None
        assert status["status"] == "done", status
        assert status["progress"] == 100
        assert manager.has_memory(source)
        assert manager.artifact(source, "memory.zip") is not None

        answer = manager.ask(source, "where is the cup?")
        assert "cup" in answer.answer.lower()
        assert answer.method.startswith("vault")

        reloaded = load_status(memory_job_dir(tmp_path, source))
        assert reloaded is not None
        assert reloaded["status"] == "done"
    finally:
        manager.close()


def test_delete_video_is_opt_in_and_only_for_uploads(tmp_path: Path) -> None:
    _make_graph_db(tmp_path / "chronicle.db")
    video = _make_video(tmp_path / "clip.mp4")
    source = _source(tmp_path)
    manager = VFMJobManager(
        tmp_path,
        captioner=_fake_captioner,
        delete_video=True,
    )
    try:
        manager.submit(source, video)
        status = manager.wait(source, timeout=60.0)
        assert status is not None and status["status"] == "done"
        # A source outside <data-dir>/uploads must never be deleted.
        assert video.is_file()
    finally:
        manager.close()


def test_cli_exposes_vfm_flags() -> None:
    from chronicle.cli import build_parser

    args = build_parser().parse_args(
        [
            "serve",
            "--qwen",
            "--no-vfm",
            "--vfm-window-s",
            "20",
            "--vfm-frames-per-window",
            "2",
            "--vfm-max-snapshots",
            "12",
            "--vfm-delete-video",
        ]
    )
    assert args.qwen is True
    assert args.no_vfm is True
    assert args.vfm_window_s == 20.0
    assert args.vfm_frames_per_window == 2
    assert args.vfm_max_snapshots == 12
    assert args.vfm_delete_video is True


def test_dashboard_routes_compile_ask_and_download(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from nicegui import app

    from chronicle.ui.app import create_app
    from chronicle.ui.uploads import load_upload_sources, new_upload_target, register_upload

    # Re-creating the NiceGUI app in one process would otherwise leave the
    # previous test's route closures in front of ours.
    app.router.routes.clear()

    video = new_upload_target(tmp_path, "clip.mp4")
    _make_video(video)
    source = register_upload(
        tmp_path,
        video,
        original_name="clip.mp4",
        vocabulary=("cup", "plate"),
    )
    _make_graph_db(video.parent / "chronicle.db")

    state = create_app(
        data_dir=tmp_path,
        source_key=source.key,
        auto_run=False,
        start_server=False,
    )
    try:
        loaded = load_upload_sources(tmp_path)[0]
        assert loaded.key == source.key
        assert state.vfm is not None
        state.vfm.captioner = _fake_captioner
        job = state.vfm.submit(loaded, video)
        assert job is not None
        status = state.vfm.wait(loaded, timeout=60.0)
        assert status is not None and status["status"] == "done", status

        client = TestClient(app)

        page = client.get("/")
        assert page.status_code == 200
        for text in ("Memory", "VFM memory (ZIP)", "Ask memory", "Open memory report"):
            assert text in page.text, text

        status_response = client.get(f"/vfm/status/{loaded.key}")
        assert status_response.status_code == 200
        payload = status_response.json()
        assert payload["enabled"] is True
        assert payload["status"]["status"] == "done"
        assert payload["files"]["vault.db"] > 0

        report = client.get(f"/vfm/report/{loaded.key}")
        assert report.status_code == 200
        assert report.headers["content-type"].startswith("text/html")
        assert "<!doctype html>" in report.text.lower()
        assert "<title>" in report.text.lower()

        archive = client.get(f"/vfm/download/{loaded.key}/memory.zip")
        assert archive.status_code == 200
        assert archive.content[:2] == b"PK"

        vault = client.get(f"/vfm/download/{loaded.key}/vault.db")
        assert vault.status_code == 200
        assert vault.content[:15] == b"SQLite format 3"

        answer = client.post(
            f"/vfm/ask/{loaded.key}",
            json={"question": "where is the cup?", "qwen": False},
        )
        assert answer.status_code == 200
        body = answer.json()
        assert "cup" in body["answer"].lower()
        assert body["method"].startswith("vault")

        evidence_file = next((state.vfm.job_dir(loaded) / "evidence").glob("*.webp"))
        evidence = client.get(f"/vfm/evidence/{loaded.key}/{evidence_file.name}")
        assert evidence.status_code == 200
        assert evidence.content[:4] == b"RIFF"
    finally:
        if state.vfm is not None:
            state.vfm.close()
        state.close()


def test_count_answers_expose_track_and_visibility_bounds(tmp_path: Path) -> None:
    db = _make_graph_db(tmp_path / "chronicle.db")
    job_dir = tmp_path / "job"
    result = compile_memory(None, db, job_dir, title="fixture", episode_id="fixture")
    assert result["vault_stats"]["counts"] == {"cup": {"tracks": 2, "peak": 2}}

    con = sqlite3.connect(str(job_dir / "vault.db"))
    con.row_factory = sqlite3.Row
    try:
        store = query.load_store(con)
        answer = query.answer(store, "how many cups are there?")
    finally:
        con.close()
    assert answer.method == "vault-count-visibility"
    assert "2 cup tracks" in answer.answer
    assert "visible at the same instant" in answer.answer

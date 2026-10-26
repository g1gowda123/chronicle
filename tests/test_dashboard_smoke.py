"""Headless smoke test for the NiceGUI dashboard and export endpoints."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_HAS_DASHBOARD_DEPS = bool(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("nicegui")
)

pytestmark = pytest.mark.skipif(
    not _HAS_DASHBOARD_DEPS,
    reason="dashboard extras not installed (uv sync --extra ui)",
)


def test_dashboard_headless_routes_and_exports(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from nicegui import app

    from chronicle.ui.app import create_app

    state = create_app(
        data_dir=tmp_path,
        source_key="synthetic",
        auto_run=False,
        start_server=False,
    )
    try:
        client = TestClient(app)

        # 1. Main UI route returns 200 with HTML title
        response = client.get("/")
        assert response.status_code == 200
        assert "Chronicle" in response.text
        assert "chronicle-dropzone" in response.text
        assert "Drag &amp; drop a video to scan live" in response.text

        # 2. JSON snapshot export route
        res_json = client.get("/export/snapshot.json")
        assert res_json.status_code == 200
        assert res_json.headers["content-type"] == "application/json"
        assert res_json.headers["content-disposition"] == (
            'attachment; filename="chronicle_snapshot.json"'
        )
        payload = json.loads(res_json.text)
        assert "nodes" in payload
        assert "edges" in payload
        assert "verification" in payload
        assert "verification_stats" in payload
        assert "stats" in payload
        assert "verification_queue_depth" in payload["stats"]

        # 3. CSV entities export route
        res_csv = client.get("/export/entities.csv")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers["content-type"]
        assert res_csv.headers["content-disposition"] == (
            'attachment; filename="chronicle_entities.csv"'
        )
        assert res_csv.text.startswith(
            "id,label,state,zone,anchor,container,first_seen,last_seen,"
            "span,track_id,confidence,colour,colour_hex,attributes"
        )

        # 4. CSV actions export route
        res_actions = client.get("/export/actions.csv")
        assert res_actions.status_code == 200
        assert "text/csv" in res_actions.headers["content-type"]
        assert res_actions.headers["content-disposition"] == (
            'attachment; filename="chronicle_actions.csv"'
        )
        assert res_actions.text.startswith(
            "id,timestamp,action,entity_id,entity,confidence,source,clip_ref,caption"
        )

        # 5. Snapshot import route
        import_payload = {
            "entities": [
                {
                    "id": "entity:imported-cup",
                    "label": "cup",
                    "state": "VISIBLE_STATIC",
                    "zone": "Kitchen",
                }
            ],
            "furniture": [
                {"anchor_id": "table-1", "label": "table", "zone_id": "Kitchen"}
            ],
        }
        res_import = client.post("/import/snapshot", json=import_payload)
        assert res_import.status_code == 200
        assert res_import.json()["status"] == "imported"
        assert res_import.json()["entities"] == 1

        # Verify imported entity is now in snapshot
        snapshot_after = json.loads(client.get("/export/snapshot.json").text)
        assert any(e["label"] == "cup" for e in snapshot_after["entities"])

        # 6. Every registered source clip is servable by the player
        repo_data = Path(__file__).resolve().parents[1] / "data"
        from chronicle.sources import SOURCES

        checked = 0
        for source in SOURCES:
            video = source.video_path(repo_data)
            if not video.is_file():
                continue
            with state.lock:
                state.source = source
                state.source_key = source.key
                state.video = video
            response = client.get("/source-video")
            assert response.status_code == 200, source.key
            assert response.headers["content-type"].startswith("video/"), source.key
            assert len(response.content) > 1024, source.key
            checked += 1
        assert checked > 0

        # 7. Missing source-video returns 404 cleanly
        with state.lock:
            state.video = tmp_path / "missing.mp4"
        res_video = client.get("/source-video")
        assert res_video.status_code == 404
    finally:
        state.close()

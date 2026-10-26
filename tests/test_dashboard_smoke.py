"""Headless smoke test for the NiceGUI dashboard and export endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from nicegui import app

from chronicle.ui.app import create_app


def test_dashboard_headless_routes_and_exports(tmp_path: Path) -> None:
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

        # 2. JSON snapshot export route
        res_json = client.get("/export/snapshot.json")
        assert res_json.status_code == 200
        assert res_json.headers["content-type"] == "application/json"
        payload = json.loads(res_json.text)
        assert "nodes" in payload
        assert "edges" in payload
        assert "verification" in payload
        assert "stats" in payload
        assert "verification_queue_depth" in payload["stats"]

        # 3. CSV entities export route
        res_csv = client.get("/export/entities.csv")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers["content-type"]
        assert res_csv.text.startswith("id,label,state,zone,anchor,container,span")

        # 4. CSV actions export route
        res_actions = client.get("/export/actions.csv")
        assert res_actions.status_code == 200
        assert "text/csv" in res_actions.headers["content-type"]
        assert res_actions.text.startswith("id,timestamp,action,entity,caption,source")

        # 5. Missing source-video returns 404 cleanly
        res_video = client.get("/source-video")
        assert res_video.status_code == 404
    finally:
        state.close()

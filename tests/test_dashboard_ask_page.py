"""The hosted one-box ask page is served with the dashboard's own styling."""

from __future__ import annotations

import importlib.util

import pytest

_HAS_DASHBOARD_DEPS = bool(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("nicegui")
)

pytestmark = pytest.mark.skipif(
    not _HAS_DASHBOARD_DEPS,
    reason="dashboard extras not installed (uv sync --extra ui)",
)


def test_ask_page_is_served() -> None:
    from fastapi.testclient import TestClient

    import dashboard.app as app_module

    client = TestClient(app_module.app)
    response = client.get("/ask")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "Ask your" in body
    assert "/api/vfm/ask" in body
    assert "/chronicle/chronicle.css" in body

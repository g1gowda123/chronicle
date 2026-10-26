"""Evidence thumbnails attached to VFM chat answers."""

from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient

import dashboard.app as app_module
from dashboard import memory


def test_chat_and_evidence_route() -> None:
    module = importlib.reload(app_module)
    client = TestClient(module.app)
    clip = "events:replays/walking.events.jsonl"
    source = memory.resolve(clip, module.DATA_DIR)
    assert source is not None
    manager = memory.manager(module.DATA_DIR)
    if not manager.has_memory(source):
        return  # walking memory is optional in a clean checkout
    evidence = manager.evidence_file(source, "does-not-exist.webp")
    assert evidence is None
    # Use a real evidence file if the compiled vault has one.
    job = manager.job_dir(source)
    files = list((job / "evidence").glob("*")) if (job / "evidence").is_dir() else []
    if not files:
        return
    response = client.get(
        "/api/vfm/evidence", params={"clip": clip, "name": files[0].name}
    )
    assert response.status_code == 200
    assert response.content

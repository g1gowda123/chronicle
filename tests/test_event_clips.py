"""Event-based clipping and per-event graph creation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.contracts import Event, parse_event
from dashboard.event_clips import (
    build_event_clips,
    list_event_clips,
    read_index,
    score_event,
    select_events,
)


def _event(kind: str, t: float, data: dict, seq: int = 0) -> Event:
    event = parse_event({"v": 1, "kind": kind, "t": t, "data": data}, seq=seq)
    assert event is not None
    return event


def _timeline() -> list[Event]:
    return [
        _event(
            "session",
            0.0,
            {
                "title": "Event clip fixture",
                "video_url": "/media/fixture.mp4",
                "duration": 12.0,
                "fps": 30,
                "width": 640,
                "height": 480,
            },
            seq=1,
        ),
        _event(
            "entity",
            0.4,
            {"id": "mug_1", "label": "Mug", "confidence": 0.91, "zone_id": "kitchen"},
            seq=2,
        ),
        _event("zone", 0.4, {"id": "kitchen", "label": "Kitchen", "active": True}, seq=3),
        _event(
            "edge",
            0.5,
            {"id": "e1", "source": "mug_1", "target": "counter", "relation": "RESTING_ON"},
            seq=4,
        ),
        _event(
            "state",
            1.2,
            {
                "entity_id": "mug_1",
                "label": "Mug",
                "from": "VISIBLE_STATIC",
                "to": "VISIBLE_MOVING",
                "trigger": {"hand_contact": False},
            },
            seq=5,
        ),
        _event(
            "interaction",
            2.2,
            {
                "entity_id": "mug_1",
                "action": "TOUCHED",
                "confidence": 0.91,
                "accepted": True,
            },
            seq=6,
        ),
        _event(
            "state",
            3.4,
            {
                "entity_id": "mug_1",
                "label": "Mug",
                "from": "INTERACTED",
                "to": "HELD",
                "trigger": {"hand_contact": True},
            },
            seq=7,
        ),
        _event(
            "interaction",
            6.8,
            {
                "entity_id": "mug_1",
                "action": "INSERT_INTO",
                "confidence": 0.42,
                "accepted": False,
                "rejection_reason": "ambiguous",
            },
            seq=8,
        ),
        _event(
            "state",
            9.5,
            {
                "entity_id": "mug_1",
                "label": "Mug",
                "from": "CONTAINED",
                "to": "LOST",
                "trigger": {},
            },
            seq=9,
        ),
    ]


def _write_timeline(path: Path, events: list[Event]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            payload = dict(event.raw or {})
            payload.setdefault("kind", event.kind)
            handle.write(json.dumps(payload) + "\n")


def test_salience_prefers_interactions_and_flags_low_confidence() -> None:
    held = _event("state", 1.0, {"entity_id": "mug", "to": "HELD"})
    moving = _event("state", 1.0, {"entity_id": "mug", "to": "VISIBLE_MOVING"})
    low = _event(
        "interaction",
        1.0,
        {"entity_id": "mug", "action": "TOUCHED", "confidence": 0.2, "accepted": False},
    )

    assert score_event(held)[0] > score_event(moving)[0]
    score, reasons = score_event(low)
    assert score > 0.6
    assert "low-confidence" in reasons
    assert "needs-review" in reasons


def test_select_events_coalesces_nearby_events_and_respects_limit() -> None:
    events = _timeline()
    plans = select_events(events, duration=12.0, pre=3.0, post=3.0, cooldown=2.5, limit=4)

    # The touch and HELD transition are one moment, not two clips.
    held_plans = [plan for plan in plans if "HELD" in plan.reasons or "state:HELD" in plan.reasons]
    assert len(held_plans) == 1
    assert plans[0].start >= 0.0
    assert all(plan.end <= 12.0 for plan in plans)
    assert all(plan.end - plan.start >= 2.0 for plan in plans)
    assert len(select_events(events, duration=12.0, limit=1)) == 1


def test_build_event_clips_writes_shifted_timeline_graph_and_manifest(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_timeline(data / "fixture.events.jsonl", _timeline())

    index = build_event_clips(
        "events:fixture.events.jsonl",
        data,
        pre=3.0,
        post=3.0,
        cut=False,
    )

    assert index["source_title"] == "Event clip fixture"
    assert index["pre_s"] == 3.0
    assert index["post_s"] == 3.0
    assert index["clips"]
    entry = index["clips"][0]
    assert entry["play"].startswith("events:event_clips/")
    assert entry["graph_nodes"] >= 1
    assert entry["graph_edges"] >= 1

    events_path = data / entry["events"]
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "session"
    assert rows[0]["data"]["source"] == "event-clip"
    assert rows[0]["data"]["event"]["score"] == entry["score"]
    assert rows[0]["t"] == 0.0
    assert all(row["t"] >= 0.0 for row in rows)
    assert all(row["t"] <= entry["duration"] + 0.01 for row in rows)

    graph = json.loads((data / entry["graph"]).read_text(encoding="utf-8"))
    assert graph["anchor_t"] == entry["anchor_t"]
    assert graph["graph"]["nodes"]
    assert any(node["id"] == "mug_1" for node in graph["graph"]["nodes"])

    manifest = json.loads(
        (data / "event_clips" / "fixture" / "index.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["clips"]) == len(index["clips"])
    assert read_index("events:fixture.events.jsonl", data) is not None
    assert list_event_clips(data)


def test_uploads_with_generic_source_names_get_separate_folders(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for run in ("run-a", "run-b"):
        folder = data / "uploads" / run
        folder.mkdir(parents=True)
        session = _event(
            "session",
            0.0,
            {
                "title": run,
                "video_url": f"/media/uploads/{run}/source.mp4",
                "duration": 12.0,
                "fps": 30,
                "width": 640,
                "height": 480,
            },
        )
        _write_timeline(folder / "source.events.jsonl", [session] + _timeline()[1:])

    first = build_event_clips("events:uploads/run-a/source.events.jsonl", data, cut=False)
    second = build_event_clips("events:uploads/run-b/source.events.jsonl", data, cut=False)

    assert first["source_title"] == "run-a"
    assert second["source_title"] == "run-b"
    assert Path(first["clips"][0]["events"]).parts[1] != Path(second["clips"][0]["events"]).parts[1]
    assert len(list_event_clips(data)) == len(first["clips"]) + len(second["clips"])


def test_build_event_clips_rejects_paths_outside_data_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_event_clips("events:../outside.events.jsonl", tmp_path)


def test_event_clips_api_builds_and_lists(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import dashboard.app as app_module

    data = tmp_path / "data"
    data.mkdir()
    _write_timeline(data / "fixture.events.jsonl", _timeline())
    monkeypatch.setattr(app_module, "DATA_DIR", data)

    client = TestClient(app_module.app)
    response = client.post(
        "/api/event-clips",
        json={"clip": "events:fixture.events.jsonl", "pre": 2, "post": 2, "cut": False},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["clips"]
    assert payload["pre_s"] == 2.0

    listed = client.get("/api/event-clips", params={"clip": "events:fixture.events.jsonl"})
    assert listed.status_code == 200
    assert len(listed.json()["clips"]) == len(payload["clips"])

    missing = client.get("/api/event-clips", params={"clip": "events:nope.events.jsonl"})
    assert missing.status_code == 200
    assert missing.json()["clips"] == []

    artifact = data / "event_clips" / "fixture" / "snapshot.jpg"
    artifact.write_bytes(b"jpeg")
    served = client.get("/api/event-clips/artifact", params={"path": "event_clips/fixture/snapshot.jpg"})
    assert served.status_code == 200
    assert served.content == b"jpeg"
    denied = client.get("/api/event-clips/artifact", params={"path": "fixture.events.jsonl"})
    assert denied.status_code == 400

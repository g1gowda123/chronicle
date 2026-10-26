"""Model-free tests for the Role D JSONL event exporter."""

from __future__ import annotations

import json
from pathlib import Path

from chronicle.event_export import (
    DEFAULT_HZ,
    RoleDEventRecorder,
    _clamp_box,
    write_events_jsonl,
)
from chronicle.telemetry import EventBus
from chronicle.types import BBox, FrameResult, MotionEstimate, TrackedObject


def _track(track_id: int = 7, label: str = "joy-con") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label=label,
        bbox=BBox(-5.0, 10.0, 700.0, 500.0),
        confidence=0.91,
        first_seen=0.0,
        last_seen=0.2,
        hits=2,
        age=2,
        entity_id=f"track_{track_id}",
        reid_score=0.88,
        reid_source="appearance",
    )


def _frame(timestamp: float, tracks: list[TrackedObject]) -> FrameResult:
    return FrameResult(
        frame_index=int(timestamp * 10),
        timestamp=timestamp,
        wall_time=0.0,
        motion=MotionEstimate(),
        detections=[],
        hands=[],
        tracks=tracks,
        interactions=[],
        interaction_events=[],
        clips=[],
        telemetry=[],
    )


def _snapshot() -> dict:
    return {
        "entities": [
            {"id": "track_7", "label": "joy-con", "state": "HELD"},
            {"id": "track_8", "label": "table", "state": "VISIBLE_STATIC"},
        ],
        "transitions": [
            {
                "entity_id": "track_7",
                "label": "joy-con",
                "from_state": "NEW",
                "to_state": "HELD",
                "timestamp": 0.2,
                "clip_ref": "clip-1",
            }
        ],
        "actions": [
            {
                "id": "action-1",
                "timestamp": 0.25,
                "action": "PICK_UP",
                "entity_id": "track_7",
                "entity": "joy-con",
                "confidence": 0.84,
                "source": "rule",
                "clip_ref": "clip-1",
                "caption": "the hand grasps the left Joy-Con",
            }
        ],
        "edges": [
            {
                "id": "edge-1",
                "from": "track_7",
                "to": "track_8",
                "label": "RESTING_ON",
                "title": "RESTING_ON @ t=0.3s",
                "dashes": False,
            }
        ],
    }


def _recorder(**overrides) -> RoleDEventRecorder:
    kwargs = {
        "width": 640,
        "height": 480,
        "fps": 10.0,
        "duration": 1.0,
        "title": "Test clip",
        "video_url": "/media/real/test.mp4",
        "zone_id": "work_table",
        "hz": 5.0,
        "notes": "unit test",
    }
    kwargs.update(overrides)
    return RoleDEventRecorder(**kwargs)


def _telemetry():
    bus = EventBus()
    bus.emit("STATE", "track_7 HELD", 0.2, entity_id="track_7")
    return bus.history()


def test_entities_are_sampled_at_the_requested_rate() -> None:
    recorder = _recorder()
    for timestamp in (0.0, 0.1, 0.2):
        recorder.on_frame(_frame(timestamp, [_track()]))
    events = recorder.finish(_snapshot())

    entities = [event for event in events if event["kind"] == "entity"]
    assert len(entities) == 2
    assert [event["t"] for event in entities] == [0.0, 0.2]
    assert [event["seq"] for event in events] == list(range(len(events)))
    assert events[0]["kind"] == "session"
    assert events[0]["data"]["video_url"] == "/media/real/test.mp4"


def test_entity_state_action_and_edge_ids_agree() -> None:
    recorder = _recorder()
    recorder.on_frame(_frame(0.2, [_track()]))
    events = recorder.finish(_snapshot())

    entity = next(event for event in events if event["kind"] == "entity")
    state = next(event for event in events if event["kind"] == "state")
    action = next(event for event in events if event["kind"] == "interaction")
    edge = next(event for event in events if event["kind"] == "edge")

    assert entity["data"]["id"] == "joy_con_7"
    assert state["data"]["entity_id"] == "joy_con_7"
    assert action["data"]["entity_id"] == "joy_con_7"
    assert edge["data"]["source"] == "joy_con_7"
    assert edge["data"]["target"] == "table_8"


def test_snapshot_and_telemetry_fields_are_preserved() -> None:
    recorder = _recorder()
    recorder.on_frame(_frame(0.2, [_track()]))
    events = recorder.finish(_snapshot(), _telemetry())

    state = next(event for event in events if event["kind"] == "state")
    action = next(event for event in events if event["kind"] == "interaction")
    telemetry = next(event for event in events if event["kind"] == "telemetry")

    assert state["data"]["from"] == "NEW"
    assert state["data"]["to"] == "HELD"
    assert state["data"]["trigger"] == {"clip_ref": "clip-1"}
    assert action["data"]["action"] == "PICK_UP"
    assert action["data"]["model"] == "rule"
    assert action["data"]["caption"].startswith("the hand")
    assert telemetry["data"]["channel"] == "STATE"
    assert telemetry["data"]["message"] == "track_7 HELD"


def test_boxes_are_clamped_to_the_frame() -> None:
    assert _clamp_box((-5, 10, 700, 500), 640, 480) == [0, 10, 640, 480]


def test_jsonl_is_utf8_without_bom_and_roundtrips(tmp_path: Path) -> None:
    recorder = _recorder()
    recorder.on_frame(_frame(0.2, [_track()]))
    events = recorder.finish(_snapshot(), _telemetry())
    path = write_events_jsonl(events, tmp_path / "nested" / "clip.events.jsonl")

    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n")
    parsed = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert parsed == events


def test_default_rate_is_dashboard_friendly() -> None:
    assert 4.0 <= DEFAULT_HZ <= 8.0

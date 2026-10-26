"""Role A -> Role B bridge: FrameResult ingestion without an adapter in tests."""

from __future__ import annotations

from shapely.geometry import Polygon

from chronicle import ChronicleMemory
from chronicle.types import (
    BBox,
    CameraState,
    FrameResult,
    HandObservation,
    MotionEstimate,
    ObjectState,
    SupportSurface,
    TrackedObject,
)

DT = 1.0 / 30.0
TABLE = SupportSurface(
    anchor_id="dining_table",
    label="dining table",
    polygon=Polygon([(0, 380), (800, 380), (800, 600), (0, 600)]),
)


def result_at(t: float, bbox: BBox = BBox(100, 300, 180, 380)) -> FrameResult:
    track = TrackedObject(
        track_id=7,
        label="cup",
        bbox=bbox,
        confidence=0.93,
        first_seen=0.0,
        last_seen=t,
        hits=int(t / DT) + 1,
        age=int(t / DT) + 1,
    )
    hand = HandObservation(
        landmarks=[
            (100.0, 300.0),
            (180.0, 300.0),
            (180.0, 380.0),
            (100.0, 380.0),
        ],
        confidence=0.9,
    )
    return FrameResult(
        frame_index=int(t / DT),
        timestamp=t,
        wall_time=t,
        motion=MotionEstimate(state=CameraState.STATIONARY),
        detections=[],
        hands=[hand] if t >= 1.2 else [],
        tracks=[track],
        interactions=[],
        interaction_events=[],
        clips=[],
        telemetry=[],
    )


def test_frame_result_drives_lifecycle_and_query() -> None:
    with ChronicleMemory(":memory:") as memory:
        for index in range(round(1.2 / DT)):
            memory.observe_result(
                result_at(index * DT),
                supports=(TABLE,),
                zone_id="kitchen",
                zone_label="kitchen",
            )

        snapshot = memory.engine.get("track_7")
        assert snapshot is not None
        assert snapshot.label == "cup"
        assert snapshot.state is ObjectState.VISIBLE_STATIC

        result = memory.query("where is the cup?")
        assert result is not None
        assert result.entity_id == "track_7"
        assert result.terminal_anchor is not None
        assert result.terminal_anchor["anchor_id"] == "anchor:dining_table"

from __future__ import annotations

import json

import numpy as np

from chronicle.types import (
    BBox,
    CameraState,
    Detection,
    EventCategory,
    FrameResult,
    HandObservation,
    MotionEstimate,
    RunSummary,
    TelemetryEvent,
    TrackedObject,
)


def hand() -> HandObservation:
    points = np.zeros((21, 2), dtype=np.float32)
    points[:, 0] = np.linspace(90, 150, 21)
    points[:, 1] = np.linspace(90, 150, 21)
    return HandObservation(landmarks=points, handedness="right")


def frame_result() -> FrameResult:
    bbox = BBox(100, 100, 140, 140)
    return FrameResult(
        frame_index=4,
        timestamp=0.133,
        wall_time=1.5,
        motion=MotionEstimate(dx=1.0, dy=-2.0, valid=True, state=CameraState.MOVING),
        detections=[
            Detection(
                label="keys",
                bbox=bbox,
                confidence=0.91234,
                track_id=3,
                embedding=np.ones(4, dtype=np.float32),
                metadata={"prompt": "keys", "frame": np.int64(4)},
            )
        ],
        hands=[hand()],
        tracks=[
            TrackedObject(
                track_id=3,
                label="keys",
                bbox=bbox,
                confidence=0.9,
                first_seen=0.0,
                last_seen=0.133,
                hits=2,
                age=2,
                embedding=np.ones(4, dtype=np.float32),
            )
        ],
        interactions=[],
        interaction_events=[],
        clips=[],
        telemetry=[
            TelemetryEvent(
                category=EventCategory.TRACK,
                message="new track #3 keys",
                presentation_time=0.133,
                frame_index=4,
                data={"track_id": 3},
            )
        ],
    )


def test_frame_result_serializes_without_pixels_or_embeddings() -> None:
    payload = frame_result().as_dict()
    encoded = json.dumps(payload)  # must not raise on numpy scalars
    assert '"embedding"' not in encoded
    assert "landmarks" not in payload["hands"][0]
    assert payload["hands"][0]["handedness"] == "Right"
    assert payload["motion"]["state"] == "MOVING"
    assert payload["camera_state"] == "MOVING"
    assert payload["detections"][0]["confidence"] == 0.9123
    assert payload["telemetry"][0]["category"] == "TRACK"


def test_optional_embeddings_and_landmarks_are_opt_in() -> None:
    payload = frame_result().as_dict(include_landmarks=True, include_embeddings=True)
    assert payload["detections"][0]["embedding"] == [1.0, 1.0, 1.0, 1.0]
    assert payload["tracks"][0]["has_embedding"] is True
    assert len(payload["hands"][0]["landmarks"]) == 21


def test_bbox_geometry_contract() -> None:
    box = BBox(10, 20, 50, 60)
    assert box.width == 40 and box.height == 40 and box.area == 1600
    assert box.centroid == (30.0, 40.0)
    assert box.iou(box) == 1.0
    assert box.ios(BBox(30, 40, 90, 100)) == 0.25
    assert box.contains_point((10, 20))
    assert not box.contains_point((9.9, 20.0))

    expanded = box.expanded(10, bounds=(55, 70))
    assert expanded.xyxy == (0.0, 10.0, 55.0, 70.0)
    assert box.clipped(30, 30).xyxy == (10.0, 20.0, 30.0, 30.0)
    assert BBox.from_xywh(5, 6, 10, 20).xyxy == (5.0, 6.0, 15.0, 26.0)

    swapped = BBox(50, 60, 10, 20)
    assert swapped.xyxy == (10.0, 20.0, 50.0, 60.0)


def test_run_summary_and_motion_helpers() -> None:
    motion = MotionEstimate(dx=3.0, dy=4.0, theta=0.0, state=CameraState.PANNING)
    assert motion.magnitude == 5.0
    assert motion.is_pan is True
    assert motion.keyframe is False

    summary = RunSummary(
        video="clip.mp4",
        frames=300,
        duration=10.0,
        elapsed=5.0,
        detections=42,
        tracks=6,
        interactions=3,
        clips=2,
        camera_states={"STATIONARY": 250, "PANNING": 50},
    )
    assert summary.fps == 60.0
    assert summary.as_dict()["processing_fps"] == 60.0

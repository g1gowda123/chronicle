from __future__ import annotations

import json

import numpy as np

from chronicle.perception.clip_buffer import FrameRingBuffer, InteractionClipRecorder
from chronicle.perception.config import ClipConfig
from chronicle.types import BBox, HandObservation, InteractionEvent, TrackedObject


def track() -> TrackedObject:
    return TrackedObject(
        track_id=7,
        label="keys",
        bbox=BBox(100, 100, 180, 160),
        confidence=0.9,
        first_seen=0.0,
        last_seen=0.0,
        hits=4,
        age=4,
    )


def hand() -> HandObservation:
    points = np.tile(np.array([[140.0, 130.0]], dtype=np.float32), (21, 1))
    points[:, 0] += np.linspace(-8, 8, 21)
    return HandObservation(landmarks=points, handedness="Right")


def start_event(timestamp: float, frame_index: int) -> InteractionEvent:
    return InteractionEvent(
        kind="start",
        track_id=7,
        label="keys",
        hand="right",
        timestamp=timestamp,
        start_timestamp=timestamp,
        duration=0.3,
        ios=0.9,
        trigger="fingertip+ios",
        bbox=BBox(100, 100, 180, 160),
        frame_index=frame_index,
    )


def test_clip_contains_thirty_frames_and_delivers_a_composite(tmp_path) -> None:
    buffer = FrameRingBuffer(maxlen=30, max_width=320)
    recorder = InteractionClipRecorder(
        ClipConfig(pre_roll=10, length=30, post_roll=5, max_width=320)
    )
    released = []
    for index in range(40):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        snapshot = buffer.append(frame, index / 30.0, index, [track()], [hand()])
        events = [start_event(index / 30.0, index)] if index == 15 else []
        released.extend(recorder.update(buffer, snapshot, events))

    assert len(released) == 1
    clip = released[0]
    assert clip.track_id == 7
    assert len(clip.frames) == 30
    assert clip.contact_index == 10
    assert clip.contact_ts is not None and abs(clip.contact_ts - 0.5) < 1e-9
    assert clip.frames[0].frame_index == 5  # ten true pre-contact frames
    assert clip.composite is not None
    assert clip.composite.shape == (180, 960, 3)

    paths = clip.save(tmp_path)
    assert (tmp_path / f"{clip.clip_id}.jpg").exists()
    assert (tmp_path / f"{clip.clip_id}.json").exists()
    metadata = json.loads((tmp_path / f"{clip.clip_id}.json").read_text())
    assert metadata["frame_count"] == 30
    assert metadata["track_id"] == 7
    assert paths["composite"].endswith(".jpg")


def test_short_interaction_waits_for_post_roll_then_finalizes() -> None:
    buffer = FrameRingBuffer(maxlen=30, max_width=320)
    recorder = InteractionClipRecorder(
        ClipConfig(pre_roll=5, length=30, post_roll=2, max_width=320)
    )
    clips = []
    for index in range(15):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        snapshot = buffer.append(frame, index / 30.0, index, [track()], [hand()])
        events = []
        if index == 5:
            events.append(start_event(index / 30.0, index))
        if index == 8:
            events.append(
                InteractionEvent(
                    kind="end",
                    track_id=7,
                    label="keys",
                    hand="right",
                    timestamp=index / 30.0,
                    start_timestamp=5 / 30.0,
                    duration=0.1,
                    ios=0.9,
                    trigger="fingertip",
                    bbox=BBox(100, 100, 180, 160),
                    frame_index=index,
                )
            )
        clips.extend(recorder.update(buffer, snapshot, events))

    assert len(clips) == 1
    clip = clips[0]
    assert clip.metadata["ended"] is True
    assert clip.frames[0].frame_index == 0  # five true pre-contact frames
    assert len(clip.frames) == 11  # frames 0..10, including two post-roll
    assert abs(clip.end_ts - 8 / 30.0) < 1e-9
    assert recorder.pending == 0

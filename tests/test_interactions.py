from __future__ import annotations

import math

import numpy as np

from chronicle.perception.config import InteractionConfig
from chronicle.perception.interactions import InteractionTracker
from chronicle.types import BBox, HandObservation, TrackedObject


def tracked(label: str = "keys", bbox: tuple[float, float, float, float] = (100, 100, 200, 200)) -> TrackedObject:
    return TrackedObject(
        track_id=1,
        label=label,
        bbox=BBox(*bbox),
        confidence=0.9,
        first_seen=0.0,
        last_seen=0.0,
        hits=5,
        age=5,
    )


def hand_at(cx: float, cy: float, radius: float = 10.0) -> HandObservation:
    angles = np.linspace(0, 2 * math.pi, 21, endpoint=False)
    points = np.stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1
    ).astype(np.float32)
    return HandObservation(landmarks=points, confidence=0.95, handedness="Right")


def test_dwell_gates_start_and_release_emits_end() -> None:
    tracker = InteractionTracker(
        InteractionConfig(
            ios_threshold=0.60,
            dwell_seconds=0.2,
            release_grace=0.05,
            fingertip_indices=(8,),
        )
    )
    obj = tracked()
    inside = hand_at(150, 150)
    observations, events = tracker.update([obj], [inside], 0.0)
    assert observations and observations[0].contact is True
    assert events == []

    _, events = tracker.update([obj], [inside], 0.1)
    assert events == []

    observations, events = tracker.update([obj], [inside], 0.21)
    assert [event.kind for event in events] == ["start"]
    assert events[0].start_timestamp == 0.0
    assert observations[0].dwell_met is True

    _, events = tracker.update([obj], [hand_at(400, 400)], 0.50)
    assert [event.kind for event in events] == ["end"]
    assert abs(events[0].duration - 0.21) < 1e-6
    assert tracker._active == {}


def test_short_tap_never_emits_start_or_end() -> None:
    tracker = InteractionTracker(InteractionConfig(dwell_seconds=0.3, release_grace=0.05))
    obj = tracked()
    tracker.update([obj], [hand_at(150, 150)], 0.0)
    _, events = tracker.update([obj], [hand_at(400, 400)], 0.1)
    assert events == []


def test_missing_track_closes_an_active_interaction() -> None:
    tracker = InteractionTracker(InteractionConfig(dwell_seconds=0.1, release_grace=0.05))
    obj = tracked()
    tracker.update([obj], [hand_at(150, 150)], 0.0)
    _, events = tracker.update([obj], [hand_at(150, 150)], 0.2)
    assert [event.kind for event in events] == ["start"]
    _, events = tracker.update([], [], 0.4)
    assert [event.kind for event in events] == ["end"]


def test_flush_closes_ongoing_interaction() -> None:
    tracker = InteractionTracker(InteractionConfig(dwell_seconds=0.1, release_grace=0.05))
    obj = tracked()
    tracker.update([obj], [hand_at(150, 150)], 0.0)
    _, events = tracker.update([obj], [hand_at(150, 150)], 0.2)
    assert [event.kind for event in events] == ["start"]
    events = tracker.flush(0.4)
    assert [event.kind for event in events] == ["end"]
    assert tracker._active == {}


def test_furniture_labels_are_ignored() -> None:
    tracker = InteractionTracker(InteractionConfig(dwell_seconds=0.0))
    table = tracked(label="table")
    observations, events = tracker.update([table], [hand_at(150, 150)], 0.0)
    assert observations == []
    assert events == []

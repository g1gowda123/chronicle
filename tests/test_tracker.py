from __future__ import annotations

import numpy as np

from chronicle.perception.config import TrackerConfig
from chronicle.perception.tracker import ByteTrackLite
from chronicle.types import BBox, Detection


def detection(
    bbox: tuple[float, float, float, float],
    *,
    label: str = "keys",
    confidence: float = 0.9,
    embedding: np.ndarray | None = None,
) -> Detection:
    return Detection(
        label=label,
        bbox=BBox(*bbox),
        confidence=confidence,
        embedding=embedding,
        source="test",
    )


def test_identity_survives_gap_via_appearance_reid() -> None:
    tracker = ByteTrackLite(
        TrackerConfig(
            min_hits=1,
            reid_threshold=0.82,
            spatial_reid_distance=500.0,
            spatial_reid_window=60.0,
        )
    )
    embedding = np.ones(8, dtype=np.float32)

    first = tracker.update(None, [detection((100, 100, 140, 140), embedding=embedding)], 0.0)
    assert len(first.tracks) == 1
    track_id = first.tracks[0].track_id

    missing = tracker.update(None, [], 0.2)
    assert missing.tracks == []
    assert len(tracker.inactive_tracks()) == 1

    reacquired = tracker.update(
        None, [detection((106, 104, 146, 144), embedding=embedding)], 0.4
    )
    assert [track.track_id for track in reacquired.tracks] == [track_id]
    assert [track.track_id for track in reacquired.rebound] == [track_id]
    assert reacquired.new_tracks == []
    assert tracker.reid_events[-1]["source"] == "appearance"


def test_spatial_reid_works_without_embeddings() -> None:
    tracker = ByteTrackLite(
        TrackerConfig(min_hits=1, reid_threshold=0.99, spatial_reid_distance=120.0)
    )
    first = tracker.update(None, [detection((100, 100, 140, 140))], 0.0)
    track_id = first.tracks[0].track_id
    tracker.update(None, [], 0.2)
    reacquired = tracker.update(None, [detection((110, 105, 150, 145))], 0.4)
    assert [track.track_id for track in reacquired.tracks] == [track_id]
    assert tracker.reid_events[-1]["source"] == "spatial"


def test_far_detection_creates_new_track() -> None:
    tracker = ByteTrackLite(
        TrackerConfig(min_hits=1, reid_threshold=0.82, spatial_reid_distance=60.0)
    )
    first = tracker.update(None, [detection((100, 100, 140, 140))], 0.0)
    tracker.update(None, [], 0.2)
    newcomer = tracker.update(None, [detection((500, 400, 540, 440))], 0.4)
    assert newcomer.new_tracks
    assert newcomer.tracks[0].track_id != first.tracks[0].track_id


def test_min_hits_suppresses_one_frame_false_positive() -> None:
    tracker = ByteTrackLite(TrackerConfig(min_hits=2))
    first = tracker.update(None, [detection((10, 10, 50, 50))], 0.0)
    assert first.tracks == []
    assert first.new_tracks == []
    second = tracker.update(None, [detection((11, 11, 51, 51))], 0.1)
    assert len(second.tracks) == 1
    assert len(second.new_tracks) == 1


def test_low_confidence_detection_extends_existing_track() -> None:
    tracker = ByteTrackLite(TrackerConfig(min_hits=1))
    first = tracker.update(None, [detection((10, 10, 50, 50), confidence=0.9)], 0.0)
    track_id = first.tracks[0].track_id
    second = tracker.update(None, [detection((12, 11, 52, 51), confidence=0.2)], 0.1)
    assert [track.track_id for track in second.tracks] == [track_id]
    assert second.tracks[0].confidence == 0.2


def test_labels_are_never_mixed() -> None:
    tracker = ByteTrackLite(TrackerConfig(min_hits=1))
    first = tracker.update(None, [detection((10, 10, 50, 50), label="keys")], 0.0)
    assert first.tracks[0].label == "keys"
    second = tracker.update(None, [detection((10, 10, 50, 50), label="phone")], 0.1)
    labels = {track.label for track in second.tracks}
    assert labels == {"phone"}
    assert all(track.label == "phone" for track in second.tracks)


def test_relative_velocity_is_filled_by_pipeline_contract() -> None:
    tracker = ByteTrackLite(TrackerConfig(min_hits=1))
    tracker.update(None, [detection((10, 10, 50, 50))], 0.0)
    second = tracker.update(None, [detection((20, 10, 60, 50))], 0.1)
    track = second.tracks[0]
    assert track.velocity[0] > 0
    assert track.relative_velocity == (0.0, 0.0)

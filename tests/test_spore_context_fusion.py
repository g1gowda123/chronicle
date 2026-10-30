"""Track spread used by adaptive context-fusion boxes."""

from __future__ import annotations

from chronicle.perception.spore import _DetectionTrack
from chronicle.types import BBox, Detection


def _track(boxes: list[BBox]) -> _DetectionTrack:
    track = _DetectionTrack()
    for index, box in enumerate(boxes):
        track.add(
            Detection(label="cup", bbox=box, confidence=0.8),
            timestamp=float(index),
            support=None,
        )
    return track


def test_stable_track_has_high_pair_iou() -> None:
    track = _track([BBox(10, 10, 30, 30), BBox(11, 11, 31, 31), BBox(10, 12, 30, 32)])
    assert track.mean_pair_iou() > 0.6


def test_moving_track_has_low_pair_iou() -> None:
    track = _track([BBox(10, 10, 30, 30), BBox(110, 110, 130, 130)])
    assert track.mean_pair_iou() == 0.0


def test_single_detection_is_stable() -> None:
    assert _track([BBox(10, 10, 30, 30)]).mean_pair_iou() == 1.0

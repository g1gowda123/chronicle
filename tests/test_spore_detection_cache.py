"""Query-time detector cache: repeated questions reuse immutable keyframes."""

from __future__ import annotations

import numpy as np

from chronicle.perception.capsules import CapsuleStore
from chronicle.perception.detector import BaseDetector
from chronicle.perception.spore import SporeConfig, SporeQueryEngine
from chronicle.types import BBox, Detection


class CountingDetector(BaseDetector):
    def __init__(self) -> None:
        self.vocabulary: list[str] = []
        self.calls = 0

    def set_vocabulary(self, vocabulary) -> None:
        self.vocabulary = list(vocabulary)

    def detect(self, frame, frame_index=0, timestamp=0.0):
        self.calls += 1
        label = self.vocabulary[0] if self.vocabulary else "object"
        return [Detection(label=label, bbox=BBox(1, 2, 3, 4), confidence=0.9)]


def test_detection_cache_reuses_keyframes_and_tracks_hits() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detector = CountingDetector()
    engine = SporeQueryEngine(
        CapsuleStore(),
        SporeConfig(detect_cache=True),
        detector=detector,
    )

    first = engine._detect_cached(
        frame, ["mug", "table"], frame_index=7, timestamp=0.7
    )
    second = engine._detect_cached(
        frame, ["mug", "table"], frame_index=7, timestamp=0.7
    )

    assert detector.calls == 1
    assert engine.detector_calls == 1
    assert engine.detection_cache_hits == 1
    assert first is second
    assert first[0].label == "mug"

    # A different prompt set is a different computation.
    engine._detect_cached(frame, ["cup"], frame_index=7, timestamp=0.7)
    assert detector.calls == 2


def test_detection_cache_can_be_disabled() -> None:
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detector = CountingDetector()
    engine = SporeQueryEngine(
        CapsuleStore(),
        SporeConfig(detect_cache=False),
        detector=detector,
    )

    engine._detect_cached(frame, ["mug"], frame_index=1, timestamp=0.1)
    engine._detect_cached(frame, ["mug"], frame_index=1, timestamp=0.1)

    assert detector.calls == 2
    assert engine.detection_cache_hits == 0

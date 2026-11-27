"""Tests for the fixed-label presence/absence verifier."""

from __future__ import annotations

import numpy as np

from chronicle.perception.absence import COCO_ALIASES, FixedLabelVerifier
from chronicle.types import BBox, Detection


class FakeContext:
    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def image(self) -> np.ndarray:
        return self.frame


class FakeStore:
    def __init__(self, timestamps: list[float]) -> None:
        self.contexts = [FakeContext(timestamp) for timestamp in timestamps]


class FakeDetector:
    labels = {"cup", "bowl", "tv"}

    def __init__(self, detections: dict[float, list[tuple[str, float]]]) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, frame, *, timestamp: float = 0.0) -> list[Detection]:
        self.calls += 1
        return [
            Detection(
                label=label,
                bbox=BBox(0.0, 0.0, 10.0, 10.0),
                confidence=confidence,
            )
            for label, confidence in self.detections.get(timestamp, [])
        ]


def make_verifier(
    timestamps: list[float], detections: dict[float, list[tuple[str, float]]]
) -> FixedLabelVerifier:
    return FixedLabelVerifier(
        FakeStore(timestamps),  # type: ignore[arg-type]
        detector=FakeDetector(detections),
        threshold=0.50,
    )


def test_scan_is_cached_and_counts_calls() -> None:
    verifier = make_verifier([0.0, 1.0, 2.0], {})
    assert verifier.scan() == 3
    assert verifier.scan() == 0
    assert verifier.scan_calls == 3
    assert verifier.scan_seconds >= 0.0


def test_present_returns_best_detection() -> None:
    verifier = make_verifier(
        [0.0, 1.0, 2.0],
        {1.0: [("cup", 0.51)], 2.0: [("cup", 0.82)]},
    )
    verdict = verifier.check("cup")
    assert verdict.verdict == "present"
    assert verdict.supported
    assert verdict.hits == 2
    assert verdict.timestamp == 2.0
    assert verdict.confidence == 0.82
    assert verdict.contexts_scanned == 3
    assert verdict.bbox is not None


def test_not_observed_is_not_absence() -> None:
    verifier = make_verifier([0.0, 1.0], {0.0: [("bowl", 0.49)]})
    verdict = verifier.check("bowl")
    assert verdict.verdict == "not_observed"
    assert verdict.supported
    assert verdict.hits == 0
    assert verdict.confidence is None
    assert verdict.as_dict()["verdict"] == "not_observed"


def test_unsupported_term_never_returns_negative() -> None:
    verifier = make_verifier([0.0], {})
    verdict = verifier.check("laptop")
    assert verdict.verdict == "unsupported"
    assert not verdict.supported
    assert verdict.labels == ("laptop",)


def test_time_window_filters_contexts() -> None:
    verifier = make_verifier(
        [0.0, 10.0, 20.0],
        {0.0: [("cup", 0.9)]},
    )
    inside = verifier.check("cup", at_time=0.5, window=2.5)
    assert inside.verdict == "present"
    assert inside.contexts_scanned == 1
    outside = verifier.check("cup", at_time=20.0, window=2.5)
    assert outside.verdict == "not_observed"
    assert outside.contexts_scanned == 1
    assert outside.window == (17.5, 22.5)


def test_alias_maps_television_to_tv() -> None:
    assert COCO_ALIASES["television"] == ("tv",)
    verifier = make_verifier([0.0], {0.0: [("tv", 0.73)]})
    verdict = verifier.check("television")
    assert verdict.verdict == "present"
    assert verdict.labels == ("tv",)


def test_check_many_preserves_order() -> None:
    verifier = make_verifier([0.0], {0.0: [("cup", 0.8)]})
    verdicts = verifier.check_many(["cup", "laptop", "bowl"])
    assert [verdict.term for verdict in verdicts] == ["cup", "laptop", "bowl"]
    assert [verdict.verdict for verdict in verdicts] == [
        "present",
        "unsupported",
        "not_observed",
    ]


def test_custom_aliases_are_supported() -> None:
    verifier = FixedLabelVerifier(
        FakeStore([0.0]),  # type: ignore[arg-type]
        detector=FakeDetector({0.0: [("tv", 0.9)]}),
        aliases={"screen": ("tv",)},
    )
    assert verifier.check("screen").verdict == "present"
    assert verifier.check("television").verdict == "unsupported"

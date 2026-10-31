"""Detector-synonym binding needs persistence, not a single label guess."""

from __future__ import annotations

import cv2
import numpy as np

from chronicle.perception.capsules import CapsuleStore, ContextFrame
from chronicle.perception.detector import BaseDetector
from chronicle.perception.spore import SporeConfig, SporeQueryEngine
from chronicle.types import BBox, Detection


class _Embedder:
    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def close(self) -> None:
        pass


class _InstantPotDetector(BaseDetector):
    def set_vocabulary(self, vocabulary) -> None:
        self.vocabulary = list(vocabulary)

    def detect(self, frame, frame_index=0, timestamp=0.0):
        return [
            Detection(
                label="instant pot",
                bbox=BBox(10, 10, 50, 50),
                confidence=0.70,
            )
        ]


def _context(timestamp: float, frame_index: int) -> ContextFrame:
    ok, encoded = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    assert ok
    return ContextFrame(
        timestamp=timestamp,
        frame_index=frame_index,
        image_jpeg=encoded.tobytes(),
    )


def _engine(contexts: list[ContextFrame]) -> SporeQueryEngine:
    store = CapsuleStore()
    store.contexts = contexts
    return SporeQueryEngine(
        store,
        SporeConfig(
            late_binding=False,
            synonym_binding=True,
            synonym_min_contexts=2,
        ),
        embedder=_Embedder(),
        detector=_InstantPotDetector(),
    )


def test_synonym_grounds_rice_cooker_after_two_contexts() -> None:
    engine = _engine([_context(0.0, 0), _context(1.0, 1)])

    bindings = engine.bind("rice cooker", at_time=1.0)

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.term == "rice cooker"
    assert binding.label == "rice cooker"
    assert binding.provenance == "synonym_track"
    assert binding.capsule_id == ""
    assert binding.views_agree == 2


def test_synonym_requires_repeated_detection() -> None:
    engine = _engine([_context(0.0, 0)])

    assert engine.bind("rice cooker", at_time=0.0) == []


def test_synonym_is_opt_in() -> None:
    store = CapsuleStore()
    store.contexts = [_context(0.0, 0), _context(1.0, 1)]
    engine = SporeQueryEngine(
        store,
        SporeConfig(late_binding=False),
        embedder=_Embedder(),
        detector=_InstantPotDetector(),
    )

    assert engine.bind("rice cooker", at_time=1.0) == []

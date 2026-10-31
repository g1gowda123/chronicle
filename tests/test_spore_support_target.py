"""Support surfaces are valid query targets, not only predicates."""

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


class _DeskDetector(BaseDetector):
    def set_vocabulary(self, vocabulary) -> None:
        self.vocabulary = list(vocabulary)

    def detect(self, frame, frame_index=0, timestamp=0.0):
        return [
            Detection(label="desk", bbox=BBox(10, 40, 90, 95), confidence=0.80),
            Detection(label="cup", bbox=BBox(30, 30, 45, 50), confidence=0.60),
        ]


def _context(timestamp: float, frame_index: int) -> ContextFrame:
    ok, encoded = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    assert ok
    return ContextFrame(
        timestamp=timestamp,
        frame_index=frame_index,
        image_jpeg=encoded.tobytes(),
    )


def _engine() -> SporeQueryEngine:
    store = CapsuleStore()
    store.contexts = [_context(1.0, 1)]
    return SporeQueryEngine(
        store,
        SporeConfig(late_binding=False),
        embedder=_Embedder(),
        detector=_DeskDetector(),
    )


def test_table_binds_to_desk_support_surface() -> None:
    engine = _engine()

    bindings = engine.bind("table", at_time=1.0)

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.term == "table"
    assert binding.label == "desk"
    assert binding.support is None
    assert binding.provenance == "support_track"
    assert binding.capsule_id == ""


def test_support_target_answer_uses_query_term() -> None:
    engine = _engine()

    answer = engine.answer("Where is the table?", at_time=1.0)

    assert not answer.abstained
    assert answer.answer == "table at 1.0s"
    assert answer.bindings[0].bbox == BBox(10, 40, 90, 95)


def test_non_support_term_still_abstains_without_capsules() -> None:
    engine = _engine()

    assert engine.bind("shoe", at_time=1.0) == []

"""Open-vocabulary scene inventory: objects on a support surface."""

from __future__ import annotations

import cv2
import numpy as np

from chronicle.perception import spore as spore_module
from chronicle.perception.capsules import CapsuleStore, ContextFrame
from chronicle.perception.detector import BaseDetector
from chronicle.perception.spore import SporeConfig, SporeQueryEngine
from chronicle.types import BBox, Detection


class SceneDetector(BaseDetector):
    def set_vocabulary(self, vocabulary) -> None:
        self.vocabulary = list(vocabulary)

    def detect(self, frame, frame_index=0, timestamp=0.0):
        return [
            Detection(label="desk", bbox=BBox(0, 40, 100, 100), confidence=0.70),
            Detection(label="joy-con", bbox=BBox(20, 60, 40, 80), confidence=0.90),
            Detection(label="person", bbox=BBox(0, 0, 100, 40), confidence=0.95),
        ]


def _context(timestamp: float, frame_index: int) -> ContextFrame:
    ok, encoded = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    assert ok
    return ContextFrame(
        timestamp=timestamp,
        frame_index=frame_index,
        image_jpeg=encoded.tobytes(),
    )


def test_inventory_question_parser() -> None:
    assert (
        SporeQueryEngine._inventory_question("What objects are on the work table?")
        == "work table"
    )
    assert SporeQueryEngine._inventory_question("What is on the table?") == "table"
    assert SporeQueryEngine._inventory_question("Which items are at the desk?") == "desk"
    assert SporeQueryEngine._inventory_question("Where is the mug?") is None


def test_inventory_lists_objects_on_support() -> None:
    store = CapsuleStore()
    store.contexts.extend([_context(1.0, 1), _context(2.0, 2)])
    engine = SporeQueryEngine(
        store,
        SporeConfig(
            inventory_labels=("joy-con", "person", "desk"),
            inventory_contexts=4,
        ),
        detector=SceneDetector(),
    )

    answer = engine.answer("What objects are on the work table?")

    assert answer.abstained is False
    assert "joy-con" in answer.answer
    assert "person" not in answer.answer
    assert answer.detector_calls >= 1
    assert answer.stats["inventory"] == ["joy-con"]


def test_inventory_builds_detector_on_first_question(monkeypatch) -> None:
    store = CapsuleStore()
    store.contexts.extend([_context(1.0, 1)])
    monkeypatch.setattr(
        spore_module, "build_detector", lambda perception: SceneDetector()
    )
    engine = SporeQueryEngine(
        store,
        SporeConfig(
            inventory_labels=("joy-con", "person", "desk"),
            inventory_contexts=2,
            late_binding=True,
        ),
    )

    answer = engine.answer("What objects are on the work table?")

    assert answer.abstained is False
    assert "joy-con" in answer.answer

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


class LateSceneDetector(SceneDetector):
    def detect(self, frame, frame_index=0, timestamp=0.0):
        if frame_index not in (3, 4):
            return []
        return super().detect(frame, frame_index=frame_index, timestamp=timestamp)


class TransientDetector(SceneDetector):
    def detect(self, frame, frame_index=0, timestamp=0.0):
        detections = super().detect(frame, frame_index=frame_index, timestamp=timestamp)
        if frame_index == 3:
            detections.append(
                Detection(label="phone", bbox=BBox(25, 65, 35, 75), confidence=0.40)
            )
            detections.append(
                Detection(label="knife", bbox=BBox(45, 65, 55, 85), confidence=0.80)
            )
        return detections


class DuplicateDetector(SceneDetector):
    def detect(self, frame, frame_index=0, timestamp=0.0):
        return [
            Detection(label="desk", bbox=BBox(0, 40, 100, 100), confidence=0.70),
            Detection(label="joy-con", bbox=BBox(20, 60, 40, 80), confidence=0.90),
            Detection(label="remote", bbox=BBox(20, 60, 40, 80), confidence=0.70),
            Detection(
                label="joy-con controllers",
                bbox=BBox(20, 60, 40, 80),
                confidence=0.80,
            ),
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
            inventory_min_hits=1,
            late_binding=True,
        ),
    )

    answer = engine.answer("What objects are on the work table?")

    assert answer.abstained is False
    assert "joy-con" in answer.answer


def test_inventory_expands_contexts_when_first_batch_is_empty() -> None:
    store = CapsuleStore()
    store.contexts.extend([_context(float(i), i) for i in range(1, 5)])
    engine = SporeQueryEngine(
        store,
        SporeConfig(
            inventory_labels=("joy-con", "person", "desk"),
            inventory_contexts=1,
            inventory_max_contexts=4,
        ),
        detector=LateSceneDetector(),
    )

    answer = engine.answer("What objects are on the work table?")

    assert answer.abstained is False
    assert answer.stats["inventory"] == ["joy-con"]


def test_inventory_drops_single_frame_false_positive() -> None:
    store = CapsuleStore()
    store.contexts.extend([_context(float(i), i) for i in range(1, 5)])
    engine = SporeQueryEngine(
        store,
        SporeConfig(
            inventory_labels=("joy-con", "phone", "knife", "person", "desk"),
            inventory_contexts=4,
            inventory_min_hits=2,
        ),
        detector=TransientDetector(),
    )

    answer = engine.answer("What objects are on the work table?")

    assert answer.stats["inventory"] == ["joy-con", "knife"]
    assert "phone" not in answer.answer


def test_inventory_merges_aliases_and_suppresses_duplicate_boxes() -> None:
    store = CapsuleStore()
    store.contexts.extend([_context(1.0, 1), _context(2.0, 2)])
    engine = SporeQueryEngine(
        store,
        SporeConfig(
            inventory_labels=("joy-con", "joy-con controllers", "remote", "desk"),
            inventory_contexts=4,
            inventory_min_hits=2,
        ),
        detector=DuplicateDetector(),
    )

    answer = engine.answer("What objects are on the work table?")

    assert answer.stats["inventory"] == ["joy-con"]
    assert "remote" not in answer.answer
    assert "joy-con controllers" not in answer.answer

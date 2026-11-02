"""Spatial relation parsing and geometric matching."""

from __future__ import annotations

import cv2
import numpy as np

from chronicle.perception.capsules import CapsuleStore, ContextFrame
from chronicle.perception.detector import BaseDetector
from chronicle.perception.spore import (
    Binding,
    SporeConfig,
    SporeQueryEngine,
    _binding_matches_relation,
    extract_relation_qualifier,
    extract_terms,
)
from chronicle.types import BBox, Detection


def _binding(box: BBox) -> Binding:
    return Binding(
        term="cup",
        label="cup",
        capsule_id="spore-1",
        confidence=0.9,
        timestamp=1.0,
        bbox=box,
    )


def test_relation_parser_strips_target_phrase() -> None:
    assert extract_relation_qualifier(
        "Where is the cup next to the plate?"
    ) == ("cup", "next to", "plate")
    assert extract_relation_qualifier(
        "Where is the cup near the plate?"
    ) == ("cup", "near", "plate")
    assert extract_relation_qualifier(
        "Where is the cup left of the plate?"
    ) == ("cup", "left of", "plate")
    assert extract_relation_qualifier(
        "Where is the cup on the shelf?"
    ) is None


def test_relation_terms_do_not_leak_into_target() -> None:
    assert extract_terms("Where is the cup next to the plate?") == ["cup"]
    assert extract_terms("Where is the cup near the plate?") == ["cup"]


def test_relation_geometry_proximity_and_direction() -> None:
    plate = _binding(BBox(100, 100, 160, 160))
    anchors = [plate]

    assert _binding_matches_relation(
        _binding(BBox(170, 110, 200, 140)), anchors, "next to"
    )
    assert not _binding_matches_relation(
        _binding(BBox(400, 400, 430, 430)), anchors, "next to"
    )
    assert _binding_matches_relation(
        _binding(BBox(40, 110, 90, 140)), anchors, "left of"
    )
    assert _binding_matches_relation(
        _binding(BBox(170, 110, 200, 140)), anchors, "right of"
    )
    assert _binding_matches_relation(
        _binding(BBox(110, 40, 140, 90)), anchors, "above"
    )
    assert _binding_matches_relation(
        _binding(BBox(110, 170, 140, 220)), anchors, "below"
    )


def test_relation_requires_anchor() -> None:
    assert not _binding_matches_relation(
        _binding(BBox(170, 110, 200, 140)), [], "next to"
    )


class _Embedder:
    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def close(self) -> None:
        pass


class _Detector(BaseDetector):
    class config:
        confidence = 0.30

    def set_vocabulary(self, vocabulary) -> None:
        self.vocabulary = list(vocabulary)

    def detect(self, frame, frame_index=0, timestamp=0.0):
        return [
            Detection(label="cup", bbox=BBox(170, 110, 200, 140), confidence=0.9),
            Detection(label="plate", bbox=BBox(100, 100, 160, 160), confidence=0.9),
        ]


def test_answer_applies_relation_filter() -> None:
    ok, encoded = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    assert ok
    store = CapsuleStore()
    store.contexts = [
        ContextFrame(
            timestamp=1.0,
            frame_index=1,
            image_jpeg=encoded.tobytes(),
        )
    ]
    engine = SporeQueryEngine(
        store,
        SporeConfig(late_binding=False, context_fallback=True),
        embedder=_Embedder(),
        detector=_Detector(),
    )

    near = engine.answer("Where is the cup next to the plate?")
    assert not near.abstained
    assert near.terms == ["cup"]
    assert near.stats["relation"]["anchor"] == "plate"

    left = engine.answer("Where is the cup left of the plate?")
    assert left.abstained
    assert left.stats["relation"]["filtered"] == 1

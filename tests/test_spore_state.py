"""Grounded state/absence parsing and answer composition."""

from __future__ import annotations

import cv2
import numpy as np

from chronicle.perception.capsules import CapsuleStore, ContextFrame
from chronicle.perception.detector import BaseDetector
from chronicle.perception.spore import (
    SporeConfig,
    SporeQueryEngine,
    extract_state_query,
    extract_terms,
)
from chronicle.types import BBox, Detection


def test_state_parser_presence_forms() -> None:
    query = extract_state_query("Is there a kettle on the countertop?")
    assert query is not None
    assert (query.kind, query.term, query.place) == ("presence", "kettle", "countertop")

    query = extract_state_query("Is the cup on the table?")
    assert query is not None
    assert (query.kind, query.term, query.place) == ("presence", "cup", "table")

    query = extract_state_query("Are there plates on the counter?")
    assert query is not None
    assert (query.kind, query.term, query.place) == ("presence", "plates", "counter")


def test_state_parser_time_clause() -> None:
    query = extract_state_query("Was the cup on the table at 20 seconds?")
    assert query is not None
    assert query.kind == "presence"
    assert query.at_time == 20.0
    assert query.term == "cup"


def test_state_parser_still_and_moved_forms() -> None:
    query = extract_state_query("Is the cup still on the shelf?")
    assert query is not None
    assert (query.kind, query.term, query.place) == ("still", "cup", "shelf")

    query = extract_state_query("Did the plate move?")
    assert query is not None
    assert (query.kind, query.term, query.place) == ("moved", "plate", None)

    query = extract_state_query("Did the plate move from the table?")
    assert query is not None
    assert (query.kind, query.term, query.place) == ("moved", "plate", "table")


def test_state_parser_leaves_ordinary_questions_alone() -> None:
    assert extract_state_query("Where is the cup?") is None
    assert extract_state_query("What is on the table?") is None
    assert extract_state_query("How many cups are on the table?") is None
    assert extract_state_query("Is the cup on the windowsill?") is None
    assert extract_terms("Is the cup on the table?") == ["cup"]


class _Embedder:
    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def close(self) -> None:
        pass


class _SceneDetector(BaseDetector):
    """Synthetic scene: cup moves from the table to the shelf at 3 s."""

    class config:
        confidence = 0.30

    def __init__(self) -> None:
        self.vocabulary: list[str] = []

    def set_vocabulary(self, vocabulary) -> None:
        self.vocabulary = [str(label).lower() for label in vocabulary]

    def detect(self, frame, frame_index=0, timestamp=0.0):
        wanted = set(self.vocabulary)
        table = BBox(0, 55, 200, 100)
        shelf = BBox(0, 0, 200, 40)
        detections: list[Detection] = []
        if wanted & {"table", "work table", "countertop"}:
            detections.append(Detection("table", table, 0.9))
        if "shelf" in wanted:
            detections.append(Detection("shelf", shelf, 0.9))
        if "cup" in wanted:
            if timestamp <= 2.0:
                detections.append(Detection("cup", BBox(20, 40, 40, 75), 0.8))
            else:
                detections.append(Detection("cup", BBox(20, 5, 40, 30), 0.8))
        if "toaster" in wanted:
            detections.append(Detection("toaster", BBox(120, 40, 160, 75), 0.8))
        return detections


def _frame() -> ContextFrame:
    ok, encoded = cv2.imencode(".jpg", np.zeros((100, 200, 3), dtype=np.uint8))
    assert ok
    return ContextFrame(timestamp=0.0, frame_index=0, image_jpeg=encoded.tobytes())


def _engine() -> SporeQueryEngine:
    store = CapsuleStore()
    store.contexts = []
    for index, timestamp in enumerate((0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5)):
        context = _frame()
        context.timestamp = timestamp
        context.frame_index = index
        store.contexts.append(context)
    return SporeQueryEngine(
        store,
        SporeConfig(
            late_binding=True,
            context_fallback=True,
            state_queries=True,
            state_contexts=16,
            state_time_window=2.0,
            state_min_confidence=0.3,
            state_min_observations=2,
        ),
        embedder=_Embedder(),
        detector=_SceneDetector(),
    )


def test_presence_yes_and_no() -> None:
    engine = _engine()
    present = engine.answer("Is there a cup on the table?", at_time=1.0)
    assert present.verdict == "present"
    assert not present.abstained
    assert "yes" in present.answer
    assert present.stats["state"]["support"] == "table"

    absent = engine.answer("Is there a kettle on the table?", at_time=1.0)
    assert absent.verdict == "not_observed"
    assert absent.abstained
    assert "no supported evidence" in absent.answer


def test_still_yes_moved_and_unobserved() -> None:
    engine = _engine()
    still = engine.answer("Is the cup still on the table?", at_time=1.0)
    assert still.verdict == "present"
    assert "still on the table" in still.answer

    moved = engine.answer("Is the cup still on the table?", at_time=4.0)
    assert moved.verdict == "moved"
    assert "countertop" not in moved.answer
    assert "shelf" in moved.answer

    on_shelf = engine.answer("Is the cup still on the shelf?", at_time=4.0)
    assert on_shelf.verdict == "present"

    never = engine.answer("Is the kettle still on the table?", at_time=4.0)
    assert never.verdict == "not_observed"
    assert "no supported evidence" in never.answer


def test_moved_and_stayed_verdicts() -> None:
    engine = _engine()
    moved = engine.answer("Did the cup move?")
    assert moved.verdict == "moved"
    assert "from the table to the shelf" in moved.answer

    stayed = engine.answer("Did the toaster move?")
    assert stayed.verdict == "stayed"
    assert "stayed on" in stayed.answer

    unknown = engine.answer("Did the kettle move?")
    assert unknown.verdict == "unknown"
    assert unknown.abstained


class _FakeSurface:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def support(self, image, bbox, label, current=None, *, key=None):
        self.calls += 1
        return self.label

    def close(self):
        pass


def test_state_surface_fallback_can_fill_missing_support() -> None:
    engine = _engine()
    engine.config.state_surface_support = True
    fake = _FakeSurface("floor")
    engine._surface = fake  # type: ignore[assignment]

    placements, _ = engine._placements("cup")
    assert placements
    assert {placement.support for placement in placements} == {"floor"}
    assert fake.calls == len(placements)


class _FakeFixedVerifier:
    def __init__(self, verdict: str, supported: bool = True) -> None:
        self.verdict = verdict
        self.supported = supported
        self.threshold = 0.5

    def check(self, term, at_time=None, window=2.5):
        return self

    def as_dict(self):
        return {
            "term": "cup",
            "verdict": self.verdict,
            "supported": self.supported,
            "threshold": self.threshold,
        }


def test_fixed_label_veto_downgrades_positive_presence() -> None:
    engine = _engine()
    engine.config.state_fixed_verify = True
    engine._fixed_verifier = _FakeFixedVerifier("not_observed")

    answer = engine.answer("Is there a cup on the table?")
    assert answer.verdict == "not_observed"
    assert answer.abstained
    assert "fixed-label verifier" in answer.answer


def test_fixed_label_unsupported_term_is_not_downgraded() -> None:
    engine = _engine()
    engine.config.state_fixed_verify = True
    engine._fixed_verifier = _FakeFixedVerifier("unsupported", supported=False)

    answer = engine.answer("Is there a cup on the table?")
    assert answer.verdict == "present"


class _ConfusedDetector(_SceneDetector):
    """Returns a competing label over the cup when the bank is present."""

    def detect(self, frame, frame_index=0, timestamp=0.0):
        detections = super().detect(frame, frame_index, timestamp)
        wanted = set(self.vocabulary)
        if "teapot" in wanted and timestamp <= 2.0:
            detections = [
                detection
                for detection in detections
                if detection.label != "cup"
            ]
            detections.append(Detection("teapot", BBox(20, 40, 40, 75), 0.95))
            detections.append(Detection("cup", BBox(20, 40, 40, 75), 0.5))
        return detections


def _engine_with(detector, **overrides) -> SporeQueryEngine:
    engine = _engine()
    engine._detector = detector
    for key, value in overrides.items():
        setattr(engine.config, key, value)
    return engine


def test_competing_label_downgrades_positive_answer() -> None:
    engine = _engine_with(
        _ConfusedDetector(),
        state_verify_labels=("teapot", "toaster"),
        state_verify_margin=0.05,
    )
    answer = engine.answer("Is there a cup on the table?")
    assert answer.verdict == "not_observed"
    assert answer.abstained
    assert "ambiguous evidence" in answer.answer
    assert answer.stats["state"]["conflict"]["label"] == "teapot"


def test_ordinary_location_question_is_not_rewritten() -> None:
    engine = _engine()
    answer = engine.answer("Where is the cup?", at_time=1.0)
    assert answer.verdict is None
    assert "state" not in answer.stats

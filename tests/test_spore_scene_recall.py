"""Detector-free scene recall over stored CLIP scene embeddings."""

from __future__ import annotations

import numpy as np

from chronicle.perception.capsules import CapsuleStore, SceneCapsule
from chronicle.perception.embedder import BaseEmbedder
from chronicle.perception.spore import SporeConfig, SporeQueryEngine


class FakeEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_text(self, text: str):
        self.texts.append(text)
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_image(self, frame):
        return np.array([1.0, 0.0], dtype=np.float32)


def _scene(timestamp: float, vector: list[float]) -> SceneCapsule:
    return SceneCapsule(
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        score=1.0,
        embedding=np.array(vector, dtype=np.float32),
    )


def test_scene_recall_ranks_by_text_similarity() -> None:
    store = CapsuleStore()
    store.add_scene(_scene(1.0, [0.0, 1.0]))
    store.add_scene(_scene(2.0, [0.8, 0.6]))
    store.add_scene(_scene(3.0, [1.0, 0.0]))
    engine = SporeQueryEngine(store, SporeConfig(), embedder=FakeEmbedder())

    hits = engine.scene_recall("stairs", top_k=2)

    assert [scene.timestamp for scene, _ in hits] == [3.0, 2.0]
    assert hits[0][1] > hits[1][1] > 0.0


def test_scene_recall_skips_scenes_without_embeddings() -> None:
    store = CapsuleStore()
    store.add_scene(_scene(1.0, [1.0, 0.0]))
    store.add_scene(SceneCapsule(timestamp=2.0, frame_index=20, score=1.0))
    engine = SporeQueryEngine(store, SporeConfig(), embedder=FakeEmbedder())

    hits = engine.scene_recall("cup")

    assert len(hits) == 1
    assert hits[0][0].timestamp == 1.0


def test_scene_recall_is_empty_without_scenes_or_term() -> None:
    engine = SporeQueryEngine(CapsuleStore(), SporeConfig(), embedder=FakeEmbedder())
    assert engine.scene_recall("stairs") == []
    assert engine.scene_recall("") == []


def test_scene_visibility_question_answers_without_detector() -> None:
    store = CapsuleStore()
    store.add_scene(_scene(1.0, [0.0, 1.0]))
    store.add_scene(_scene(3.0, [1.0, 0.0]))
    engine = SporeQueryEngine(store, SporeConfig(), embedder=FakeEmbedder())

    answer = engine.answer("When was the stairs visible?")

    assert answer.answer.startswith("saw stairs at 3.0s")
    assert answer.bindings[0].provenance == "scene"
    assert answer.stats["scene_recall"][0]["timestamp"] == 3.0
    assert answer.detector_calls == 0


def test_scene_parser_does_not_hijack_touch_questions() -> None:
    assert SporeQueryEngine._scene_question("When did I first pick up the cup?") is None
    assert SporeQueryEngine._scene_question("When did I see the stairs?") == "stairs"
    assert SporeQueryEngine._scene_question("When was the door visible?") == "door"


def test_scene_answer_abstains_below_threshold() -> None:
    store = CapsuleStore()
    store.add_scene(_scene(2.0, [1.0, 0.0]))
    engine = SporeQueryEngine(
        store,
        SporeConfig(scene_recall_min_score=1.1),
        embedder=FakeEmbedder(),
    )

    answer = engine.answer("When did I see the stairs?")

    assert answer.abstained is True
    assert answer.answer == "not enough evidence"
    assert answer.bindings[0].provenance == "scene"

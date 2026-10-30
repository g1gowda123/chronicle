from __future__ import annotations

import numpy as np

from chronicle.perception.capsules import CapsuleStore, ObjectCapsule
from chronicle.perception.spore import SporePerceiver, _Candidate, _descriptor_similarity, _nms
from chronicle.types import BBox


def _candidate(box: tuple[float, float, float, float], score: float) -> _Candidate:
    bbox = BBox(*box)
    return _Candidate(
        bbox=bbox,
        source="test",
        score=score,
        detail=0.0,
        hand_overlap=0.0,
        motion=0.0,
        saturation=0.0,
    )


def _greedy_nms(candidates: list[_Candidate], threshold: float) -> list[_Candidate]:
    kept: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(candidate.bbox.iou(other.bbox) < threshold for other in kept):
            kept.append(candidate)
    return kept


def test_vectorized_nms_matches_greedy_reference() -> None:
    random = np.random.default_rng(7)
    for _ in range(20):
        candidates = []
        for _ in range(24):
            x = float(random.uniform(0, 140))
            y = float(random.uniform(0, 100))
            width = float(random.uniform(8, 60))
            height = float(random.uniform(8, 60))
            candidates.append(
                _candidate((x, y, x + width, y + height), float(random.uniform(0, 1)))
            )
        expected = [item.bbox for item in _greedy_nms(candidates, 0.38)]
        actual = [item.bbox for item in _nms(candidates, 0.38)]
        assert actual == expected


def test_vectorized_novelty_matches_pairwise_cosine() -> None:
    store = CapsuleStore()
    descriptors = np.random.default_rng(3).normal(size=(12, 64)).astype(np.float32)
    for index, descriptor in enumerate(descriptors):
        capsule = ObjectCapsule(
            capsule_id=f"c{index}",
            created_at=float(index),
            last_seen=float(index),
            last_bbox=BBox(0, 0, 10, 10),
        )
        capsule.metadata["descriptor"] = (descriptor / np.linalg.norm(descriptor)).tolist()
        store.capsules[capsule.capsule_id] = capsule

    perceiver = SporePerceiver()
    perceiver.store = store
    candidates = [
        _candidate((0, 0, 10, 10), 1.0),
        _candidate((5, 5, 15, 15), 1.0),
    ]
    for index, candidate in enumerate(candidates):
        vector = np.random.default_rng(11 + index).normal(size=64).astype(np.float32)
        candidate.descriptor = vector / np.linalg.norm(vector)
    perceiver._annotate_novelty(candidates)

    for candidate in candidates:
        best = max(
            _descriptor_similarity(candidate.descriptor, capsule.metadata.get("descriptor"))
            for capsule in store.capsules.values()
        )
        assert abs(candidate.novelty - max(0.0, 1.0 - best)) < 1e-6

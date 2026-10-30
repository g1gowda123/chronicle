"""Tests for split capsule evidence storage."""

from __future__ import annotations

import numpy as np

from chronicle.perception.capsules import CapsuleStore, ContextFrame, SporeView
from chronicle.types import BBox


def _view(timestamp: float) -> SporeView:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    return SporeView(
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        bbox=BBox(10, 10, 40, 40),
        score=0.5,
        image=image,
        embedding=np.ones(8, dtype=np.float32) / np.sqrt(8),
    )


def test_split_store_keeps_index_small_and_round_trips(tmp_path) -> None:
    store = CapsuleStore()
    store.add_capsule(1.0, _view(1.0))
    store.add_context(
        ContextFrame(
            timestamp=1.0,
            frame_index=10,
            image_jpeg=b"\xff\xd8\xff\xd9",
            metadata={"frame_size": [32, 24]},
        )
    )

    full = store.save(tmp_path / "full.json")
    split = store.save(tmp_path / "index.json", split=True)
    loaded = CapsuleStore.load(split)

    assert split.stat().st_size < full.stat().st_size
    assert (tmp_path / "index_evidence").is_dir()
    assert loaded.stats()["capsules"] == 1
    assert loaded.stats()["contexts"] == 1
    view = next(iter(loaded.capsules.values())).best_views[0]
    assert view.image is not None
    assert view.image.shape == (24, 32, 3)
    assert loaded.contexts[0].image_jpeg == b"\xff\xd8\xff\xd9"

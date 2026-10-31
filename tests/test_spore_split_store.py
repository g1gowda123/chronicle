"""Tests for split capsule evidence storage."""

from __future__ import annotations

import numpy as np

from chronicle.perception.capsules import (
    CapsuleStore,
    ContextFrame,
    SceneCapsule,
    SporeView,
)
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


def test_split_store_can_drop_view_pixels_but_keep_contexts(tmp_path) -> None:
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

    with_views = store.save(tmp_path / "with_views.json", split=True)
    without_views = store.save(
        tmp_path / "no_views.json",
        split=True,
        include_view_images=False,
    )
    loaded = CapsuleStore.load(without_views)

    assert without_views.stat().st_size < with_views.stat().st_size
    view = next(iter(loaded.capsules.values())).best_views[0]
    assert view.image is None
    assert view.embedding is not None
    assert loaded.contexts[0].image_jpeg == b"\xff\xd8\xff\xd9"


def test_split_store_can_drop_scene_pixels_but_keep_embeddings(tmp_path) -> None:
    store = CapsuleStore()
    store.add_scene(
        SceneCapsule(
            timestamp=1.0,
            frame_index=10,
            score=0.9,
            embedding=np.ones(8, dtype=np.float32) / np.sqrt(8),
            image=np.zeros((24, 32, 3), dtype=np.uint8),
        )
    )

    full = store.save(tmp_path / "with_scenes.json", split=True)
    compact = store.save(
        tmp_path / "without_scenes.json",
        split=True,
        include_scene_images=False,
    )
    loaded = CapsuleStore.load(compact)

    assert compact.stat().st_size < full.stat().st_size
    assert loaded.scenes[0].image is None
    assert loaded.scenes[0].embedding is not None

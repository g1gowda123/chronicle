"""Tests for SPORE capsule memory bounds."""

from __future__ import annotations

import base64

import numpy as np

from chronicle.perception.capsules import ActionCapsule, CapsuleStore, ObjectCapsule, SporeView
from chronicle.types import BBox


def _view(timestamp: float, *, hand_overlap: float = 0.0) -> SporeView:
    return SporeView(
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        bbox=BBox(10, 10, 40, 40),
        score=0.5,
        hand_overlap=hand_overlap,
    )


def test_view_image_roundtrips_as_webp() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 200
    view = _view(1.0)
    view.image = image

    payload = view.as_dict()
    raw = base64.b64decode(payload["image"])

    assert raw[:4] == b"RIFF"
    assert raw[8:12] == b"WEBP"
    restored = SporeView.from_dict(payload)
    assert restored.image is not None
    assert restored.image.shape == image.shape


def test_evict_to_keeps_touched_objects_before_recent_background() -> None:
    store = CapsuleStore()
    touched = store.add_capsule(1.0, _view(1.0))
    touched.add_contact(1.0)
    store.add_capsule(9.0, _view(9.0))
    store.add_capsule(8.0, _view(8.0, hand_overlap=0.4))

    removed = store.evict_to(1)

    assert removed == 2
    assert list(store.capsules) == [touched.capsule_id]


def test_evict_to_is_a_noop_below_limit() -> None:
    store = CapsuleStore()
    capsule = store.add_capsule(1.0, _view(1.0))

    assert store.evict_to(5) == 0
    assert list(store.capsules) == [capsule.capsule_id]


def test_consolidate_actions_merges_same_object_contacts() -> None:
    store = CapsuleStore()
    store.add_action(
        ActionCapsule(
            action_id="action-1",
            start=1.0,
            end=2.0,
            hand_id="hand:0",
            capsule_ids=["spore-1"],
        )
    )
    store.add_action(
        ActionCapsule(
            action_id="action-2",
            start=2.3,
            end=3.0,
            hand_id="hand:1",
            capsule_ids=["spore-1"],
        )
    )
    store.add_action(
        ActionCapsule(
            action_id="action-3",
            start=2.4,
            end=3.2,
            hand_id="hand:0",
            capsule_ids=["spore-2"],
        )
    )

    removed = store.consolidate_actions(max_gap=0.4)

    assert removed == 1
    assert set(store.actions) == {"action-1", "action-3"}
    assert store.actions["action-1"].end == 3.0


def test_consolidate_actions_folds_short_still_fragments() -> None:
    store = CapsuleStore()
    store.add_action(
        ActionCapsule(
            action_id="action-1",
            start=1.0,
            end=2.0,
            hand_id="hand:0",
            capsule_ids=["spore-1"],
            displacement=(20.0, 0.0),
            metadata={"path_length": 20.0},
        )
    )
    store.add_action(
        ActionCapsule(
            action_id="action-2",
            start=2.1,
            end=2.5,
            hand_id="hand:0",
            capsule_ids=["spore-1"],
            metadata={"path_length": 0.0},
        )
    )

    removed = store.consolidate_actions(max_gap=0.4)

    assert removed == 1
    assert set(store.actions) == {"action-1"}
    assert store.actions["action-1"].end == 2.5
    assert store.actions["action-1"].metadata["merged_actions"] == 2


def test_consolidate_actions_keeps_long_still_holds() -> None:
    store = CapsuleStore()
    store.add_action(
        ActionCapsule(
            action_id="action-1",
            start=1.0,
            end=2.0,
            hand_id="hand:0",
            capsule_ids=["spore-1"],
            metadata={"path_length": 20.0},
        )
    )
    store.add_action(
        ActionCapsule(
            action_id="action-2",
            start=2.1,
            end=3.4,
            hand_id="hand:0",
            capsule_ids=["spore-2"],
            metadata={"path_length": 0.0},
        )
    )

    removed = store.consolidate_actions(max_gap=0.4)

    assert removed == 0
    assert set(store.actions) == {"action-1", "action-2"}


def test_consolidate_actions_does_not_fold_across_hands() -> None:
    store = CapsuleStore()
    store.add_action(
        ActionCapsule(
            action_id="action-1",
            start=1.0,
            end=2.0,
            hand_id="hand:0",
            capsule_ids=["spore-1"],
            metadata={"path_length": 20.0},
        )
    )
    store.add_action(
        ActionCapsule(
            action_id="action-2",
            start=2.1,
            end=2.5,
            hand_id="hand:1",
            capsule_ids=["spore-2"],
            metadata={"path_length": 0.0},
        )
    )

    removed = store.consolidate_actions(max_gap=0.4)

    assert removed == 0
    assert set(store.actions) == {"action-1", "action-2"}


def test_capsule_trajectory_is_normalised_and_roundtrips() -> None:
    capsule = ObjectCapsule(
        capsule_id="spore-1",
        created_at=0.0,
        last_seen=4.0,
        best_views=[
            SporeView(
                timestamp=0.0,
                frame_index=0,
                bbox=BBox(100, 100, 200, 200),
                score=0.8,
                metadata={"frame_size": [1000, 500]},
            ),
            SporeView(
                timestamp=2.0,
                frame_index=20,
                bbox=BBox(250, 100, 350, 200),
                score=0.7,
                metadata={"frame_size": [1000, 500]},
            ),
            SporeView(
                timestamp=4.0,
                frame_index=40,
                bbox=BBox(400, 150, 500, 250),
                score=0.6,
                metadata={"frame_size": [1000, 500]},
            ),
        ],
    )

    trajectory = capsule.trajectory
    assert trajectory is not None
    assert trajectory["start"] == [0.15, 0.3]
    assert trajectory["end"] == [0.45, 0.4]
    assert trajectory["direction"] == [0.3, 0.1]
    assert trajectory["displacement"] == 0.3162
    assert trajectory["path"] == 0.3303
    assert trajectory["start_time"] == 0.0
    assert trajectory["end_time"] == 4.0
    assert trajectory["samples"] == 3
    assert trajectory["moved"] is True

    payload = capsule.as_dict(include_images=False)
    assert payload["trajectory"] == trajectory
    restored = ObjectCapsule.from_dict(payload)
    assert restored.trajectory == trajectory


def test_capsule_trajectory_needs_two_sized_views() -> None:
    capsule = ObjectCapsule(
        capsule_id="spore-1",
        created_at=0.0,
        last_seen=1.0,
        best_views=[
            SporeView(
                timestamp=0.0,
                frame_index=0,
                bbox=BBox(10, 10, 20, 20),
                score=0.8,
                metadata={"frame_size": [640, 480]},
            )
        ],
    )

    assert capsule.trajectory is None

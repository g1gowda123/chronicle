"""Tests for SPORE capsule memory bounds."""

from __future__ import annotations

from chronicle.perception.capsules import ActionCapsule, CapsuleStore, SporeView
from chronicle.types import BBox


def _view(timestamp: float, *, hand_overlap: float = 0.0) -> SporeView:
    return SporeView(
        timestamp=timestamp,
        frame_index=int(timestamp * 10),
        bbox=BBox(10, 10, 40, 40),
        score=0.5,
        hand_overlap=hand_overlap,
    )


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

"""Spatial qualifiers constrain a where-question to the queried surface."""

from __future__ import annotations

from chronicle.perception.capsules import CapsuleStore
from chronicle.perception.spore import Binding, SporeConfig, SporeQueryEngine
from chronicle.types import BBox


def _cup_on_shelf() -> Binding:
    return Binding(
        term="cup",
        label="cup",
        capsule_id="capsule-1",
        confidence=0.8,
        timestamp=1.0,
        bbox=BBox(0, 0, 10, 10),
        support="shelf",
    )


def test_answer_rejects_a_binding_on_the_wrong_surface(monkeypatch) -> None:
    engine = SporeQueryEngine(CapsuleStore(), SporeConfig(surface_support=False))
    binding = _cup_on_shelf()
    monkeypatch.setattr(
        engine,
        "bind",
        lambda term, *, budget, at_time: [binding],
    )
    monkeypatch.setattr(
        engine,
        "route",
        lambda terms, *, budget, at_time: [],
    )

    wrong = engine.answer("Where is the cup on the table?")
    right = engine.answer("Where is the cup on the shelf?")

    assert wrong.abstained
    assert wrong.answer == "not enough evidence"
    assert not right.abstained
    assert "on the shelf" in right.answer

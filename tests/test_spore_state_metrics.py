"""Scoring helpers for the state/absence evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from chronicle.perception.state_metrics import (
    score_default,
    score_state,
    support_matches,
    support_matches_strict,
)


@dataclass
class _Binding:
    support: str | None = None


@dataclass
class _Answer:
    verdict: str | None = None
    support: str | None = None
    abstained: bool = True
    best: _Binding | None = None

    @property
    def stats(self) -> dict:
        return {"state": {"support": self.support}} if self.verdict else {}


def test_support_alias_versus_strict() -> None:
    assert support_matches("desk", "table")
    assert not support_matches_strict("desk", "table")
    assert support_matches_strict("countertop", "counter top")
    assert support_matches_strict("shelf", "shelf")
    assert not support_matches("shelf", None)
    assert support_matches(None, None)


def test_presence_scoring() -> None:
    question = {"kind": "presence", "expected": "yes", "expected_support": "table"}
    assert score_state(question, _Answer("present", "table")) == (True, "")
    assert score_state(question, _Answer("present", "shelf")) == (
        False,
        "wrong_support",
    )
    assert score_state(question, _Answer("not_observed", None)) == (
        False,
        "missed_presence",
    )

    absent = {"kind": "presence", "expected": "no", "expected_support": None}
    assert score_state(absent, _Answer("not_observed", None)) == (True, "")
    assert score_state(absent, _Answer("present", "table")) == (
        False,
        "false_presence",
    )
    assert score_state(absent, _Answer("unknown", None)) == (
        False,
        "unexpected_verdict",
    )


def test_still_and_moved_scoring() -> None:
    still = {"kind": "still", "expected": "yes", "expected_support": "table"}
    assert score_state(still, _Answer("present", "table")) == (True, "")
    assert score_state(still, _Answer("moved", "shelf")) == (
        False,
        "missed_presence",
    )

    stayed = {"kind": "moved", "expected": "stayed", "expected_support": "table"}
    assert score_state(stayed, _Answer("stayed", "table")) == (True, "")
    assert score_state(stayed, _Answer("moved", "shelf")) == (
        False,
        "false_move",
    )

    unknown = {"kind": "moved", "expected": "unknown", "expected_support": None}
    assert score_state(unknown, _Answer("unknown", None)) == (True, "")
    assert score_state(unknown, _Answer("present", "table")) == (
        False,
        "false_presence",
    )


def test_default_binding_scoring() -> None:
    question = {"kind": "presence", "expected": "yes", "expected_support": "table"}
    assert score_default(question, _Answer(best=_Binding("table"), abstained=False)) == (
        True,
        "",
    )
    assert score_default(question, _Answer(best=None, abstained=True)) == (
        False,
        "missed_presence",
    )

    absent = {"kind": "presence", "expected": "no", "expected_support": None}
    assert score_default(absent, _Answer(best=None, abstained=True)) == (True, "")
    assert score_default(absent, _Answer(best=_Binding("table"), abstained=False)) == (
        False,
        "false_presence",
    )
    assert score_default({"kind": "still", "expected": "yes"}, _Answer()) == (
        None,
        "not_applicable",
    )

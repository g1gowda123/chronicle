"""Metric helpers for grounded state/absence evaluation.

These functions are deliberately free of torch/detector imports so the scoring
rules can be unit-tested in the lightweight project environment and reused by a
host runtime.  They score a ``SporeAnswer``-shaped object (``verdict``,
``stats["state"]``, ``best``, ``abstained``) against a probe row.
"""

from __future__ import annotations

from typing import Any

from chronicle.perception.spore import _normalise_surface, _support_aliases


def support_matches(actual: str | None, expected: str | None) -> bool:
    """Alias-aware support match used by the shipped query routing."""
    if expected is None:
        return actual is None
    if actual is None:
        return False
    aliases = {_normalise_surface(alias) for alias in _support_aliases(expected)}
    return _normalise_surface(actual) in aliases


def support_matches_strict(actual: str | None, expected: str | None) -> bool:
    """Exact support label, without the table/desk/countertop routing aliases.

    A state answer that says "desk" for a dining table is still wrong even
    though the routing layer treats both as the same place bucket.
    """
    if expected is None:
        return actual is None
    if actual is None:
        return False
    return _normalise_surface(actual) == _normalise_surface(expected)


def state_support(answer: Any) -> str | None:
    return (answer.stats.get("state") or {}).get("support")


def score_state(question: dict, answer: Any) -> tuple[bool | None, str]:
    """Score a state-path answer; returns ``(correct, error_tag)``."""
    expected = str(question.get("expected", "")).lower()
    verdict = answer.verdict
    support = state_support(answer)
    expected_support = question.get("expected_support")
    kind = question.get("kind")
    if kind == "presence":
        if expected == "yes":
            if verdict != "present":
                return False, "missed_presence"
            if not support_matches(support, expected_support):
                return False, "wrong_support"
            return True, ""
        if expected == "no":
            if verdict in {"present", "moved", "stayed"}:
                return False, "false_presence"
            if verdict == "not_observed":
                return True, ""
            return False, "unexpected_verdict"
        return None, "unknown_label"
    if kind == "still":
        if expected == "yes":
            if verdict != "present":
                return False, "missed_presence"
            if not support_matches(support, expected_support):
                return False, "wrong_support"
            return True, ""
        if expected == "no":
            if verdict == "present" and support_matches(support, expected_support):
                return True, ""
            if verdict == "not_observed":
                return True, ""
            return False, "wrong_state"
        return None, "unknown_label"
    if kind == "moved":
        if expected == "unknown":
            return (
                (verdict == "unknown"),
                "" if verdict == "unknown" else "false_presence",
            )
        if expected in {"stayed", "no_move"}:
            if verdict == "moved":
                return False, "false_move"
            if verdict == "stayed":
                if not support_matches(support, expected_support):
                    return False, "wrong_support"
                return True, ""
            return False, "unexpected_verdict"
        if expected in {"moved", "yes"}:
            if verdict != "moved":
                return False, "missed_move"
            if expected_support and not support_matches(support, expected_support):
                return False, "wrong_support"
            return True, ""
        return None, "unknown_label"
    return None, "unknown_kind"


def score_default(question: dict, answer: Any) -> tuple[bool | None, str]:
    """Baseline: the ordinary binding path has no state verdict."""
    expected = str(question.get("expected", "")).lower()
    if question.get("kind") != "presence":
        return None, "not_applicable"
    support = answer.best.support if answer.best is not None else None
    if expected == "yes":
        if answer.abstained:
            return False, "missed_presence"
        if not support_matches(support, question.get("expected_support")):
            return False, "wrong_support"
        return True, ""
    if expected == "no":
        return (answer.abstained), "" if answer.abstained else "false_presence"
    return None, "unknown_label"

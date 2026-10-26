"""Confidence calibration for action verification.

The VLM/rule classifier returns a confidence per action, but a single global
cutoff is a poor fit: ``HOLD`` is routinely over-confident while ``OPEN`` is
under-confident.  This module turns a small labelled set of captions into
per-action acceptance thresholds, and scores the calibrated output.

Usage (offline, no models)::

    from chronicle.semantic.calibration import calibrate_rule_classifier

    result = calibrate_rule_classifier()
    print(result.report.to_dict())
    config = SemanticConfig(action_thresholds=result.thresholds)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .actions import (
    DEFAULT_ACTION_THRESHOLDS,
    ActionCalibrator,
    ActionPrimitive,
    RuleBasedActionClassifier,
)


@dataclass(frozen=True)
class CalibrationSample:
    """One prediction with its ground-truth verdict."""

    action: ActionPrimitive
    confidence: float
    correct: bool
    caption: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class ActionMetrics:
    support: int = 0
    correct: int = 0
    accepted: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "support": self.support,
            "correct": self.correct,
            "accepted": self.accepted,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass(frozen=True)
class CalibrationReport:
    samples: int
    accepted: int
    thresholds: tuple[tuple[str, float], ...]
    metrics: tuple[tuple[str, ActionMetrics], ...]

    @property
    def true_positives(self) -> int:
        return sum(item.true_positives for _, item in self.metrics)

    @property
    def false_positives(self) -> int:
        return sum(item.false_positives for _, item in self.metrics)

    @property
    def false_negatives(self) -> int:
        return sum(item.false_negatives for _, item in self.metrics)

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "accepted": self.accepted,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "thresholds": {name: round(value, 4) for name, value in self.thresholds},
            "per_action": {name: item.to_dict() for name, item in self.metrics},
        }


def _threshold_map(
    thresholds: Mapping[str, float] | Iterable[tuple[str, float]] | None,
) -> dict[str, float]:
    if thresholds is None:
        return dict(DEFAULT_ACTION_THRESHOLDS)
    items = thresholds.items() if isinstance(thresholds, Mapping) else thresholds
    return {str(name).upper(): float(value) for name, value in items}


def _accepts(action: ActionPrimitive, confidence: float, thresholds: Mapping[str, float], floor: float) -> bool:
    if action is ActionPrimitive.OTHER:
        return False
    cutoff = max(float(floor), float(thresholds.get(action.value, floor)))
    return float(confidence) >= cutoff


def evaluate_calibration(
    samples: Sequence[CalibrationSample],
    *,
    thresholds: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
    floor: float = 0.45,
) -> CalibrationReport:
    """Score accepted/rejected predictions against labelled samples."""
    threshold_map = _threshold_map(thresholds)
    buckets: dict[str, dict[str, int]] = {}
    accepted_total = 0
    for sample in samples:
        bucket = buckets.setdefault(
            sample.action.value,
            {
                "support": 0,
                "correct": 0,
                "accepted": 0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
        )
        bucket["support"] += 1
        bucket["correct"] += int(sample.correct)
        accepted = _accepts(sample.action, sample.confidence, threshold_map, floor)
        if accepted:
            accepted_total += 1
            bucket["accepted"] += 1
            if sample.correct:
                bucket["true_positives"] += 1
            else:
                bucket["false_positives"] += 1
        elif sample.correct:
            bucket["false_negatives"] += 1

    metrics = tuple(
        (name, ActionMetrics(**values))
        for name, values in sorted(buckets.items())
    )
    return CalibrationReport(
        samples=len(samples),
        accepted=accepted_total,
        thresholds=tuple(sorted(threshold_map.items())),
        metrics=metrics,
    )


def suggest_thresholds(
    samples: Sequence[CalibrationSample],
    *,
    floor: float = 0.45,
    target_precision: float = 0.9,
) -> tuple[tuple[str, float], ...]:
    """Pick a per-action cutoff from labelled confidences.

    For each action, candidate cutoffs are the observed confidences.  We choose
    the lowest cutoff whose accepted subset meets ``target_precision`` (maximise
    recall at that precision).  If no cutoff reaches the target, fall back to
    the cutoff with the best (precision, recall) pair.  Actions without labels
    keep their default threshold, and ``OTHER`` is always rejected.
    """
    grouped: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.action.value, []).append(sample)

    suggested: dict[str, float] = dict(DEFAULT_ACTION_THRESHOLDS)
    for action_name, items in grouped.items():
        if action_name == ActionPrimitive.OTHER.value:
            suggested[action_name] = 1.0
            continue
        candidates = sorted({float(item.confidence) for item in items}, reverse=True)
        candidates.append(1.0)
        best_eligible: tuple[float, float] | None = None  # (recall, threshold)
        fallback: tuple[float, float, float] | None = None
        for cutoff in candidates:
            accepted = [item for item in items if item.confidence >= cutoff]
            if not accepted:
                continue
            tp = sum(1 for item in accepted if item.correct)
            precision = tp / len(accepted)
            recall = tp / max(1, sum(1 for item in items if item.correct))
            if fallback is None or (precision, recall) > (fallback[0], fallback[1]):
                fallback = (precision, recall, cutoff)
            if precision >= target_precision:
                if best_eligible is None or recall > best_eligible[0]:
                    best_eligible = (recall, cutoff)
        chosen = (
            best_eligible[1]
            if best_eligible is not None
            else (fallback[2] if fallback is not None else floor)
        )
        suggested[action_name] = max(float(floor), float(chosen))

    return tuple(sorted(suggested.items()))


# A compact labelled regression set used by the offline demo/tests.  Captions
# are phrased the way Florence-2 tends to describe the three-panel strip.
CALIBRATION_CAPTIONS: tuple[tuple[str, ActionPrimitive], ...] = (
    ("The hand picks up the red mug from the desk.", ActionPrimitive.PICK_UP),
    ("A hand grasps the keys and lifts them.", ActionPrimitive.PICK_UP),
    ("The person places the mug down on the counter.", ActionPrimitive.PUT_DOWN),
    ("The hand sets the cup onto the table.", ActionPrimitive.PUT_DOWN),
    ("The hand puts the keys into the open drawer.", ActionPrimitive.INSERT_INTO),
    ("A hand places the charger inside the bag.", ActionPrimitive.INSERT_INTO),
    ("The hand takes the charger out of the drawer.", ActionPrimitive.REMOVE_FROM),
    ("A hand pulls the cable out from the box.", ActionPrimitive.REMOVE_FROM),
    ("The hand opens the lid of the box.", ActionPrimitive.OPEN),
    ("The person closes the drawer.", ActionPrimitive.CLOSE),
    ("The hand pushes the door closed.", ActionPrimitive.PUSH),
    ("The hand pulls the drawer open.", ActionPrimitive.PULL),
    ("The person hands the mug to another person.", ActionPrimitive.HAND_OFF),
    ("The hand holds the bottle while walking.", ActionPrimitive.HOLD),
    ("A first-person view of a room.", ActionPrimitive.OTHER),
    ("The scene is mostly dark and unclear.", ActionPrimitive.OTHER),
)


@dataclass(frozen=True)
class CalibrationResult:
    samples: tuple[CalibrationSample, ...]
    thresholds: tuple[tuple[str, float], ...]
    report: CalibrationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": {name: round(value, 4) for name, value in self.thresholds},
            "report": self.report.to_dict(),
        }


def calibrate_rule_classifier(
    *,
    classifier: RuleBasedActionClassifier | None = None,
    labels: Sequence[tuple[str, ActionPrimitive]] | None = None,
    floor: float = 0.45,
    target_precision: float = 0.9,
) -> CalibrationResult:
    """Run the deterministic classifier over labelled captions and calibrate."""
    active = classifier or RuleBasedActionClassifier()
    entries = list(labels or CALIBRATION_CAPTIONS)
    samples: list[CalibrationSample] = []
    for caption, expected in entries:
        prediction = active.verification(
            caption, "object", min_confidence=floor
        )
        samples.append(
            CalibrationSample(
                action=prediction.action,
                confidence=prediction.confidence,
                correct=prediction.action is expected,
                caption=caption,
            )
        )
    thresholds = suggest_thresholds(
        samples, floor=floor, target_precision=target_precision
    )
    report = evaluate_calibration(samples, thresholds=thresholds, floor=floor)
    return CalibrationResult(
        samples=tuple(samples), thresholds=thresholds, report=report
    )


def calibrator_from_samples(
    samples: Sequence[CalibrationSample],
    *,
    floor: float = 0.45,
    target_precision: float = 0.9,
    reject_ambiguous: bool = True,
) -> ActionCalibrator:
    thresholds = suggest_thresholds(
        samples, floor=floor, target_precision=target_precision
    )
    return ActionCalibrator(
        thresholds=thresholds,
        floor=floor,
        reject_ambiguous=reject_ambiguous,
    )

"""Score verified actions against a hand-labelled action timeline.

Role C's acceptance bar is: *the verified actions for a clip match the labelled
timeline within tolerance*.  GraphRAG answers are scored by
:mod:`chronicle.semantic.evaluation`; this module scores the other half of the
contract — the action facts Role C writes into the graph.

Inputs are ``ActionEvent`` objects (or plain mappings), so the same code can
score the built-in demo, a JSON label file, or a dashboard snapshot export::

    from chronicle.integration import ChronicleRuntime
    from chronicle.semantic import evaluate_action_timeline, load_action_labels

    runtime = ChronicleRuntime("data/demo/chronicle.db", load=True)
    labels = load_action_labels("runs/hololens_labels.json")
    report = evaluate_action_timeline(runtime.snapshot(), labels, tolerance=2.0)
    print("\\n".join(report.summary_lines()))

CLI::

    python -m chronicle.semantic.timeline \
        --labels docs/action_timeline.example.json \
        --snapshot runs/chronicle_snapshot.json --tolerance 2.0

Exit code is 0 only when every label is matched, there are no extra accepted
predictions, and (where comparable) entity/container names agree.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import ActionPrimitive, coerce_action

DEFAULT_TIMELINE_TOLERANCE = 2.0
_FILLER_WORDS = frozenset({"the", "a", "an", "of", "my", "your", "his", "her", "their"})


def _first(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _tokens(value: Any) -> set[str]:
    words = re.split(r"[^a-z0-9]+", str(value).casefold())
    return {word for word in words if word and word not in _FILLER_WORDS}


def _text_match(
    predicted: str | None,
    expected: str | None,
) -> bool | None:
    """Token match; ``None`` when either side is missing.

    ``"keys"`` matches ``"the keys"`` and ``"drawer"`` matches ``"open
    drawer"``, while ``"cup"`` does not match ``"cupboard"``.
    """
    if not predicted or not expected:
        return None
    left, right = _tokens(predicted), _tokens(expected)
    if not left or not right:
        return None
    if left == right:
        return True
    return left < right or right < left


def _rate(values: Iterable[bool | None]) -> float | None:
    relevant = [value for value in values if value is not None]
    return sum(1 for value in relevant if value) / len(relevant) if relevant else None


@dataclass(frozen=True)
class ActionEvent:
    """One labelled or predicted action at a timestamp."""

    timestamp: float
    action: ActionPrimitive
    entity: str = ""
    container: str | None = None
    confidence: float | None = None
    accepted: bool = True
    caption: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", float(self.timestamp))
        if not isinstance(self.action, ActionPrimitive):
            object.__setattr__(self, "action", coerce_action(self.action))
        object.__setattr__(self, "entity", "" if self.entity is None else str(self.entity))
        if self.container is not None:
            text = str(self.container).strip()
            object.__setattr__(self, "container", text or None)
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))

    @classmethod
    def from_mapping(
        cls, item: Mapping[str, Any], *, index: int | None = None
    ) -> ActionEvent:
        timestamp = _first(item, "timestamp", "t", "t_start", "time")
        action = _first(item, "action", "primitive", "label")
        if timestamp is None or action is None:
            where = f"entry {index}" if index is not None else "entry"
            raise ValueError(f"{where} needs a timestamp and an action")
        evidence = item.get("evidence")
        caption = _first(item, "caption", "evidence_text")
        if caption is None and isinstance(evidence, Mapping):
            caption = evidence.get("caption")
        return cls(
            timestamp=float(timestamp),
            action=coerce_action(action),
            entity=_first(item, "entity", "entity_label", "object", "object_label") or "",
            container=_first(item, "container", "target_container"),
            confidence=_first(item, "confidence", "conf"),
            accepted=bool(item.get("accepted", True)),
            caption=str(caption or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "action": self.action.value,
            "entity": self.entity,
            "container": self.container,
            "confidence": self.confidence,
            "accepted": self.accepted,
        }


def _events(items: Iterable[ActionEvent | Mapping[str, Any]]) -> tuple[ActionEvent, ...]:
    events: list[ActionEvent] = []
    for index, item in enumerate(items):
        if isinstance(item, ActionEvent):
            events.append(item)
        elif isinstance(item, Mapping):
            events.append(ActionEvent.from_mapping(item, index=index))
        else:
            raise ValueError(f"entry {index} must be an ActionEvent or a mapping")
    return tuple(events)


def predictions_from_snapshot(
    snapshot: Mapping[str, Any],
) -> tuple[ActionEvent, ...]:
    """Extract predictions from ``ChronicleRuntime.snapshot()`` or an export."""
    rows = snapshot.get("actions")
    if rows is None:
        rows = snapshot.get("predictions")
    if rows is None:
        raise ValueError("snapshot needs an 'actions' or 'predictions' list")
    return _events(rows)


@dataclass(frozen=True)
class TimelineMatch:
    label: ActionEvent
    prediction: ActionEvent
    timestamp_error: float
    entity_ok: bool | None
    container_ok: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label.to_dict(),
            "prediction": self.prediction.to_dict(),
            "timestamp_error": round(self.timestamp_error, 3),
            "entity_ok": self.entity_ok,
            "container_ok": self.container_ok,
        }


@dataclass(frozen=True)
class TimelineReport:
    labels: int
    predictions: int
    rejected_predictions: int
    matches: tuple[TimelineMatch, ...]
    unmatched_labels: tuple[ActionEvent, ...]
    unmatched_predictions: tuple[ActionEvent, ...]
    tolerance: float

    @property
    def matched(self) -> int:
        return len(self.matches)

    @property
    def true_positives(self) -> int:
        return self.matched

    @property
    def precision(self) -> float:
        denominator = self.matched + len(self.unmatched_predictions)
        return self.matched / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        return self.matched / self.labels if self.labels else 1.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    @property
    def mean_timestamp_error(self) -> float | None:
        if not self.matches:
            return None
        return sum(match.timestamp_error for match in self.matches) / len(self.matches)

    @property
    def max_timestamp_error(self) -> float | None:
        if not self.matches:
            return None
        return max(match.timestamp_error for match in self.matches)

    @property
    def entity_accuracy(self) -> float | None:
        return _rate(match.entity_ok for match in self.matches)

    @property
    def container_accuracy(self) -> float | None:
        return _rate(match.container_ok for match in self.matches)

    @property
    def container_comparable(self) -> int:
        return sum(1 for match in self.matches if match.container_ok is not None)

    @property
    def passed(self) -> bool:
        entity_ok = self.entity_accuracy is None or self.entity_accuracy >= 1.0 - 1e-9
        container_ok = (
            self.container_accuracy is None or self.container_accuracy >= 1.0 - 1e-9
        )
        return (
            self.precision >= 1.0 - 1e-9
            and self.recall >= 1.0 - 1e-9
            and entity_ok
            and container_ok
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "predictions": self.predictions,
            "rejected_predictions": self.rejected_predictions,
            "matched": self.matched,
            "unmatched_labels": self.labels - self.matched,
            "unmatched_predictions": len(self.unmatched_predictions),
            "tolerance": self.tolerance,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_timestamp_error": _round(self.mean_timestamp_error),
            "max_timestamp_error": _round(self.max_timestamp_error),
            "entity_accuracy": _round(self.entity_accuracy),
            "container_accuracy": _round(self.container_accuracy),
            "container_comparable": self.container_comparable,
            "passed": self.passed,
            "matches": [match.to_dict() for match in self.matches],
            "extras": [event.to_dict() for event in self.unmatched_predictions],
            "misses": [event.to_dict() for event in self.unmatched_labels],
        }

    def summary_lines(self) -> list[str]:
        mean_error = self.mean_timestamp_error
        lines = [
            f"Action timeline: {self.matched}/{self.labels} labels matched within "
            f"{self.tolerance:g} s (precision {self.precision:.0%}, recall "
            f"{self.recall:.0%})"
        ]
        if mean_error is not None:
            lines.append(
                f"  mean timestamp error {mean_error:.2f} s "
                f"(max {self.max_timestamp_error:.2f} s)"
            )
        entity, container = self.entity_accuracy, self.container_accuracy
        if entity is not None:
            lines.append(f"  entity accuracy   {entity:.0%}")
        if container is not None:
            lines.append(
                f"  container accuracy {container:.0%} "
                f"({self.container_comparable} comparable)"
            )
        if self.rejected_predictions:
            lines.append(f"  rejected predictions skipped: {self.rejected_predictions}")
        for event in self.unmatched_labels:
            lines.append(
                f"  MISS  {event.action.value} {event.entity!r} at t={event.timestamp:.2f}s"
            )
        for event in self.unmatched_predictions:
            lines.append(
                f"  EXTRA {event.action.value} {event.entity!r} at t={event.timestamp:.2f}s"
            )
        lines.append(f"  result: {'PASS' if self.passed else 'FAIL'}")
        return lines


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def evaluate_action_timeline(
    predictions: Iterable[ActionEvent | Mapping[str, Any]],
    labels: Iterable[ActionEvent | Mapping[str, Any]],
    *,
    tolerance: float = DEFAULT_TIMELINE_TOLERANCE,
    accepted_only: bool = True,
    compare_entities: bool = True,
) -> TimelineReport:
    """Greedy nearest-time match of same-action predictions to labels.

    Matching requires the same action primitive and ``|Δt| <= tolerance``.
    Wrong-action or out-of-window predictions stay in ``unmatched_predictions``
    (they lower precision); unmatched labels lower recall.

    ``compare_entities=False`` skips the entity/container name agreement used by
    :attr:`TimelineReport.passed`.  It is useful when the ground truth uses a
    different vocabulary from the runtime's detector labels (the HoloAssist
    clip labels everything ``joy-con`` while the colour fallback detects
    ``cup``/``mug``).
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    predicted_events = _events(predictions)
    rejected = 0
    if accepted_only:
        kept = tuple(event for event in predicted_events if event.accepted)
        rejected = len(predicted_events) - len(kept)
        predicted_events = kept
    label_events = _events(labels)

    candidates: list[tuple[float, int, int]] = []
    for label_index, label in enumerate(label_events):
        for prediction_index, prediction in enumerate(predicted_events):
            if prediction.action is not label.action:
                continue
            delta = abs(prediction.timestamp - label.timestamp)
            if delta <= tolerance:
                candidates.append((delta, label_index, prediction_index))
    candidates.sort()

    used_labels: set[int] = set()
    used_predictions: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for delta, label_index, prediction_index in candidates:
        if label_index in used_labels or prediction_index in used_predictions:
            continue
        used_labels.add(label_index)
        used_predictions.add(prediction_index)
        pairs.append((label_index, prediction_index, delta))

    pairs.sort(key=lambda item: label_events[item[0]].timestamp)
    matches = tuple(
        TimelineMatch(
            label=label_events[label_index],
            prediction=predicted_events[prediction_index],
            timestamp_error=delta,
            entity_ok=_text_match(
                predicted_events[prediction_index].entity,
                label_events[label_index].entity,
            )
            if compare_entities
            else None,
            container_ok=_text_match(
                predicted_events[prediction_index].container,
                label_events[label_index].container,
            )
            if compare_entities
            else None,
        )
        for label_index, prediction_index, delta in pairs
    )
    return TimelineReport(
        labels=len(label_events),
        predictions=len(predicted_events),
        rejected_predictions=rejected,
        matches=matches,
        unmatched_labels=tuple(
            event
            for index, event in enumerate(label_events)
            if index not in used_labels
        ),
        unmatched_predictions=tuple(
            event
            for index, event in enumerate(predicted_events)
            if index not in used_predictions
        ),
        tolerance=float(tolerance),
    )


def _read_json(path: str | Path) -> Any:
    file = Path(path).expanduser()
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{file}: invalid JSON ({exc})") from exc


def _event_rows(data: Any) -> Sequence[Any]:
    if isinstance(data, Mapping):
        for key in ("labels", "actions", "predictions"):
            if key in data:
                rows = data[key]
                break
        else:
            raise ValueError("expected a list, or a 'labels'/'actions'/'predictions' key")
    else:
        rows = data
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("expected a list of action events")
    return rows


def load_action_labels(
    source: str | Path | Mapping[str, Any] | Sequence[Any],
) -> tuple[ActionEvent, ...]:
    """Load ``[{timestamp, action, entity?, container?}]`` from JSON or memory."""
    data = _read_json(source) if isinstance(source, (str, Path)) else source
    return _events(_event_rows(data))


def _annotation_rows(data: Any) -> Sequence[Any]:
    if isinstance(data, Mapping):
        ground_truth = data.get("ground_truth")
        if isinstance(ground_truth, Mapping) and "actions" in ground_truth:
            rows = ground_truth["actions"]
        elif "actions" in data:
            rows = data["actions"]
        else:
            raise ValueError(
                "annotations need a 'ground_truth.actions' or 'actions' list"
            )
    else:
        rows = data
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("expected a list of annotation actions")
    return rows


def load_annotation_actions(
    source: str | Path | Mapping[str, Any] | Sequence[Any],
) -> tuple[ActionEvent, ...]:
    """Load a HoloAssist annotation file as ground-truth actions.

    HoloAssist rows carry a ``label`` (the entity), a ``time`` and an ordered
    ``actions`` list of candidate primitives, e.g. ``["PICK_UP", "TOUCH"]``.
    The first candidate that maps to a Chronicle primitive wins.  Rows whose
    candidates are all movement-only (``MOVE`` / ``VISIBLE_MOVING``) cannot be
    verified by Role C and are skipped.

    Both ``{"ground_truth": {"actions": [...]}}`` (the bundled
    ``data/real/holoassist_switch_34s.json`` shape) and a bare action list are
    accepted.
    """
    data = _read_json(source) if isinstance(source, (str, Path)) else source
    events: list[ActionEvent] = []
    for index, row in enumerate(_annotation_rows(data)):
        if not isinstance(row, Mapping):
            raise ValueError(f"annotation entry {index} must be a mapping")
        timestamp = _first(row, "time", "timestamp", "t", "start")
        if timestamp is None:
            raise ValueError(f"annotation entry {index} needs a timestamp ('time')")
        candidates = row.get("actions")
        if isinstance(candidates, str):
            candidates = [candidates]
        if candidates is None:
            single = _first(row, "action", "primitive")
            candidates = [single] if single is not None else []
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise ValueError(f"annotation entry {index} needs an 'actions' list")
        action: ActionPrimitive | None = None
        for candidate in candidates:
            try:
                action = coerce_action(candidate)
            except (TypeError, ValueError):
                continue
            break
        if action is None:
            continue
        events.append(
            ActionEvent(
                timestamp=float(timestamp),
                action=action,
                entity=_first(row, "label", "entity", "entity_label", "object") or "",
                caption=str(_first(row, "source", "caption", "event") or ""),
            )
        )
    return tuple(events)


def load_action_predictions(
    source: str | Path | Mapping[str, Any] | Sequence[Any],
) -> tuple[ActionEvent, ...]:
    """Load predictions from a list, a snapshot export, or ``{"predictions"}``."""
    data = _read_json(source) if isinstance(source, (str, Path)) else source
    if isinstance(data, Mapping) and "actions" in data and "labels" not in data:
        return predictions_from_snapshot(data)
    if isinstance(data, Mapping) and "predictions" in data:
        return _events(_event_rows(data))
    return _events(_event_rows(data))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: compare a labels file with a predictions file or snapshot export."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Match verified actions against a labelled action timeline"
    )
    truth = parser.add_mutually_exclusive_group(required=True)
    truth.add_argument("--labels", help="ground-truth JSON file")
    truth.add_argument(
        "--annotations",
        help="HoloAssist annotation JSON; uses ground_truth.actions",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", help="prediction JSON file")
    source.add_argument(
        "--snapshot", help="ChronicleRuntime.snapshot() JSON export"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TIMELINE_TOLERANCE,
        help="maximum |Δt| in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="score predictions that Role C marked accepted=false",
    )
    parser.add_argument(
        "--ignore-entity",
        action="store_true",
        help="ignore entity/container name agreement when scoring and deciding PASS",
    )
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args(argv)

    labels = (
        load_annotation_actions(args.annotations)
        if args.annotations
        else load_action_labels(args.labels)
    )
    predictions = (
        predictions_from_snapshot(_read_json(args.snapshot))
        if args.snapshot
        else load_action_predictions(args.predictions)
    )
    report = evaluate_action_timeline(
        predictions,
        labels,
        tolerance=args.tolerance,
        accepted_only=not args.include_rejected,
        compare_entities=not args.ignore_entity,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for line in report.summary_lines():
            print(line)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

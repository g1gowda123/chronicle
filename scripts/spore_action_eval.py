"""Evaluate SPORE action capsules against a real interaction timeline.

The HoloAssist switch clip ships a timeline of annotated interaction
intervals.  SPORE never sees those labels during ingest; it only records
hand-object contact geometry.  This script measures how much of the annotated
timeline those unlabelled intervals recover, without tuning to the timeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from chronicle.perception.capsules import CapsuleStore  # noqa: E402


def _temporal_iou(a: float, b: float, c: float, d: float) -> float:
    intersection = max(0.0, min(b, d) - max(a, c))
    union = max(1e-6, max(b, d) - min(a, c))
    return intersection / union


def evaluate(store_path: str, annotation_path: str) -> dict[str, Any]:
    store = CapsuleStore.load(store_path)
    annotation = json.loads(Path(annotation_path).read_text())
    timeline = annotation.get("timeline") or []
    actions = list(store.actions.values())
    thresholds = (0.1, 0.2, 0.3, 0.5)
    best_iou: list[float] = []
    for event in timeline:
        best_iou.append(
            max(
                (
                    _temporal_iou(event["t"], event["end"], action.start, action.end)
                    for action in actions
                ),
                default=0.0,
            )
        )
    recall: dict[str, float] = {}
    precision: dict[str, float] = {}
    for threshold in thresholds:
        recovered = sum(value >= threshold for value in best_iou)
        matched = sum(
            1
            for action in actions
            if any(
                _temporal_iou(event["t"], event["end"], action.start, action.end)
                >= threshold
                for event in timeline
            )
        )
        recall[str(threshold)] = recovered / max(1, len(timeline))
        precision[str(threshold)] = matched / max(1, len(actions))

    covered = 0.0
    for event in timeline:
        start, end = float(event["t"]), float(event["end"])
        if end <= start:
            continue
        spans = sorted(
            (max(start, action.start), min(end, action.end))
            for action in actions
            if action.end > start and action.start < end
        )
        merged = 0.0
        cursor = start
        for span_start, span_end in spans:
            if span_end <= cursor:
                continue
            merged += span_end - max(cursor, span_start)
            cursor = max(cursor, span_end)
        covered += min(1.0, merged / (end - start))

    return {
        "store": store_path,
        "annotation": annotation_path,
        "events": len(timeline),
        "actions": len(actions),
        "mean_best_iou": round(sum(best_iou) / max(1, len(best_iou)), 4),
        "timeline_coverage": round(covered / max(1, len(timeline)), 4),
        "recall": recall,
        "precision": precision,
        "per_event": [
            {
                "event": event.get("event", ""),
                "start": event["t"],
                "end": event["end"],
                "best_iou": round(value, 4),
            }
            for event, value in zip(timeline, best_iou, strict=False)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = evaluate(args.store, args.annotation)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

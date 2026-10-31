#!/usr/bin/env python3
"""Evaluate SPORE object location against Aria scene annotations.

The multi-domain result in ``spore_domains.py`` uses dense YOLO-World tracks as
both the question source and the reference boxes.  That measures retention and
re-binding, but it is circular: a label the dense detector missed never becomes
a question.

Aria Gen 2 Pilot also ships per-frame 2D scene boxes with object categories,
generated independently of this project.  This script:

* loads those boxes for a clip's main RGB stream;
* asks about the largest static object categories at two viewpoints;
* queries SPORE at the same clip-relative timestamps;
* scores semantic agreement and box IoU against the dataset annotation.

It is still development validation, not a human benchmark.  The scene annotation
is machine-generated, and the four clips are short windows from one capture
rig.  The point is to remove YOLO from the question/reference loop and to expose
failures on categories the dense vocabulary did not cover.

WARNING: the released Aria 2D boxes are projections of 3D scene cuboids.  They
are coarse, can cover occluded space, and their stream/camera alignment with
the preview has not been verified.  On the current clips they produce low IoU
even for visually correct answers.  Treat this script as a diagnostic for
alignment work, not as a benchmark or a basis for an accuracy claim.

Example::

    python scripts/aria_scene_eval.py --sequence eat_0 --dry-run
    python scripts/aria_scene_eval.py --allow-missing
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.perception.detector import build_detector  # noqa: E402
from chronicle.perception.embedder import build_embedder  # noqa: E402
from chronicle.perception.hands import build_hands  # noqa: E402
from chronicle.perception.spore import (  # noqa: E402
    SporeConfig,
    SporePerceiver,
    SporeQueryEngine,
)
from chronicle.types import BBox  # noqa: E402

DEFAULT_MANIFEST = ROOT / "data/aria_gen2pilot.json"
DEFAULT_ANNOTATION_ROOT = Path.home() / ".cache/chronicle/aria_gen2pilot_annotations"
FULL_WIDTH = 2560.0
FULL_HEIGHT = 1920.0
TARGET_WIDTH = 640.0
TARGET_HEIGHT = 480.0
SKIP_CATEGORIES = frozenset({"", "unknown", "other", "background"})


@dataclass(frozen=True, slots=True)
class SceneBox:
    category: str
    timestamp: float
    bbox: BBox
    object_uid: str


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("sequences"), list) or not payload["sequences"]:
        raise ValueError(f"{path}: expected a non-empty 'sequences' list")
    return payload


def select(manifest: dict, names: list[str] | None) -> list[dict]:
    if not names:
        return list(manifest["sequences"])
    wanted = set(names)
    chosen = [item for item in manifest["sequences"] if item["name"] in wanted]
    missing = wanted - {item["name"] for item in chosen}
    if missing:
        raise KeyError(f"unknown sequence(s): {', '.join(sorted(missing))}")
    return chosen


def _ratio(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return -1.0


def load_scene_boxes(annotation_root: Path, item: dict) -> tuple[list[SceneBox], dict]:
    """Load main-RGB scene boxes and convert them to the 640x480 query space.

    Aria scene timestamps are absolute sequence seconds.  The preview videos
    start at (or within about a second of) the first annotated frame, so the
    first timestamp is used as the clip-relative origin.  Only static scene
    objects are retained; the annotation does not cover the objects being
    manipulated.
    """
    sequence = str(item["sequence"])
    scene_dir = annotation_root / sequence / "scene" / "scene"
    instances_path = scene_dir / "instances.json"
    boxes_path = scene_dir / "2d_bounding_box.csv"
    if not instances_path.exists() or not boxes_path.exists():
        raise FileNotFoundError(
            f"missing Aria scene annotations under {scene_dir}; "
            "fetch the scene zip from the Dataset Explorer"
        )
    instances = json.loads(instances_path.read_text())
    categories = {
        str(uid).strip(): str(payload.get("category", "")).strip()
        for uid, payload in instances.items()
    }
    rows: list[dict] = []
    with boxes_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("stream_id", "").strip() != str(item.get("main_stream", "214-1")):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"{boxes_path}: no rows for stream {item.get('main_stream')}")

    timestamps = sorted({_ratio(row["timestamp[ns]"]) / 1e9 for row in rows})
    origin = timestamps[0]
    duration = float(item["duration_s"])
    scale_x = TARGET_WIDTH / FULL_WIDTH
    scale_y = TARGET_HEIGHT / FULL_HEIGHT
    boxes: list[SceneBox] = []
    for row in rows:
        uid = row["object_uid"].strip()
        category = categories.get(uid, "")
        if category.lower() in SKIP_CATEGORIES or category.lower().startswith("other_"):
            continue
        timestamp = _ratio(row["timestamp[ns]"]) / 1e9 - origin
        if timestamp < 2.0 or timestamp > duration - 2.0:
            continue
        try:
            bbox = BBox(
                _ratio(row["x_min[pixel]"]) * scale_x,
                _ratio(row["y_min[pixel]"]) * scale_y,
                _ratio(row["x_max[pixel]"]) * scale_x,
                _ratio(row["y_max[pixel]"]) * scale_y,
            )
        except (KeyError, ValueError):
            continue
        if bbox.area <= 1.0:
            continue
        boxes.append(SceneBox(category=category, timestamp=timestamp, bbox=bbox, object_uid=uid))
    meta = {
        "sequence": sequence,
        "stream_id": str(item.get("main_stream", "214-1")),
        "origin_s": round(origin, 3),
        "annotation_span_s": round(timestamps[-1] - timestamps[0], 3),
        "clip_duration_s": duration,
        "boxes": len(boxes),
    }
    return boxes, meta


def boxes_near(
    boxes: list[SceneBox],
    category: str,
    timestamp: float,
    *,
    tolerance: float = 1.0,
    cluster: float = 0.16,
) -> list[BBox]:
    """Return every annotated instance of ``category`` near ``timestamp``.

    Aria repeats each projected 3D box at ~10 Hz.  A query such as "Where is
    the table?" is ambiguous when several tables are visible, so scoring uses
    the closest annotation group and accepts a match to any instance in it.
    Returns an empty list when the nearest annotation is farther than
    ``tolerance`` seconds.
    """
    candidates = [box for box in boxes if box.category == category]
    if not candidates:
        return []
    nearest = min(abs(box.timestamp - timestamp) for box in candidates)
    if nearest > tolerance:
        return []
    anchor = min(candidates, key=lambda box: abs(box.timestamp - timestamp)).timestamp
    chosen: dict[str, tuple[float, BBox]] = {}
    for box in candidates:
        gap = abs(box.timestamp - anchor)
        if gap > cluster:
            continue
        previous = chosen.get(box.object_uid)
        if previous is None or gap < previous[0]:
            chosen[box.object_uid] = (gap, box.bbox)
    return [bbox for _, bbox in chosen.values()]


def select_questions(
    boxes: list[SceneBox],
    *,
    duration: float,
    max_categories: int,
    max_per_category: int,
    min_area: float,
) -> list[dict]:
    """Pick large, visible categories at two viewpoints per clip."""
    by_category: dict[str, list[SceneBox]] = defaultdict(list)
    for box in boxes:
        if box.bbox.area >= min_area:
            by_category[box.category].append(box)
    ranked = sorted(
        by_category.items(),
        key=lambda pair: (-statistics.median(box.bbox.area for box in pair[1]), pair[0]),
    )[:max_categories]
    questions: list[dict] = []
    fractions = (0.25, 0.65)
    for category, category_boxes in ranked:
        for fraction in fractions[:max_per_category]:
            target_time = duration * fraction
            chosen = min(category_boxes, key=lambda box: abs(box.timestamp - target_time))
            if abs(chosen.timestamp - target_time) > 2.0:
                continue
            nearby_by_uid: dict[str, tuple[float, SceneBox]] = {}
            for box in category_boxes:
                gap = abs(box.timestamp - chosen.timestamp)
                if gap > 0.16:
                    continue
                previous = nearby_by_uid.get(box.object_uid)
                if previous is None or gap < previous[0]:
                    nearby_by_uid[box.object_uid] = (gap, box)
            nearby = [box for _, box in nearby_by_uid.values()]
            if not nearby:
                continue
            # Prefer the selected object, then any other instance at the same
            # instant.  Aria repeats each box at ~10 Hz, so dedupe by object_uid
            # before deciding whether the prompt is ambiguous.
            targets = [chosen.bbox] + [box.bbox for box in nearby if box is not chosen]
            questions.append(
                {
                    "query": (
                        f"Where is a {category}?"
                        if len(nearby) >= 2
                        else f"Where is the {category}?"
                    ),
                    "category": category,
                    "timestamp": round(float(chosen.timestamp), 3),
                    "target_bbox": [round(float(value), 2) for value in chosen.bbox.xyxy],
                    "target_boxes": [
                        [round(float(value), 2) for value in target.xyxy] for target in targets
                    ],
                    "simultaneous_instances": len(nearby),
                    "source": "aria_scene_annotation",
                }
            )
    questions.sort(key=lambda question: (question["timestamp"], question["category"]))
    return questions


def _bbox(values: list[float]) -> BBox:
    return BBox.from_sequence(values)


def evaluate_clip(item: dict, args) -> dict:
    boxes, meta = load_scene_boxes(args.annotation_root, item)
    questions = select_questions(
        boxes,
        duration=float(item["duration_s"]),
        max_categories=args.max_categories,
        max_per_category=args.max_per_category,
        min_area=args.min_area,
    )
    if not questions:
        raise ValueError(f"{item['name']}: no scene questions selected")
    if args.dry_run:
        return {
            "name": item["name"],
            "domain": item["domain"],
            "meta": meta,
            "questions": questions,
        }

    perception = PerceptionConfig(model_dir=str(ROOT / "models"))
    config = SporeConfig(
        sample_fps=args.sample_fps,
        max_width=640,
        binding_mode=args.mode,
        context_binding=not args.no_context,
        query_capsules=args.budget,
        context_fusion_box=args.context_fusion_box,
    )
    categories = sorted({question["category"] for question in questions})
    detector = build_detector(perception, source_name="tabletop")
    detector.set_vocabulary(categories)
    embedder = build_embedder(perception, source_name="tabletop")
    hands = build_hands(perception, source_name="tabletop")
    perceiver = SporePerceiver(config, perception, embedder=embedder, hands=hands)
    started = time.perf_counter()
    store = perceiver.ingest(str((ROOT / item["clip"]).resolve()), source_name="tabletop")
    ingest_seconds = time.perf_counter() - started
    engine = SporeQueryEngine(
        store,
        config,
        perception,
        embedder=embedder,
        detector=detector,
    )

    rows = []
    for question in questions:
        target = _bbox(question["target_bbox"])
        before = engine.detector_calls
        answer = engine.answer(question["query"], budget=args.budget, at_time=question["timestamp"])
        calls = engine.detector_calls - before
        text = (answer.answer or "").lower()
        semantic = question["category"].lower() in text
        predicted = answer.best.bbox if answer.best is not None else None
        predicted_time = (
            float(answer.best.timestamp) if answer.best is not None else question["timestamp"]
        )
        # Score at the timestamp the system actually answered, and accept any
        # visible instance of the category.  The question is category-level
        # ("Where is the table?") and multiple tables/sofas are often visible.
        answer_targets = boxes_near(
            boxes,
            question["category"],
            predicted_time,
            tolerance=args.target_tolerance,
        )
        iou = predicted.iou(target) if predicted is not None else 0.0
        iou_answer = (
            max((predicted.iou(box) for box in answer_targets), default=0.0)
            if predicted is not None
            else 0.0
        )
        center_hit = False
        if predicted is not None:
            center = predicted.centroid
            center_hit = any(
                box.x1 <= center[0] <= box.x2 and box.y1 <= center[1] <= box.y2
                for box in answer_targets
            )
        rows.append(
            {
                **question,
                "answer": answer.answer,
                "abstained": answer.abstained,
                "semantic": semantic,
                "iou": round(float(iou), 4),
                "iou_any_instance": round(float(iou_answer), 4),
                "iou_at_answer_time": round(float(iou_answer), 4),
                "target_boxes_at_answer": [
                    [round(float(value), 2) for value in box.xyxy] for box in answer_targets
                ],
                "predicted_time": round(predicted_time, 3),
                "time_delta": round(predicted_time - question["timestamp"], 3),
                "center_hit": center_hit,
                "calls": calls,
                "predicted_bbox": (
                    [round(float(value), 2) for value in predicted.xyxy]
                    if predicted is not None
                    else None
                ),
            }
        )
        print(
            f"{question['query']:<30} t={question['timestamp']:5.2f} "
            f"sem={int(semantic)} iou={iou:.2f} any={iou_answer:.2f} "
            f"center={int(center_hit)} calls={calls} -> {answer.answer}",
            flush=True,
        )

    values = [row["iou_any_instance"] for row in rows]
    summary = {
        "name": item["name"],
        "domain": item["domain"],
        "clip": str((ROOT / item["clip"]).resolve()),
        "source": "aria_scene_annotation",
        "meta": meta,
        "questions": len(rows),
        "semantic_accuracy": round(
            sum(1 for row in rows if row["semantic"]) / max(1, len(rows)), 3
        ),
        "localization_30": round(
            sum(1 for value in values if value >= 0.3) / max(1, len(values)), 3
        ),
        "mean_iou": round(sum(values) / max(1, len(values)), 3),
        "center_accuracy": round(
            sum(1 for row in rows if row["center_hit"]) / max(1, len(rows)), 3
        ),
        "abstention": round(sum(1 for row in rows if row["abstained"]) / max(1, len(rows)), 3),
        "mean_calls": round(sum(row["calls"] for row in rows) / max(1, len(rows)), 2),
        "ingest_seconds": round(ingest_seconds, 2),
        "capsules": len(store.capsules),
        "store_bytes": store.memory_bytes(),
        "questions_detail": rows,
    }
    return summary


def aggregate(summaries: list[dict]) -> dict:
    rows = []
    for summary in summaries:
        rows.append(
            {
                key: summary[key]
                for key in (
                    "name",
                    "domain",
                    "questions",
                    "semantic_accuracy",
                    "localization_30",
                    "mean_iou",
                    "center_accuracy",
                    "abstention",
                    "mean_calls",
                )
            }
        )
    n = max(1, sum(row["questions"] for row in rows))
    return {
        "source": "aria_scene_annotation",
        "clips": len(rows),
        "questions": n,
        "semantic_accuracy": round(
            sum(row["semantic_accuracy"] * row["questions"] for row in rows) / n, 3
        ),
        "localization_30": round(
            sum(row["localization_30"] * row["questions"] for row in rows) / n, 3
        ),
        "mean_iou": round(sum(row["mean_iou"] * row["questions"] for row in rows) / n, 3),
        "center_accuracy": round(
            sum(row["center_accuracy"] * row["questions"] for row in rows) / n, 3
        ),
        "abstention": round(sum(row["abstention"] * row["questions"] for row in rows) / n, 3),
        "mean_calls": round(sum(row["mean_calls"] * row["questions"] for row in rows) / n, 2),
        "per_clip": rows,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# SPORE independent scene-annotation evaluation",
        "",
        "Reference boxes come from Aria scene annotations, not YOLO-World.",
        "Machine-generated annotations; development validation only.",
        "",
        "| clip | domain | questions | semantic | loc@.30 | mean IoU | center | calls/query |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_clip"]:
        lines.append(
            f"| {row['name']} | {row['domain']} | {row['questions']} | "
            f"{row['semantic_accuracy']:.3f} | {row['localization_30']:.3f} | "
            f"{row['mean_iou']:.3f} | {row['center_accuracy']:.3f} | {row['mean_calls']:.2f} |"
        )
    lines.append(
        f"| **all** | | {payload['questions']} | {payload['semantic_accuracy']:.3f} | "
        f"{payload['localization_30']:.3f} | {payload['mean_iou']:.3f} | "
        f"{payload['center_accuracy']:.3f} | {payload['mean_calls']:.2f} |"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/aria_scene_eval")
    parser.add_argument("--mode", choices=("clip", "detector", "hybrid"), default="hybrid")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--max-categories", type=int, default=6)
    parser.add_argument("--max-per-category", type=int, default=2)
    parser.add_argument("--min-area", type=float, default=1200.0)
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=1.0,
        help="seconds allowed between the answer time and the nearest annotation",
    )
    parser.add_argument(
        "--context-fusion-box", default="adaptive", choices=("mean", "representative", "adaptive")
    )
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    chosen = select(manifest, args.sequence)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for item in chosen:
        try:
            summary = evaluate_clip(item, args)
        except (FileNotFoundError, ValueError) as exc:
            if args.allow_missing:
                print(f"[skip] {item['name']}: {exc}", flush=True)
                continue
            parser.error(str(exc))
        if args.dry_run:
            print(f"\n[{item['name']}] {summary['meta']}")
            for question in summary["questions"]:
                print(
                    f"  {question['query']:<30} t={question['timestamp']:5.2f} "
                    f"n={question['simultaneous_instances']}"
                )
            continue
        summaries.append(summary)
        (args.out_dir / f"{item['name']}.json").write_text(json.dumps(summary, indent=2))
        print(
            f"\n[{item['name']}] {json.dumps({k: v for k, v in summary.items() if k != 'questions_detail'}, indent=2)}"
        )

    if args.dry_run or not summaries:
        return
    payload = aggregate(summaries)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    markdown = render_markdown(payload)
    (args.out_dir / "summary.md").write_text(markdown)
    print(markdown)
    print(f"[report] {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

"""Self-supervised SPORE evaluation from dense pseudo-ground-truth.

The manually written QA sets are small and easy to overfit.  This script
builds questions automatically from dense YOLO-World detections: it tracks
detections through the clip and asks only about objects that the dense
detector saw repeatedly.  No human picks the question, timestamp, or label,
so the resulting score is a cleaner measure of whether SPORE's memory and
late binding generalise.

The questions are deliberately boring (``Where is the <label>?``).  Object
location is the one skill we can score without inventing action semantics,
and it exercises routing, context retention, detector binding, and
abstention together.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chronicle.perception.capsules import CapsuleStore  # noqa: E402
from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.perception.detector import build_detector  # noqa: E402
from chronicle.perception.embedder import build_embedder  # noqa: E402
from chronicle.perception.hands import build_hands  # noqa: E402
from chronicle.perception.sources import VideoSource  # noqa: E402
from chronicle.perception.spore import (  # noqa: E402
    SporeConfig,
    SporePerceiver,
    SporeQueryEngine,
)
from chronicle.types import BBox  # noqa: E402

DEFAULT_VOCABULARY = (
    "knife",
    "cup",
    "plate",
    "bowl",
    "toaster",
    "rice cooker",
    "cutting board",
    "bottle",
    "book",
    "chair",
    "shoe",
    "table",
    "joy-con",
    "switch console",
    "game card",
    "wrist strap",
    "comfort grip",
    "mug",
    "spatula",
    "tray",
    "box",
    "shelf",
)


def _box_list(box: BBox) -> list[float]:
    return [round(float(value), 2) for value in (box.x1, box.y1, box.x2, box.y2)]


def _iou(a: BBox, b: BBox) -> float:
    return a.iou(b)


def _resize(frame, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / float(width)
    resized = cv2.resize(
        frame,
        (max_width, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def build_tracks(dense, *, min_frames: int, min_confidence: float):
    """Greedy IoU tracker over dense detections, one track list per label."""
    tracks: list[dict] = []
    active: dict[str, list[dict]] = {}
    for timestamp, detections in dense:
        for detection in detections:
            if detection.confidence < min_confidence:
                continue
            label = detection.label.strip().lower()
            if not label:
                continue
            candidates = active.get(label, [])
            best = None
            best_iou = 0.25
            for track in candidates:
                if timestamp - track["last_t"] > 1.2:
                    continue
                score = _iou(track["last_box"], detection.bbox)
                if score >= best_iou:
                    best = track
                    best_iou = score
            if best is None:
                best = {
                    "label": label,
                    "frames": [],
                    "last_box": detection.bbox,
                    "last_t": timestamp,
                }
                tracks.append(best)
                active.setdefault(label, []).append(best)
            best["frames"].append(
                {
                    "timestamp": float(timestamp),
                    "confidence": float(detection.confidence),
                    "bbox": detection.bbox,
                }
            )
            best["last_box"] = detection.bbox
            best["last_t"] = timestamp
        for label in list(active):
            active[label] = [t for t in active[label] if timestamp - t["last_t"] <= 1.2]
    kept = []
    for track in tracks:
        if len(track["frames"]) < min_frames:
            continue
        mean_conf = sum(f["confidence"] for f in track["frames"]) / len(track["frames"])
        if mean_conf < min_confidence:
            continue
        kept.append(track)
    kept.sort(key=lambda t: (-len(t["frames"]), t["label"], t["frames"][0]["timestamp"]))
    return kept


def make_questions(tracks, *, limit: int) -> list[dict]:
    questions = []
    used_labels: set[str] = set()
    # One question per label first, then additional instances if budget remains.
    ordered = sorted(tracks, key=lambda t: (-len(t["frames"]), t["label"]))
    for track in ordered:
        label = track["label"]
        if label in used_labels:
            continue
        used_labels.add(label)
        questions.append(_question_for(track))
        if len(questions) >= limit:
            return questions
    for track in ordered:
        if len(questions) >= limit:
            break
        if track["label"] in {q["label"] for q in questions}:
            # Only add a second instance when it is spatially distinct; the
            # greedy tracker already separates instances, so this is safe.
            pass
        question = _question_for(track)
        if not any(
            abs(question["timestamp"] - q["timestamp"]) < 0.5
            and question["label"] == q["label"]
            for q in questions
        ):
            questions.append(question)
    return questions[:limit]


def _question_for(track) -> dict:
    frames = track["frames"]
    # Middle frame is less likely to be a track edge (partial object, blur).
    frame = frames[len(frames) // 2]
    return {
        "query": f"Where is the {track['label']}?",
        "label": track["label"],
        "timestamp": float(frame["timestamp"]),
        "bbox": _box_list(frame["bbox"]),
        "track_frames": len(frames),
        "source": "dense_track",
    }


def _nearest_target(dense, timestamp: float, label: str):
    best = None
    for stamp, detections in dense:
        gap = abs(stamp - timestamp)
        if best is not None and gap >= best[0]:
            continue
        for detection in detections:
            if detection.label.strip().lower() != label:
                continue
            if best is None or gap < best[0]:
                best = (gap, detection.bbox)
    return best[1] if best is not None and best[0] <= 0.5 else None


def _covered(view_box: BBox, target: BBox) -> bool:
    return view_box.intersection_area(target) / max(1.0, target.area) >= 0.50 or view_box.iou(target) >= 0.30


def _same_label_boxes(dense, timestamp: float, label: str, window: float = 0.35) -> list[BBox]:
    boxes: list[BBox] = []
    for stamp, detections in dense:
        if abs(stamp - timestamp) > window:
            continue
        for detection in detections:
            if detection.label.strip().lower() != label:
                continue
            # YOLO-World can emit several overlapping boxes for one object.
            # Ambiguity should mean distinct instances, not duplicate prompts.
            if any(detection.bbox.iou(box) >= 0.50 for box in boxes):
                continue
            boxes.append(detection.bbox)
    return boxes


def evaluate(args) -> dict:
    perception = PerceptionConfig(model_dir=str(ROOT / "models"))
    config = SporeConfig(
        sample_fps=args.sample_fps,
        max_width=640,
        binding_mode=args.mode,
        context_binding=not args.no_context,
        query_capsules=args.budget,
    )
    vocabulary = [term.strip() for term in args.vocabulary.split(",") if term.strip()]
    detector = build_detector(perception, source_name="tabletop")
    detector.set_vocabulary(vocabulary)

    dense = []
    started = time.perf_counter()
    interval = 1.0 / args.sample_fps
    last_time = -1e9
    for frame, timestamp, _ in VideoSource(args.clip):
        if timestamp - last_time < interval - 1e-6:
            continue
        last_time = timestamp
        small, _ = _resize(frame, 640)
        dense.append(
            (
                float(timestamp),
                detector.detect(small, frame_index=len(dense), timestamp=timestamp),
            )
        )
    dense_seconds = time.perf_counter() - started
    print(f"[dense] frames={len(dense)} calls={len(dense)} seconds={dense_seconds:.2f}", flush=True)

    tracks = build_tracks(
        dense,
        min_frames=args.min_track_frames,
        min_confidence=args.min_confidence,
    )
    questions = make_questions(tracks, limit=args.max_questions)
    print(f"[questions] {len(questions)} from {len(tracks)} tracks", flush=True)
    for question in questions:
        print(
            f"  {question['label']:<18} t={question['timestamp']:.2f} "
            f"frames={question['track_frames']}",
            flush=True,
        )

    embedder = build_embedder(perception, source_name="tabletop")
    hands = build_hands(perception, source_name="tabletop")
    if args.store:
        store = CapsuleStore.load(args.store)
        ingest_seconds = 0.0
        print(f"[store] {len(store.capsules)} capsules {store.memory_bytes()} bytes", flush=True)
    else:
        perceiver = SporePerceiver(config, perception, embedder=embedder, hands=hands)
        start = time.perf_counter()
        store = perceiver.ingest(args.clip, source_name="tabletop")
        ingest_seconds = time.perf_counter() - start
        print(f"[ingest] {ingest_seconds:.2f}s", flush=True)

    engine = SporeQueryEngine(
        store,
        config,
        perception,
        embedder=embedder if args.mode == "clip" else None,
        detector=detector if args.mode in {"detector", "hybrid"} else None,
    )

    rows = []
    for question in questions:
        target = BBox.from_sequence(question["bbox"])
        simultaneous = _same_label_boxes(dense, question["timestamp"], question["label"])
        ambiguous = len(simultaneous) >= 2
        # With several identical objects in view, "the X" has no unique
        # referent.  Ask for "a X" and score any instance; otherwise score the
        # specific track the question was generated from.
        query = f"Where is a {question['label']}?" if ambiguous else question["query"]
        calls_before = engine.detector_calls
        answer = engine.answer(query, budget=args.budget, at_time=question["timestamp"])
        calls = engine.detector_calls - calls_before
        text = (answer.answer or "").lower()
        semantic = question["label"] in text
        iou = 0.0
        iou_any = 0.0
        if answer.best is not None:
            iou = answer.best.bbox.iou(target)
            same = _same_label_boxes(dense, answer.best.timestamp, question["label"])
            iou_any = max((answer.best.bbox.iou(box) for box in same), default=0.0)
        rows.append(
            {
                "query": question["query"],
                "label": question["label"],
                "timestamp": question["timestamp"],
                "target_bbox": _box_list(target),
                "simultaneous_instances": len(simultaneous),
                "ambiguous": len(simultaneous) >= 2,
                "answer": answer.answer,
                "abstained": answer.abstained,
                "semantic": semantic,
                "iou": round(float(iou), 4),
                "iou_any_instance": round(float(iou_any), 4),
                "calls": calls,
            }
        )
        print(
            f"{question['query']:<34} t={question['timestamp']:5.2f} "
            f"n={len(simultaneous)} sem={int(semantic)} iou={iou:.2f} "
            f"any={iou_any:.2f} calls={calls} -> {answer.answer}",
            flush=True,
        )

    scored = [row for row in rows if row["target_bbox"]]
    # Ambiguous prompts ask for "a X", so any instance is a correct answer;
    # unique prompts still score the exact track the question came from.
    primary = [
        row["iou_any_instance"] if row["ambiguous"] else row["iou"]
        for row in scored
    ]
    summary = {
        "clip": args.clip,
        "mode": args.mode,
        "context_binding": not args.no_context,
        "questions": len(rows),
        "scored": len(scored),
        "ambiguous_questions": sum(1 for row in rows if row["ambiguous"]),
        "semantic_accuracy": round(sum(1 for row in rows if row["semantic"]) / max(1, len(rows)), 3),
        "localization_30": round(sum(1 for value in primary if value >= 0.3) / max(1, len(primary)), 3),
        "localization_50": round(sum(1 for value in primary if value >= 0.5) / max(1, len(primary)), 3),
        "mean_iou": round(sum(primary) / max(1, len(primary)), 3),
        "localization_any_30": round(
            sum(1 for row in scored if row["iou_any_instance"] >= 0.3) / max(1, len(scored)), 3
        ),
        "localization_any_50": round(
            sum(1 for row in scored if row["iou_any_instance"] >= 0.5) / max(1, len(scored)), 3
        ),
        "mean_iou_any": round(sum(row["iou_any_instance"] for row in scored) / max(1, len(scored)), 3),
        "abstention": round(sum(1 for row in rows if row["abstained"]) / max(1, len(rows)), 3),
        "mean_calls": round(sum(row["calls"] for row in rows) / max(1, len(rows)), 2),
        "dense_calls": len(dense),
        "dense_seconds": round(dense_seconds, 2),
        "ingest_seconds": round(ingest_seconds, 2),
        "capsules": len(store.capsules),
        "store_bytes": store.memory_bytes(),
        "questions_detail": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--store", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("clip", "detector", "hybrid"), default="hybrid")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=16)
    parser.add_argument("--min-track-frames", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--vocabulary", default=",".join(DEFAULT_VOCABULARY))
    args = parser.parse_args()
    summary = evaluate(args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "questions_detail"}, indent=2))
    print(f"[report] {path}")


if __name__ == "__main__":
    main()

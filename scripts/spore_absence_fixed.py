#!/usr/bin/env python3
"""Fixed-label detector check for the presence/absence probe set.

The main absence calibration uses YOLO-World, whose prompt-conditioned label
space substitutes confusable categories (a glass becomes `mug`, a plate of
chips becomes `bowl`).  This helper runs one pass of a fixed-label detector
(COCO YOLOv8 by default) over the same retained contexts and reports the same
presence/absence counts on the terms the fixed label space covers.

It is a post-hoc diagnostic, not part of the pre-registered protocol: run it
after ``spore_absence.py`` and pass that report to get a side-by-side table.

Usage::

    python scripts/spore_absence_fixed.py \
        --store research/spore/results/eat1_absence/store.json \
        --probes data/qa/eat_1_absence.json \
        --world-report research/spore/results/eat1_absence/report.json \
        --out research/spore/results/eat1_absence/fixed_label.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chronicle.perception.capsules import CapsuleStore  # noqa: E402

# COCO class names for probe terms whose wording differs.
COCO_ALIASES = {
    "television": ("tv",),
    "phone": ("cell phone",),
    "sofa": ("couch",),
    "table": ("dining table",),
    "glass": ("wine glass",),
}


def load_probes(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    probes = payload.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError(f"{path}: expected a non-empty 'probes' list")
    return probes


def scan(model_name: str, contexts, probes: list[dict], floor: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_name)
    names = set(model.names.values())
    rows = {probe["term"]: [] for probe in probes}
    started = time.perf_counter()
    for index, context in enumerate(contexts):
        frame = context.image()
        if frame is None:
            continue
        result = model(frame, conf=floor, iou=0.45, imgsz=640, verbose=False)
        detections = [
            (model.names[int(box.cls)], float(box.conf)) for box in result[0].boxes
        ]
        for probe in probes:
            term = probe["term"]
            wanted = COCO_ALIASES.get(term, (term,))
            scores = [score for label, score in detections if label in wanted]
            rows[term].append(
                {
                    "timestamp": round(float(context.timestamp), 2),
                    "score": round(max(scores), 4) if scores else None,
                }
            )
        if (index + 1) % 10 == 0:
            print(f"[scan] {index + 1}/{len(contexts)} contexts", flush=True)
    seconds = time.perf_counter() - started

    terms = []
    for probe in probes:
        term = probe["term"]
        wanted = COCO_ALIASES.get(term, (term,))
        scores = [row["score"] for row in rows[term] if row["score"] is not None]
        terms.append(
            {
                "term": term,
                "present": bool(probe["present"]),
                "covered": any(label in names for label in wanted),
                "labels": list(wanted),
                "hits": {
                    str(threshold): sum(
                        1
                        for row in rows[term]
                        if (row["score"] or 0.0) >= threshold
                    )
                    for threshold in (0.25, 0.50)
                },
                "max_confidence": round(max(scores), 4) if scores else None,
            }
        )
    return {
        "model": model_name,
        "floor": floor,
        "contexts": len(contexts),
        "seconds": round(seconds, 2),
        "terms": terms,
    }


def comparison(
    world_report: dict | None,
    fixed: dict,
    *,
    audited: bool,
) -> list[dict]:
    if world_report is None:
        return []
    world = {row["term"]: row for row in world_report["term_rows"]}
    covered = [row for row in fixed["terms"] if row["covered"]]
    present = [
        row["term"]
        for row in covered
        if (row["term"] == "television" if audited else False)
        or world[row["term"]]["present"]
    ]
    absent = [row["term"] for row in covered if row["term"] not in present]
    rows = []

    def hit_count(term: str, key: str, floor: float) -> bool:
        if key == "world":
            return world[term]["hits"][str(floor)] > 0
        row = next(item for item in covered if item["term"] == term)
        return row["hits"][str(floor)] > 0

    for key, floors in (("world", (0.22, 0.30, 0.50)), ("fixed", (0.25, 0.50))):
        for floor in floors:
            found = sum(1 for term in present if hit_count(term, key, floor))
            clean = sum(1 for term in absent if not hit_count(term, key, floor))
            rows.append(
                {
                    "detector": "YOLO-World-S" if key == "world" else fixed["model"],
                    "floor": floor,
                    "present_found": found,
                    "present_total": len(present),
                    "absent_clean": clean,
                    "absent_total": len(absent),
                    "missed": [
                        term for term in present if not hit_count(term, key, floor)
                    ],
                    "false_positive": [
                        term for term in absent if hit_count(term, key, floor)
                    ],
                }
            )
    return rows


def render_markdown(payload: dict) -> str:
    lines = ["# Fixed-label detector check", ""]
    lines.append(
        f"Model `{payload['fixed']['model']}`, {payload['fixed']['contexts']} "
        f"contexts, floor {payload['fixed']['floor']}. "
        f"Post-hoc diagnostic; not part of the pre-registered protocol."
    )
    lines.append("")
    lines.append("| detector | floor | present found | absent clean |")
    lines.append("|---|---:|---:|---:|")
    for row in payload["comparison_audited"]:
        lines.append(
            f"| {row['detector']} | {row['floor']} | "
            f"{row['present_found']}/{row['present_total']} | "
            f"{row['absent_clean']}/{row['absent_total']} |"
        )
    lines.append("")
    lines.append("Per covered term (fixed-label):")
    lines.append("")
    lines.append("| term | human | hits@.25 | hits@.50 | max |")
    lines.append("|---|---|---:|---:|---:|")
    for row in payload["fixed"]["terms"]:
        if not row["covered"]:
            continue
        lines.append(
            f"| {row['term']} | {'present' if row['present'] else 'absent'} | "
            f"{row['hits']['0.25']} | {row['hits']['0.5']} | "
            f"{row['max_confidence']} |"
        )
    uncovered = [
        row["term"] for row in payload["fixed"]["terms"] if not row["covered"]
    ]
    lines.append("")
    lines.append(f"No fixed-label class: {', '.join(uncovered)}.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--world-report", default="")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    store = CapsuleStore.load(args.store)
    probes = load_probes(Path(args.probes))
    print(f"[store] {len(store.contexts)} contexts, {len(probes)} probes")
    fixed = scan(args.model, store.contexts, probes, args.floor)
    world_report = (
        json.loads(Path(args.world_report).read_text()) if args.world_report else None
    )
    payload = {
        "fixed": fixed,
        "comparison_original": comparison(world_report, fixed, audited=False),
        "comparison_audited": comparison(world_report, fixed, audited=True),
    }
    out = Path(args.out) if args.out else Path(args.store).parent / "fixed_label.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    out.with_suffix(".md").write_text(render_markdown(payload))
    print(render_markdown(payload))


if __name__ == "__main__":
    main()

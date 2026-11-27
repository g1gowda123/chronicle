#!/usr/bin/env python3
"""Crop-level CLIP verification probe for fixed-label presence claims.

For every ``present`` verdict from the fixed-label verifier, crop the box and
compare CLIP image similarity against the term and a small set of confusable
competitors (a plate vs a bowl, a window vs a television).  The question is
whether the margin can veto confident look-alike false positives without
hurting true positives.

This is a diagnostic, not a shipped filter: with two false positives and a few
dozen true positives the sample cannot calibrate a threshold.

Usage::

    python scripts/spore_absence_crop_check.py \
        --store research/spore/results/eat3_absence/store.json \
        --probes data/qa/eat_3_absence.json \
        --out research/spore/results/eat3_absence/crop_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chronicle.perception.absence import FixedLabelVerifier  # noqa: E402
from chronicle.perception.capsules import CapsuleStore  # noqa: E402
from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.perception.embedder import build_embedder, crop_image  # noqa: E402

COMPETITORS = {
    "bowl": ["plate", "dish", "saucer"],
    "plate": ["bowl", "dish"],
    "television": ["window", "picture frame", "monitor", "painting"],
    "window": ["door", "wall", "television"],
    "cup": ["glass", "mug", "bottle"],
    "glass": ["cup", "bottle", "mug"],
    "mug": ["cup", "glass"],
    "bottle": ["cup", "glass", "jar"],
    "pitcher": ["jug", "vase", "bottle"],
    "sandwich": ["bread", "toast", "pizza"],
    "table": ["desk", "countertop"],
    "sofa": ["chair", "bench", "bed"],
    "cabinet": ["shelf", "dresser"],
    "refrigerator": ["cabinet", "freezer", "oven"],
    "microwave": ["oven", "toaster"],
    "napkin": ["paper", "towel", "cloth"],
    "fork": ["knife", "spoon"],
    "knife": ["fork", "spoon"],
    "spoon": ["fork", "knife"],
    "book": ["laptop", "box", "magazine"],
    "phone": ["remote", "tablet"],
    "laptop": ["book", "tablet", "keyboard"],
    "bicycle": ["motorcycle", "scooter"],
    "umbrella": ["parasol", "canopy"],
}


def evaluate(args: argparse.Namespace) -> dict:
    store = CapsuleStore.load(args.store)
    probes = json.loads(Path(args.probes).read_text())["probes"]
    embedder = build_embedder(PerceptionConfig(model_dir=ROOT / "models"))

    def text_vectors(texts: list[str]) -> list[np.ndarray]:
        vectors = []
        for text in texts:
            vector = embedder.embed_text(text)
            if vector is not None:
                vectors.append(vector)
        return vectors

    verifier = FixedLabelVerifier(
        store, model=args.model, threshold=args.threshold
    )
    verifier.scan()
    rows = []
    for probe in probes:
        term = probe["term"]
        verdict = verifier.check(term)
        if verdict.verdict != "present" or verdict.bbox is None:
            continue
        context = min(
            store.contexts,
            key=lambda item: abs(float(item.timestamp) - float(verdict.timestamp or 0.0)),
        )
        frame = context.image()
        if frame is None:
            continue
        height, width = frame.shape[:2]
        pad = 0.15 * max(verdict.bbox.width, verdict.bbox.height)
        crop = crop_image(frame, verdict.bbox.expanded(pad, bounds=(width, height)))
        if crop is None or crop.size == 0:
            continue
        image = embedder.embed_image(crop)
        positive_texts = [term, f"a photo of a {term}", f"a close-up photo of a {term}"]
        positive = max(
            float(np.dot(image, vector)) for vector in text_vectors(positive_texts)
        )
        competitors = COMPETITORS.get(term, [])
        negative_vectors = text_vectors(competitors)
        negative = (
            max(float(np.dot(image, vector)) for vector in negative_vectors)
            if negative_vectors
            else -1.0
        )
        rows.append(
            {
                "term": term,
                "human_present": bool(probe.get("present", False)),
                "label": "TP" if probe.get("present") else "FP",
                "detector_confidence": round(float(verdict.confidence or 0.0), 4),
                "timestamp": verdict.timestamp,
                "positive_similarity": round(positive, 4),
                "competitor_similarity": round(negative, 4),
                "margin": round(positive - negative, 4),
                "competitors": competitors,
            }
        )
    tps = [row["margin"] for row in rows if row["label"] == "TP"]
    fps = [row["margin"] for row in rows if row["label"] == "FP"]
    return {
        "store": args.store,
        "probes": args.probes,
        "model": args.model,
        "threshold": args.threshold,
        "rows": rows,
        "tp_margin_range": [min(tps), max(tps)] if tps else None,
        "fp_margins": fps,
        "veto_threshold_candidate": (
            round(max(fps), 3) if fps and tps and max(fps) < min(tps) else None
        ),
    }


def render_markdown(payload: dict) -> str:
    lines = ["# Crop-level CLIP verification probe", ""]
    lines.append(
        f"Store `{payload['store']}`, model `{payload['model']}`, threshold "
        f"{payload['threshold']}. Diagnostic only."
    )
    lines.append("")
    lines.append("| term | label | detector conf | positive sim | competitor sim | margin |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in payload["rows"]:
        lines.append(
            f"| {row['term']} | {row['label']} | {row['detector_confidence']} | "
            f"{row['positive_similarity']} | {row['competitor_similarity']} | "
            f"{row['margin']} |"
        )
    lines.append("")
    lines.append(
        f"TP margin range: {payload['tp_margin_range']}. FP margins: "
        f"{payload['fp_margins']}. A single veto threshold exists: "
        f"{payload['veto_threshold_candidate']}."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--model", default="yolov8l.pt")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    payload = evaluate(args)
    out = Path(args.out) if args.out else Path(args.store).parent / "crop_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    out.with_suffix(".md").write_text(render_markdown(payload))
    print(render_markdown(payload))


if __name__ == "__main__":
    main()

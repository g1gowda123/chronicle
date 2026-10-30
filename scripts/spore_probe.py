"""Inspect SPORE proposals on a single video frame.

Usage::

    python scripts/spore_probe.py --clip data/real/aria_cook_45s.mp4 \
        --time 12.0 --box 342,437,430,478
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.perception.embedder import build_embedder  # noqa: E402
from chronicle.perception.hands import build_hands  # noqa: E402
from chronicle.perception.sources import VideoSource  # noqa: E402
from chronicle.perception.spore import (  # noqa: E402
    SporeConfig,
    SporePerceiver,
    _coverage_top,
    _resize_max_width,
)
from chronicle.types import BBox  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--box", default="")
    parser.add_argument("--top", type=int, default=16)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    perception = PerceptionConfig(model_dir=str(ROOT / "models"))
    config = SporeConfig()
    perceiver = SporePerceiver(
        config,
        perception,
        embedder=build_embedder(perception, source_name="tabletop"),
        hands=build_hands(perception, source_name="tabletop"),
    )
    frame = None
    for candidate, timestamp, _ in VideoSource(args.clip):
        if timestamp >= args.time:
            frame = candidate
            break
    if frame is None:
        raise SystemExit("frame not found")
    frame, _ = _resize_max_width(frame, config.max_width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = perceiver._hands.detect(frame, timestamp=args.time) if perceiver._hands else []
    candidates = perceiver._proposals(frame, gray, None, hands)
    target = BBox.from_sequence([float(v) for v in args.box.split(",")]) if args.box else None
    print(f"frame={frame.shape} hands={[hand.bbox.int_xyxy for hand in hands]}")
    print(f"candidates={len(candidates)} target={target.int_xyxy if target else None}")
    selected = _coverage_top(
        candidates,
        config.patches_per_keyframe,
        config.grid_cell,
        hand_slots=config.hand_slots,
    )
    for index, candidate in enumerate(selected):
        coverage = (
            candidate.bbox.intersection_area(target) / target.area if target is not None else 0.0
        )
        print(
            f"{index:2d} score={candidate.score:.3f} src={candidate.source:10s} "
            f"box={candidate.bbox.int_xyxy} cov={coverage:.2f} "
            f"hand={candidate.hand_overlap:.2f} detail={candidate.detail:.2f}"
        )
    if args.box:
        x1, y1, x2, y2 = target.int_xyxy
        cv2.imwrite(
            str(ROOT / "research" / "spore" / f"probe_{args.time:.1f}_target.jpg"),
            frame[max(0, y1 - 12) : y2 + 12, max(0, x1 - 12) : x2 + 12],
        )


if __name__ == "__main__":
    main()

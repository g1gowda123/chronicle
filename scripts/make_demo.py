#!/usr/bin/env python3
"""Render the offline synthetic demo clip used by the palette backends.

The clip is deliberately simple: four flat-colour objects, a skin-colour hand
that reaches for the keys for a few seconds, and a short camera pan. It is not
a benchmark; it lets every Role A stage (motion gate, detection, tracking,
IoS/dwell interaction, 30-frame clip) run without downloading models.

    python scripts/make_demo.py
    python -m chronicle.perception data/demo/chronicle_demo.mp4 \
        --detector color --hands color --embedder color

Requires ``opencv-python`` and ``numpy`` only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronicle.perception.palette import PALETTE  # noqa: E402

WIDTH, HEIGHT = 640, 360
WALL_HEIGHT = 140
BACKGROUND_BGR = (48, 66, 88)  # dark wood, far from every palette colour

OBJECTS = {
    "keys": (300, 250, 120, 55),
    "phone": (120, 235, 65, 110),
    "mug": (470, 255, 65, 75),
    "backpack": (552, 175, 62, 120),
}


def rounded_rect(
    image: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int], radius: int = 8
) -> None:
    cv2.rectangle(image, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(image, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in (
        (x + radius, y + radius),
        (x + w - radius, y + radius),
        (x + radius, y + h - radius),
        (x + w - radius, y + h - radius),
    ):
        cv2.circle(image, (cx, cy), radius, color, -1)


def background(rng: np.random.Generator) -> np.ndarray:
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:WALL_HEIGHT] = (112, 104, 96)
    frame[WALL_HEIGHT:] = BACKGROUND_BGR

    # Grain and high-frequency speckle give the affine estimator plenty of
    # corners even after the object and hand boxes are masked out.
    grain = rng.normal(0, 12, size=(HEIGHT, WIDTH, 1)).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    for _ in range(320):
        x = int(rng.integers(4, WIDTH - 4))
        y = int(rng.integers(4, HEIGHT - 4))
        radius = int(rng.integers(1, 3))
        shade = int(rng.integers(25, 105))
        color = (shade, shade + 14, shade + 30) if y >= WALL_HEIGHT else (shade, shade - 6, shade - 12)
        cv2.circle(frame, (x, y), radius, color, -1)
    for _ in range(120):
        x = int(rng.integers(8, WIDTH - 40))
        y = int(rng.integers(8, HEIGHT - 8))
        cv2.line(
            frame,
            (x, y),
            (x + int(rng.integers(10, 35)), y + int(rng.integers(-2, 3))),
            (75, 92, 112),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "CHRONICLE  //  SYNTHETIC DEMO",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 160, 170),
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_objects(frame: np.ndarray) -> None:
    x, y, w, h = OBJECTS["keys"]
    rounded_rect(frame, x, y, w, h, PALETTE["keys"], radius=10)
    for index in range(3):
        cv2.rectangle(frame, (x + 18 + index * 30, y + 10), (x + 32 + index * 30, y + h - 10), (40, 80, 120), -1)
    cv2.circle(frame, (x + w - 16, y + h // 2), 9, (40, 80, 120), -1)

    x, y, w, h = OBJECTS["phone"]
    rounded_rect(frame, x, y, w, h, PALETTE["phone"], radius=8)
    cv2.rectangle(frame, (x + 10, y + 16), (x + w - 10, y + h - 18), (30, 30, 40), -1)
    cv2.circle(frame, (x + w // 2, y + h - 9), 4, (30, 30, 40), -1)

    x, y, w, h = OBJECTS["mug"]
    rounded_rect(frame, x, y, w, h, PALETTE["mug"], radius=8)
    cv2.ellipse(frame, (x + w + 8, y + h // 2), (16, 22), 0, -90, 90, PALETTE["mug"], 8)
    cv2.ellipse(frame, (x + w + 10, y + h // 2), (9, 13), 0, -90, 90, BACKGROUND_BGR, 8)

    x, y, w, h = OBJECTS["backpack"]
    rounded_rect(frame, x, y, w, h, PALETTE["backpack"], radius=14)
    cv2.rectangle(frame, (x + 14, y + 42), (x + w - 14, y + h - 26), (10, 90, 10), 3)
    cv2.line(frame, (x + 16, y + 20), (x + w - 16, y + 20), (10, 90, 10), 4)


def draw_hand(frame: np.ndarray, center: tuple[float, float]) -> None:
    cx, cy = int(round(center[0])), int(round(center[1]))
    skin = PALETTE["hand"]
    edge = (70, 105, 150)
    cv2.ellipse(frame, (cx, cy), (28, 36), 0, 0, 360, skin, -1)
    for dx, dy, radius in ((-18, -34, 9), (-6, -41, 9), (6, -41, 9), (18, -34, 9), (-31, -4, 10)):
        cv2.circle(frame, (cx + dx, cy + dy), radius, skin, -1)
    cv2.ellipse(frame, (cx, cy), (28, 36), 0, 0, 360, edge, 2)
    for dx, dy in ((-18, -34), (-6, -41), (6, -41), (18, -34)):
        cv2.circle(frame, (cx + dx, cy + dy), 9, edge, 1)


def hand_position(index: int, fps: int) -> tuple[float, float] | None:
    """Reach for the keys, hover for ~4 s, then withdraw."""
    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * max(0.0, min(1.0, t))

    start, approach_end = int(0.8 * fps), int(2.8 * fps)
    hover_end, exit_end = int(7.2 * fps), int(9.2 * fps)
    offscreen = (505.0, 430.0)
    # Overlap the right half of the keys so the remaining yellow region stays
    # one connected component (a centred hand splits it into two boxes).
    hover = (395.0, 272.0)

    if index < start:
        return None
    if index < approach_end:
        t = (index - start) / max(1, approach_end - start)
        return (lerp(offscreen[0], hover[0], t), lerp(offscreen[1], hover[1], t))
    if index < hover_end:
        wobble = 2.0 * np.sin(index / 9.0)
        return (hover[0] + wobble, hover[1] + 0.5 * np.sin(index / 7.0))
    if index < exit_end:
        t = (index - hover_end) / max(1, exit_end - hover_end)
        return (lerp(hover[0], offscreen[0], t), lerp(hover[1], offscreen[1], t))
    return None


def render(out: Path, *, fps: int, seconds: float) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out}")

    rng = np.random.default_rng(23)
    base = background(rng)
    draw_objects(base)
    total = int(round(seconds * fps))
    pan_start = int(10.0 * fps)
    pan_frames = max(4, int(0.4 * fps))  # 12 px/frame -> classified as PANNING

    for index in range(total):
        frame = base.copy()
        hand = hand_position(index, fps)
        if hand is not None:
            draw_hand(frame, hand)
        if index >= pan_start:
            progress = min(1.0, (index - pan_start) / pan_frames)
            offset = int(round(progress * 12.0 * pan_frames))
            if offset:
                matrix = np.float32([[1, 0, -offset], [0, 1, 0]])
                # Constant borders avoid reflecting objects back into view,
                # which would create duplicate detections during the pan.
                frame = cv2.warpAffine(
                    frame,
                    matrix,
                    (WIDTH, HEIGHT),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=BACKGROUND_BGR,
                )
        writer.write(frame)
    writer.release()
    print(f"wrote {out} ({total} frames @ {fps} fps, {seconds:.1f}s)")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/demo/chronicle_demo.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=12.0)
    args = parser.parse_args(argv)
    render(Path(args.out), fps=max(1, args.fps), seconds=max(1.0, args.seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

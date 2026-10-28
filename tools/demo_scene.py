"""Synthetic egocentric scene used by the judge demo.

This is *not* part of the production pipeline.  It renders a deterministic
26-second kitchen-counter episode so the Role D dashboard can be demonstrated
independently of Roles A/B/C, while emitting exactly the same event contract
the real pipeline uses.

Rendered with Pillow only (no model weights, no network).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCREEN_W = 1280
SCREEN_H = 720
FPS = 15
DURATION = 26.0

CAMERA_MAX = 260.0
PAN_IN_END = 5.0
PAN_OUT_START = 23.0

# World-space boxes (x1, y1, x2, y2). The camera only pans horizontally.
BASE_BOXES: dict[str, tuple[float, float, float, float]] = {
    "wallet": (350, 192, 452, 252),
    "bottle": (40, 470, 112, 612),
    "notebook": (260, 470, 430, 590),
    "mug": (500, 505, 585, 610),
    "phone": (680, 520, 745, 645),
    "keys": (840, 570, 950, 615),
    "tote": (1030, 455, 1220, 700),
    "backpack": (1330, 415, 1545, 652),
}

# Held intervals drive both the renderer and the event timeline.
HELD_KEYS = (10.9, 16.9)
HELD_PHONE = (19.4, 21.7)

SKIN = (214, 158, 116)
SKIN_DARK = (142, 92, 61)
SKIN_LIGHT = (236, 189, 150)
SLEEVE = (31, 40, 52)
SLEEVE_LIGHT = (57, 72, 88)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smoothstep(value: float) -> float:
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def camera_x(t: float) -> float:
    if t <= PAN_IN_END:
        return CAMERA_MAX * smoothstep(t / PAN_IN_END)
    if t <= PAN_OUT_START:
        return CAMERA_MAX
    return CAMERA_MAX * (1.0 - smoothstep((t - PAN_OUT_START) / (DURATION - PAN_OUT_START)))


def _sx(world_x: float, t: float) -> float:
    return world_x - camera_x(t)


def screen_bbox(name: str, t: float) -> list[float] | None:
    """Screen-space bbox in pixels, including held-object motion."""
    hand = hand_at(t)
    if name == "keys" and held("keys", t) and hand is not None:
        return held_bbox("keys", hand)
    if name == "phone" and held("phone", t) and hand is not None:
        return held_bbox("phone", hand)
    box = BASE_BOXES.get(name)
    if box is None:
        return None
    cam = camera_x(t)
    return [box[0] - cam, box[1], box[2] - cam, box[3]]


def normalised_bbox(name: str, t: float) -> list[float] | None:
    box = screen_bbox(name, t)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    x1, x2 = sorted((clamp(x1, 0, SCREEN_W), clamp(x2, 0, SCREEN_W)))
    y1, y2 = sorted((clamp(y1, 0, SCREEN_H), clamp(y2, 0, SCREEN_H)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [round(x1 / SCREEN_W, 4), round(y1 / SCREEN_H, 4), round(x2 / SCREEN_W, 4), round(y2 / SCREEN_H, 4)]


def held(name: str, t: float) -> bool:
    if name == "keys":
        return HELD_KEYS[0] <= t < HELD_KEYS[1]
    if name == "phone":
        return HELD_PHONE[0] <= t < HELD_PHONE[1]
    return False


# --------------------------------------------------------------------------- #
# Hand path
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Hand:
    x: float
    y: float
    grip: float

    @property
    def visible(self) -> bool:
        return 60 < self.x < SCREEN_W + 220 and self.y < SCREEN_H + 150


HAND_KEYS: list[tuple[float, float, float, float]] = [
    (7.70, 1245, 815, 0.00),
    (8.00, 300, 600, 0.25),
    (8.20, 300, 590, 0.95),
    (8.45, 320, 600, 0.95),
    (8.85, 1200, 815, 0.00),
    (9.80, 1245, 815, 0.00),
    (10.15, 640, 605, 0.25),
    (10.45, 640, 600, 0.95),
    (10.90, 650, 520, 1.00),
    (12.40, 760, 485, 1.00),
    (14.40, 800, 500, 1.00),
    (16.40, 865, 525, 1.00),
    (16.90, 865, 555, 0.35),
    (17.35, 905, 620, 0.05),
    (17.80, 1210, 815, 0.00),
    (18.40, 1245, 815, 0.00),
    (18.85, 450, 605, 0.25),
    (19.15, 450, 595, 0.95),
    (19.40, 465, 520, 1.00),
    (20.30, 520, 420, 1.00),
    (20.85, 545, 430, 1.00),
    (21.35, 465, 575, 1.00),
    (21.70, 450, 600, 0.30),
    (22.05, 475, 615, 0.05),
    (22.60, 1210, 815, 0.00),
    (26.00, 1245, 815, 0.00),
]


def hand_at(t: float) -> Hand | None:
    if t < HAND_KEYS[0][0] or t > HAND_KEYS[-1][0]:
        return None
    for i in range(len(HAND_KEYS) - 1):
        t0, x0, y0, g0 = HAND_KEYS[i]
        t1, x1, y1, g1 = HAND_KEYS[i + 1]
        if t0 <= t <= t1 and t1 > t0:
            u = smoothstep((t - t0) / (t1 - t0))
            return Hand(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, g0 + (g1 - g0) * u)
    return None


def held_anchor(name: str, hand: Hand) -> tuple[float, float]:
    if name == "keys":
        return hand.x - 40, hand.y - 30
    return hand.x - 20, hand.y - 50


def held_bbox(name: str, hand: Hand) -> list[float]:
    cx, cy = held_anchor(name, hand)
    if name == "keys":
        return [cx - 58, cy - 28, cx + 58, cy + 28]
    return [cx - 40, cy - 64, cx + 40, cy + 64]


# --------------------------------------------------------------------------- #
# Fonts / static resources
# --------------------------------------------------------------------------- #


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10
        return ImageFont.load_default()


FONT_16 = _font(16)
FONT_14 = _font(14)
FONT_12 = _font(12)
FONT_22 = _font(22)

_WALL_GRADIENT = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.float32)
for _y in range(SCREEN_H):
    _u = _y / SCREEN_H
    _top = np.array([27, 42, 54], dtype=np.float32)
    _bottom = np.array([11, 18, 25], dtype=np.float32)
    _WALL_GRADIENT[_y, :, :] = _top * (1 - _u) + _bottom * _u

_yy, _xx = np.mgrid[0:SCREEN_H, 0:SCREEN_W]
_r = np.sqrt(((_xx - SCREEN_W / 2) / (SCREEN_W * 0.62)) ** 2 + ((_yy - SCREEN_H / 2) / (SCREEN_H * 0.62)) ** 2)
_VIGNETTE = np.clip(1.06 - 0.55 * _r**2.2, 0.46, 1.0)[:, :, None].astype(np.float32)


# --------------------------------------------------------------------------- #
# Primitive helpers
# --------------------------------------------------------------------------- #


def _rr(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _shadow(draw: ImageDraw.ImageDraw, box, spread=14, y_offset=12, color=(24, 18, 13)):
    x1, y1, x2, y2 = box
    draw.ellipse([x1 - spread, y2 - 8 + y_offset, x2 + spread, y2 + 16 + y_offset], fill=color)


def _capsule(draw: ImageDraw.ImageDraw, p0, p1, radius, fill, outline=None, width=2):
    draw.line([p0, p1], fill=outline or fill, width=int(radius * 2 + width * 2))
    draw.line([p0, p1], fill=fill, width=int(radius * 2))
    for point in (p0, p1):
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            fill=fill,
            outline=outline,
            width=width,
        )


# --------------------------------------------------------------------------- #
# Set dressing
# --------------------------------------------------------------------------- #


def _draw_room(img: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
    cam = camera_x(t)
    base = Image.fromarray(_WALL_GRADIENT.astype(np.uint8), mode="RGB")
    img.paste(base, (0, 0))

    # Wall panels (move with camera).
    for world_x in range(0, 1800, 240):
        x = world_x - cam
        draw.line([x, 0, x, 430], fill=(78, 104, 118), width=1)

    # Window with cool daylight.
    wx1, wy1, wx2, wy2 = 560 - cam, 62, 940 - cam, 344
    draw.rounded_rectangle([wx1 - 12, wy1 - 12, wx2 + 12, wy2 + 12], radius=10, fill=(15, 22, 29))
    for i in range(24):
        u = i / 23
        color = (
            int(64 + 96 * u),
            int(109 + 80 * u),
            int(126 + 46 * u),
        )
        draw.rectangle([wx1, wy1 + (wy2 - wy1) * i / 24, wx2, wy1 + (wy2 - wy1) * (i + 1) / 24 + 1], fill=color)
    draw.rectangle([wx1, wy1, wx2, wy2], outline=(20, 30, 38), width=8)
    mid = (wx1 + wx2) / 2
    draw.rectangle([mid - 4, wy1, mid + 4, wy2], fill=(22, 32, 40))
    draw.rectangle([wx1, (wy1 + wy2) / 2 - 4, wx2, (wy1 + wy2) / 2 + 4], fill=(22, 32, 40))

    # Pendant lamp.
    lx = 1130 - cam
    draw.line([lx, 0, lx, 72], fill=(58, 54, 46), width=2)
    draw.polygon([(lx - 42, 128), (lx + 42, 128), (lx + 22, 64), (lx - 22, 64)], fill=(52, 47, 39))
    draw.ellipse([lx - 30, 118, lx + 30, 142], fill=(246, 206, 142))
    draw.ellipse([lx - 46, 128, lx + 46, 146], fill=(92, 77, 55))

    # Backsplash tiles.
    for world_x in range(0, 1700, 64):
        x = world_x - cam
        draw.line([x, 356, x, 430], fill=(45, 58, 61), width=1)
    for y in range(356, 430, 24):
        draw.line([0, y, SCREEN_W, y], fill=(45, 58, 61), width=1)

    # Countertop: warm wood with grain.
    draw.rectangle([0, 430, SCREEN_W, 720], fill=(73, 53, 37))
    for i, color in enumerate(((86, 63, 43), (79, 57, 39), (68, 49, 35))):
        draw.rectangle([0, 430 + i * 74, SCREEN_W, 504 + i * 74], fill=color)
    draw.line([0, 430, SCREEN_W, 430], fill=(138, 112, 82), width=3)
    draw.line([0, 438, SCREEN_W, 438], fill=(52, 37, 27), width=2)
    for world_x in range(-80, 1760, 83):
        x = world_x - cam
        phase = math.sin(world_x * 0.031) * 5
        draw.line([x, 444, x + phase, 700], fill=(58, 41, 30), width=1)
        if world_x % 166 == 0:
            draw.line([x + 13, 448, x + 13 + phase, 704], fill=(62, 44, 31), width=1)
    draw.rectangle([0, 664, SCREEN_W, 720], fill=(47, 34, 25))
    draw.line([0, 664, SCREEN_W, 664], fill=(92, 69, 47), width=2)

    # Shelf.
    sx1, _, sx2, sy2 = 60 - cam, 140, 460 - cam, 300
    draw.rectangle([sx1, sy2 - 18, sx2, sy2], fill=(96, 71, 47))
    draw.rectangle([sx1, sy2 - 18, sx2, sy2 - 11], fill=(124, 95, 64))
    draw.line([sx1, sy2 - 2, sx2, sy2 - 2], fill=(44, 32, 23), width=2)
    # Jars and greenery.
    for _i, (jx, jw, jh, fill) in enumerate(
        ((104, 42, 58, (111, 136, 92)), (156, 34, 44, (128, 92, 61)), (212, 46, 66, (90, 112, 126)))
    ):
        x = _sx(jx, t)
        top = sy2 - 18 - jh
        draw.rounded_rectangle([x, top, x + jw, sy2 - 18], radius=5, fill=(204, 214, 208), outline=(222, 230, 226))
        draw.rectangle([x + 4, top + 14, x + jw - 4, sy2 - 24], fill=fill)
        draw.rectangle([x + jw * 0.28, top - 7, x + jw * 0.72, top], fill=(70, 62, 52))
    plant_x = 286 - cam
    draw.rectangle([plant_x, 232, plant_x + 44, 282], fill=(151, 110, 78))
    for angle in range(-60, 61, 26):
        rad = math.radians(angle - 90)
        draw.line(
            [plant_x + 22, 236, plant_x + 22 + math.cos(rad) * 40, 236 + math.sin(rad) * 40],
            fill=(71, 102, 70),
            width=7,
        )
    # A second, smaller shelf on the right.
    rx1, rx2 = 1190 - cam, 1400 - cam
    draw.rectangle([rx1, 300, rx2, 322], fill=(92, 67, 45))
    draw.rectangle([rx1, 300, rx2, 307], fill=(120, 92, 62))
    draw.rectangle([rx1 + 34, 254, rx1 + 78, 300], fill=(72, 78, 70))
    draw.rectangle([rx2 - 74, 264, rx2 - 26, 300], fill=(122, 82, 49))


# --------------------------------------------------------------------------- #
# Objects
# --------------------------------------------------------------------------- #


def _draw_wallet(draw: ImageDraw.ImageDraw, t: float) -> None:
    x1, y1, x2, y2 = (v - camera_x(t) if i % 2 == 0 else v for i, v in enumerate(BASE_BOXES["wallet"]))
    _shadow(draw, [x1, y1, x2, y2], spread=8, y_offset=6)
    _rr(draw, [x1, y1, x2, y2], 8, (103, 71, 45), (70, 47, 30), 2)
    draw.line([x1 + 8, y1 + 14, x2 - 8, y1 + 14], fill=(132, 94, 61), width=2)
    for x in range(int(x1) + 12, int(x2) - 8, 12):
        draw.line([x, y2 - 8, x + 4, y2 - 4], fill=(187, 154, 111), width=1)
    draw.ellipse([x2 - 24, y1 + 22, x2 - 12, y1 + 34], fill=(188, 152, 80))


def _draw_bottle(draw: ImageDraw.ImageDraw, t: float) -> None:
    box = screen_bbox("bottle", t)
    if box is None:
        return
    x1, y1, x2, y2 = box
    if x2 < -20 or x1 > SCREEN_W + 20:
        return
    _shadow(draw, box, spread=9, y_offset=8)
    neck = (x2 - x1) * 0.32
    draw.rounded_rectangle([x1, y1 + 34, x2, y2], radius=18, fill=(69, 116, 133), outline=(37, 70, 84), width=2)
    draw.rectangle([x1 + neck, y1 + 4, x2 - neck, y1 + 44], fill=(79, 128, 144), outline=(37, 70, 84), width=2)
    draw.rectangle([x1 + neck + 4, y1, x2 - neck - 4, y1 + 12], fill=(43, 52, 56))
    draw.rectangle([x1 + 10, y1 + 62, x2 - 10, y2 - 46], fill=(226, 215, 190))
    draw.line([x1 + 16, y1 + 70, x1 + 16, y2 - 54], fill=(246, 240, 220), width=4)
    draw.line([x1 + neck + 8, y1 + 12, x1 + neck + 8, y1 + 40], fill=(160, 200, 210), width=2)


def _draw_backpack(draw: ImageDraw.ImageDraw, t: float) -> None:
    box = screen_bbox("backpack", t)
    if box is None:
        return
    x1, y1, x2, y2 = box
    if x2 < -30 or x1 > SCREEN_W + 30:
        return
    _shadow(draw, box, spread=18, y_offset=10)
    _rr(draw, [x1, y1 + 22, x2, y2], 26, (55, 71, 56), (33, 45, 35), 3)
    _rr(draw, [x1 + 16, y1, x2 - 16, y1 + 92], 24, (64, 82, 63), (33, 45, 35), 3)
    draw.line([x1 + 24, y1 + 34, x2 - 24, y1 + 34], fill=(35, 48, 37), width=3)
    _rr(draw, [x1 + 24, y1 + 108, x2 - 24, y2 - 24], 16, (46, 61, 48), (33, 45, 35), 2)
    draw.line([x1 + 24, y2 - 66, x2 - 24, y2 - 66], fill=(96, 113, 81), width=2)
    for sx in (x1 + 18, x2 - 30):
        draw.line([sx, y1 + 20, sx, y2 - 10], fill=(37, 50, 39), width=8)
    for bx in (x1 + 44, x2 - 60):
        _rr(draw, [bx, y1 + 96, bx + 16, y1 + 120], 4, (160, 135, 77), (104, 86, 47), 1)


def _draw_tote(draw: ImageDraw.ImageDraw, t: float) -> None:
    box = screen_bbox("tote", t)
    if box is None:
        return
    x1, y1, x2, y2 = box
    if x2 < -30 or x1 > SCREEN_W + 30:
        return
    _shadow(draw, box, spread=16, y_offset=8)
    draw.polygon(
        [(x1 + 12, y1 + 42), (x2 - 12, y1 + 42), (x2, y2 - 16), (x2 - 18, y2), (x1 + 18, y2), (x1, y2 - 16)],
        fill=(190, 160, 117),
    )
    draw.line([(x1 + 12, y1 + 42), (x1, y2 - 16)], fill=(136, 108, 75), width=3)
    draw.line([(x2 - 12, y1 + 42), (x2, y2 - 16)], fill=(136, 108, 75), width=3)
    draw.ellipse([x1 + 10, y1 + 28, x2 - 10, y1 + 76], fill=(54, 42, 30), outline=(116, 90, 61), width=3)
    draw.ellipse([x1 + 14, y1 + 32, x2 - 14, y1 + 62], fill=(40, 31, 22))

    # Handles.
    draw.arc([x1 + 24, y1 - 54, x1 + 104, y1 + 46], 180, 355, fill=(118, 89, 60), width=9)
    draw.arc([x2 - 104, y1 - 54, x2 - 24, y1 + 46], 185, 360, fill=(118, 89, 60), width=9)
    # Stitching.
    for y in range(int(y1) + 96, int(y2) - 14, 18):
        draw.line([x1 + 12, y, x1 + 22, y + 6], fill=(219, 197, 163), width=2)
        draw.line([x2 - 22, y, x2 - 12, y + 6], fill=(219, 197, 163), width=2)
    draw.line([x2 - 46, y1 + 100, x2 - 46, y2 - 24], fill=(163, 133, 95), width=2)


def _draw_notebook(draw: ImageDraw.ImageDraw, t: float) -> None:
    box = screen_bbox("notebook", t)
    if box is None:
        return
    x1, y1, x2, y2 = box
    if x2 < -30 or x1 > SCREEN_W + 30:
        return
    _shadow(draw, box, spread=15, y_offset=8)
    _rr(draw, [x1, y1, x2, y2], 8, (38, 57, 88), (22, 34, 54), 2)
    draw.rectangle([x1 + 6, y1 + 4, x1 + 22, y2 - 4], fill=(28, 43, 68))
    draw.rectangle([x2 - 6, y1 + 4, x2 - 1, y2 - 4], fill=(225, 217, 197))
    for y in range(int(y1) + 12, int(y2) - 6, 5):
        draw.line([x2 - 6, y, x2 - 1, y], fill=(180, 171, 150), width=1)
    draw.line([x2 - 58, y1 + 4, x2 - 58, y2 - 4], fill=(203, 112, 63), width=7)
    draw.line([x1 + 20, y1 + 8, x2 - 12, y1 + 8], fill=(56, 79, 116), width=2)


def _draw_mug(draw: ImageDraw.ImageDraw, t: float) -> None:
    box = screen_bbox("mug", t)
    if box is None:
        return
    x1, y1, x2, y2 = box
    if x2 < -30 or x1 > SCREEN_W + 30:
        return
    _shadow(draw, box, spread=12, y_offset=6)
    draw.ellipse([x2 - 22, y1 + 28, x2 + 22, y2 - 30], outline=(176, 164, 144), width=10)
    _rr(draw, [x1, y1 + 8, x2, y2], 18, (226, 218, 202), (174, 161, 139), 2)
    draw.ellipse([x1 + 1, y1 + 2, x2 - 1, y1 + 26], fill=(87, 58, 40), outline=(196, 184, 162), width=2)
    draw.ellipse([x1 + 8, y1 + 7, x2 - 26, y1 + 19], fill=(60, 39, 27))
    draw.line([x1 + 12, y1 + 34, x1 + 12, y2 - 16], fill=(247, 241, 228), width=6)
    draw.line([x2 - 12, y1 + 34, x2 - 12, y2 - 16], fill=(199, 188, 168), width=3)


def _draw_phone(draw: ImageDraw.ImageDraw, t: float) -> None:
    box = screen_bbox("phone", t)
    if box is None:
        return
    x1, y1, x2, y2 = box
    if x2 < -30 or x1 > SCREEN_W + 30:
        return
    _shadow(draw, box, spread=8, y_offset=5)
    _rr(draw, [x1, y1, x2, y2], 14, (17, 21, 27), (7, 9, 12), 2)
    _rr(draw, [x1 + 5, y1 + 8, x2 - 5, y2 - 10], 10, (20, 40, 61))
    draw.polygon([(x1 + 8, y1 + 18), (x1 + 8, y1 + 62), (x2 - 16, y1 + 10), (x2 - 30, y1 + 10)], fill=(41, 79, 105))
    draw.ellipse([x1 + 20, y1 + 14, x1 + 28, y1 + 22], fill=(7, 9, 12))
    draw.rounded_rectangle([x1 + 22, y2 - 20, x2 - 22, y2 - 15], radius=2, fill=(98, 112, 126))
    draw.line([x1 + 5, y1 + 40, x2 - 5, y1 + 24], fill=(52, 96, 122), width=2)


def _draw_keys(draw: ImageDraw.ImageDraw, t: float) -> None:
    box = screen_bbox("keys", t)
    if box is None:
        return
    x1, y1, x2, y2 = box
    if x2 < -30 or x1 > SCREEN_W + 30:
        return
    if not held("keys", t):
        _shadow(draw, box, spread=9, y_offset=5)
    draw.ellipse([x1, y1 + 2, x1 + 34, y1 + 34], outline=(192, 154, 62), width=5)
    draw.ellipse([x1 + 9, y1 + 11, x1 + 25, y1 + 25], outline=(226, 192, 108), width=2)
    for _index, (dy, length) in enumerate(((6, 58), (18, 66), (30, 52))):
        draw.rounded_rectangle(
            [x1 + 28, y1 + dy, x1 + 28 + length, y1 + dy + 10],
            radius=4,
            fill=(201, 163, 73),
            outline=(143, 109, 41),
        )
        draw.rectangle([x1 + 28 + length - 16, y1 + dy + 10, x1 + 28 + length - 8, y1 + dy + 17], fill=(201, 163, 73))
        draw.rectangle([x1 + 28 + length - 26, y1 + dy + 10, x1 + 28 + length - 18, y1 + dy + 15], fill=(201, 163, 73))
    draw.line([x1 + 4, y1 + 6, x1 + 24, y1 + 8], fill=(242, 214, 140), width=2)


OBJECT_DRAWERS = {
    "wallet": _draw_wallet,
    "bottle": _draw_bottle,
    "backpack": _draw_backpack,
    "tote": _draw_tote,
    "notebook": _draw_notebook,
    "mug": _draw_mug,
    "phone": _draw_phone,
    "keys": _draw_keys,
}


# --------------------------------------------------------------------------- #
# Hand
# --------------------------------------------------------------------------- #


def _draw_hand(img: Image.Image, hand: Hand, t: float) -> None:
    if not hand.visible:
        return
    draw = ImageDraw.Draw(img)
    entry = (1255.0, 812.0)
    wx, wy = hand.x, hand.y

    # Sleeve + forearm.
    dx, dy = wx - entry[0], wy - entry[1]
    length = max(math.hypot(dx, dy), 1.0)
    cuff = (entry[0] + dx * 0.22, entry[1] + dy * 0.22)
    draw.line([entry, cuff], fill=(18, 25, 33), width=152)
    draw.line([entry, cuff], fill=SLEEVE, width=138)
    draw.line([entry[0] + 8, entry[1] - 4, cuff[0] + 8, cuff[1] - 4], fill=SLEEVE_LIGHT, width=5)
    draw.line([cuff, (wx, wy)], fill=SKIN_DARK, width=128)
    draw.line([cuff, (wx, wy)], fill=SKIN, width=118)
    draw.line([cuff, (wx, wy)], fill=SKIN_LIGHT, width=34)
    # Cuff seam.
    draw.line(
        [cuff[0] - dy / length * 52, cuff[1] + dx / length * 52, cuff[0] + dy / length * 52, cuff[1] - dx / length * 52],
        fill=(86, 103, 120),
        width=7,
    )

    # Hand local frame: fingers point up-left toward the counter.
    angle = math.radians(-142)
    ca, sa = math.cos(angle), math.sin(angle)

    def local(lx: float, ly: float) -> tuple[float, float]:
        return (wx + lx * ca - ly * sa, wy + lx * sa + ly * ca)

    palm = [local(-72, -58), local(28, -66), local(62, -42), local(68, 38), local(24, 64), local(-62, 58), local(-84, 0)]
    draw.polygon(palm, fill=SKIN, outline=SKIN_DARK)
    draw.line([local(-46, 0), local(28, 0)], fill=SKIN_LIGHT, width=3)

    held_name = "keys" if held("keys", t) else "phone" if held("phone", t) else None
    if held_name is not None:
        if held_name == "keys":
            # Draw the keys directly at the palm (bypasses the world bbox).
            x1, y1, x2, y2 = held_bbox("keys", hand)
            draw.ellipse([x1 + 6, y1 + 4, x1 + 42, y1 + 40], outline=(200, 162, 70), width=6)
            for _index, (dy2, length2) in enumerate(((6, 58), (18, 66), (30, 52))):
                draw.rounded_rectangle(
                    [x1 + 32, y1 + dy2, x1 + 32 + length2, y1 + dy2 + 10],
                    radius=4,
                    fill=(209, 172, 82),
                    outline=(143, 109, 41),
                )
                draw.rectangle([x1 + 32 + length2 - 16, y1 + dy2 + 10, x1 + 32 + length2 - 8, y1 + dy2 + 17], fill=(209, 172, 82))
        else:
            x1, y1, x2, y2 = held_bbox("phone", hand)
            _rr(draw, [x1, y1, x2, y2], 14, (17, 21, 27), (6, 8, 11), 2)
            _rr(draw, [x1 + 5, y1 + 10, x2 - 5, y2 - 10], 9, (22, 43, 65))
            draw.polygon([(x1 + 8, y1 + 22), (x1 + 8, y1 + 62), (x2 - 14, y1 + 12), (x2 - 28, y1 + 12)], fill=(46, 86, 112))
            draw.ellipse([x1 + 16, y1 + 16, x1 + 23, y1 + 23], fill=(8, 10, 13))

    # Fingers (over the held object -> looks grasped).
    for index, ly in enumerate((-46, -16, 16, 46)):
        start = local(52, ly * 0.82)
        if hand.grip > 0.55:
            end = local(96 - abs(ly) * 0.18, ly * 0.22)
        else:
            end = local(148 + (8 if index == 1 else 0), ly * 1.14)
        _capsule(draw, start, end, 13, SKIN, SKIN_DARK, 2)
        nail = local(144 + (6 if index == 1 else 0), ly * 1.1) if hand.grip <= 0.55 else local(92 - abs(ly) * 0.2, ly * 0.2)
        draw.ellipse([nail[0] - 5, nail[1] - 4, nail[0] + 5, nail[1] + 4], fill=(238, 203, 170))

    # Thumb.
    thumb_start = local(-4, -54)
    if hand.grip > 0.55:
        thumb_end = local(52, -66)
    else:
        thumb_end = local(64, -92)
    _capsule(draw, thumb_start, thumb_end, 16, SKIN, SKIN_DARK, 2)

    # Soft wrist shadow.
    draw.arc([wx - 52, wy + 36, wx + 52, wy + 105], 200, 340, fill=(168, 112, 77), width=2)


# --------------------------------------------------------------------------- #
# HUD / frame composition
# --------------------------------------------------------------------------- #


def _draw_hud(draw: ImageDraw.ImageDraw, t: float) -> None:
    draw.rectangle([0, 0, SCREEN_W, 44], fill=(7, 11, 16))
    draw.ellipse([22, 15, 34, 27], fill=(237, 76, 76))
    minutes = int(t // 60)
    seconds = t - minutes * 60
    draw.text((44, 13), f"REC  {minutes:02d}:{seconds:04.1f}", font=FONT_14, fill=(226, 235, 243))
    draw.text((178, 13), "EGO-01 / KITCHEN COUNTER", font=FONT_14, fill=(126, 143, 156))
    right = "CHRONICLE  //  1280x720  /  15 FPS"
    width = draw.textlength(right, font=FONT_14)
    draw.text((SCREEN_W - width - 24, 13), right, font=FONT_14, fill=(126, 143, 156))
    draw.line([0, 44, SCREEN_W, 44], fill=(36, 48, 58), width=1)


def render_frame(t: float) -> Image.Image:
    t = clamp(t, 0.0, DURATION)
    img = Image.new("RGB", (SCREEN_W, SCREEN_H), (10, 16, 22))
    draw = ImageDraw.Draw(img)
    _draw_room(img, draw, t)

    # Shelf item.
    _draw_wallet(draw, t)

    # Counter objects. Held objects are drawn with the hand.
    if t < HELD_KEYS[1] and not held("keys", t):
        _draw_keys(draw, t)
    if not held("phone", t):
        _draw_phone(draw, t)
    _draw_bottle(draw, t)
    _draw_backpack(draw, t)
    _draw_tote(draw, t)
    _draw_notebook(draw, t)
    _draw_mug(draw, t)

    hand = hand_at(t)
    if hand is not None:
        _draw_hand(img, hand, t)

    _draw_hud(draw, t)
    arr = np.asarray(img, dtype=np.float32)
    arr *= _VIGNETTE
    arr[:, :, 0] *= 1.035
    arr[:, :, 2] *= 0.985
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


# --------------------------------------------------------------------------- #
# Crop helpers for the drawer
# --------------------------------------------------------------------------- #


def crop_region(t: float, box: list[float] | None, *, pad: int = 18) -> Image.Image:
    frame = render_frame(t)
    if box is None:
        return frame
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = clamp(x1 - pad, 0, SCREEN_W - 4)
    y1 = clamp(y1 - pad, 0, SCREEN_H - 4)
    x2 = clamp(x2 + pad, x1 + 4, SCREEN_W)
    y2 = clamp(y2 + pad, y1 + 4, SCREEN_H)
    return frame.crop((int(x1), int(y1), int(x2), int(y2)))


def save_crop(t: float, name: str, path: Path, *, pad: int = 18) -> None:
    box = screen_bbox(name, t)
    image = crop_region(t, box, pad=pad)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.width < 40 or image.height < 40:
        image = image.resize((max(80, image.width * 2), max(80, image.height * 2)))
    image.save(path)


def save_strip(t: float, name: str, path: Path) -> None:
    box = screen_bbox(name, t)
    if box is None:
        base = BASE_BOXES.get(name)
        if base is not None:
            cam = camera_x(t)
            box = [base[0] - cam, base[1], base[2] - cam, base[3]]
    panels = []
    labels = ("PRE", "CONTACT", "POST")
    if box is not None:
        bx1, bx2 = sorted((box[0], box[2]))
        by1, by2 = sorted((box[1], box[3]))
        width = max((bx2 - bx1) + 240, ((by2 - by1) + 240) * 16 / 9)
        height = width * 9 / 16
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        strip_box = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]
    else:
        strip_box = [0, 0, SCREEN_W, SCREEN_H]

    for offset, label in zip((-0.55, 0.0, 0.55), labels, strict=False):
        frame_t = clamp(t + offset, 0, DURATION)
        image = crop_region(frame_t, strip_box, pad=0).convert("RGB")
        image = image.resize((360, 202), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, 112, 24], fill=(7, 11, 16))
        draw.text((8, 5), f"{label}  {frame_t:04.1f}s", font=FONT_12, fill=(226, 235, 243))
        panels.append(image)
    strip = Image.new("RGB", (360 * 3 + 8, 202), (7, 11, 16))
    for index, panel in enumerate(panels):
        strip.paste(panel, (index * 364, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(path)

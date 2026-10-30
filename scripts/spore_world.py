"""Procedural egocentric world for SPORE perception experiments.

The world is a top-down-ish tabletop viewed by a smoothly moving, slightly
shaking camera.  A scripted hand enters, picks up objects and moves them.  The
renderer knows the exact bounding box of every object at every frame, which
makes it a controlled testbed for evidence retention: did the perceiver keep a
crop of the object *before* it was asked about it?

Nothing here is a benchmark of video understanding.  It is a microscope for
perception mechanisms (gating, proposal quality, capsule matching, late
binding) where the ground truth is cheap and exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from chronicle.types import BBox

HORIZON = 220
WALL = (196, 198, 202)


def _draw_hexnut(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    r = max(5, s // 2)
    points = np.array(
        [
            [x + int(r * math.cos(a)), y + int(r * math.sin(a))]
            for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [points], color)
    cv2.circle(img, (x, y), max(2, int(r * 0.42)), (38, 40, 42), -1)
    return BBox(x - r, y - r, x + r, y + r)


def _draw_band(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    r = max(5, s // 2)
    cv2.ellipse(img, (x, y), (r, int(r * 0.72)), 0, 0, 360, color, -1)
    cv2.ellipse(img, (x, y), (max(2, int(r * 0.48)), max(2, int(r * 0.32))), 0, 0, 360, (45, 45, 45), -1)
    return BBox(x - r, y - int(r * 0.72), x + r, y + int(r * 0.72))


def _draw_tag(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    w, h = s, max(6, int(s * 0.68))
    cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, -1)
    cv2.circle(img, (x, y - h // 2 + max(3, h // 5)), max(2, h // 6), (40, 40, 40), -1)
    return BBox(x - w // 2, y - h // 2, x + w // 2, y + h // 2)


def _draw_clamp(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    r = max(6, s // 2)
    thickness = max(4, s // 5)
    cv2.ellipse(img, (x, y), (r, r), 0, 40, 320, color, thickness)
    cv2.rectangle(img, (x + r - thickness, y), (x + r + thickness // 2, y + r), color, -1)
    return BBox(x - r, y - r, x + r + thickness, y + r)


def _draw_lens(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    r = max(6, s // 2)
    cv2.circle(img, (x, y), r, color, -1)
    cv2.circle(img, (x, y), max(3, int(r * 0.62)), (30, 34, 36), -1)
    cv2.circle(img, (x, y), max(2, int(r * 0.3)), (155, 190, 210), -1)
    return BBox(x - r, y - r, x + r, y + r)


def _draw_cap(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    r = max(6, s // 2)
    cv2.ellipse(img, (x, y + r // 4), (r, int(r * 0.62)), 0, 180, 360, color, -1)
    cv2.rectangle(img, (x - r, y + r // 4), (x + r, y + r // 2), color, -1)
    return BBox(x - r, y - r // 2, x + r, y + r // 2)


def _draw_token(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    r = max(6, s // 2)
    cv2.circle(img, (x, y), r, color, -1)
    cv2.rectangle(img, (x - max(2, r // 4), y - r), (x + max(2, r // 4), y - r + r // 2), (45, 45, 45), -1)
    return BBox(x - r, y - r, x + r, y + r)


def _draw_cube(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    r = max(6, s // 2)
    top = tuple(min(255, int(c * 1.18)) for c in color)
    side = tuple(int(c * 0.72) for c in color)
    front = BBox(x - r, y - r // 2, x + r, y + r)
    cv2.rectangle(img, front.int_xyxy[:2], front.int_xyxy[2:], color, -1)
    top_pts = np.array([[x - r, y - r // 2], [x - r // 2, y - r], [x + r, y - r], [x + r // 2, y - r // 2]], np.int32)
    side_pts = np.array([[x + r // 2, y - r // 2], [x + r, y - r], [x + r, y + r - r // 2], [x + r // 2, y + r]], np.int32)
    cv2.fillPoly(img, [top_pts], top)
    cv2.fillPoly(img, [side_pts], side)
    return BBox(x - r, y - r, x + r, y + r)


def _draw_book(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    w, h = int(s * 1.3), int(s * 0.5)
    cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, -1)
    cv2.line(img, (x - w // 2, y), (x + w // 2, y), (230, 230, 230), max(1, h // 8))
    return BBox(x - w // 2, y - h // 2, x + w // 2, y + h // 2)


def _draw_mug(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    r = max(6, s // 2)
    cv2.rectangle(img, (x - r, y - r), (x + r, y + r), color, -1)
    cv2.ellipse(img, (x, y - r), (r, max(2, r // 3)), 0, 0, 360, color, -1)
    cv2.ellipse(img, (x + r, y), (max(3, r // 2), max(3, r // 2)), 0, -80, 80, color, max(2, r // 4))
    return BBox(x - r, y - r - r // 3, x + r + r // 2, y + r)


def _draw_bottle(img: np.ndarray, x: int, y: int, s: int, color: tuple[int, int, int]) -> BBox:
    r = max(4, s // 3)
    h = int(s * 0.9)
    cv2.rectangle(img, (x - r, y - h), (x + r, y + r), color, -1)
    cv2.rectangle(img, (x - r // 2, y - h - r), (x + r // 2, y - h), color, -1)
    return BBox(x - r, y - h - r, x + r, y + r)


SHAPES = {
    "hexnut": _draw_hexnut,
    "band": _draw_band,
    "tag": _draw_tag,
    "clamp": _draw_clamp,
    "lens": _draw_lens,
    "cap": _draw_cap,
    "token": _draw_token,
    "cube": _draw_cube,
    "book": _draw_book,
    "mug": _draw_mug,
    "bottle": _draw_bottle,
}


@dataclass
class WorldObject:
    name: str
    shape: str
    color: tuple[int, int, int]
    size: int
    home: np.ndarray
    touched: bool = True
    action_index: int = -1

    # Filled by the world at run time.
    target: np.ndarray = field(default_factory=lambda: np.zeros(2))
    action_start: float = 0.0


@dataclass
class SynthEvent:
    time: float
    name: str
    question: str
    bbox: BBox


class SynthWorld:
    """Deterministic tabletop world with scripted pick-and-place actions."""

    def __init__(
        self,
        *,
        seed: int = 7,
        width: int = 640,
        height: int = 480,
        duration: float = 30.0,
        fps: float = 12.0,
        hand: bool = True,
        objects: list[WorldObject] | None = None,
    ) -> None:
        self.seed = seed
        self.width = width
        self.height = height
        self.duration = duration
        self.fps = fps
        self.hand_enabled = hand
        self.rng = np.random.default_rng(seed)
        self.margin_x = 180
        self.margin_y = 90
        self.world_w = width + 2 * self.margin_x
        self.world_h = height + 2 * self.margin_y
        self.objects = objects or self._default_objects()
        self._layout_actions()
        self.background = self._make_background()
        self._noise = self.rng.normal(0, 1.2, (height, width, 3)).astype(np.float32)
        self.events = self._make_events()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _default_objects(self) -> list[WorldObject]:
        specs = [
            ("hexnut", "hexnut", (40, 40, 205), 46, (330, 515), True),
            ("band", "band", (30, 170, 60), 54, (470, 520), True),
            ("tag", "tag", (215, 110, 30), 44, (610, 510), True),
            ("clamp", "clamp", (190, 70, 180), 50, (750, 520), True),
            ("lens", "lens", (35, 190, 215), 42, (340, 405), True),
            ("cap", "cap", (200, 90, 40), 46, (500, 410), True),
            ("coin", "token", (80, 60, 200), 40, (660, 405), False),
            ("cube", "cube", (70, 170, 70), 44, (790, 415), False),
            ("mug", "mug", (60, 60, 190), 48, (410, 455), False),
            ("bottle", "bottle", (150, 190, 40), 44, (700, 455), False),
            ("book", "book", (90, 70, 50), 46, (280, 450), False),
        ]
        return [
            WorldObject(name=name, shape=shape, color=color, size=size, home=np.array(pos, float), touched=touched, action_index=i)
            for i, (name, shape, color, size, pos, touched) in enumerate(specs)
        ]

    def _layout_actions(self) -> None:
        touched = [obj for obj in self.objects if obj.touched]
        for slot, obj in enumerate(touched):
            start = 3.0 + slot * 3.9
            obj.action_start = start
            direction = np.array([40.0 + 11.0 * slot, 18.0 - 6.0 * (slot % 3)])
            obj.target = np.clip(obj.home + direction, [60, HORIZON + 40], [self.world_w - 60, self.world_h - 40])

    def _make_background(self) -> np.ndarray:
        img = np.zeros((self.world_h, self.world_w, 3), np.uint8)
        # Wall with a soft vertical gradient and a couple of frames.
        for y in range(HORIZON):
            shade = 205 + int(24 * (1.0 - y / max(1, HORIZON)))
            img[y, :] = (shade - 8, shade, shade + 2)
        img[HORIZON:, :] = (58, 96, 148)
        # Table planks receding towards the horizon.
        for k in range(14):
            y = HORIZON + int((self.world_h - HORIZON) * (k / 14.0) ** 1.35)
            cv2.line(img, (0, y), (self.world_w, y), (48, 82, 132), 1)
        for x in range(0, self.world_w, 110):
            cv2.line(img, (x, HORIZON), (x - 30, self.world_h), (52, 88, 138), 1)
        # A window and a poster give static texture and contrast edges.
        cv2.rectangle(img, (60, 40), (240, 190), (150, 176, 190), -1)
        cv2.rectangle(img, (60, 40), (240, 190), (120, 140, 150), 3)
        cv2.line(img, (150, 40), (150, 190), (120, 140, 150), 3)
        cv2.rectangle(img, (self.world_w - 260, 30), (self.world_w - 80, 170), (170, 160, 130), -1)
        cv2.rectangle(img, (self.world_w - 260, 30), (self.world_w - 80, 170), (140, 130, 100), 3)
        cv2.circle(img, (self.world_w - 170, 100), 38, (120, 140, 80), -1)
        # Static clutter on the table.
        for _ in range(22):
            x = int(self.rng.uniform(40, self.world_w - 40))
            y = int(self.rng.uniform(HORIZON + 50, self.world_h - 30))
            if self._near_object(x, y, self.objects, 75):
                continue
            r = int(self.rng.uniform(4, 10))
            color = tuple(int(c) for c in self.rng.integers(60, 190, 3))
            if self.rng.random() < 0.5:
                cv2.circle(img, (x, y), r, color, -1)
            else:
                cv2.rectangle(img, (x - r, y - r), (x + r, y + r), color, -1)
        # Slight vignette on the table.
        return img

    @staticmethod
    def _near_object(x: float, y: float, objects: list[WorldObject], margin: float) -> bool:
        for obj in objects:
            if abs(x - obj.home[0]) < margin and abs(y - obj.home[1]) < margin:
                return True
        return False

    def _make_events(self) -> list[SynthEvent]:
        events: list[SynthEvent] = []
        for obj in self.objects:
            if obj.touched:
                when = obj.action_start + 3.0
            else:
                when = 2.0
            frame, gt = self.render(when)
            if obj.name in gt:
                events.append(
                    SynthEvent(
                        time=when,
                        name=obj.name,
                        question=f"Where is the {obj.name}?",
                        bbox=gt[obj.name],
                    )
                )
        return events

    # ------------------------------------------------------------------
    # timeline
    # ------------------------------------------------------------------
    def camera(self, t: float) -> tuple[float, float]:
        cx = self.margin_x + 20 + 42 * math.sin(2 * math.pi * t / 17.0 + self.seed * 0.3)
        cx += 12 * math.sin(2 * math.pi * t / 6.1 + 0.7)
        cy = self.margin_y + 15 + 26 * math.sin(2 * math.pi * t / 21.0 + 1.3)
        cy += 8 * math.sin(2 * math.pi * t / 5.3)
        cx = float(np.clip(cx, 0, self.world_w - self.width))
        cy = float(np.clip(cy, 0, self.world_h - self.height))
        # Handheld jitter, deterministic in frame index.
        frame_index = int(round(t * self.fps))
        jitter = np.random.default_rng(self.seed * 100003 + frame_index).normal(0, 0.9, 2)
        return float(cx + jitter[0]), float(cy + jitter[1])

    def object_state(self, obj: WorldObject, t: float) -> tuple[np.ndarray, bool]:
        """Return (world position, held) at time ``t``."""
        if not obj.touched:
            return obj.home, False
        start = obj.action_start
        approach = 0.8
        carry = 1.8
        if t <= start:
            return obj.home, False
        if t < start + approach:
            return obj.home, False
        if t < start + approach + carry:
            progress = (t - start - approach) / carry
            progress = progress * progress * (3 - 2 * progress)
            return obj.home + (obj.target - obj.home) * progress, True
        return obj.target, False

    def hand_position(self, t: float) -> tuple[np.ndarray, float, bool]:
        """Return (palm world position, scale, visible)."""
        for obj in self.objects:
            if not obj.touched:
                continue
            start = obj.action_start
            if t < start - 0.15 or t > start + 3.25:
                continue
            pos, _ = self.object_state(obj, t)
            scale = self._scale(pos[1])
            hover = np.array([0.0, 62.0 * scale])
            if t < start + 0.8:
                u = (t - (start - 0.15)) / 0.95
                u = float(np.clip(u, 0.0, 1.0)) ** 0.7
                entry = obj.home + np.array([0.0, 210.0 * scale])
                palm = entry + (obj.home + hover - entry) * u
            elif t < start + 2.6:
                palm = pos + hover
            else:
                u = float(np.clip((t - (start + 2.6)) / 0.65, 0.0, 1.0))
                exit_pos = obj.target + hover + np.array([0.0, 210.0 * scale])
                palm = (obj.target + hover) + (exit_pos - (obj.target + hover)) * u
            return palm, scale, True
        return np.zeros(2), 1.0, False

    def _scale(self, y: float) -> float:
        return 0.55 + 0.75 * float(np.clip((y - HORIZON) / (self.world_h - HORIZON), 0, 1))

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _render_hand(self, view: np.ndarray, palm: np.ndarray, scale: float, camera: tuple[float, float]) -> None:
        cx, cy = camera
        px, py = int(round(palm[0] - cx)), int(round(palm[1] - cy))
        skin = (120, 165, 220)
        dark = (98, 138, 190)
        if py < -80 or px < -80 or px > view.shape[1] + 80 or py > view.shape[0] + 80:
            return
        wrist = (px + int(8 * scale), py + int(230 * scale))
        cv2.line(view, wrist, (px, py), skin, max(12, int(58 * scale)))
        cv2.ellipse(view, (px, py), (int(34 * scale), int(42 * scale)), 0, 0, 360, skin, -1)
        # Four fingers reaching up/forward and a thumb to the side.
        for i, offset in enumerate((-0.55, -0.18, 0.18, 0.55)):
            base = (px + int(offset * 30 * scale), py - int(24 * scale))
            tip = (base[0] + int(offset * 16 * scale), base[1] - int((68 - abs(offset) * 14) * scale))
            cv2.line(view, base, tip, skin if i % 2 else dark, max(4, int(13 * scale)))
            cv2.circle(view, tip, max(3, int(7 * scale)), skin, -1)
        thumb_base = (px - int(28 * scale), py + int(6 * scale))
        thumb_tip = (px - int(58 * scale), py - int(28 * scale))
        cv2.line(view, thumb_base, thumb_tip, skin, max(5, int(15 * scale)))
        cv2.circle(view, thumb_tip, max(3, int(8 * scale)), skin, -1)
        # Knuckle shading.
        cv2.ellipse(view, (px, py + int(10 * scale)), (int(24 * scale), int(10 * scale)), 0, 0, 360, dark, max(1, int(3 * scale)))

    def render(self, t: float, *, with_hand: bool | None = None) -> tuple[np.ndarray, dict[str, BBox]]:
        camera = self.camera(t)
        world = self.background.copy()
        positions: dict[str, np.ndarray] = {}
        held: dict[str, bool] = {}
        for obj in self.objects:
            pos, is_held = self.object_state(obj, t)
            positions[obj.name] = pos
            held[obj.name] = is_held
        world_boxes: dict[str, BBox] = {}
        for obj in sorted(self.objects, key=lambda item: positions[item.name][1]):
            x, y = int(round(positions[obj.name][0])), int(round(positions[obj.name][1]))
            size = int(round(obj.size * self._scale(positions[obj.name][1])))
            world_boxes[obj.name] = SHAPES[obj.shape](world, x, y, size, obj.color)
        # Crop to the moving view.
        cx, cy = camera
        x0, y0 = int(round(cx)), int(round(cy))
        x0 = int(np.clip(x0, 0, self.world_w - self.width))
        y0 = int(np.clip(y0, 0, self.world_h - self.height))
        view = world[y0 : y0 + self.height, x0 : x0 + self.width].copy()
        # Ground truth in view coordinates.
        gt: dict[str, BBox] = {}
        for obj in self.objects:
            box = world_boxes[obj.name]
            shifted = BBox(box.x1 - x0, box.y1 - y0, box.x2 - x0, box.y2 - y0)
            clipped = shifted.clipped(self.width, self.height)
            if clipped.area > 0.55 * shifted.area and clipped.area > 120:
                gt[obj.name] = clipped
        hand_visible = self.hand_enabled if with_hand is None else with_hand
        palm, scale, present = self.hand_position(t)
        if hand_visible and present:
            self._render_hand(view, palm, scale, (x0, y0))
        noise = np.random.default_rng(self.seed * 7919 + int(round(t * self.fps))).normal(0, 1.0, view.shape).astype(np.float32)
        view = np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return view, gt

    # ------------------------------------------------------------------
    # streaming / evaluation helpers
    # ------------------------------------------------------------------
    def frames(self, *, fps: float | None = None) -> list[tuple[float, np.ndarray]]:
        rate = fps or self.fps
        count = int(round(self.duration * rate))
        return [(index / rate, self.render(index / rate)[0]) for index in range(count)]

    def events_at(self, t: float, tolerance: float = 0.6) -> list[SynthEvent]:
        return [event for event in self.events if abs(event.time - t) <= tolerance]

"""Shared synthetic-scene helpers for the Role B test suite.

No video and no models: frames are generated geometrically so every threshold in
the state machine can be asserted to the millisecond.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import pytest
from shapely.geometry import Polygon

from chronicle.graph.memory import ChronicleMemory
from chronicle.state_machine import LifecycleEngine
from chronicle.types import (
    CameraMotion,
    FrameContext,
    HandObservation,
    ObjectObservation,
    SupportSurface,
)

DT = 1.0 / 30.0
TABLE_POLY = Polygon([(0, 380), (800, 380), (800, 600), (0, 600)])
CUP_BBOX = (100.0, 300.0, 180.0, 380.0)


def table_surface(
    anchor_id: str = "dining_table",
    label: str = "dining table",
    polygon: Polygon | None = None,
    zone_id: str | None = None,
) -> SupportSurface:
    return SupportSurface(
        anchor_id=anchor_id,
        label=label,
        polygon=polygon or TABLE_POLY,
        zone_id=zone_id,
    )


def hand_at(x: float, y: float, size: float = 80.0) -> HandObservation:
    """Square hand hull whose area overlaps a similarly sized object."""

    points = [
        (x, y),
        (x + size, y),
        (x + size, y + size),
        (x, y + size),
        (x + size / 2, y + size / 2),
    ]
    return HandObservation.from_landmarks(points, hand_id="hand:0")


def cup(
    entity_id: str = "cup",
    label: str = "cup",
    bbox: tuple[float, float, float, float] = CUP_BBOX,
    **kwargs,
) -> ObjectObservation:
    return ObjectObservation(entity_id=entity_id, label=label, bbox=bbox, **kwargs)


def make_frame(
    t: float,
    *,
    objects: Sequence[ObjectObservation] = (),
    hands: Sequence[HandObservation] = (),
    supports: Sequence[SupportSurface] | None = None,
    camera: CameraMotion | None = None,
    zone_id: str | None = "kitchen",
    zone_label: str | None = None,
    frame_size: tuple[int, int] | None = (800, 600),
) -> FrameContext:
    return FrameContext(
        t=t,
        objects=tuple(objects),
        hands=tuple(hands),
        supports=(table_surface(),) if supports is None else tuple(supports),
        camera=camera or CameraMotion(),
        zone_id=zone_id,
        frame_size=frame_size,
        attributes={"zone_label": zone_label} if zone_label else {},
    )


def feed(
    memory: ChronicleMemory,
    t0: float,
    t1: float,
    dt: float,
    objects_fn: Callable[[float], Sequence[ObjectObservation]],
    hands_fn: Callable[[float], Sequence[HandObservation]] | None = None,
    supports_fn: Callable[[float], Sequence[SupportSurface]] | None = None,
    camera_fn: Callable[[float], CameraMotion] | None = None,
    zone_id: str | None = "kitchen",
    zone_label: str | None = None,
) -> list:
    """Feed ``[t0, t1)`` into memory and return every transition fired."""

    transitions = []
    steps = round((t1 - t0) / dt)
    for index in range(steps):
        t = t0 + index * dt
        frame = make_frame(
            t,
            objects=objects_fn(t),
            hands=hands_fn(t) if hands_fn else (),
            supports=supports_fn(t) if supports_fn else None,
            camera=camera_fn(t) if camera_fn else None,
            zone_id=zone_id,
            zone_label=zone_label,
        )
        transitions.extend(memory.observe(frame))
    return transitions


def feed_engine(
    engine: LifecycleEngine,
    t0: float,
    t1: float,
    dt: float,
    objects_fn: Callable[[float], Sequence[ObjectObservation]],
    hands_fn: Callable[[float], Sequence[HandObservation]] | None = None,
    supports_fn: Callable[[float], Sequence[SupportSurface]] | None = None,
    camera_fn: Callable[[float], CameraMotion] | None = None,
    zone_id: str | None = "kitchen",
) -> list:
    transitions = []
    steps = round((t1 - t0) / dt)
    for index in range(steps):
        t = t0 + index * dt
        frame = make_frame(
            t,
            objects=objects_fn(t),
            hands=hands_fn(t) if hands_fn else (),
            supports=supports_fn(t) if supports_fn else None,
            camera=camera_fn(t) if camera_fn else None,
            zone_id=zone_id,
        )
        transitions.extend(engine.update(frame))
    return transitions


def lift_scenario(t0: float = 1.6, speed: float = 180.0):
    """Return object/hand functions for a lift-and-replace motion."""

    def objects(t: float) -> Sequence[ObjectObservation]:
        if t < t0:
            return [cup()]
        if t < t0 + 1.0:
            dy = (t - t0) * speed
            x1, y1, x2, y2 = CUP_BBOX
            return [cup(bbox=(x1, y1 - dy, x2, y2 - dy))]
        return [cup(bbox=(400.0, 300.0, 480.0, 380.0))]

    def hands(t: float) -> Sequence[HandObservation]:
        if t < t0:
            return ()
        if t < t0 + 1.0:
            dy = (t - t0) * speed
            return [hand_at(100.0, 300.0 - dy)]
        return ()

    return objects, hands


def run_lift_scene(memory: ChronicleMemory, dt: float = DT) -> list:
    """Static -> hand contact -> lift -> replace. Returns all transitions."""

    transitions = []
    transitions.extend(feed(memory, 0.0, 1.0, dt, lambda t: [cup()]))
    transitions.extend(
        feed(
            memory,
            1.0,
            1.6,
            dt,
            lambda t: [cup()],
            lambda t: [hand_at(100.0, 300.0)],
        )
    )
    objects_fn, hands_fn = lift_scenario()
    transitions.extend(feed(memory, 1.6, 2.6, dt, objects_fn, hands_fn))
    transitions.extend(
        feed(
            memory,
            2.6,
            3.6,
            dt,
            lambda t: [cup(bbox=(400.0, 300.0, 480.0, 380.0))],
        )
    )
    return transitions


@pytest.fixture()
def memory() -> Iterable[ChronicleMemory]:
    mem = ChronicleMemory(":memory:")
    try:
        yield mem
    finally:
        mem.close()

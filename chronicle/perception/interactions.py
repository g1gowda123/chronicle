"""Hand-object contact geometry and temporal interaction gating.

Role A reports raw, time-qualified interactions; the lifecycle state machine
(Role B) decides whether an object becomes INTERACTED / HELD / CONTAINED.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon, box

from chronicle.perception.config import InteractionConfig
from chronicle.types import (
    BBox,
    HandObservation,
    InteractionEvent,
    InteractionObservation,
    TrackedObject,
)


def bbox_polygon(bbox: BBox) -> Polygon:
    return box(bbox.x1, bbox.y1, bbox.x2, bbox.y2)


def hand_polygon(hand: HandObservation) -> Polygon:
    points = np.asarray(hand.landmarks, dtype=float)
    if len(points) == 0:
        return Polygon()
    if len(points) < 3:
        return MultiPoint([tuple(point) for point in points]).convex_hull.buffer(1.0)
    hull = MultiPoint([tuple(point) for point in points]).convex_hull
    return hull if not hull.is_empty else Polygon()


def intersection_over_smaller(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    intersection = a.intersection(b).area
    smaller = min(a.area, b.area)
    return float(intersection / smaller) if smaller > 0 else 0.0


@dataclass(slots=True)
class HandObjectContact:
    ios: float
    fingertip_inside: bool
    trigger: str
    contact: bool
    distance: float

    def as_dict(self) -> dict[str, object]:
        return {
            "ios": round(self.ios, 4),
            "fingertip_inside": self.fingertip_inside,
            "trigger": self.trigger,
            "contact": self.contact,
            "distance": round(self.distance, 2),
        }


def hand_object_contact(
    hand: HandObservation,
    obj_bbox: BBox,
    config: InteractionConfig,
) -> HandObjectContact:
    obj_poly = bbox_polygon(obj_bbox)
    hand_poly = hand_polygon(hand)
    ios = intersection_over_smaller(hand_poly, obj_poly)

    fingertip_inside = False
    distance = float("inf")
    for tip in hand.fingertips(config.fingertip_indices):
        point = Point(float(tip[0]), float(tip[1]))
        point_distance = float(obj_poly.distance(point))
        if point_distance <= config.fingertip_tolerance_px:
            fingertip_inside = True
        distance = min(distance, point_distance)

    ios_contact = ios >= config.ios_threshold
    trigger = "fingertip+ios" if fingertip_inside and ios_contact else (
        "fingertip" if fingertip_inside else ("ios" if ios_contact else "none")
    )
    return HandObjectContact(
        ios=ios,
        fingertip_inside=fingertip_inside,
        trigger=trigger,
        contact=fingertip_inside or ios_contact,
        distance=distance,
    )


@dataclass
class _ActiveInteraction:
    track_id: int
    label: str
    hand: str
    bbox: BBox
    start_ts: float
    last_contact_ts: float
    max_ios: float
    trigger: str
    emitted: bool = False


class InteractionTracker:
    """Turn per-frame hand/object contact into dwell-gated start/end events."""

    def __init__(self, config: InteractionConfig | None = None) -> None:
        self.config = config or InteractionConfig()
        self._active: dict[tuple[int, str], _ActiveInteraction] = {}

    def reset(self) -> None:
        self._active.clear()

    def update(
        self,
        tracks: list[TrackedObject],
        hands: list[HandObservation],
        timestamp: float,
        frame_index: int = 0,
    ) -> tuple[list[InteractionObservation], list[InteractionEvent]]:
        observations: list[InteractionObservation] = []
        events: list[InteractionEvent] = []
        if not self.config.enabled:
            return observations, events

        ignored = {label.lower() for label in self.config.ignore_labels}
        active_tracks = {
            track.track_id: track
            for track in tracks
            if track.label.lower() not in ignored
        }

        seen: set[tuple[int, str]] = set()
        for track in active_tracks.values():
            for hand in hands:
                hand_key = hand.handedness.lower()
                if not hand_key:
                    continue
                contact = hand_object_contact(hand, track.bbox, self.config)
                key = (track.track_id, hand_key)
                seen.add(key)
                state = self._active.get(key)
                if contact.contact:
                    if state is None:
                        state = _ActiveInteraction(
                            track_id=track.track_id,
                            label=track.label,
                            hand=hand_key,
                            bbox=track.bbox,
                            start_ts=timestamp,
                            last_contact_ts=timestamp,
                            max_ios=contact.ios,
                            trigger=contact.trigger,
                        )
                        self._active[key] = state
                    else:
                        state.last_contact_ts = timestamp
                        state.bbox = track.bbox
                        state.label = track.label
                        state.max_ios = max(state.max_ios, contact.ios)
                        if contact.trigger != "none":
                            state.trigger = contact.trigger
                    duration = max(0.0, timestamp - state.start_ts)
                    dwell_met = duration >= self.config.dwell_seconds
                    if dwell_met and not state.emitted:
                        state.emitted = True
                        events.append(
                            InteractionEvent(
                                kind="start",
                                track_id=track.track_id,
                                label=track.label,
                                hand=hand_key,
                                timestamp=timestamp,
                                start_timestamp=state.start_ts,
                                duration=duration,
                                ios=state.max_ios,
                                trigger=state.trigger,
                                bbox=track.bbox,
                                frame_index=frame_index,
                            )
                        )
                    observations.append(
                        InteractionObservation(
                            track_id=track.track_id,
                            label=track.label,
                            hand=hand_key,
                            contact=True,
                            ios=contact.ios,
                            fingertips_inside=contact.fingertip_inside,
                            active_since=state.start_ts,
                            duration=duration,
                            dwell_met=dwell_met,
                            bbox=track.bbox,
                        )
                    )
                elif state is not None and timestamp - state.last_contact_ts <= self.config.release_grace:
                    # A dropped hand frame should not split one interaction.
                    observations.append(
                        InteractionObservation(
                            track_id=track.track_id,
                            label=track.label,
                            hand=hand_key,
                            contact=False,
                            ios=contact.ios,
                            fingertips_inside=contact.fingertip_inside,
                            active_since=state.start_ts,
                            duration=max(0.0, state.last_contact_ts - state.start_ts),
                            dwell_met=state.emitted,
                            bbox=track.bbox,
                        )
                    )

        # Close interactions whose contact has been absent past the grace
        # window. This covers both "hand still visible but not touching" and
        # "hand/track disappeared entirely".
        for key, state in list(self._active.items()):
            if timestamp - state.last_contact_ts <= self.config.release_grace:
                continue
            if state.emitted:
                events.append(
                    InteractionEvent(
                        kind="end",
                        track_id=state.track_id,
                        label=state.label,
                        hand=state.hand,
                        timestamp=state.last_contact_ts,
                        start_timestamp=state.start_ts,
                        duration=max(0.0, state.last_contact_ts - state.start_ts),
                        ios=state.max_ios,
                        trigger=state.trigger,
                        bbox=state.bbox,
                        frame_index=frame_index,
                    )
                )
            del self._active[key]

        return observations, events

    def flush(self, timestamp: float | None = None, frame_index: int = 0) -> list[InteractionEvent]:
        events: list[InteractionEvent] = []
        for state in list(self._active.values()):
            if state.emitted:
                end = timestamp if timestamp is not None else state.last_contact_ts
                events.append(
                    InteractionEvent(
                        kind="end",
                        track_id=state.track_id,
                        label=state.label,
                        hand=state.hand,
                        timestamp=end,
                        start_timestamp=state.start_ts,
                        duration=max(0.0, end - state.start_ts),
                        ios=state.max_ios,
                        trigger=state.trigger,
                        bbox=state.bbox,
                        frame_index=frame_index,
                    )
                )
        self._active.clear()
        return events


def guess_support_label(label: str) -> bool:
    return label.lower() in {
        "table",
        "desk",
        "counter",
        "countertop",
        "sofa",
        "couch",
        "shelf",
        "floor",
        "bed",
    }

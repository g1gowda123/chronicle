"""The nine-state resilient object lifecycle (spec section 4).

Design invariants
-----------------
* **Wall-clock only.**  Every timer stores a presentation timestamp and compares
  ``t - since >= dwell``.  No frame counters are used anywhere, so the same clip
  sampled at 10 fps or 60 fps yields the same lifecycle and the same anchor
  timestamps (within one sample).
* **Deterministic priority.**  When several conditions hold simultaneously the
  lifecycle resolves CONTAINED > HELD > INTERACTED > VISIBLE_MOVING >
  VISIBLE_STATIC.  Each candidate must be sustained for its own dwell gate
  before the transition fires, which suppresses flicker.
* **Missing tracks are explicit.**  A track that disappears is held in its last
  state for ``occlusion_dwell_s``, then classified as OCCLUDED (camera stable /
  registered occluder) or OUT_OF_VIEW (camera pan / left frustum).  OUT_OF_VIEW
  for ten minutes becomes LOST without ever destroying the node.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon

from chronicle.config import CONTAINER_LABELS, DEFAULT_THRESHOLDS, Thresholds
from chronicle.geometry import (
    any_point_inside,
    bbox_to_polygon,
    center_outside_frame,
    cosine_similarity,
    ema_time_aware,
    ios,
    polyline_velocity,
    support_contact,
)
from chronicle.telemetry import EventBus
from chronicle.types import (
    CameraMotion,
    CameraState,
    EntitySnapshot,
    FrameContext,
    HandObservation,
    ObjectObservation,
    ObjectState,
    StateTransition,
)

_EPS = 1e-9


@dataclass
class _Measurements:
    """Per-frame geometric evidence for one entity."""

    hand_ios: float = 0.0
    hand_contact: bool = False
    hand_id: str | None = None
    fingertip_contact: bool = False
    supported: bool = False
    support_verified: bool = False
    support_id: str | None = None
    support_score: float = 0.0
    container_id: str | None = None
    container_ios: float = 0.0
    abs_speed: float = 0.0
    rel_speed: float = 0.0
    vel_alignment: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hand_ios": round(self.hand_ios, 3),
            "hand_contact": self.hand_contact,
            "fingertip_contact": self.fingertip_contact,
            "supported": self.supported,
            "support_verified": self.support_verified,
            "support_id": self.support_id,
            "container_id": self.container_id,
            "container_ios": round(self.container_ios, 3),
            "rel_speed_px_s": round(self.rel_speed, 2),
            "vel_alignment": round(self.vel_alignment, 3),
        }


@dataclass
class _EntityRuntime:
    entity_id: str
    label: str
    state: ObjectState
    state_since: float
    first_seen: float
    last_seen: float
    bbox: tuple[float, float, float, float] | None = None
    polygon: Polygon | None = None
    centroid_history: deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=256)
    )
    support_id: str | None = None
    container_id: str | None = None
    hand_ios: float = 0.0
    rel_speed: float = 0.0
    zone_id: str | None = None
    candidate: ObjectState | None = None
    candidate_since: float = 0.0
    candidate_reason: str = ""
    missing_since: float | None = None
    out_of_view_since: float | None = None
    reacquired_count: int = 0
    reid_similarity: float | None = None
    seen_count: int = 0

    def hull(self) -> Polygon | None:
        if self.polygon is not None and not self.polygon.is_empty:
            return self.polygon
        if self.bbox is not None:
            return bbox_to_polygon(self.bbox)
        return None


class LifecycleEngine:
    """Deterministic, frame-rate independent state machine.

    The engine is pure: it owns no database, no graph and no I/O.  Feed it
    :class:`FrameContext` objects in presentation-timestamp order and it returns
    the transitions that fired on that frame.
    """

    def __init__(
        self,
        thresholds: Thresholds | None = None,
        *,
        container_labels: Iterable[str] = CONTAINER_LABELS,
        strict_support: bool = False,
        lost_from_occluded: bool = False,
        bus: EventBus | None = None,
    ) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.container_labels = frozenset(label.strip().lower() for label in container_labels)
        self.strict_support = strict_support
        self.lost_from_occluded = lost_from_occluded
        self.bus = bus or EventBus()

        self._entities: dict[str, _EntityRuntime] = {}
        self._hand_history: dict[str, deque[tuple[float, float, float]]] = {}
        self._camera_history: deque[tuple[float, float, float]] = deque(maxlen=256)
        self._camera_positions: dict[float, tuple[float, float]] = {}
        self._last_t: float | None = None

        self._camera_state = CameraState.STATIONARY
        self._camera_mag = 0.0
        self._camera_x = 0.0
        self._camera_y = 0.0
        self._camera_vx = 0.0
        self._camera_vy = 0.0
        self._warned_no_support = False

    # ------------------------------------------------------------------ public
    @property
    def camera_state(self) -> CameraState:
        return self._camera_state

    @property
    def last_timestamp(self) -> float | None:
        return self._last_t

    @property
    def warnings(self) -> dict[str, Any]:
        return {"no_support_surface_seen": self._warned_no_support}

    def update(self, frame: FrameContext) -> list[StateTransition]:
        """Advance the machine to ``frame.t`` and return fired transitions."""

        t = frame.t
        dt = self._advance_clock(t)
        self._update_camera(frame.camera, t, dt)

        for hand in frame.hands:
            self._track_hand(hand, t)

        seen: set[str] = set()
        transitions: list[StateTransition] = []
        for obj in frame.objects:
            if obj.entity_id in seen:
                self.bus.geom(
                    "duplicate observation ignored",
                    t,
                    entity=obj.entity_id,
                    reason="entity observed twice in one frame",
                )
                continue
            seen.add(obj.entity_id)
            transitions.extend(self._update_seen(obj, frame, t))

        for entity_id, runtime in self._entities.items():
            if entity_id in seen:
                continue
            transitions.extend(self._update_missing(runtime, frame, t))

        self._last_t = t
        return transitions

    def snapshot(self) -> list[EntitySnapshot]:
        return [
            self._snapshot(rt) for rt in sorted(self._entities.values(), key=lambda r: r.entity_id)
        ]

    def get(self, entity_id: str) -> EntitySnapshot | None:
        rt = self._entities.get(entity_id)
        return self._snapshot(rt) if rt else None

    def state_of(self, entity_id: str) -> ObjectState | None:
        rt = self._entities.get(entity_id)
        return rt.state if rt else None

    def entities_in(self, *states: ObjectState) -> list[EntitySnapshot]:
        wanted = set(states)
        return [s for s in self.snapshot() if s.state in wanted]

    def seed(
        self,
        entity_id: str,
        label: str,
        state: ObjectState | str,
        *,
        state_since: float,
        first_seen: float,
        last_seen: float,
        bbox: tuple[float, float, float, float] | None = None,
        support_id: str | None = None,
        container_id: str | None = None,
        zone_id: str | None = None,
    ) -> None:
        """Restore a runtime from durable storage after a process restart.

        Timers are re-based on the stored presentation timestamps, so a LOST
        countdown or an occlusion grace period continues across restarts.
        """

        state_enum = ObjectState(state)
        runtime = _EntityRuntime(
            entity_id=entity_id,
            label=label,
            state=state_enum,
            state_since=float(state_since),
            first_seen=float(first_seen),
            last_seen=float(last_seen),
            bbox=bbox,
            polygon=bbox_to_polygon(bbox) if bbox else None,
            support_id=support_id,
            container_id=container_id,
            zone_id=zone_id,
            missing_since=float(last_seen)
            if state_enum in {ObjectState.OCCLUDED, ObjectState.OUT_OF_VIEW, ObjectState.LOST}
            else None,
            out_of_view_since=float(last_seen)
            if state_enum in {ObjectState.OUT_OF_VIEW, ObjectState.LOST}
            else None,
        )
        self._entities[entity_id] = runtime
        self._last_t = max(self._last_t or 0.0, float(last_seen))

    def reset(self) -> None:
        self._entities.clear()
        self._hand_history.clear()
        self._camera_history.clear()
        self._camera_positions.clear()
        self._last_t = None
        self._camera_state = CameraState.STATIONARY
        self._camera_mag = 0.0
        self._camera_x = 0.0
        self._camera_y = 0.0
        self._camera_vx = 0.0
        self._camera_vy = 0.0

    # ------------------------------------------------------------------ clock
    def _advance_clock(self, t: float) -> float:
        if self._last_t is None:
            self._last_t = t
            return 0.0
        dt = t - self._last_t
        if dt < 0:
            self.bus.geom(
                "non-monotonic presentation timestamp clamped",
                t,
                previous_t=self._last_t,
            )
            dt = 0.0
        return dt

    def _update_camera(self, motion: CameraMotion, t: float, dt: float) -> None:
        th = self.thresholds
        magnitude = motion.magnitude
        self._camera_mag = ema_time_aware(self._camera_mag, magnitude, dt, th.camera_smoothing_s)

        # Accumulated rigid translation behaves like a virtual centroid, so the
        # same windowed velocity estimator used for objects also compensates the
        # camera.  Using the identical estimator/window matters: a constant pan
        # then cancels exactly and background-attached objects never look like
        # they are moving independently.
        self._camera_x += motion.dx
        self._camera_y += motion.dy
        self._camera_history.append((t, self._camera_x, self._camera_y))
        self._prune(self._camera_history, t)
        self._camera_positions[t] = (self._camera_x, self._camera_y)
        cutoff = t - 2.0 * th.velocity_window_s
        for stamp in [stamp for stamp in self._camera_positions if stamp < cutoff]:
            del self._camera_positions[stamp]
        samples = [
            sample for sample in self._camera_history if t - sample[0] <= th.velocity_window_s
        ]
        span = samples[-1][0] - samples[0][0] if len(samples) > 1 else 0.0
        if span >= th.velocity_min_span_s:
            self._camera_vx, self._camera_vy, _ = polyline_velocity(samples)
        else:
            self._camera_vx = 0.0
            self._camera_vy = 0.0

        previous = self._camera_state
        if self._camera_mag > th.tau_pan_px:
            self._camera_state = CameraState.PANNING
        elif self._camera_mag < th.tau_still_px:
            self._camera_state = CameraState.STATIONARY
        # between the two thresholds the hysteresis keeps the previous verdict

        if self._camera_state is not previous:
            if self._camera_state is CameraState.PANNING:
                self.bus.geom(
                    "camera pan detected",
                    t,
                    magnitude_px=round(self._camera_mag, 2),
                    tau_pan_px=th.tau_pan_px,
                )
            else:
                self.bus.geom(
                    "camera stationary",
                    t,
                    magnitude_px=round(self._camera_mag, 2),
                    tau_still_px=th.tau_still_px,
                )

    def _track_hand(self, hand: HandObservation, t: float) -> None:
        centre = hand.centroid
        if centre is None:
            return
        history = self._hand_history.setdefault(hand.hand_id, deque(maxlen=256))
        history.append((t, centre[0], centre[1]))
        self._prune(history, t)

    def _prune(self, history: deque[tuple[float, float, float]], t: float) -> None:
        cutoff = t - self.thresholds.velocity_window_s
        while len(history) > 2 and history[0][0] < cutoff:
            history.popleft()

    def _camera_delta(self, t_start: float, t_end: float) -> tuple[float, float] | None:
        """Rigid camera displacement over exactly the object's sample span."""

        start = self._camera_positions.get(t_start)
        end = self._camera_positions.get(t_end)
        if start is None or end is None:
            return None
        return end[0] - start[0], end[1] - start[1]

    # ------------------------------------------------------------------ update
    def _update_seen(
        self, obj: ObjectObservation, frame: FrameContext, t: float
    ) -> list[StateTransition]:
        created = obj.entity_id not in self._entities
        rt = self._entities.get(obj.entity_id)
        if rt is None:
            rt = _EntityRuntime(
                entity_id=obj.entity_id,
                label=obj.label,
                state=ObjectState.NEW,
                state_since=t,
                first_seen=t,
                last_seen=t,
                reid_similarity=obj.reid_similarity,
            )
            self._entities[obj.entity_id] = rt
            self.bus.state(
                "entity minted",
                t,
                entity=obj.entity_id,
                label=obj.label,
                reid_similarity=obj.reid_similarity,
            )

        was_missing = rt.missing_since is not None
        if was_missing:
            rt.reacquired_count += 1
            # Stale samples describe a different spatial epoch; keeping them
            # would fabricate velocity on the first frame back.
            rt.centroid_history.clear()
            self.bus.state(
                "entity re-acquired",
                t,
                entity=rt.entity_id,
                label=rt.label,
                after_s=round(t - (rt.missing_since or t), 3),
            )
        rt.missing_since = None
        rt.out_of_view_since = None
        rt.last_seen = t
        rt.seen_count += 1
        rt.label = obj.label or rt.label
        rt.bbox = obj.bbox
        rt.polygon = obj.hull()
        if frame.zone_id:
            rt.zone_id = frame.zone_id
        if obj.reid_similarity is not None:
            rt.reid_similarity = obj.reid_similarity

        cx, cy = obj.centroid
        rt.centroid_history.append((t, cx, cy))
        self._prune(rt.centroid_history, t)

        measurements = self._measure(obj, frame, rt, t)
        rt.hand_ios = measurements.hand_ios
        rt.rel_speed = measurements.rel_speed
        rt.support_id = measurements.support_id
        rt.container_id = measurements.container_id

        target, reason = self._target_state(rt, measurements)
        first_transition = (
            rt.state is ObjectState.NEW and abs(rt.state_since - rt.first_seen) <= _EPS
        )
        transition = self._maybe_transition(
            rt,
            target,
            t,
            evidence={
                **measurements.as_dict(),
                "camera": self._camera_state.value,
                "camera_magnitude_px": round(self._camera_mag, 2),
                "bbox": [round(v, 2) for v in obj.bbox],
                "track_id": obj.track_id,
                "new_entity": first_transition,
                "observed_first_time": created,
                "reacquired": was_missing,
                "reacquired_count": rt.reacquired_count,
                "reid_similarity": rt.reid_similarity,
                "zone_id": rt.zone_id,
            },
            reason=reason,
            force=was_missing,
        )
        return [transition] if transition else []

    def _update_missing(
        self, rt: _EntityRuntime, frame: FrameContext, t: float
    ) -> list[StateTransition]:
        if rt.state is ObjectState.LOST:
            return []

        # A contained object is expected to be undetectable while the container
        # persists: do not demote it to OCCLUDED.
        if (
            rt.state is ObjectState.CONTAINED
            and rt.container_id
            and self._container_visible(rt.container_id, frame)
        ):
            rt.missing_since = None
            return []

        if rt.missing_since is None:
            rt.missing_since = t

        # LOST is reached from OUT_OF_VIEW after the ten-minute archival window.
        if (
            rt.state is ObjectState.OUT_OF_VIEW
            and rt.out_of_view_since is not None
            and t - rt.out_of_view_since >= self.thresholds.lost_after_s
        ):
            return [self._force_state(rt, ObjectState.LOST, t, {"reason": "archival_timeout"})]
        if (
            self.lost_from_occluded
            and rt.state is ObjectState.OCCLUDED
            and t - rt.missing_since >= self.thresholds.lost_after_s
        ):
            return [
                self._force_state(rt, ObjectState.LOST, t, {"reason": "archival_timeout_occluded"})
            ]

        exit_state, reason, evidence = self._classify_missing(rt, frame)
        transition = self._maybe_transition(
            rt,
            exit_state,
            t,
            evidence={
                **evidence,
                "camera": self._camera_state.value,
                "camera_magnitude_px": round(self._camera_mag, 2),
                "missing_since": rt.missing_since,
                "missing_for_s": round(t - rt.missing_since, 3),
                "last_state": rt.state.value,
                "bbox": [round(v, 2) for v in rt.bbox] if rt.bbox else None,
                "support_id": rt.support_id,
                "container_id": rt.container_id,
            },
            reason=reason,
        )
        return [transition] if transition else []

    # ------------------------------------------------------------- measurement
    def _measure(
        self, obj: ObjectObservation, frame: FrameContext, rt: _EntityRuntime, t: float
    ) -> _Measurements:
        th = self.thresholds
        poly = obj.hull()
        m = _Measurements()

        # --- hand contact -----------------------------------------------------
        best_hand: HandObservation | None = None
        for hand in frame.hands:
            hull = hand.hull()
            if hull is None:
                continue
            contact_ios = ios(hull, poly)
            fingertip = any_point_inside(poly, hand.fingertips())
            if contact_ios > m.hand_ios or fingertip:
                m.hand_ios = max(m.hand_ios, contact_ios)
                m.fingertip_contact = m.fingertip_contact or fingertip
                best_hand = hand
        m.hand_contact = m.fingertip_contact or m.hand_ios >= th.ios_interact
        if best_hand is not None:
            m.hand_id = best_hand.hand_id

        # --- support surface --------------------------------------------------
        if frame.supports:
            supported: list[Any] = []
            for surface in frame.supports:
                contact = support_contact(
                    poly,
                    surface.polygon,
                    surface.anchor_id,
                    tolerance_px=th.support_tolerance_px,
                    min_edge_overlap=th.support_min_edge_overlap,
                    min_ios=th.support_min_ios,
                )
                if contact.supported:
                    supported.append(contact)
            if supported:
                best = max(supported, key=lambda c: c.score)
                m.supported = True
                m.support_verified = True
                m.support_id = best.anchor_id
                m.support_score = best.score
        elif not self.strict_support:
            # No surface detector configured: keep the pipeline alive but flag
            # the missing geometric proof in every transition's evidence.
            m.supported = True
            m.support_verified = False
            if not self._warned_no_support:
                self._warned_no_support = True
                self.bus.geom(
                    "no support surfaces in frame; static gate degraded",
                    t,
                    entity=obj.entity_id,
                    strict_support=self.strict_support,
                )

        # --- containment ------------------------------------------------------
        for other in frame.objects:
            if other.entity_id == obj.entity_id:
                continue
            if other.label not in self.container_labels:
                continue
            container_poly = other.hull()
            contained_ios = ios(poly, container_poly)
            if contained_ios >= th.ios_contained and contained_ios > m.container_ios:
                m.container_ios = contained_ios
                m.container_id = other.entity_id

        # --- velocities (background relative) ---------------------------------
        samples = [
            sample for sample in rt.centroid_history if t - sample[0] <= th.velocity_window_s
        ]
        span = samples[-1][0] - samples[0][0] if len(samples) > 1 else 0.0
        if span < th.velocity_min_span_s:
            # Not enough motion evidence yet (e.g. first frames after a
            # re-acquisition). Do not subtract a camera velocity we cannot
            # corroborate with object displacement.
            abs_speed = 0.0
            m.abs_speed = 0.0
            m.rel_speed = 0.0
        else:
            vx, vy, abs_speed = polyline_velocity(samples)
            camera_delta = self._camera_delta(samples[0][0], samples[-1][0])
            if camera_delta is None:
                cam_vx, cam_vy = self._camera_vx, self._camera_vy
            else:
                cam_vx = camera_delta[0] / span
                cam_vy = camera_delta[1] / span
            m.abs_speed = abs_speed
            m.rel_speed = math.hypot(vx - cam_vx, vy - cam_vy)

        if m.hand_id:
            history = self._hand_history.get(m.hand_id)
            if history:
                hvx, hvy, hspeed = polyline_velocity(
                    [s for s in history if t - s[0] <= th.velocity_window_s]
                )
                if hspeed > th.tau_move_px_s and abs_speed > th.tau_move_px_s:
                    m.vel_alignment = cosine_similarity((vx, vy), (hvx, hvy))

        return m

    def _target_state(self, rt: _EntityRuntime, m: _Measurements) -> tuple[ObjectState | None, str]:
        th = self.thresholds

        if m.container_id is not None:
            return ObjectState.CONTAINED, "enclosed_by_container"

        sticky_held = (
            rt.state is ObjectState.HELD
            and m.hand_ios >= th.ios_held
            and m.rel_speed > th.tau_static_px_s
        )
        if sticky_held:
            return ObjectState.HELD, "held_continuity"

        if (
            m.hand_ios >= th.ios_held
            and m.vel_alignment >= th.held_velocity_cosine
            and m.abs_speed > th.tau_move_px_s
        ):
            return ObjectState.HELD, "lockstep_hand_motion"

        if m.hand_contact:
            return ObjectState.INTERACTED, "hand_contact"

        if m.rel_speed > th.tau_move_px_s and m.hand_ios < th.ios_interact:
            reason = (
                "independent_motion"
                if m.hand_ios < th.ios_hand_free
                else "independent_motion_relaxed"
            )
            return ObjectState.VISIBLE_MOVING, reason

        if m.rel_speed <= th.tau_static_px_s and m.supported:
            return ObjectState.VISIBLE_STATIC, (
                "resting_on_support" if m.support_verified else "resting_no_surface_detector"
            )

        return None, "ambiguous"

    # ------------------------------------------------------------- transitions
    def _dwell(self, state: ObjectState) -> float:
        th = self.thresholds
        return {
            ObjectState.VISIBLE_STATIC: th.static_dwell_s,
            ObjectState.INTERACTED: th.interaction_dwell_s,
            ObjectState.VISIBLE_MOVING: th.moving_dwell_s,
            ObjectState.HELD: th.held_dwell_s,
            ObjectState.CONTAINED: th.contained_dwell_s,
            ObjectState.OCCLUDED: th.occlusion_dwell_s,
            ObjectState.OUT_OF_VIEW: th.occlusion_dwell_s,
            ObjectState.LOST: 0.0,
        }.get(state, 0.0)

    def _maybe_transition(
        self,
        rt: _EntityRuntime,
        target: ObjectState | None,
        t: float,
        *,
        evidence: dict[str, Any],
        reason: str,
        force: bool = False,
    ) -> StateTransition | None:
        if target is None:
            rt.candidate = None
            rt.candidate_reason = ""
            return None
        if target is rt.state:
            rt.candidate = None
            rt.candidate_reason = ""
            return None

        if rt.candidate is not target:
            rt.candidate = target
            rt.candidate_since = t
            rt.candidate_reason = reason

        dwell = 0.0 if force else self._dwell(target)
        sustained = t - rt.candidate_since
        if sustained + _EPS < dwell:
            return None

        return self._commit(
            rt, target, t, evidence, rt.candidate_reason or reason, dwell, sustained
        )

    def _force_state(
        self,
        rt: _EntityRuntime,
        target: ObjectState,
        t: float,
        evidence: dict[str, Any],
    ) -> StateTransition:
        return self._commit(rt, target, t, evidence, evidence.get("reason", ""), 0.0, 0.0)

    def _commit(
        self,
        rt: _EntityRuntime,
        target: ObjectState,
        t: float,
        evidence: dict[str, Any],
        reason: str,
        dwell: float,
        sustained: float,
    ) -> StateTransition:
        previous = rt.state
        rt.state = target
        rt.state_since = t
        rt.candidate = None
        rt.candidate_reason = ""

        if target is ObjectState.OUT_OF_VIEW:
            rt.out_of_view_since = t
        elif target is ObjectState.LOST:
            rt.out_of_view_since = None

        payload = dict(evidence)
        payload.update(
            {
                "reason": reason,
                "dwell_s": round(dwell, 3),
                "sustained_s": round(sustained, 3),
            }
        )
        transition = StateTransition(
            entity_id=rt.entity_id,
            label=rt.label,
            from_state=previous,
            to_state=target,
            t_event=t,
            evidence=payload,
        )
        self.bus.state(
            f"{rt.label}: {previous.value} -> {target.value}",
            t,
            entity=rt.entity_id,
            reason=reason,
            hand_ios=round(rt.hand_ios, 2),
            support=rt.support_id,
            container=rt.container_id,
        )
        return transition

    # ------------------------------------------------------------------ helpers
    def _classify_missing(
        self, rt: _EntityRuntime, frame: FrameContext
    ) -> tuple[ObjectState, str, dict[str, Any]]:
        th = self.thresholds
        hull = rt.hull()
        evidence: dict[str, Any] = {"occluder_id": None, "occluder_ios": 0.0}

        if self._camera_state is CameraState.PANNING:
            return ObjectState.OUT_OF_VIEW, "camera_pan", evidence

        if (
            hull is not None
            and frame.frame_size is not None
            and center_outside_frame(hull, frame.frame_size)
        ):
            return ObjectState.OUT_OF_VIEW, "left_frustum", evidence

        if hull is not None:
            best_ios = 0.0
            best_id: str | None = None
            for other in frame.objects:
                if other.entity_id == rt.entity_id:
                    continue
                value = ios(hull, other.hull())
                if value > best_ios:
                    best_ios, best_id = value, other.entity_id
            for surface in frame.supports:
                value = ios(hull, surface.polygon)
                if value > best_ios:
                    best_ios, best_id = value, surface.anchor_id
            evidence["occluder_id"] = best_id
            evidence["occluder_ios"] = round(best_ios, 3)
            if best_ios >= th.occluder_min_ios:
                return ObjectState.OCCLUDED, "occluder_overlap", evidence

        return ObjectState.OCCLUDED, "line_of_sight_lost", evidence

    @staticmethod
    def _container_visible(container_id: str, frame: FrameContext) -> bool:
        return any(obj.entity_id == container_id for obj in frame.objects)

    def _snapshot(self, rt: _EntityRuntime) -> EntitySnapshot:
        return EntitySnapshot(
            entity_id=rt.entity_id,
            label=rt.label,
            state=rt.state,
            state_since=rt.state_since,
            last_seen=rt.last_seen,
            bbox=rt.bbox,
            support_id=rt.support_id,
            container_id=rt.container_id,
            hand_ios=rt.hand_ios,
            velocity_px_s=rt.rel_speed,
            zone_id=rt.zone_id,
        )

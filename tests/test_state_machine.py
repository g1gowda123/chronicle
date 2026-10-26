"""Nine-state lifecycle: geometry gates, dwell gating and frame-rate independence."""

from __future__ import annotations

import pytest
from conftest import (
    CUP_BBOX,
    DT,
    cup,
    feed_engine,
    hand_at,
    lift_scenario,
    make_frame,
    run_lift_scene,
)

from chronicle.graph.memory import ChronicleMemory
from chronicle.state_machine import LifecycleEngine
from chronicle.types import CameraMotion, ObjectObservation, ObjectState


def test_static_requires_full_wall_clock_dwell() -> None:
    engine = LifecycleEngine()
    early = feed_engine(engine, 0.0, 0.45, DT, lambda t: [cup()])
    assert early == []
    assert engine.state_of("cup") is ObjectState.NEW

    later = feed_engine(engine, 0.45, 0.85, DT, lambda t: [cup()])
    assert engine.state_of("cup") is ObjectState.VISIBLE_STATIC
    assert later[0].t_event == pytest.approx(0.5, abs=0.05)
    assert later[0].evidence["new_entity"] is True
    assert later[0].evidence["dwell_s"] == pytest.approx(0.5)
    assert later[0].evidence["support_verified"] is True
    assert later[0].evidence["support_id"] == "dining_table"


def test_strict_support_never_promotes_unanchored_object() -> None:
    engine = LifecycleEngine(strict_support=True)
    feed_engine(
        engine,
        0.0,
        1.5,
        DT,
        lambda t: [cup()],
        supports_fn=lambda t: (),
    )
    assert engine.state_of("cup") is ObjectState.NEW


def test_missing_surface_detector_degrades_static_gate() -> None:
    engine = LifecycleEngine(strict_support=False)
    transitions = feed_engine(
        engine,
        0.0,
        0.8,
        DT,
        lambda t: [cup()],
        supports_fn=lambda t: (),
    )
    assert engine.state_of("cup") is ObjectState.VISIBLE_STATIC
    assert transitions[0].evidence["support_verified"] is False
    assert transitions[0].evidence["reason"] == "resting_no_surface_detector"
    assert engine.warnings["no_support_surface_seen"] is True


def test_interaction_is_gated_at_300ms() -> None:
    engine = LifecycleEngine()
    feed_engine(engine, 0.0, 1.0, DT, lambda t: [cup()])
    assert engine.state_of("cup") is ObjectState.VISIBLE_STATIC

    feed_engine(
        engine,
        1.0,
        1.2,
        DT,
        lambda t: [cup()],
        lambda t: [hand_at(100.0, 300.0)],
    )
    assert engine.state_of("cup") is ObjectState.VISIBLE_STATIC

    transitions = feed_engine(
        engine,
        1.2,
        1.45,
        DT,
        lambda t: [cup()],
        lambda t: [hand_at(100.0, 300.0)],
    )
    assert engine.state_of("cup") is ObjectState.INTERACTED
    interaction = transitions[-1]
    assert interaction.to_state is ObjectState.INTERACTED
    assert interaction.t_event == pytest.approx(1.3, abs=0.04)
    assert interaction.evidence["hand_ios"] >= 0.6


def test_held_requires_lockstep_hand_velocity() -> None:
    engine = LifecycleEngine()
    feed_engine(engine, 0.0, 1.0, DT, lambda t: [cup()])
    feed_engine(
        engine,
        1.0,
        1.6,
        DT,
        lambda t: [cup()],
        lambda t: [hand_at(100.0, 300.0)],
    )
    objects_fn, hands_fn = lift_scenario()
    transitions = feed_engine(engine, 1.6, 2.6, DT, objects_fn, hands_fn)

    states = [tr.to_state for tr in transitions]
    assert ObjectState.HELD in states
    held = next(tr for tr in transitions if tr.to_state is ObjectState.HELD)
    assert held.t_event < 2.1
    assert held.evidence["vel_alignment"] >= 0.8
    assert held.evidence["hand_ios"] > 0.9


def test_containment_outranks_support_and_persists_while_container_visible() -> None:
    engine = LifecycleEngine()
    backpack = ObjectObservation("backpack", "backpack", (80.0, 280.0, 220.0, 420.0))
    feed_engine(engine, 0.0, 0.35, DT, lambda t: [cup(), backpack])
    assert engine.state_of("cup") is ObjectState.CONTAINED

    silent = feed_engine(engine, 0.35, 1.0, DT, lambda t: [backpack])
    assert [tr for tr in silent if tr.entity_id == "cup"] == []
    assert engine.state_of("cup") is ObjectState.CONTAINED

    transitions = feed_engine(engine, 1.0, 1.5, DT, lambda t: [])
    assert engine.state_of("cup") is ObjectState.OCCLUDED
    assert transitions[-1].to_state is ObjectState.OCCLUDED


def test_occlusion_versus_out_of_view_versus_lost() -> None:
    engine = LifecycleEngine()
    feed_engine(engine, 0.0, 1.0, DT, lambda t: [cup()])

    occluder = ObjectObservation("occluder", "box", (80.0, 280.0, 200.0, 400.0))
    feed_engine(engine, 1.0, 1.5, DT, lambda t: [occluder])
    assert engine.state_of("cup") is ObjectState.OCCLUDED
    occluded = engine.get("cup")
    assert occluded is not None

    # A camera pan reclassifies the absence as OUT_OF_VIEW.
    feed_engine(
        engine,
        1.5,
        2.0,
        DT,
        lambda t: [],
        camera_fn=lambda t: CameraMotion(dx=30.0, dy=0.0),
    )
    assert engine.state_of("cup") is ObjectState.OUT_OF_VIEW

    transitions = engine.update(make_frame(700.0, objects=[], camera=CameraMotion(), supports=()))
    assert transitions[-1].to_state is ObjectState.LOST
    assert transitions[-1].evidence["reason"] == "archival_timeout"


def test_reacquisition_is_immediate_and_keeps_identity() -> None:
    engine = LifecycleEngine()
    feed_engine(engine, 0.0, 1.0, DT, lambda t: [cup()])
    occluder = ObjectObservation("occluder", "box", (80.0, 280.0, 200.0, 400.0))
    feed_engine(engine, 1.0, 1.5, DT, lambda t: [occluder])
    assert engine.state_of("cup") is ObjectState.OCCLUDED

    transitions = feed_engine(engine, 1.5, 1.6, DT, lambda t: [cup()])
    assert engine.state_of("cup") is ObjectState.VISIBLE_STATIC
    reacquired = next(tr for tr in transitions if tr.entity_id == "cup")
    assert reacquired.evidence["reacquired"] is True
    assert len([s for s in engine.snapshot() if s.entity_id == "cup"]) == 1


def test_snapshot_reports_live_measurements() -> None:
    engine = LifecycleEngine()
    feed_engine(engine, 0.0, 0.8, DT, lambda t: [cup()])
    snapshot = engine.get("cup")
    assert snapshot is not None
    assert snapshot.state is ObjectState.VISIBLE_STATIC
    assert snapshot.support_id == "dining_table"
    assert snapshot.bbox == CUP_BBOX
    assert engine.snapshot()[0].entity_id == "cup"


def test_camera_pan_velocity_is_compensated() -> None:
    engine = LifecycleEngine()
    feed_engine(engine, 0.0, 1.0, DT, lambda t: [cup()])
    assert engine.state_of("cup") is ObjectState.VISIBLE_STATIC

    def panned_cup(t: float):
        # A background-attached object moves with the camera from the first pan
        # frame, i.e. the frame at t=1.0 already contains the 30 px translation.
        shift = (t - (1.0 - DT)) * 30.0 / DT
        x1, y1, x2, y2 = CUP_BBOX
        return [cup(bbox=(x1 + shift, y1, x2 + shift, y2))]

    transitions = feed_engine(
        engine,
        1.0,
        1.7,
        DT,
        panned_cup,
        camera_fn=lambda t: CameraMotion(dx=30.0, dy=0.0),
    )
    # The object is glued to the background: zero relative velocity, no motion.
    assert [tr for tr in transitions if tr.entity_id == "cup"] == []
    assert engine.state_of("cup") is ObjectState.VISIBLE_STATIC


def test_framerate_independence() -> None:
    def run(dt: float):
        memory = ChronicleMemory(":memory:")
        try:
            transitions = run_lift_scene(memory, dt=dt)
            anchor = memory.store.terminal_anchor("cup")
            cup_events = [tr for tr in transitions if tr.entity_id == "cup"]
            times = {
                tr.to_state.value: tr.t_event
                for tr in cup_events
                if tr.to_state is not ObjectState.VISIBLE_MOVING
            }
            return (
                [tr.to_state for tr in cup_events],
                times,
                anchor.t_start if anchor else None,
                memory.engine.state_of("cup"),
            )
        finally:
            memory.close()

    fast_states, fast_times, fast_anchor, fast_state = run(1.0 / 30.0)
    slow_states, slow_times, slow_anchor, slow_state = run(1.0 / 10.0)

    # Wall-clock gating: the same conditions fire at the same presentation time
    # regardless of sample rate (within one sample interval).
    for state in (ObjectState.VISIBLE_STATIC, ObjectState.INTERACTED, ObjectState.HELD):
        assert state in fast_times and state in slow_times
        assert abs(fast_times[state] - slow_times[state]) <= 1.0 / 10.0 + 1e-6

    assert fast_state is ObjectState.VISIBLE_STATIC
    assert slow_state is ObjectState.VISIBLE_STATIC
    assert ObjectState.VISIBLE_MOVING in fast_states
    assert fast_states[-1] is slow_states[-1] is ObjectState.VISIBLE_STATIC
    assert fast_anchor is not None and slow_anchor is not None
    assert abs(fast_anchor - slow_anchor) <= 0.12

"""End-to-end Role B flows: zone relocalization, hooks and telemetry."""

from __future__ import annotations

import csv
import io
import json
import time
from types import SimpleNamespace

from conftest import DT, TABLE_POLY, cup, feed, hand_at, make_frame, run_lift_scene

from chronicle.graph.memory import ChronicleMemory
from chronicle.graph.query import QueryResult
from chronicle.integration import ChronicleRuntime
from chronicle.telemetry import TelemetryTag
from chronicle.types import CameraMotion, EventCategory, ObjectState, SupportSurface, TelemetryEvent


def work_desk_surface() -> SupportSurface:
    return SupportSurface(
        anchor_id="work_desk",
        label="work desk",
        polygon=TABLE_POLY,
        zone_id="work_desk",
    )


def test_pan_relocalization_moves_terminal_anchor_to_new_zone(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    assert memory.active_zone == "zone:kitchen"
    assert memory.background.zone_count() == 1

    # Head rotation: the object leaves the frustum while the camera pans.
    feed(
        memory,
        4.0,
        4.8,
        DT,
        lambda t: [],
        camera_fn=lambda t: CameraMotion(dx=30.0, dy=0.0),
    )
    assert memory.engine.state_of("cup") is ObjectState.OUT_OF_VIEW

    transitions = feed(
        memory,
        4.8,
        5.8,
        DT,
        lambda t: [cup(bbox=(500.0, 300.0, 580.0, 380.0))],
        supports_fn=lambda t: (work_desk_surface(),),
        zone_id="work_desk",
        zone_label="work desk",
    )

    states = [tr.to_state for tr in transitions if tr.entity_id == "cup"]
    assert ObjectState.VISIBLE_STATIC in states
    reacquired = next(
        tr
        for tr in transitions
        if tr.entity_id == "cup" and tr.to_state is ObjectState.VISIBLE_STATIC
    )
    assert reacquired.evidence["reacquired"] is True
    assert reacquired.evidence["support_id"] == "work_desk"

    assert memory.active_zone == "zone:work_desk"
    assert memory.background.zone_count() == 2
    assert memory.background.zone_of("work_desk") == "zone:work_desk"

    anchors = memory.store.load_anchors()
    assert len(anchors) == 3
    assert [anchor.anchor_id for anchor in anchors] == [
        "anchor:dining_table",
        "anchor:dining_table",
        "anchor:work_desk",
    ]
    assert [anchor.is_terminal for anchor in anchors] == [False, False, True]
    assert anchors[0].t_end is not None
    assert anchors[-1].t_end is None

    result = memory.query("where is my cup?")
    assert result is not None
    assert result.zone_label == "work desk"
    assert result.terminal_anchor is not None
    assert result.terminal_anchor["label"] == "work desk"
    assert "work desk" in result.fallback_answer()


def test_transition_hook_emits_verified_interaction_candidates(memory: ChronicleMemory) -> None:
    received = []
    for index in range(45):
        frame = make_frame(index * DT, objects=[cup()])
        memory.observe(frame, on_transition=lambda tr, fr: received.append(tr))

    assert received
    assert received[-1].to_state is ObjectState.VISIBLE_STATIC
    assert received[-1].t_event > 0.0


def test_telemetry_channels_are_labeled(memory: ChronicleMemory) -> None:
    seen: list[TelemetryTag] = []
    memory.bus.subscribe(lambda event: seen.append(event.tag))

    run_lift_scene(memory)
    feed(
        memory,
        4.0,
        4.5,
        DT,
        lambda t: [],
        camera_fn=lambda t: CameraMotion(dx=30.0, dy=0.0),
    )

    assert TelemetryTag.GEOM in seen
    assert TelemetryTag.STATE in seen
    assert TelemetryTag.GRAPH in seen
    rendered = [event.format() for event in memory.bus.history()]
    assert any(line.startswith("[STATE]") for line in rendered)
    assert any(line.startswith("[GRAPH]") for line in rendered)


def test_multi_entity_scene_keeps_independent_lifecycles(memory: ChronicleMemory) -> None:
    keys = cup("keys", "keys", (100.0, 300.0, 180.0, 380.0))
    phone = cup("phone", "phone", (400.0, 300.0, 480.0, 380.0))

    feed(memory, 0.0, 1.0, DT, lambda t: [keys, phone])
    assert memory.engine.state_of("keys") is ObjectState.VISIBLE_STATIC
    assert memory.engine.state_of("phone") is ObjectState.VISIBLE_STATIC

    # Only the phone is picked up.
    def objects(t: float):
        dy = max(0.0, (t - 1.0) * 200.0)
        return [
            keys,
            cup("phone", "phone", (400.0, 300.0 - dy, 480.0, 380.0 - dy)),
        ]

    def hands(t: float):
        if t < 1.0:
            return ()
        dy = max(0.0, (t - 1.0) * 200.0)
        return [hand_at(400.0, 300.0 - dy)]

    feed(memory, 1.0, 2.2, DT, objects, hands)

    assert memory.engine.state_of("keys") is ObjectState.VISIBLE_STATIC
    assert memory.engine.state_of("phone") is ObjectState.HELD
    anchors = memory.store.load_anchors()
    assert {anchor.entity_id for anchor in anchors} == {"keys", "phone"}
    assert memory.store.terminal_anchor("keys").t_end is None
    assert memory.store.terminal_anchor("phone").t_end is not None
    assert memory.query("where are my keys?") is not None
    assert memory.query("where is my phone?") is not None


def test_runtime_facade_answers_from_the_team_graph() -> None:
    runtime = ChronicleRuntime(
        ":memory:",
        semantic_enabled=False,
        zone_label="kitchen",
        reset=True,
    )
    try:
        for index in range(30):
            runtime.memory.observe(
                make_frame(index * DT, objects=[cup()], zone_label="kitchen")
            )

        snapshot = runtime.snapshot()
        assert snapshot["stats"]["entities"] >= 1
        assert snapshot["entities"][0]["label"] == "cup"
        assert snapshot["running"] is False

        answer = runtime.ask("where is my cup?")
        assert answer.grounded is True
        assert "cup" in answer.answer.lower()
        assert answer.method == "graph-template"
        assert answer.entity == "cup"
        assert answer.zone == "kitchen"
    finally:
        runtime.close()


def test_runtime_normalizes_role_b_edges_for_role_c() -> None:
    runtime = ChronicleRuntime(":memory:", semantic_enabled=False)
    try:
        result = QueryResult(
            query="where are my keys?",
            entity_id="keys",
            node_id="entity:keys",
            label="keys",
            state="VISIBLE_STATIC",
            state_since=1.0,
            last_seen=8.0,
            match_score=0.95,
            zone_label="work desk",
            nodes=[
                {"node_id": "entity:keys", "label": "keys"},
                {"node_id": "anchor:work_desk", "label": "work desk"},
            ],
            edges=[
                {
                    "relation": "ACTION",
                    "src": "agent:user",
                    "dst": "entity:keys",
                    "t_start": 4.2,
                    "action": "PICK_UP",
                    "confidence": 0.91,
                    "source": "florence-2",
                },
                {
                    "relation": "RESTING_ON",
                    "src": "entity:keys",
                    "dst": "anchor:work_desk",
                    "t_start": 7.5,
                    "confidence": 0.8,
                },
            ],
        )

        context = runtime.graph_context(result)
        facts = {(fact.subject, fact.relation, fact.object_label) for fact in context.facts}
        assert ("USER", "PICK_UP", "keys") in facts
        assert ("keys", "RESTING_ON", "work desk") in facts
        assert context.entity_label == "keys"
        assert context.zone == "work desk"
        assert context.last_seen == 8.0
    finally:
        runtime.close()


def test_runtime_exports_snapshot_json() -> None:
    runtime = ChronicleRuntime(
        ":memory:",
        semantic_enabled=False,
        zone_label="kitchen",
        reset=True,
    )
    try:
        for index in range(30):
            runtime.memory.observe(
                make_frame(index * DT, objects=[cup()], zone_label="kitchen")
            )

        payload = json.loads(runtime.export_snapshot_json())
        assert payload["entities"][0]["label"] == "cup"
        assert payload["stats"]["entities"] >= 1
        assert payload["running"] is False
        assert "verification_stats" in payload["stats"]
        assert "backends" in payload
    finally:
        runtime.close()


def test_runtime_exports_entities_csv() -> None:
    runtime = ChronicleRuntime(
        ":memory:",
        semantic_enabled=False,
        zone_label="kitchen",
        reset=True,
    )
    try:
        for index in range(30):
            runtime.memory.observe(
                make_frame(index * DT, objects=[cup()], zone_label="kitchen")
            )

        text = runtime.export_entities_csv()
        reader = csv.DictReader(io.StringIO(text))
        assert reader.fieldnames == [
            "id",
            "label",
            "state",
            "zone",
            "anchor",
            "container",
            "first_seen",
            "last_seen",
            "span",
            "track_id",
            "confidence",
            "colour",
            "colour_hex",
            "attributes",
        ]
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["id"] == "cup"
        assert rows[0]["label"] == "cup"
        assert rows[0]["zone"] == "kitchen"
        assert rows[0]["state"] == "VISIBLE_STATIC"
    finally:
        runtime.close()


def test_runtime_exports_actions_csv() -> None:
    runtime = ChronicleRuntime(
        ":memory:",
        semantic_enabled=False,
        zone_label="kitchen",
        reset=True,
    )
    try:
        for index in range(30):
            runtime.memory.observe(
                make_frame(index * DT, objects=[cup()], zone_label="kitchen")
            )
        runtime.memory.foreground.add_action(
            "cup",
            "PICK_UP",
            1.5,
            confidence=0.92,
            source="test",
            clip_ref="clip-1",
            evidence={"caption": "the hand picks up the cup"},
        )

        text = runtime.export_actions_csv()
        reader = csv.DictReader(io.StringIO(text))
        assert reader.fieldnames == [
            "id",
            "timestamp",
            "action",
            "entity_id",
            "entity",
            "confidence",
            "source",
            "clip_ref",
            "caption",
        ]
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["action"] == "PICK_UP"
        assert rows[0]["entity"] == "cup"
        assert rows[0]["source"] == "test"
        assert rows[0]["caption"] == "the hand picks up the cup"
    finally:
        runtime.close()


def test_count_groups_detector_label_aliases() -> None:
    runtime = ChronicleRuntime.__new__(ChronicleRuntime)
    runtime.memory = SimpleNamespace(
        foreground=SimpleNamespace(
            entities=lambda: [
                {
                    "label": "joy-con",
                    "state": "VISIBLE_STATIC",
                    "zone_id": "zone:living_room",
                    "last_seen": 3.0,
                },
                {
                    "label": "controller",
                    "state": "VISIBLE_STATIC",
                    "zone_id": "zone:living_room",
                    "last_seen": 4.0,
                },
            ]
        ),
        background=SimpleNamespace(zone_label=lambda zone: "living room"),
        active_zone=None,
    )

    answer = runtime._count_answer("how many controllers are there?", time.monotonic())

    assert answer is not None
    assert answer.answer == "There are 2 controllers in this memory."
    assert answer.grounded is True


def test_perception_telemetry_merges_duplicate_frame_index() -> None:
    runtime = ChronicleRuntime(":memory:", semantic_enabled=False)
    try:
        # Clip payloads can include their own ``frame_index``; the runtime must
        # not pass it to the bus twice (regression for the YOLO-World smoke).
        event = TelemetryEvent(
            category=EventCategory.CLIP,
            message="cup clip",
            presentation_time=1.25,
            frame_index=42,
            data={"frame_index": 7, "clip_id": "clip-1"},
        )
        runtime._on_perception_event(event)

        emitted = runtime.bus.history()[-1]
        assert emitted.tag is TelemetryTag.VLM
        assert emitted.t == 1.25
        assert emitted.data["frame_index"] == 42
        assert emitted.data["clip_id"] == "clip-1"
    finally:
        runtime.close()

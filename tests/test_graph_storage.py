"""Dual-graph topology, persistence invariants and retention policy."""

from __future__ import annotations

from pathlib import Path

from conftest import DT, cup, feed, run_lift_scene

from chronicle.graph.foreground import ForegroundGraph
from chronicle.graph.memory import ChronicleMemory
from chronicle.storage.records import EdgeRecord
from chronicle.types import ObjectState, Relation


def test_dual_graph_structure(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    foreground = memory.foreground.graph

    assert foreground.has_node("entity:cup")
    assert foreground.has_node("agent:user")
    assert memory.background.graph.has_node("zone:kitchen")
    assert memory.background.graph.has_node("anchor:dining_table")
    assert foreground.nodes["entity:cup"]["state"] == ObjectState.VISIBLE_STATIC.value

    assert memory.background.zone_of("dining_table") == "zone:kitchen"
    assert foreground.nodes["entity:cup"]["zone_id"] == "zone:kitchen"

    resting = [
        data
        for _, _, data in foreground.out_edges("entity:cup", data=True)
        if data["relation"] == Relation.RESTING_ON.value
    ]
    assert len(resting) == 2
    assert sum(1 for edge in resting if edge["is_terminal"]) == 1
    assert sum(1 for edge in resting if edge["t_end"] is None) == 1

    transitions = [
        data
        for _, _, data in foreground.out_edges("entity:cup", data=True)
        if data["relation"] == Relation.STATE_TRANSITION.value
    ]
    assert transitions[-1]["to_state"] == ObjectState.VISIBLE_STATIC.value
    assert transitions[-1]["t_end"] is None
    assert all(edge["t_end"] is not None for edge in transitions[:-1])

    assert memory.store.count("state_transitions") == len(transitions)
    assert memory.store.count("anchors") == 2


def test_terminal_anchor_is_latest_but_history_is_retained(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    anchors = memory.store.load_anchors()
    assert len(anchors) == 2
    assert anchors[0].retained is True
    assert anchors[0].is_terminal is False
    assert anchors[0].t_end is not None
    assert anchors[1].retained is True
    assert anchors[1].is_terminal is True
    assert anchors[1].t_end is None

    terminal = memory.store.terminal_anchor("cup")
    assert terminal is not None
    assert terminal.anchor_id == anchors[1].anchor_id
    assert memory.terminal_anchor("cup").t_start == terminal.t_start


def test_sqlite_keeps_exactly_one_terminal_placement(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    placements = [
        edge
        for edge in memory.store.load_edges()
        if edge.relation in {Relation.RESTING_ON.value, Relation.INSIDE.value}
    ]
    assert len(placements) == 2
    terminal = [edge for edge in placements if edge.is_terminal]
    assert len(terminal) == 1
    assert terminal[0].t_end is None
    assert all(edge.t_end is not None for edge in placements if not edge.is_terminal)


def test_relation_switch_keeps_one_terminal_placement(memory: ChronicleMemory) -> None:
    feed(memory, 0.0, 1.0, DT, lambda t: [cup()])
    assert memory.engine.state_of("cup") is ObjectState.VISIBLE_STATIC

    backpack = cup("backpack", "backpack", (80.0, 280.0, 220.0, 420.0))
    inside = cup(bbox=(100.0, 340.0, 180.0, 420.0))
    feed(memory, 1.0, 1.5, DT, lambda t: [inside, backpack])
    assert memory.engine.state_of("cup") is ObjectState.CONTAINED

    placements = [
        edge
        for edge in memory.store.load_edges()
        if edge.relation in {Relation.RESTING_ON.value, Relation.INSIDE.value}
    ]
    assert len(placements) == 2
    terminal = [edge for edge in placements if edge.is_terminal]
    assert len(terminal) == 1
    assert terminal[0].relation == Relation.INSIDE.value
    assert terminal[0].t_end is None
    assert all(edge.t_end is not None for edge in placements if not edge.is_terminal)


def test_foreground_restore_normalizes_inconsistent_terminal_flags() -> None:
    foreground = ForegroundGraph()
    foreground.upsert_entity(
        "cup",
        "cup",
        state=ObjectState.VISIBLE_STATIC,
        state_since=0.0,
        first_seen=0.0,
        last_seen=2.0,
    )
    for row_id, t_start, anchor in ((1, 0.0, "anchor:a"), (2, 1.0, "anchor:b")):
        foreground.add_edge_record(
            EdgeRecord(
                id=row_id,
                src="entity:cup",
                dst=anchor,
                relation=Relation.RESTING_ON.value,
                t_start=t_start,
                t_end=None,
                is_terminal=True,
                props={},
            )
        )

    foreground.normalize_intervals("cup")
    placements = [
        data
        for _, _, data in foreground.graph.out_edges("entity:cup", data=True)
        if data.get("relation") == Relation.RESTING_ON.value
    ]
    assert len(placements) == 2
    assert sum(1 for edge in placements if edge["is_terminal"]) == 1
    assert max(placements, key=lambda edge: edge["t_start"])["is_terminal"] is True
    assert all(
        edge["t_end"] is not None
        for edge in placements
        if not edge["is_terminal"]
    )


def test_retention_purges_trajectories_only(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    observations = memory.store.count("observations")
    assert observations > 0

    deleted = memory.store.purge_trajectories(now=10_000.0, retention_s=600.0)
    assert deleted == observations
    assert memory.store.count("observations") == 0
    assert memory.store.count("anchors") == 2
    assert memory.store.count("entities") == 1


def test_action_multi_edges_and_query_interaction(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    memory.record_action(
        "cup",
        "PICK_UP",
        t=5.0,
        confidence=0.91,
        source="florence-2",
        clip_ref="clip-42",
    )
    recent = memory.foreground.recent_action("cup")
    assert recent is not None
    assert recent["action"] == "PICK_UP"
    assert recent["source"] == "florence-2"

    result = memory.query("where is my cup?")
    assert result is not None
    assert result.last_interaction is not None
    assert result.last_interaction["kind"] == "PICK_UP"
    assert memory.store.count("edges") >= 3


def test_observations_are_throttled(memory: ChronicleMemory) -> None:
    feed(memory, 0.0, 1.0, DT, lambda t: [cup()])
    # 1 s at 30 fps should record ~10 samples, not 30.
    assert 8 <= memory.store.count("observations") <= 14


def test_zone_relocalization_threshold(memory: ChronicleMemory) -> None:
    kitchen = memory.enter_zone("kitchen", 0.0, embedding=[1.0, 0.0, 0.0])
    assert kitchen == "zone:kitchen"
    assert memory.relocalize_zone([0.99, 0.05, 0.0], 1.0) == "zone:kitchen"
    assert memory.relocalize_zone([0.0, 1.0, 0.0], 2.0) is None
    assert memory.store.count("zones") == 1

    created = memory.enter_zone("work desk", 3.0, embedding=[0.0, 1.0, 0.0])
    assert created == "zone:work-desk"
    assert memory.store.count("zones") == 2


def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "chronicle.db"
    first = ChronicleMemory(path)
    run_lift_scene(first)
    anchor_before = first.store.terminal_anchor("cup")
    zone_before = first.active_zone
    assert anchor_before is not None
    first.close()

    second = ChronicleMemory(path)
    try:
        assert second.engine.state_of("cup") is ObjectState.VISIBLE_STATIC
        anchor_after = second.store.terminal_anchor("cup")
        assert anchor_after is not None
        assert anchor_after.t_start == anchor_before.t_start
        assert second.active_zone == zone_before
        assert second.background.zone_of("dining_table") == "zone:kitchen"

        result = second.query("where did I leave the cup?")
        assert result is not None
        assert "dining table" in result.fallback_answer()

        anchors = second.store.count("anchors")
        transitions = feed(
            second,
            10.0,
            11.0,
            DT,
            lambda t: [cup(bbox=(400.0, 300.0, 480.0, 380.0))],
        )
        assert [tr for tr in transitions if tr.entity_id == "cup"] == []
        assert second.store.count("anchors") == anchors
        assert second.store.count("entities") == 1
    finally:
        second.close()


def test_reopen_repairs_duplicate_terminal_placement_rows(tmp_path: Path) -> None:
    path = tmp_path / "chronicle.db"
    first = ChronicleMemory(path)
    run_lift_scene(first)
    with first.store.transaction() as connection:
        connection.execute(
            "UPDATE edges SET is_terminal=1, t_end=NULL WHERE relation='RESTING_ON'"
        )
    first.close()

    second = ChronicleMemory(path)
    try:
        rows = [
            edge
            for edge in second.store.load_edges()
            if edge.relation == Relation.RESTING_ON.value
        ]
        assert len(rows) == 2
        assert sum(1 for edge in rows if edge.is_terminal) == 1
        assert sum(1 for edge in rows if edge.t_end is None) == 1

        foreground_rows = [
            data
            for _, _, data in second.foreground.graph.out_edges(
                "entity:cup", data=True
            )
            if data.get("relation") == Relation.RESTING_ON.value
        ]
        assert sum(1 for edge in foreground_rows if edge["is_terminal"]) == 1
    finally:
        second.close()


def test_visualization_payload_for_dashboard(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    payload = memory.visualization_snapshot()
    node_ids = {node["id"] for node in payload["nodes"]}
    assert {"entity:cup", "agent:user", "anchor:dining_table", "zone:kitchen"} <= node_ids
    nodes = {node["id"]: node for node in payload["nodes"]}
    assert nodes["entity:cup"]["zone_id"] == "zone:kitchen"
    assert nodes["anchor:dining_table"]["zone_id"] == "zone:kitchen"
    assert nodes["zone:kitchen"]["zone_id"] == "zone:kitchen"
    relations = {edge["label"] for edge in payload["edges"]}
    assert Relation.RESTING_ON.value in relations
    assert Relation.IN_ZONE.value in relations
    assert payload["stats"]["entities"] == 1
    assert payload["stats"]["states"]["VISIBLE_STATIC"] == 1

"""GraphRAG retrieval: entity resolution, 2-hop serialization, prompt building."""

from __future__ import annotations

import json

from conftest import DT, cup, feed, hand_at, lift_scenario, make_frame, run_lift_scene

from chronicle.graph.memory import ChronicleMemory
from chronicle.text import canonical_label, match_details, match_score, singular
from chronicle.types import CameraMotion, ObjectObservation, ObjectState


def test_plural_query_resolves_entity(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    result = memory.query("where are my cups?")
    assert result is not None
    assert result.entity_id == "cup"
    assert result.state == ObjectState.VISIBLE_STATIC.value
    assert result.terminal_anchor is not None
    assert result.terminal_anchor["label"] == "dining table"
    assert result.zone_label == "kitchen"
    assert result.match_score >= 0.8


def test_two_hop_subgraph_includes_parent_zone(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    result = memory.query("where is the cup")
    assert result is not None
    node_ids = {node["node_id"] for node in result.nodes}
    assert "entity:cup" in node_ids
    assert "anchor:dining_table" in node_ids
    assert "zone:kitchen" in node_ids
    relations = {edge.get("relation") for edge in result.edges}
    assert "RESTING_ON" in relations
    assert "IN_ZONE" in relations
    assert any(node.get("kind") == "zone" for node in result.nodes)


def test_markdown_table_and_prompt_citations(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    result = memory.query("where is the cup")
    assert result is not None

    table = result.to_table()
    assert "| t (s) |" in table
    assert "RESTING_ON" in table
    assert "STATE_TRANSITION" in table

    prompt = result.to_prompt("Where did I leave the cup?")
    assert "QUESTION: Where did I leave the cup?" in prompt
    assert "CURRENT STATE: VISIBLE_STATIC" in prompt
    assert "TERMINAL ANCHOR:" in prompt
    assert "dining table" in prompt
    assert "kitchen" in prompt
    assert "t=" in prompt
    assert "2-HOP SUBGRAPH" in prompt


def test_answer_context_is_json_serializable(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    result = memory.query("where is the cup")
    assert result is not None
    encoded = json.dumps(result.as_dict())
    assert "entity:cup" in encoded
    assert "embedding" not in encoded


def test_query_miss_returns_none(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    assert memory.query("where is my spaceship?") is None


def test_clip_embedding_fallback_resolution(memory: ChronicleMemory) -> None:
    memory.foreground.upsert_entity(
        "gadget",
        "gadget",
        state=ObjectState.VISIBLE_STATIC,
        state_since=0.0,
        first_seen=0.0,
        last_seen=1.0,
        embedding=[1.0, 0.0, 0.0],
    )
    result = memory.query("where is it", embedder=lambda text: [1.0, 0.0, 0.0])
    assert result is not None
    assert result.entity_id == "gadget"


def test_contained_answer_names_container_and_zone(memory: ChronicleMemory) -> None:
    backpack = ObjectObservation("backpack", "backpack", (80.0, 280.0, 220.0, 420.0))
    feed(memory, 0.0, 0.5, DT, lambda t: [cup(), backpack])
    assert memory.engine.state_of("cup") is ObjectState.CONTAINED

    result = memory.query("where is the cup?")
    assert result is not None
    assert result.state == "CONTAINED"
    assert result.container_label == "backpack"
    answer = result.fallback_answer()
    assert "inside the backpack" in answer
    assert "kitchen" in answer


def test_held_answer(memory: ChronicleMemory) -> None:
    feed(memory, 0.0, 1.0, DT, lambda t: [cup()])
    feed(
        memory,
        1.0,
        1.6,
        DT,
        lambda t: [cup()],
        lambda t: [hand_at(100.0, 300.0)],
    )
    objects_fn, hands_fn = lift_scenario()
    feed(memory, 1.6, 2.0, DT, objects_fn, hands_fn)
    assert memory.engine.state_of("cup") is ObjectState.HELD

    result = memory.query("where is the cup?")
    assert result is not None
    assert result.state == "HELD"
    assert result.fallback_answer().startswith("You are holding the cup")


def test_lost_object_still_resolves_via_terminal_anchor(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    feed(
        memory,
        4.0,
        4.5,
        DT,
        lambda t: [],
        camera_fn=lambda t: CameraMotion(dx=30.0, dy=0.0),
    )
    assert memory.engine.state_of("cup") is ObjectState.OUT_OF_VIEW

    memory.observe(
        make_frame(
            700.0,
            objects=[],
            supports=(),
            zone_id="kitchen",
            camera=CameraMotion(),
        )
    )
    assert memory.engine.state_of("cup") is ObjectState.LOST

    result = memory.query("where is the cup?")
    assert result is not None
    assert result.state == "LOST"
    assert result.terminal_anchor is not None
    assert "dining table" in result.fallback_answer()


def test_summary_table(memory: ChronicleMemory) -> None:
    run_lift_scene(memory)
    rows = memory.state_table()
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "cup"
    assert rows[0]["anchor"] == "dining table"
    assert rows[0]["zone"] == "kitchen"


def test_text_matching_heuristics() -> None:
    assert singular("keys") == "key"
    assert singular("glasses") == "glass"
    assert match_score("where are my keys", "keys") >= 0.8
    assert match_score("where is the key", "keys") >= 0.8
    assert match_score("cup", "coffee mug") == 0.0
    assert match_score("the red mug", "mug") >= 0.7


def test_label_aliases_are_symmetric_and_plural_aware() -> None:
    assert match_score("tv", "television") == 0.85
    assert match_score("televisions", "tv") == 0.85
    assert match_score("couch", "sofa") == 0.85
    assert match_score("where is my purse?", "wallet") == 0.85
    assert match_details("tv", "television")[1] == "label alias"
    assert canonical_label("tv") == canonical_label("television")
    assert canonical_label("purses") == canonical_label("wallet")
    # Deliberately not aliases: the existing retrieval contract treats these
    # as different object categories.
    assert match_score("cup", "coffee mug") == 0.0

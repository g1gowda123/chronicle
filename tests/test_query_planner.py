"""Event planner and object-focus resolution over the episodic graph."""

from __future__ import annotations

from chronicle.graph.memory import ChronicleMemory
from chronicle.graph.planner import answer_query_plan, plan_query
from chronicle.graph.query import focus_phrases
from chronicle.types import ObjectState


def _entity(
    memory: ChronicleMemory,
    entity_id: str,
    label: str,
    *,
    last_seen: float = 10.0,
    embedding: list[float] | None = None,
) -> None:
    memory.foreground.upsert_entity(
        entity_id,
        label,
        state=ObjectState.VISIBLE_STATIC,
        state_since=0.0,
        first_seen=0.0,
        last_seen=last_seen,
        embedding=embedding,
    )


def test_focus_phrases_put_the_object_after_the_subject() -> None:
    phrases = focus_phrases("What did the person do with the joy-con?")
    assert phrases == ["person", "joy-con"]


def test_object_focus_beats_the_subject_person(memory: ChronicleMemory) -> None:
    _entity(memory, "person", "person", last_seen=12.0)
    _entity(memory, "joy1", "joy-con", last_seen=10.0)
    _entity(memory, "joy2", "joy-con", last_seen=11.0)
    _entity(memory, "console", "switch console", last_seen=9.0)

    asking_about_object = memory.query("What did the person do with the joy-con?")
    assert asking_about_object is not None
    assert asking_about_object.label == "joy-con"

    asking_about_console = memory.query("Where is the switch console?")
    assert asking_about_console is not None
    assert asking_about_console.label == "switch console"


def test_planner_detects_event_question_shapes() -> None:
    assert plan_query("What did the person do with the cup?").kind == "activity"
    assert plan_query("When did the person first pick up the cup?").kind == "first_action"
    assert plan_query("How many cups are in the memory?").kind == "count"
    assert plan_query("Where did the plate go?").kind == "previous_location"
    assert plan_query("What is the person slicing?").kind == "activity_object"
    assert plan_query("Where is the cup?") is None


def test_activity_and_first_action_use_timestamped_edges(memory: ChronicleMemory) -> None:
    _entity(memory, "joy1", "joy-con")
    _entity(memory, "joy2", "joy-con")
    memory.foreground.add_action("joy1", "PICK_UP", 4.2, confidence=0.7)
    memory.foreground.add_action("joy2", "PICK_UP", 5.3, confidence=0.7)

    activity = answer_query_plan(
        "What did the person do with the joy-con?", engine=memory.query_engine
    )
    assert activity is not None
    assert "picked up" in activity.answer
    assert activity.method == "event-activity"

    first = answer_query_plan(
        "When did the person first pick up the joy-con?", engine=memory.query_engine
    )
    assert first is not None
    assert "4.2 seconds" in first.answer
    assert first.method == "event-first-action"


def test_count_deduplicates_near_duplicate_tracks(memory: ChronicleMemory) -> None:
    _entity(memory, "joy1", "joy-con", last_seen=9.0, embedding=[1.0, 0.0, 0.0])
    _entity(memory, "joy2", "joy-con", last_seen=10.0, embedding=[0.0, 1.0, 0.0])
    _entity(memory, "joy3", "joy-con", last_seen=11.0, embedding=[0.0, 1.0, 0.01])
    _entity(memory, "ctl", "controller", last_seen=12.0, embedding=[0.0, 1.0, 0.02])

    answer = answer_query_plan(
        "How many joy-con controllers are in the memory?",
        engine=memory.query_engine,
    )
    assert answer is not None
    assert "There are 2 joy-cons" in answer.answer
    assert "4 detector tracks" in answer.answer


def test_previous_location_reports_destination(memory: ChronicleMemory) -> None:
    _entity(memory, "plate", "plate")
    _entity(memory, "bowl", "bowl")
    memory.background.ensure_furniture("anchor:counter", "countertop", "zone:kitchen", 0.0)
    memory.foreground.open_resting("plate", "anchor:counter", 5.0)
    memory.foreground.open_inside("plate", "bowl", 20.0)

    answer = answer_query_plan(
        "Where did the plate go?", engine=memory.query_engine
    )
    assert answer is not None
    assert "inside the bowl" in answer.answer
    assert "20.0 seconds" in answer.answer


def test_missing_object_is_explicitly_unknown(memory: ChronicleMemory) -> None:
    _entity(memory, "person", "person")

    answer = answer_query_plan(
        "How many cups are in the memory?", engine=memory.query_engine
    )
    assert answer is not None
    assert answer.answer == "I don't have that in my memory."
    assert answer.grounded is False

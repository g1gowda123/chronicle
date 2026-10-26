"""Model-free tests for the source benchmark report."""

from __future__ import annotations

import json

from chronicle.benchmark import SourceBenchmark, check_snapshot, format_markdown


def _summary(**overrides):
    payload = {
        "video": "cook.mp4",
        "frames": 10,
        "duration": 0.33,
        "elapsed": 0.25,
        "entities": 1,
        "transitions": 2,
        "actions": 1,
        "zones": 1,
        "furniture": 0,
        "clips": 1,
        "detections": 12,
        "db_path": "runs/benchmark/cook.db",
    }
    payload.update(overrides)
    return payload


def _snapshot(**overrides):
    payload = {
        "entities": [{"id": "cup", "label": "cup", "state": "VISIBLE_STATIC"}],
        "actions": [
            {
                "id": "action:1",
                "timestamp": 0.2,
                "action": "PICK_UP",
                "entity_id": "cup",
                "entity": "cup",
            }
        ],
        "transitions": [
            {
                "id": "transition:1",
                "entity_id": "cup",
                "from_state": "NEW",
                "to_state": "VISIBLE_STATIC",
                "timestamp": 0.1,
            },
            {
                "id": "transition:2",
                "entity_id": "cup",
                "from_state": "VISIBLE_STATIC",
                "to_state": "HELD",
                "timestamp": 0.2,
            },
        ],
        "stats": {"frames": 10},
    }
    payload.update(overrides)
    return payload


def _answers():
    return [
        {"query": "where is the cup?", "answer": "In the kitchen.", "grounded": True},
        {
            "query": "what did I see?",
            "answer": "I don't have that in my memory.",
            "grounded": False,
        },
    ]


def test_consistent_snapshot_passes_every_check() -> None:
    checks = check_snapshot(_summary(), _snapshot(), _answers())
    assert checks == {
        "snapshot_serializable": True,
        "entities_unique": True,
        "actions_reference_entities": True,
        "transitions_reference_entities": True,
        "transition_states_known": True,
        "transition_times_ordered": True,
        "summary_matches_snapshot": True,
        "questions_answered": True,
    }


def test_unknown_state_is_rejected() -> None:
    snapshot = _snapshot()
    snapshot["transitions"][0]["to_state"] = "MAGIC"
    assert check_snapshot(_summary(), snapshot, _answers())["transition_states_known"] is False


def test_dangling_action_is_rejected() -> None:
    snapshot = _snapshot()
    snapshot["actions"][0]["entity_id"] = "ghost"
    checks = check_snapshot(_summary(), snapshot, _answers())
    assert checks["actions_reference_entities"] is False


def test_unordered_transitions_are_rejected() -> None:
    snapshot = _snapshot()
    snapshot["transitions"][1]["timestamp"] = 0.05
    checks = check_snapshot(_summary(), snapshot, _answers())
    assert checks["transition_times_ordered"] is False


def test_duplicate_entity_ids_are_rejected() -> None:
    snapshot = _snapshot()
    snapshot["entities"].append(dict(snapshot["entities"][0]))
    checks = check_snapshot(_summary(), snapshot, _answers())
    assert checks["entities_unique"] is False


def test_summary_mismatch_and_blank_answer_are_rejected() -> None:
    checks = check_snapshot(_summary(frames=9), _snapshot(), [{"answer": " "}])
    assert checks["summary_matches_snapshot"] is False
    assert checks["questions_answered"] is False


def test_markdown_table_lists_metrics_and_checks() -> None:
    result = SourceBenchmark(
        source="cook",
        label="Aria cook",
        video="cook.mp4",
        metadata={"width": 640, "height": 360, "fps": 30.0, "frames": 10, "duration": 0.33},
        summary=_summary(),
        backends={"detector": "color", "hands": "color", "embedder": "color"},
        wall_time=0.5,
        answers=tuple(_answers()),
        checks=check_snapshot(_summary(), _snapshot(), _answers()),
    )
    table = format_markdown([result])
    assert "| cook | 640x360 @ 30.0 fps | 10 | 0.50s | 20.0 fps |" in table
    assert "1/2 | 8/8 |" in table
    assert json.dumps(result.to_dict())  # the report payload is JSON-ready

"""Replay clips without a VLM interaction table must still answer timeline questions."""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.query import QueryEngine
from dashboard.runtime import load_demo_events
from dashboard.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
WALKING_REPLAY = ROOT / "data" / "replays" / "walking.events.jsonl"
pytestmark = pytest.mark.skipif(
    not WALKING_REPLAY.is_file(),
    reason="walking replay fixture is not tracked in a clean checkout",
)

def _walking_engine() -> QueryEngine:
    events, _, _ = load_demo_events(WALKING_REPLAY)
    store = MemoryStore()
    for event in events:
        store.apply(event)
    return QueryEngine(store, llm_backend="none")


def test_time_hint_parses_around() -> None:
    assert QueryEngine._time_hint("what happened around 10s?") == 10.0
    assert QueryEngine._time_hint("what happened at 42.5 seconds?") == 42.5
    assert QueryEngine._time_hint("what happened?") is None


def test_timeline_falls_back_to_state_events() -> None:
    query = _walking_engine()
    result = query.answer("what happened around 10s?", 23.98)
    assert result["route"] == "timeline"
    assert "No VLM-verified interactions" not in result["answer"]
    assert "Around 10.0s" in result["answer"]
    assert result["citations"]
    assert any("person" in str(citation.get("label", "")).lower() for citation in result["citations"])


def test_entity_labels_are_available_early() -> None:
    query = _walking_engine()
    labels = {entity.label for entity in query.store.entities.values()}
    assert "person" in labels
    assert "cup" in labels
    assert "Track" not in labels

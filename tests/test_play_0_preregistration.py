"""Pre-registration sanity checks for the play_0 surface-head test."""

from __future__ import annotations

import json
from pathlib import Path

from chronicle.perception.spore import _known_place

ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "data/qa/play_0_state_45s.json"
MANIFEST_PATH = ROOT / "data/aria_gen2pilot.json"


def test_play_0_manifest_matches_probe_and_run_config() -> None:
    payload = json.loads(MANIFEST_PATH.read_text())
    entry = next(item for item in payload["sequences"] if item["name"] == "play_0")
    qa = json.loads(QA_PATH.read_text())
    assert entry["clip"] == qa["clip"]
    assert entry["start_s"] == qa["window"]["start_s"]
    assert entry["duration_s"] == qa["window"]["duration_s"]
    assert qa["run"]["state_surface_support"] is True
    assert "surface-head" in qa["protocol"]


def test_play_0_probe_is_well_formed() -> None:
    qa = json.loads(QA_PATH.read_text())
    questions = qa["questions"]
    assert len(questions) == 12
    assert len({item["id"] for item in questions}) == len(questions)
    assert {item["kind"] for item in questions} == {"presence", "still", "moved"}
    for item in questions:
        assert item["query"].endswith("?")
        assert item["expected"] in {"yes", "no", "stayed", "moved", "unknown"}
        if item["expected_support"] is not None:
            assert _known_place(item["expected_support"]) == item["expected_support"]
    positives = [item for item in questions if item["expected"] == "yes"]
    absences = [
        item
        for item in questions
        if item["kind"] == "presence" and item["expected"] == "no"
    ]
    assert len(positives) == 5
    assert len(absences) == 5

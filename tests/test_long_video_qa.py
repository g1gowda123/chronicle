"""Structural checks for the long-video HoloAssist QA diagnostic."""

import json
from collections import Counter
from pathlib import Path

QA_PATH = Path(__file__).resolve().parents[1] / "data" / "qa" / "holoassist_z210_40q.json"


def load_qa():
    return json.loads(QA_PATH.read_text(encoding="utf-8"))


def test_long_video_qa_shape():
    data = load_qa()
    questions = data["questions"]
    assert len(questions) == 40
    assert data["video_id"] == "holoassist_z210_257s"
    assert data["vocabulary"]
    assert all(q["query"] and q["answer"] for q in questions)
    assert all(0 <= q["timestamp"] <= 256.988 for q in questions)
    assert len({q["query"] for q in questions}) == len(questions)


def test_long_video_qa_categories():
    questions = load_qa()["questions"]
    counts = Counter(q["category"] for q in questions)
    assert counts == {
        "object_identity": 8,
        "object_location": 6,
        "counting": 6,
        "temporal_order": 6,
        "activity": 6,
        "error_correction": 8,
    }

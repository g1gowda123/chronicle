from __future__ import annotations

from scripts.audit_location import audit


def test_strict_audit_rejects_zone_only_location_answer() -> None:
    questions = [
        {"query": "Where is the knife?", "answer": "countertop"},
        {"query": "Where is the joy-con?", "answer": "Work Table"},
    ]
    rows = [
        {
            "query": "Where is the knife?",
            "category": "object_location",
            "predicted": "Knife is in the Kitchen.",
            "correct": True,
        },
        {
            "query": "Where is the joy-con?",
            "category": "object_location",
            "predicted": "The joy-con is on the Work Table.",
            "correct": True,
        },
    ]

    report = audit(rows, questions, zone_label="Kitchen")

    assert report["total"] == 2
    assert report["official_correct"] == 2
    # "Kitchen" is an accepted alias but not the queried support.
    assert report["strict_correct"] == 1
    assert report["records"][0]["strict_support"] is False
    assert report["records"][1]["strict_support"] is True

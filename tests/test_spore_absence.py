"""Tests for the presence/absence calibration helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from spore_absence import (  # noqa: E402
    occupancy,
    present_in_intervals,
    rank_auc,
    sample_curve,
)


def test_present_in_intervals_boundaries() -> None:
    intervals = [[6.0, 10.0], [41.0, 45.0]]
    assert not present_in_intervals(5.9, intervals)
    assert present_in_intervals(6.0, intervals)
    assert present_in_intervals(8.0, intervals)
    assert present_in_intervals(10.0, intervals)
    assert not present_in_intervals(10.1, intervals)
    assert present_in_intervals(45.0, intervals)
    assert not present_in_intervals(0.0, [])


def test_rank_auc_is_rank_based() -> None:
    assert rank_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert rank_auc([0.1, 0.2], [0.8, 0.9]) == 0.0
    assert rank_auc([0.5], [0.5]) == 0.5
    assert rank_auc([], [0.5]) is None


def test_occupancy() -> None:
    assert occupancy({"intervals": []}) == "absent"
    assert occupancy({"intervals": [[0.0, 45.0]]}) == "45.0s"
    assert occupancy({"intervals": [[6.0, 10.0], [41.0, 45.0]]}) == "8.0s"


def _row(term: str, present: bool, scores: list[float | None], stamps: list[float]) -> dict:
    return {
        "term": term,
        "present": present,
        "intervals": [[0.0, 45.0]] if present else [],
        "contexts": [
            {
                "timestamp": stamp,
                "score": score,
                "present_at": present,
            }
            for stamp, score in zip(stamps, scores)
        ],
    }


def test_sample_curve_separates_present_and_absent() -> None:
    stamps = [float(index) for index in range(8)]
    rows = [
        _row("cup", True, [0.8] * 8, stamps),
        _row("laptop", False, [None] * 8, stamps),
    ]
    curve = sample_curve(rows, floor=0.22, k_values=(2, 4, 8))
    for row in curve:
        assert row["term_false_negative"] == 0.0
        assert row["term_false_positive"] == 0.0
        assert row["context_sensitivity"] == 1.0
        assert (row["context_false_positive"] or 0.0) == 0.0


def test_sample_curve_counts_missed_present_term() -> None:
    stamps = [float(index) for index in range(8)]
    rows = [
        # Only the last context scores; k=4 samples miss it in most phases.
        _row("fork", True, [None] * 7 + [0.7], stamps),
    ]
    curve = {row["k"]: row for row in sample_curve(rows, floor=0.22, k_values=(2, 8))}
    assert curve[8]["term_false_negative"] == 0.0
    assert curve[2]["term_false_negative"] == 0.75
    assert curve[2]["context_sensitivity"] == 0.125

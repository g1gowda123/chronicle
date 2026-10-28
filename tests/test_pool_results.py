from __future__ import annotations

from scripts.pool_results import pool


def test_pool_concatenates_rows_and_recomputes_counts() -> None:
    left = {
        "evaluated_at": "2026-01-01",
        "rows": [
            {"query": "a", "category": "location", "correct": True, "query_latency_ms": 1.0},
            {"query": "b", "category": "location", "correct": False, "query_latency_ms": 3.0},
        ],
    }
    right = {
        "evaluated_at": "2026-01-02",
        "rows": [
            {"query": "c", "category": "counting", "correct": True, "query_latency_ms": 2.0},
        ],
    }

    result = pool([left, right], name="combined")

    assert result["total"] == 3
    assert result["correct"] == 2
    assert result["recall_at_1"] == 2 / 3
    assert result["avg_query_latency_ms"] == 2.0
    assert result["categories"]["location"] == {
        "correct": 1,
        "total": 2,
        "recall_at_1": 0.5,
    }
    assert result["categories"]["counting"]["correct"] == 1

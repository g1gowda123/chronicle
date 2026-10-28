"""Event summaries must stay honest when a clip has no confirmed placement."""

from __future__ import annotations

import unittest

from chronicle.graph.query import QueryResult


def _result(edges: list[dict[str, object]]) -> QueryResult:
    return QueryResult(
        query="what did I do?",
        entity_id="keys",
        node_id="entity:keys",
        label="keys",
        state="INTERACTED",
        state_since=2.8,
        last_seen=3.9,
        match_score=1.0,
        edges=edges,
    )


class EventSummaryTest(unittest.TestCase):
    def test_interaction_without_placement_is_reported(self) -> None:
        result = _result(
            [
                {
                    "relation": "ACTION",
                    "src": "agent:user",
                    "dst": "entity:keys",
                    "t_start": 2.0,
                }
            ]
        )
        summary = result.event_summary()
        self.assertIsNotNone(summary)
        self.assertIn("interacted with the keys", summary or "")
        self.assertIn("no placement was recorded", summary or "")

    def test_static_entity_without_events_has_no_summary(self) -> None:
        result = _result(
            [
                {
                    "relation": "IN_ZONE",
                    "src": "entity:keys",
                    "dst": "zone:kitchen",
                    "t_start": 0.0,
                }
            ]
        )
        self.assertIsNone(result.event_summary())


if __name__ == "__main__":
    unittest.main()

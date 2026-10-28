from __future__ import annotations

import unittest

from chronicle.semantic.config import SemanticConfig
from chronicle.semantic.graphrag import (
    GraphContext,
    GraphRAG,
    extract_citations,
    format_context,
    serialize_subgraph,
    template_answer,
)
from chronicle.semantic.llm import LLMUnavailableError

CONTEXT = {
    "entity": {"id": "keys", "label": "keys", "state": "CONTAINED", "zone": "work desk"},
    "facts": [
        {
            "subject": "keys",
            "relation": "RESTING_ON",
            "object": "work desk",
            "timestamp": 12.0,
            "zone": "work desk",
            "confidence": 0.92,
        },
        {
            "subject": "user",
            "relation": "PICK_UP",
            "object": "keys",
            "timestamp": 14.2,
            "zone": "work desk",
            "confidence": 0.9,
        },
        {
            "subject": "keys",
            "relation": "INSERT_INTO",
            "object": "drawer",
            "timestamp": 16.4,
            "zone": "work desk",
            "confidence": 0.88,
        },
    ],
}

CUP_CONTEXT = {
    "entity": {"id": "cup", "label": "cup", "zone": "kitchen"},
    "facts": [
        {
            "subject": "cup",
            "relation": "RESTING_ON",
            "object": "work desk",
            "timestamp": 5.0,
            "zone": "kitchen",
            "confidence": 0.9,
        },
        {
            "subject": "user",
            "relation": "PICK_UP",
            "object": "cup",
            "timestamp": 8.0,
            "zone": "kitchen",
            "confidence": 0.88,
        },
        {
            "subject": "cup",
            "relation": "PLACED",
            "object": "table",
            "timestamp": 10.0,
            "zone": "kitchen",
            "confidence": 0.86,
        },
    ],
}


class ContextTest(unittest.TestCase):
    def test_mapping_coercion(self) -> None:
        context = GraphContext.from_mapping(CONTEXT)
        self.assertEqual(context.entity_id, "keys")
        self.assertEqual(context.entity_label, "keys")
        self.assertEqual(context.state, "CONTAINED")
        self.assertEqual(context.zone, "work desk")
        self.assertEqual(context.last_seen, 16.4)
        self.assertEqual(len(context.facts), 3)
        self.assertEqual(context.facts[0].object_label, "work desk")

    def test_zone_inferred_from_spatial_fact(self) -> None:
        data = {
            "entity": "mug",
            "facts": [
                {"subject": "mug", "relation": "RESTING_ON", "object": "coffee table", "timestamp": 3.0}
            ],
        }
        context = GraphContext.from_mapping(data)
        self.assertEqual(context.zone, "coffee table")
        self.assertEqual(context.last_seen, 3.0)

    def test_format_context_table(self) -> None:
        text = format_context(GraphContext.from_mapping(CONTEXT))
        self.assertIn("Entity: keys", text)
        self.assertIn("| t (s) | subject | relation | object | zone | confidence |", text)
        self.assertIn("| 16.4 | keys | INSERT_INTO | drawer | work desk | 0.88 |", text)

    def test_fact_limit_keeps_latest(self) -> None:
        context = GraphContext.from_mapping(CONTEXT).limited_facts(2)
        self.assertEqual([fact.timestamp for fact in context], [14.2, 16.4])


class FakeGraph:
    def __init__(self) -> None:
        self.nodes = {
            "keys": {"label": "keys", "state": "CONTAINED", "zone": "work desk"},
            "work_desk": {"label": "work desk"},
            "drawer": {"label": "drawer"},
            "user": {"label": "user"},
        }
        self._edges = [
            ("keys", "work_desk", {"relation": "RESTING_ON", "timestamp": 12.0, "zone": "work desk"}),
            ("user", "keys", {"relation": "PICK_UP", "timestamp": 14.2}),
            ("keys", "drawer", {"relation": "INSERT_INTO", "timestamp": 16.4}),
        ]

    def in_edges(self, node: str, data: bool = True):
        return [edge for edge in self._edges if edge[1] == node]

    def out_edges(self, node: str, data: bool = True):
        return [edge for edge in self._edges if edge[0] == node]


class SerializeSubgraphTest(unittest.TestCase):
    def test_networkx_like_graph(self) -> None:
        context = serialize_subgraph(FakeGraph(), "keys")
        self.assertEqual(context.entity_label, "keys")
        self.assertEqual(context.state, "CONTAINED")
        self.assertEqual(len(context.facts), 3)
        relations = {fact.relation for fact in context.facts}
        self.assertEqual(relations, {"RESTING_ON", "PICK_UP", "INSERT_INTO"})

    def test_mapping_passthrough(self) -> None:
        context = serialize_subgraph(CONTEXT, "keys")
        self.assertEqual(context.entity_id, "keys")
        self.assertEqual(len(context.facts), 3)

    def test_missing_entity(self) -> None:
        with self.assertRaises(KeyError):
            serialize_subgraph(FakeGraph(), "phone")


class TemplateAnswerTest(unittest.TestCase):
    def test_where_query_uses_latest_location(self) -> None:
        answer = template_answer("Where are my keys?", GraphContext.from_mapping(CONTEXT))
        self.assertIn("drawer", answer.answer)
        self.assertIn("[t=16.4s]", answer.answer)
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.citations[0].timestamp, 16.4)

    def test_action_query(self) -> None:
        answer = template_answer("What did I do with the keys?", GraphContext.from_mapping(CONTEXT))
        self.assertIn("keys", answer.answer.lower())
        self.assertIn("[t=", answer.answer)

    def test_where_query_without_location_facts_abstains(self) -> None:
        context = GraphContext.from_mapping(
            {
                "entity": {"id": "keys", "label": "keys", "zone": "work desk"},
                "facts": [
                    {
                        "subject": "keys",
                        "relation": "ACTION",
                        "object": "user",
                        "timestamp": 2.8,
                        "zone": "work desk",
                    }
                ],
            }
        )
        answer = template_answer("Where are my keys?", context)
        self.assertIn("don't have a recorded location", answer.answer)
        self.assertIn("[t=2.8s]", answer.answer)
        self.assertFalse(answer.grounded)

    def test_unknown_entity(self) -> None:
        answer = template_answer("Where is my passport?", GraphContext.from_mapping({"entity": "passport"}))
        self.assertEqual(answer.answer, "I don't have that in my memory.")
        self.assertFalse(answer.grounded)

    def test_zone_is_not_repeated_when_target_is_the_zone(self) -> None:
        context = GraphContext.from_mapping(
            {
                "entity": {"id": "keys", "label": "keys"},
                "facts": [
                    {
                        "subject": "keys",
                        "relation": "RESTING_ON",
                        "object": "desk",
                        "timestamp": 4.2,
                        "zone": "desk",
                    }
                ],
            }
        )
        answer = template_answer("Where are my keys?", context)
        self.assertNotIn("in the desk at", answer.answer)
        self.assertIn("on the desk at [t=4.2s]", answer.answer)


class BeforeQueryTest(unittest.TestCase):
    def test_before_placement_walks_back_to_prior_location(self) -> None:
        answer = template_answer(
            "Where was the cup before it was placed on the table?",
            GraphContext.from_mapping(CUP_CONTEXT),
        )
        self.assertIn("before it was placed on the table", answer.answer.lower())
        self.assertIn("work desk", answer.answer)
        self.assertIn("[t=5.0s]", answer.answer)
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.citations[0].timestamp, 5.0)
        self.assertEqual(answer.citations[0].relation, "RESTING_ON")
        self.assertEqual(answer.citations[0].zone, "kitchen")

    def test_before_pickup_uses_prior_location(self) -> None:
        answer = template_answer(
            "Where was the cup before it was picked up?",
            GraphContext.from_mapping(CUP_CONTEXT),
        )
        self.assertIn("picked up", answer.answer.lower())
        self.assertIn("work desk", answer.answer)
        self.assertEqual(answer.citations[0].timestamp, 5.0)

    def test_prior_to_phrasing(self) -> None:
        answer = template_answer(
            "Where was the cup prior to being placed on the table?",
            GraphContext.from_mapping(CUP_CONTEXT),
        )
        self.assertIn("[t=5.0s]", answer.answer)
        self.assertEqual(answer.citations[0].relation, "RESTING_ON")

    def test_before_accepts_explicit_timestamp(self) -> None:
        answer = template_answer(
            "Where was the cup before t=9s?",
            GraphContext.from_mapping(CUP_CONTEXT),
        )
        self.assertIn("work desk", answer.answer)
        self.assertEqual(answer.citations[0].timestamp, 5.0)

    def test_before_without_earlier_fact_is_unknown(self) -> None:
        context = GraphContext.from_mapping(
            {
                "entity": {"id": "cup", "label": "cup", "zone": "kitchen"},
                "facts": [
                    {
                        "subject": "cup",
                        "relation": "PLACED",
                        "object": "table",
                        "timestamp": 10.0,
                        "zone": "kitchen",
                    }
                ],
            }
        )
        answer = template_answer("Where was the cup before it was placed on the table?", context)
        self.assertIn("don't have an earlier location", answer.answer.lower())
        self.assertFalse(answer.grounded)
        self.assertEqual(answer.citations, ())


class CitationTest(unittest.TestCase):
    def test_extracts_known_timestamp(self) -> None:
        context = GraphContext.from_mapping(CONTEXT)
        citations = extract_citations("It is in the drawer [t=16.4s].", context)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].relation, "INSERT_INTO")
        self.assertEqual(citations[0].zone, "work desk")

    def test_hallucinated_timestamp_is_dropped(self) -> None:
        context = GraphContext.from_mapping(CONTEXT)
        citations = extract_citations("It was moved at [t=99.0s].", context)
        self.assertEqual(citations, ())

    def test_plain_timestamp_format(self) -> None:
        context = GraphContext.from_mapping(CONTEXT)
        citations = extract_citations("Picked up at t=14.2s.", context)
        self.assertEqual(len(citations), 1)


class FakeChat:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return self.response


class GraphRAGTest(unittest.TestCase):
    def test_llm_answer_with_valid_citation(self) -> None:
        chat = FakeChat("Your keys are in the work desk drawer [t=16.4s].")
        rag = GraphRAG(chat_client=chat, config=SemanticConfig())
        answer = rag.answer("Where are my keys?", CONTEXT)
        self.assertEqual(answer.source, "llm")
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.citations[0].timestamp, 16.4)
        self.assertIn("Entity: keys", chat.calls[0][1])

    def test_llm_multihop_temporal_answer(self) -> None:
        chat = FakeChat("Before being placed in the drawer, your keys were resting on the work desk [t=12.0s].")
        rag = GraphRAG(chat_client=chat, config=SemanticConfig())
        answer = rag.answer("Where were the keys before the drawer?", CONTEXT)
        self.assertEqual(answer.source, "llm")
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.citations[0].timestamp, 12.0)
        self.assertEqual(answer.citations[0].relation, "RESTING_ON")

    def test_uncited_llm_answer_falls_back(self) -> None:
        rag = GraphRAG(chat_client=FakeChat("They are in the drawer."))
        answer = rag.answer("Where are my keys?", CONTEXT)
        self.assertEqual(answer.source, "template")
        self.assertIn("[t=16.4s]", answer.answer)

    def test_llm_failure_falls_back(self) -> None:
        rag = GraphRAG(chat_client=FakeChat(error=LLMUnavailableError("offline")))
        answer = rag.answer("Where are my keys?", CONTEXT)
        self.assertEqual(answer.source, "template")
        self.assertIn("drawer", answer.answer)

    def test_no_client_uses_template(self) -> None:
        answer = GraphRAG(chat_client=None).answer("Where are my keys?", CONTEXT)
        self.assertEqual(answer.source, "template")

    def test_answer_json_roundtrip(self) -> None:
        import json

        payload = json.loads(
            GraphRAG(chat_client=None).answer_json("Where are my keys?", CONTEXT)
        )
        self.assertEqual(payload["entity"], "keys")
        self.assertTrue(payload["grounded"])
        self.assertEqual(payload["citations"][0]["timestamp"], 16.4)

    def test_empty_query_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraphRAG().answer("  ", CONTEXT)


if __name__ == "__main__":
    unittest.main()

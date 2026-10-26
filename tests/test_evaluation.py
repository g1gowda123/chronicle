"""Role C: GraphRAG evaluation set and citation precision."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from chronicle.semantic import (
    DEFAULT_GRAPHRAG_CASES,
    GraphRAG,
    GraphRAGCase,
    SemanticConfig,
    SemanticEngine,
    evaluate_graphrag,
    evaluation,
)
from chronicle.semantic.florence import StubCaptioner
from chronicle.semantic.graphrag import GraphContext, template_answer


def make_engine(chat_client=None) -> SemanticEngine:
    return SemanticEngine(
        config=SemanticConfig(llm_backend="template"),
        captioner=StubCaptioner(),
        chat_client=chat_client,
        embedder_backend="deterministic",
        captioner_backend="stub",
    )


class FakeChat:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def complete(self, system_prompt, user_prompt, *, json_mode=False):
        self.calls += 1
        return self.response


class DefaultEvaluationTest(unittest.TestCase):
    def test_template_path_passes_every_case(self) -> None:
        report = evaluate_graphrag(make_engine())
        self.assertEqual(report.total, len(DEFAULT_GRAPHRAG_CASES))
        self.assertEqual(report.passed, report.total, report.summary_lines())
        self.assertEqual(report.pass_rate, 1.0)
        self.assertEqual(report.location_accuracy, 1.0)
        self.assertEqual(report.timestamp_accuracy, 1.0)
        self.assertEqual(report.relation_accuracy, 1.0)
        self.assertEqual(report.unknown_accuracy, 1.0)

    def test_citations_are_fully_grounded(self) -> None:
        report = evaluate_graphrag(make_engine())
        self.assertGreater(report.citation_total, 0)
        self.assertEqual(report.citation_precision, 1.0)
        self.assertEqual(report.citation_valid, report.citation_total)

    def test_report_is_json_serializable(self) -> None:
        report = evaluate_graphrag(make_engine())
        payload = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(payload["passed"], payload["total"])
        self.assertEqual(len(payload["cases"]), payload["total"])
        self.assertIn("citation_precision", payload)

    def test_unknown_case_is_ungrounded(self) -> None:
        report = evaluate_graphrag(make_engine())
        unknown = [case for case in report.cases if case.name == "unknown-entity"][0]
        self.assertTrue(unknown.unknown_ok)
        self.assertFalse(unknown.grounded)
        self.assertIn("don't have", unknown.answer.lower())

    def test_appearance_cases_pass(self) -> None:
        report = evaluate_graphrag(make_engine())
        appearance = [
            case for case in report.cases if case.name.startswith("appearance-")
        ]
        self.assertEqual(len(appearance), 4)
        self.assertTrue(all(case.passed for case in appearance), report.summary_lines())
        colour = next(case for case in appearance if case.name == "appearance-colour")
        self.assertIn("red", colour.answer.lower())
        self.assertTrue(colour.relation_ok)
        self.assertTrue(colour.timestamp_ok)


class HallucinationFallbackTest(unittest.TestCase):
    def test_hallucinated_timestamp_falls_back_to_template(self) -> None:
        chat = FakeChat("The keys are in the kitchen at [t=99.9s].")
        engine = make_engine(chat)
        report = evaluate_graphrag(engine)
        self.assertEqual(report.passed, report.total, report.summary_lines())
        self.assertEqual(report.citation_precision, 1.0)
        self.assertTrue(all(case.source == "template" for case in report.cases))
        self.assertGreaterEqual(chat.calls, 1)
        self.assertTrue(
            all("99.9" not in case.answer for case in report.cases)
        )

    def test_cited_llm_answer_is_used(self) -> None:
        chat = FakeChat(
            "The keys were placed inside the drawer at [t=16.4s]."
        )
        context = DEFAULT_GRAPHRAG_CASES[0].context
        answer = GraphRAG(
            chat_client=chat, config=SemanticConfig(llm_backend="template")
        ).answer("Where are my keys?", context)
        self.assertEqual(answer.source, "llm")
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.citations[0].timestamp, 16.4)


class CustomCaseTest(unittest.TestCase):
    def test_custom_cases(self) -> None:
        case = GraphRAGCase(
            name="cup-on-table",
            query="Where is the cup?",
            context={
                "entity": {"id": "cup", "label": "cup"},
                "facts": [
                    {
                        "subject": "cup",
                        "relation": "RESTING_ON",
                        "object": "table",
                        "timestamp": 5.0,
                        "zone": "living room",
                    }
                ],
            },
            expected_location="table",
            expected_timestamp=5.0,
            expected_relation="RESTING_ON",
        )
        report = evaluate_graphrag(make_engine(), [case])
        self.assertEqual(report.total, 1)
        self.assertTrue(report.cases[0].passed)

    def test_template_reference_answer_is_grounded(self) -> None:
        case = DEFAULT_GRAPHRAG_CASES[0]
        answer = template_answer(case.query, GraphContext.from_mapping(case.context))
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.source, "template")
        self.assertEqual(answer.citations[0].relation, "INSERT_INTO")

    def test_cli_json_output(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = evaluation.main(["--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

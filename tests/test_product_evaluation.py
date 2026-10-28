"""Deterministic product smoke evaluation."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from chronicle.cli import main
from chronicle.evaluation import load_cases, run_evaluation


class DemoSuiteTest(unittest.TestCase):
    def test_bundled_suite_is_unique_and_balanced(self) -> None:
        cases = load_cases(
            Path(__file__).parents[1]
            / "chronicle"
            / "evaluation_data"
            / "demo_questions.jsonl"
        )
        self.assertEqual(len(cases), 14)
        self.assertEqual(len({case.id for case in cases}), 14)
        categories = {case.category for case in cases}
        self.assertTrue({"location", "event", "temporal", "counting", "abstention"} <= categories)

    def test_bundled_demo_scores_full_marks(self) -> None:
        report = run_evaluation()
        self.assertEqual(report["total"], 14)
        self.assertEqual(report["passed"], 14, report["cases"])
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["score"], 1.0)
        self.assertEqual(report["kind"], "deterministic product smoke suite")

    def test_a_wrong_expectation_is_reported_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "bad.jsonl"
            suite.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "category": "location",
                        "question": "where is my cup?",
                        "expect_all": ["mars"],
                        "expect_method": "template",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = run_evaluation(suite_path=suite)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["score"], 0.0)
        self.assertIn("missing 'mars'", report["cases"][0]["failures"][0])

    def test_cli_eval_json_is_machine_readable(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["eval", "--json"])
        self.assertEqual(code, 0)
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["passed"], report["total"])

    def test_cli_eval_text_prints_score(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["eval"])
        self.assertEqual(code, 0)
        self.assertIn("Score: 14/14 (100.0%)", buffer.getvalue())


class InvalidSuiteTest(unittest.TestCase):
    def test_duplicate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "dupes.jsonl"
            line = json.dumps(
                {
                    "id": "same",
                    "category": "location",
                    "question": "where is my cup?",
                }
            )
            suite.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_cases(suite)

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_evaluation("/tmp/does-not-exist-chronicle.db")


if __name__ == "__main__":
    unittest.main()

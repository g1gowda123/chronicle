"""The zero-model onboarding path: demo, bundled memory and doctor."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from chronicle import cli
from chronicle.demo import run_demo
from chronicle.demo_data import demo_db_path
from chronicle.doctor import collect_report, render_report


class DemoTest(unittest.TestCase):
    def test_run_demo_builds_and_reopens_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "demo.db"
            result = run_demo(db)
            self.assertTrue(db.is_file())
            self.assertEqual(result["stats"]["entities"], 1)
            self.assertGreater(result["stats"]["observations"], 0)
            self.assertIn("kitchen counter", result["restored_answer"])
            self.assertEqual(result["restored_entities"], 1)

    def test_bundled_demo_memory_is_packaged(self) -> None:
        path = demo_db_path()
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)

    def test_cli_demo_json(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(cli.main(["demo", "--json"]), 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("restored_answer", payload)
        self.assertIn("queries", payload)


class DoctorTest(unittest.TestCase):
    def test_report_contains_core_and_demo(self) -> None:
        report = collect_report()
        self.assertIn("core", report)
        self.assertIn("optional", report)
        self.assertTrue(report["demo_memory"]["exists"])
        self.assertIn("Status:", render_report(report))

    def test_cli_doctor_json(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            code = cli.main(["doctor", "--json"])
        self.assertIn(code, (0, 1))
        payload = json.loads(stdout.getvalue())
        self.assertIn("cuda", payload)
        self.assertIn("model_weights", payload)


class AskDefaultTest(unittest.TestCase):
    def test_ask_without_db_uses_bundled_memory(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["ask", "where is my cup?", "--llm", "template"])
        self.assertEqual(code, 0)
        self.assertIn("using bundled demo memory", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["entity"], "cup")
        self.assertIn("kitchen counter", payload["answer"])


if __name__ == "__main__":
    unittest.main()

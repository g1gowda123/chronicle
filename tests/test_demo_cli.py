"""The zero-model onboarding path: demo, bundled memory and doctor."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

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
            self.assertEqual(result["stats"]["entities"], 4)
            self.assertGreater(result["stats"]["observations"], 0)
            self.assertIn("kitchen counter", result["restored_answer"])
            self.assertEqual(result["restored_entities"], 4)

    def test_demo_event_answer_is_grounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_demo(Path(tmp) / "demo.db")
        event = result["queries"]["what happened to my cup?"]
        self.assertIn("picked up the cup", event["answer"])
        self.assertIn("dining table", event["answer"])
        self.assertIn("kitchen counter", event["answer"])
        self.assertEqual(event["anchor"]["label"], "kitchen counter")

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

    def test_cli_version(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as raised:
            cli.main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertRegex(stdout.getvalue().strip(), r"^chronicle \d+\.\d+\.\d+")


class DoctorTest(unittest.TestCase):
    def test_report_contains_core_and_demo(self) -> None:
        report = collect_report()
        self.assertIn("version", report)
        self.assertIn("core", report)
        self.assertIn("optional", report)
        self.assertTrue(report["demo_memory"]["exists"])
        self.assertTrue(report["demo_memory"]["query_ok"])
        self.assertIn("kitchen counter", report["demo_memory"]["answer"])
        self.assertIn("Status:", render_report(report))

    def test_cli_doctor_json(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            code = cli.main(["doctor", "--json"])
        self.assertIn(code, (0, 1))
        payload = json.loads(stdout.getvalue())
        self.assertIn("cuda", payload)
        self.assertIn("model_weights", payload)

    def test_doctor_reads_a_read_only_bundled_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.db"
            shutil.copy2(demo_db_path(), path)
            path.chmod(0o444)
            with patch("chronicle.demo_data.demo_db_path", return_value=path):
                report = collect_report()
        self.assertTrue(report["demo_memory"]["query_ok"])
        self.assertIn("kitchen counter", report["demo_memory"]["answer"])


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

    def test_ask_reads_a_read_only_bundled_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.db"
            shutil.copy2(demo_db_path(), path)
            path.chmod(0o444)
            stdout = StringIO()
            stderr = StringIO()
            with patch("chronicle.demo_data.demo_db_path", return_value=path):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = cli.main(["ask", "where is my cup?", "--llm", "template"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "template")
        self.assertIn("kitchen counter", payload["answer"])

    def test_ask_event_uses_event_summary(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["ask", "what happened to my cup?", "--llm", "template"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-template-event")
        self.assertIn("picked up the cup", payload["answer"])
        self.assertIn("dining table", payload["answer"])
        self.assertIn("kitchen counter", payload["answer"])

    def test_ask_generic_event_uses_overview(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["ask", "what did I do?", "--llm", "template"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-template-event")
        self.assertIn("picked up the cup", payload["answer"])
        self.assertIn("kitchen counter", payload["answer"])

    def test_ask_first_and_last_events_are_ordered(self) -> None:
        for question, expected, unexpected in (
            ("what happened first?", "picked up the cup", "picked up the bottle"),
            ("what did I do first?", "picked up the cup", "picked up the bottle"),
            ("what was the first thing I did?", "picked up the cup", "picked up the bottle"),
            ("what happened last?", "picked up the bottle", "picked up the cup"),
            ("what did I do last?", "picked up the bottle", "picked up the cup"),
            ("what was the last thing I did?", "picked up the bottle", "picked up the cup"),
            ("what happened most recently?", "picked up the bottle", "picked up the cup"),
        ):
            with self.subTest(question=question):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = cli.main(["ask", question, "--llm", "template"])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertIn(expected, payload["answer"])
                self.assertNotIn(unexpected, payload["answer"])

    def test_ask_count_uses_tracked_entities(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["ask", "how many cups are there?", "--llm", "template"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-count")
        self.assertTrue(payload["grounded"])
        self.assertEqual(payload["entity"], "cup")
        self.assertEqual(payload["answer"], "There is 1 cup in this memory.")

    def test_ask_count_all_objects(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["ask", "how many objects are there?", "--llm", "template"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-count")
        self.assertTrue(payload["grounded"])
        self.assertEqual(payload["answer"], "There are 4 objects in this memory.")

    def test_ask_count_unknown_object_abstains(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["ask", "how many elephants?", "--llm", "template"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-count")
        self.assertFalse(payload["grounded"])
        self.assertIn("don't have any elephants", payload["answer"])

    def test_ask_count_scopes_to_interacted_objects(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                ["ask", "how many objects did I touch?", "--llm", "template"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-count")
        self.assertEqual(payload["answer"], "You interacted with 3 objects in this memory.")

    def test_ask_count_scopes_to_placed_objects(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                ["ask", "how many objects did I put down?", "--llm", "template"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-count")
        self.assertEqual(payload["answer"], "You put down 3 objects in this memory.")

    def test_ask_action_overview_scopes_moved_objects_to_surface(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                [
                    "ask",
                    "what did I move to the kitchen counter?",
                    "--llm",
                    "template",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-action")
        self.assertIn("cup", payload["answer"])
        self.assertIn("bottle", payload["answer"])
        self.assertNotIn("keys", payload["answer"])

    def test_ask_count_uses_placement_verb(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                ["ask", "how many objects did I place?", "--llm", "template"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-count")
        self.assertEqual(payload["answer"], "You placed 3 objects in this memory.")

    def test_ask_temporal_pickup_with_pronoun(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                [
                    "ask",
                    "where was the cup before I picked it up?",
                    "--llm",
                    "template",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("dining table", payload["answer"])
        self.assertNotIn("Before the cup, the cup", payload["answer"])

    def test_ask_action_overview_lists_moved_objects(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                ["ask", "what did I move?", "--llm", "template"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-action")
        self.assertEqual(
            payload["answer"], "You moved the cup, the keys and the bottle."
        )

    def test_ask_action_overview_lists_interacted_objects(self) -> None:
        for question, verb in (
            ("what did I pick up?", "picked up"),
            ("what did I put down?", "put down"),
            ("what did I interact with?", "interacted with"),
        ):
            with self.subTest(question=question):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = cli.main(["ask", question, "--llm", "template"])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["method"], "graph-action")
                self.assertEqual(
                    payload["answer"],
                    f"You {verb} the cup, the keys and the bottle.",
                )

    def test_ask_action_overview_respects_before_reference(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                ["ask", "what did I pick up before the keys?", "--llm", "template"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-action")
        self.assertEqual(payload["answer"], "You picked up the cup.")

    def test_ask_action_overview_respects_after_reference(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                ["ask", "what did I put down after the cup?", "--llm", "template"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-action")
        self.assertEqual(
            payload["answer"], "You put down the keys and the bottle."
        )

    def test_ask_action_overview_supports_surface_and_before_reference(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                [
                    "ask",
                    "what did I put on the kitchen counter before the bottle?",
                    "--llm",
                    "template",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-action")
        self.assertEqual(payload["answer"], "You put down the cup on the kitchen counter.")

    def test_ask_action_sequence_before_reference(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                [
                    "ask",
                    "what did I do before I put down the keys?",
                    "--llm",
                    "template",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-action-sequence")
        self.assertEqual(
            payload["answer"],
            "Before you put down the keys, you picked up the cup at t=1.9s, "
            "placed the cup on the kitchen counter at t=4.2s, and picked up "
            "the keys at t=5.1s.",
        )

    def test_ask_action_sequence_after_reference(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                [
                    "ask",
                    "what happened after I picked up the cup?",
                    "--llm",
                    "template",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-action-sequence")
        self.assertTrue(payload["answer"].startswith("After you picked up the cup,"))
        self.assertIn("placed the cup on the kitchen counter at t=4.2s", payload["answer"])
        self.assertIn("placed the bottle on the kitchen counter at t=10.6s", payload["answer"])

    def test_ask_action_overview_matches_zone_as_well_as_anchor(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                ["ask", "what did I put in the kitchen?", "--llm", "template"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-action")
        self.assertEqual(
            payload["answer"],
            "You put down the cup, the keys and the bottle in the kitchen.",
        )

    def test_ask_count_scopes_to_a_surface(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                [
                    "ask",
                    "how many objects are on the kitchen counter?",
                    "--llm",
                    "template",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-count")
        self.assertEqual(payload["answer"], "There are 2 objects on the kitchen counter.")

    def test_ask_action_can_scope_to_a_surface(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                [
                    "ask",
                    "what did I place on the kitchen counter?",
                    "--llm",
                    "template",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-action")
        self.assertEqual(
            payload["answer"],
            "You put down the cup and the bottle on the kitchen counter.",
        )

    def test_ask_action_can_order_first_and_last(self) -> None:
        for question, answer in (
            (
                "which object did I touch first?",
                "The first object you interacted with was the cup at t=1.9s.",
            ),
            (
                "which object did I pick up first?",
                "The first object you picked up was the cup at t=1.9s.",
            ),
            (
                "which object did I put down last?",
                "The last object you put down was the bottle at t=10.6s.",
            ),
        ):
            with self.subTest(question=question):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = cli.main(["ask", question, "--llm", "template"])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["method"], "graph-action")
                self.assertEqual(payload["answer"], answer)

    def test_ask_yes_no_location_and_interaction(self) -> None:
        cases = (
            (
                "is the cup on the kitchen counter?",
                "Yes. The cup is on the kitchen counter.",
            ),
            (
                "is the cup on the dining table?",
                "No. The cup is on the kitchen counter.",
            ),
            (
                "did I move the notebook?",
                "No. I don't have a recorded interaction with the notebook.",
            ),
        )
        for question, answer in cases:
            with self.subTest(question=question):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = cli.main(["ask", question, "--llm", "template"])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["method"], "graph-confirm")
                self.assertEqual(payload["answer"], answer)

    def test_ask_yes_no_movement_destination(self) -> None:
        cases = (
            (
                "did I move the cup to the dining table?",
                "No. The cup is on the kitchen counter.",
            ),
            (
                "did I put down the keys on the dining table?",
                "Yes. You picked up the keys at t=5.1s and placed them on the "
                "dining table at t=7.4s.",
            ),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = cli.main(["ask", question, "--llm", "template"])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["method"], "graph-confirm")
                self.assertIn(expected, payload["answer"])

    def test_ask_yes_no_pickup_source(self) -> None:
        cases = (
            (
                "did I pick up the cup from the kitchen counter?",
                "No. The cup was on the dining table before it was picked up.",
            ),
            (
                "did I pick up the cup from the dining table?",
                "Yes. You picked up the cup at t=1.9s",
            ),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = cli.main(["ask", question, "--llm", "template"])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["method"], "graph-confirm")
                self.assertIn(expected, payload["answer"])

    def test_ask_temporal_before_move_pronoun(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                [
                    "ask",
                    "where was the bottle before I moved it?",
                    "--llm",
                    "template",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("dining table", payload["answer"])
        self.assertIn("kitchen counter", payload["answer"])

    def test_ask_yes_no_interaction_starts_with_yes(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["ask", "did I move the cup?", "--llm", "template"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "graph-confirm")
        self.assertTrue(payload["answer"].startswith("Yes. You picked up the cup"))

    def test_ask_inventory_summary_uses_singular_nouns(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["ask", "which objects did I see?", "--llm", "template"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["method"], "inventory")
        self.assertIn("1 cup", payload["answer"])
        self.assertIn("1 key", payload["answer"])
        self.assertNotIn("1 keys", payload["answer"])

    def test_ask_temporal_before_short_placement_clause(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(
                ["ask", "where was the cup before it was placed?", "--llm", "template"]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("dining table", payload["answer"])
        self.assertIn("kitchen counter", payload["answer"])

    def test_ask_unknown_entity_abstains(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(["ask", "where is my toaster?", "--llm", "template"])
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIsNone(payload["entity"])
        self.assertEqual(payload["answer"], "I don't have that in my memory.")


if __name__ == "__main__":
    unittest.main()

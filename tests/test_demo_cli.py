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
        self.assertIn("Last picked up at t=1.9s.", payload["answer"])

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
        answer = payload["answer"]
        self.assertIn("picked up the cup", answer)
        self.assertIn("kitchen counter", answer)
        self.assertLess(answer.index("picked up the cup"), answer.index("picked up the keys"))
        self.assertLess(
            answer.index("picked up the keys"), answer.index("picked up the bottle")
        )

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

    def test_ask_action_sequence_accepts_gerund_reference(self) -> None:
        cases = (
            (
                "what did I do before placing the cup?",
                "Before you placed the cup, you picked up the cup at t=1.9s.",
            ),
            (
                "what did I do after picking up the keys?",
                "After you picked up the keys, you placed the keys on the "
                "dining table at t=7.4s, picked up the bottle at t=8.3s, and "
                "placed the bottle on the kitchen counter at t=10.6s.",
            ),
            (
                "what did I do before I moved the cup?",
                "Before you moved the cup, you picked up the cup at t=1.9s.",
            ),
            (
                "what did I do after I picked the cup up?",
                "After you picked up the cup, you placed the cup on the "
                "kitchen counter at t=4.2s, picked up the keys at t=5.1s, "
                "placed the keys on the dining table at t=7.4s, picked up "
                "the bottle at t=8.3s, and placed the bottle on the kitchen "
                "counter at t=10.6s.",
            ),
            (
                "what did I do before I put the keys down?",
                "Before you put down the keys, you picked up the cup at "
                "t=1.9s, placed the cup on the kitchen counter at t=4.2s, "
                "and picked up the keys at t=5.1s.",
            ),
            (
                "what happened after the cup?",
                "After you interacted with the cup, you picked up the keys at "
                "t=5.1s, placed the keys on the dining table at t=7.4s, "
                "picked up the bottle at t=8.3s, and placed the bottle on the "
                "kitchen counter at t=10.6s.",
            ),
            (
                "what happened before the cup?",
                "I don't have any recorded actions before you interacted with "
                "the cup in this memory.",
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
                self.assertEqual(payload["method"], "graph-action-sequence")
                self.assertEqual(payload["answer"], answer)

    def test_ask_action_at_timestamp(self) -> None:
        cases = (
            (
                "what did I do at 5 seconds?",
                "At t=5.1s you picked up the keys.",
            ),
            (
                "what was happening at 4.2 seconds?",
                "At t=4.2s you placed the cup on the kitchen counter.",
            ),
            (
                "what did I do around t=5?",
                "At t=5.1s you picked up the keys.",
            ),
            (
                "what did I do at 99 seconds?",
                "I don't have an action at t=99.0s. The closest recorded "
                "action was at t=10.6s: you placed the bottle on the kitchen "
                "counter.",
            ),
            (
                "what did I do at the beginning?",
                "At t=1.9s you picked up the cup.",
            ),
            (
                "what happened at the start?",
                "At t=1.9s you picked up the cup.",
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
                self.assertEqual(payload["method"], "graph-action-at")
                self.assertTrue(payload["grounded"])
                self.assertEqual(payload["answer"], expected)

    def test_ask_action_time_window(self) -> None:
        cases = (
            (
                "what did I do between 5 and 8 seconds?",
                "Between t=5.0s and t=8.0s, you picked up the keys at "
                "t=5.1s and placed the keys on the dining table at t=7.4s.",
            ),
            (
                "what happened from 5 to 8 seconds?",
                "Between t=5.0s and t=8.0s, you picked up the keys at "
                "t=5.1s and placed the keys on the dining table at t=7.4s.",
            ),
            (
                "what did I do after 5 seconds?",
                "After t=5.0s, you picked up the keys at t=5.1s, placed the "
                "keys on the dining table at t=7.4s, picked up the bottle at "
                "t=8.3s, and placed the bottle on the kitchen counter at "
                "t=10.6s.",
            ),
            (
                "what did I do before 5 seconds?",
                "Before t=5.0s, you picked up the cup at t=1.9s and placed "
                "the cup on the kitchen counter at t=4.2s.",
            ),
            (
                "what did I do between 99 and 100 seconds?",
                "I don't have any recorded actions between t=99.0s and "
                "t=100.0s.",
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
                self.assertEqual(payload["method"], "graph-action-window")
                self.assertEqual(payload["grounded"], "99" not in question)
                self.assertEqual(payload["answer"], expected)

    def test_ask_temporal_location(self) -> None:
        cases = (
            (
                "where was the cup at 3 seconds?",
                "At t=3.0s the cup was not resting on a surface; it was last "
                "picked up at t=1.9s.",
            ),
            (
                "where was the cup at 5 seconds?",
                "At t=5.0s the cup was on the kitchen counter.",
            ),
            (
                "where was the cup at 0.5 seconds?",
                "At t=0.5s the cup was on the dining table.",
            ),
            (
                "where was the bottle at 9 seconds?",
                "At t=9.0s the bottle was not resting on a surface; it was "
                "last picked up at t=8.3s.",
            ),
            (
                "where were the keys at 6 seconds?",
                "At t=6.0s the keys were not resting on a surface; they were "
                "last picked up at t=5.1s.",
            ),
            (
                "where were the keys at 8 seconds?",
                "At t=8.0s the keys were on the dining table.",
            ),
            (
                "where was the wallet at 3 seconds?",
                "I don't have that in my memory.",
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
                self.assertEqual(payload["method"], "graph-temporal-location")
                self.assertEqual(payload["grounded"], "wallet" not in question)
                self.assertEqual(payload["answer"], expected)

    def test_ask_yes_no_temporal_action_order(self) -> None:
        cases = (
            (
                "did I move the cup after picking up the keys?",
                "No. You moved the cup at t=4.2s, before you picked up the "
                "keys at t=5.1s.",
            ),
            (
                "did I pick up the keys after the cup?",
                "Yes. You picked up the keys at t=5.1s, after you interacted "
                "with the cup at t=4.2s.",
            ),
            (
                "did I put down the keys before the bottle?",
                "Yes. You placed the keys at t=7.4s, before you interacted "
                "with the bottle at t=10.6s.",
            ),
            (
                "did I pick up the keys before the cup?",
                "No. You picked up the keys at t=5.1s, after you interacted "
                "with the cup at t=4.2s.",
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
                self.assertEqual(payload["answer"], expected)

    def test_ask_movement_source_and_destination(self) -> None:
        cases = (
            (
                "where did I move the cup to?",
                "You moved the cup to the kitchen counter at t=4.2s.",
                "graph-object-destination",
            ),
            (
                "where did I move the cup from?",
                "You moved the cup from the dining table at t=1.9s.",
                "graph-object-source",
            ),
        )
        for question, expected, method in cases:
            with self.subTest(question=question):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = cli.main(["ask", question, "--llm", "template"])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["method"], method)
                self.assertEqual(payload["answer"], expected)

    def test_ask_natural_action_order_phrase(self) -> None:
        cases = (
            (
                "what was the last thing I touched?",
                "The last object you interacted with was the bottle at t=8.3s.",
            ),
            (
                "what was the first thing I picked up?",
                "The first object you picked up was the cup at t=1.9s.",
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
                self.assertEqual(payload["method"], "graph-action")
                self.assertEqual(payload["answer"], expected)

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
        for question in (
            "what did I place on the kitchen counter?",
            "what did I set on the kitchen counter?",
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
                    "You put down the cup and the bottle on the kitchen counter.",
                )

    def test_ask_pickup_can_scope_to_source_surface(self) -> None:
        for question, expected in (
            (
                "what did I pick up from the dining table?",
                "You picked up the cup and the bottle from the dining table.",
            ),
            (
                "what did I grab from the kitchen counter?",
                "You picked up the keys from the kitchen counter.",
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
                self.assertEqual(payload["answer"], expected)

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
            (
                "what did I put down first?",
                "The first object you put down was the cup at t=4.2s.",
            ),
            (
                "what did I grab first?",
                "The first object you picked up was the cup at t=1.9s.",
            ),
            (
                "what did I put down second?",
                "The second object you put down was the keys at t=7.4s.",
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
            (
                "did I put the cup on the counter?",
                "Yes. You picked up the cup at t=1.9s and placed it on the "
                "kitchen counter at t=4.2s. It had been on the dining table "
                "at t=0.5s.",
            ),
            (
                "did I put the cup on the dining table?",
                "No. The cup is on the kitchen counter.",
            ),
            (
                "did I place the keys on the kitchen counter?",
                "No. The keys are on the dining table.",
            ),
            (
                "are the keys on the dining table?",
                "Yes. The keys are on the dining table.",
            ),
            (
                "were the keys on the kitchen counter before I picked them up?",
                "Yes. The keys were on the kitchen counter before they were "
                "picked up.",
            ),
            (
                "were the keys on the dining table before I picked them up?",
                "No. The keys were on the kitchen counter before they were "
                "picked up.",
            ),
            (
                "was the bottle on the dining table before I moved it?",
                "Yes. The bottle was on the dining table before it was moved.",
            ),
            (
                "was the bottle on the kitchen counter before I moved it?",
                "No. The bottle was on the dining table before it was moved.",
            ),
            (
                "was the cup on the dining table before I placed it?",
                "No. The cup is on the kitchen counter.",
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
            (
                "did I place the bottle on the dining table?",
                "No. The bottle is on the kitchen counter.",
            ),
            (
                "did I place the bottle on the kitchen counter?",
                "Yes. You picked up the bottle at t=8.3s",
            ),
            (
                "did I set down the bottle?",
                "Yes. You picked up the bottle at t=8.3s",
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
            (
                "did I pick up the keys from the dining table?",
                "No. The keys were on the kitchen counter before they were "
                "picked up.",
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

    def test_ask_where_a_specific_object_was_put(self) -> None:
        cases = (
            (
                "what did I put the keys on?",
                "You put the keys on the dining table at t=7.4s.",
            ),
            (
                "where did I place the bottle?",
                "You placed the bottle on the kitchen counter at t=10.6s.",
            ),
            (
                "when did I put the cup on the kitchen counter?",
                "You put the cup on the kitchen counter at t=4.2s.",
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
                self.assertEqual(payload["method"], "graph-object-destination")
                self.assertEqual(payload["answer"], expected)

    def test_ask_where_a_specific_object_was_picked_up_from(self) -> None:
        cases = (
            (
                "what did I pick the cup up from?",
                "You picked up the cup from the dining table at t=1.9s.",
            ),
            (
                "where did I pick up the bottle from?",
                "You picked up the bottle from the dining table at t=8.3s.",
            ),
            (
                "when did I pick up the keys?",
                "You picked up the keys from the kitchen counter at t=5.1s.",
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
                self.assertEqual(payload["method"], "graph-object-source")
                self.assertEqual(payload["answer"], expected)

    def test_ask_object_source_and_destination_abstain_without_events(self) -> None:
        cases = (
            (
                "what did I grab the notebook from?",
                "graph-object-source",
                "I don't have a recorded pick-up for the notebook in this memory.",
            ),
            (
                "where did I pick up the notebook?",
                "graph-object-source",
                "I don't have a recorded pick-up for the notebook in this memory.",
            ),
            (
                "what did I put the mug on?",
                "graph-object-destination",
                "I don't have that in my memory.",
            ),
        )
        for question, method, expected in cases:
            with self.subTest(question=question):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = cli.main(["ask", question, "--llm", "template"])
                self.assertEqual(code, 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["method"], method)
                self.assertFalse(payload["grounded"])
                self.assertEqual(payload["answer"], expected)

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

"""CLI config assembly: source vocabulary and detector overrides (TASK-F4)."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from chronicle import cli
from chronicle.sources import source_for_key

ROOT = Path(__file__).resolve().parents[1]


class PerceptionConfigTest(unittest.TestCase):
    def test_source_vocabulary_and_overrides(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "run",
                "--source",
                "tabletop",
                "--conf",
                "0.10",
                "--imgsz",
                "960",
            ]
        )
        source = source_for_key("tabletop", ROOT / "data")
        config = cli._perception_config(args, source)
        self.assertEqual(config.vocabulary, source.vocabulary)
        self.assertIn("joy-con", config.vocabulary)
        self.assertNotIn("phone", config.vocabulary)
        self.assertAlmostEqual(config.detector.confidence, 0.10)
        self.assertEqual(config.detector.imgsz, 960)

    def test_defaults_without_source(self) -> None:
        args = cli.build_parser().parse_args(["run", "--source", "synthetic"])
        config = cli._perception_config(args)
        self.assertIn("keys", config.vocabulary)
        self.assertEqual(config.detector.confidence, 0.30)
        self.assertEqual(config.detector.imgsz, 640)

    def test_cmd_run_passes_source_vocabulary_to_runtime(self) -> None:
        captured: dict[str, object] = {}

        class FakeSummary:
            def as_dict(self):
                return {"frames": 0}

        class FakeRuntime:
            def __init__(self, *args, **kwargs) -> None:
                captured["config"] = kwargs["perception_config"]

            def ingest_video(self, *args, **kwargs):
                return FakeSummary()

            def close(self) -> None:
                pass

        args = cli.build_parser().parse_args(
            ["run", "--source", "tabletop", "--reset"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            args.db = str(Path(tmp) / "cli.db")
            with mock.patch.object(cli, "ChronicleRuntime", FakeRuntime):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.cmd_run(args), 0)
        config = captured["config"]
        self.assertIn("joy-con", config.vocabulary)
        self.assertNotIn("phone", config.vocabulary)


if __name__ == "__main__":
    unittest.main()

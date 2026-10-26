"""Test source registry resolution and CLI sources command."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from chronicle import cli
from chronicle.sources import (
    SOURCES,
    default_source,
    source_for_key,
    source_for_video,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo" / "chronicle_demo.mp4"


class SourceResolutionTest(unittest.TestCase):
    def test_sources_contain_synthetic_and_real(self) -> None:
        keys = {s.key for s in SOURCES}
        self.assertIn("synthetic", keys)
        self.assertIn("tabletop", keys)
        self.assertIn("cook", keys)
        self.assertIn("play", keys)
        self.assertIn("walking", keys)
        self.assertNotIn("live", keys)

    def test_synthetic_source_available(self) -> None:
        synthetic = source_for_key("synthetic", ROOT / "data")
        self.assertEqual(synthetic.key, "synthetic")
        self.assertFalse(synthetic.real)
        self.assertTrue(synthetic.available(ROOT / "data"))

    def test_default_source_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(default_source(tmp).key, "synthetic")

    def test_resolve_source_synthetic(self) -> None:
        source, path = cli._resolve_source(ROOT / "data", source_key="synthetic")
        self.assertEqual(source.key, "synthetic")
        self.assertIsInstance(path, Path)
        self.assertTrue(path.is_file())

    def test_source_for_video_matching(self) -> None:
        matched = source_for_video(DEMO, ROOT / "data")
        self.assertIsNotNone(matched)
        assert matched is not None
        self.assertEqual(matched.key, "synthetic")


class SourcesCliTest(unittest.TestCase):
    def test_sources_command_output(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--data-dir", str(ROOT / "data"), "sources"])
        self.assertEqual(code, 0)
        rows = {row["key"]: row for row in json.loads(buffer.getvalue())}
        self.assertIn("synthetic", rows)
        self.assertTrue(rows["synthetic"]["available"])
        self.assertNotIn("live", rows)

    def test_run_synthetic_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "smoke.db"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(
                    [
                        "--data-dir",
                        str(ROOT / "data"),
                        "run",
                        "--source",
                        "synthetic",
                        "--max-frames",
                        "6",
                        "--reset",
                        "--db",
                        str(db),
                        "--detector",
                        "color",
                        "--hands",
                        "color",
                        "--embedder",
                        "color",
                    ]
                )
            self.assertEqual(code, 0)
            payload = json.loads(buffer.getvalue())
            self.assertEqual(payload["frames"], 6)
            self.assertTrue(db.is_file())


if __name__ == "__main__":
    unittest.main()

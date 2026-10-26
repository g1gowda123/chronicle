"""Live stream source resolution and CLI wiring (TASK-F3)."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from chronicle import cli
from chronicle.sources import (
    available_sources,
    default_source,
    resolve_live_stream,
    source_for_key,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo" / "chronicle_demo.mp4"


class LiveStreamResolutionTest(unittest.TestCase):
    def test_defaults_to_camera_zero(self) -> None:
        with mock.patch.dict("os.environ", {"CHRONICLE_LIVE_STREAM": ""}):
            self.assertEqual(resolve_live_stream(), 0)

    def test_explicit_inputs(self) -> None:
        self.assertEqual(resolve_live_stream("1"), 1)
        self.assertEqual(resolve_live_stream(" rtsp://cam.local/stream "), "rtsp://cam.local/stream")
        self.assertEqual(resolve_live_stream("data/demo/chronicle_demo.mp4"), "data/demo/chronicle_demo.mp4")

    def test_environment_fallback(self) -> None:
        with mock.patch.dict("os.environ", {"CHRONICLE_LIVE_STREAM": "2"}):
            self.assertEqual(resolve_live_stream(), 2)
        with mock.patch.dict("os.environ", {"CHRONICLE_LIVE_STREAM": "http://cam/feed"}):
            self.assertEqual(resolve_live_stream(), "http://cam/feed")

    def test_live_source_is_available_without_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            live = source_for_key("live", tmp)
            self.assertTrue(live.live)
            self.assertTrue(live.available(tmp))
            self.assertIn("live", {source.key for source in available_sources(tmp)})

    def test_live_is_not_the_default_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(default_source(tmp).key, "synthetic")

    def test_resolve_source_returns_stream_not_path(self) -> None:
        source, stream = cli._resolve_source(Path("data"), source_key="live")
        self.assertEqual(source.key, "live")
        self.assertNotIsInstance(stream, Path)


class LiveRunCliTest(unittest.TestCase):
    def test_run_live_source_from_file_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "live.db"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(
                    [
                        "--data-dir",
                        str(ROOT / "data"),
                        "run",
                        "--source",
                        "live",
                        "--live-stream",
                        str(DEMO),
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
            self.assertIn("chronicle_demo.mp4", payload["video"])
            self.assertTrue(db.is_file())

    def test_sources_lists_live_feed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with mock.patch.dict("os.environ", {"CHRONICLE_LIVE_STREAM": ""}):
                with redirect_stdout(buffer):
                    code = cli.main(["--data-dir", tmp, "sources"])
            self.assertEqual(code, 0)
            rows = {row["key"]: row for row in json.loads(buffer.getvalue())}
            self.assertTrue(rows["live"]["available"])
            self.assertEqual(rows["live"]["video"], "camera:0")


if __name__ == "__main__":
    unittest.main()

"""Model-free tests for the demo preflight script."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "demo_preflight.py"


def _load_preflight():
    spec = importlib.util.spec_from_file_location("demo_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load_preflight()


class ProbeVideoTest(unittest.TestCase):
    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.mp4"
            with self.assertRaises(RuntimeError):
                preflight.probe_video(path)

    def test_non_video_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-video.mp4"
            path.write_text("not a video", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    preflight.probe_video(path)

    def test_bundled_synthetic_clip_decodes(self) -> None:
        video = REPO_ROOT / "data" / "demo" / "chronicle_demo.mp4"
        if not video.is_file():
            self.skipTest("synthetic fixture is not checked out")
        info = preflight.probe_video(video)
        self.assertGreater(info.width, 0)
        self.assertGreater(info.height, 0)
        self.assertGreater(info.fps, 0)
        self.assertGreater(info.frames, 0)
        self.assertIsNotNone(info.frame)


class CheckVideosTest(unittest.TestCase):
    def test_missing_video_is_fail_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            source = replace(
                preflight.source_for_key("synthetic", data_dir),
                relative_path="missing/clip.mp4",
            )
            checks, infos = preflight.check_videos(
                [source], data_dir, explicit=True
            )
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].status, preflight.FAIL)
        self.assertIn("git lfs pull", checks[0].command or "")
        self.assertEqual(infos, {})

    def test_missing_video_is_warn_when_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            source = replace(
                preflight.source_for_key("synthetic", data_dir),
                relative_path="missing/clip.mp4",
            )
            checks, _ = preflight.check_videos(
                [source], data_dir, explicit=False
            )
        self.assertEqual(checks[0].status, preflight.WARN)


class SQLiteCheckTest(unittest.TestCase):
    def test_missing_database_is_created_in_scratch_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            scratch = root / "scratch"
            scratch.mkdir()
            source = preflight.source_for_key("synthetic", data_dir)
            result = preflight.check_sqlite(
                [source], data_dir, scratch
            )
        self.assertEqual(result.status, preflight.PASS)
        self.assertIn("WAL ok", result.detail)
        self.assertIn("created on first run", result.detail)


class DataDirCheckTest(unittest.TestCase):
    def test_writable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = preflight.check_data_dir(Path(tmp))
        self.assertEqual(result.status, preflight.PASS)

    def test_missing_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = preflight.check_data_dir(Path(tmp) / "missing")
        self.assertEqual(result.status, preflight.FAIL)


class SummaryTest(unittest.TestCase):
    def test_warning_is_ready_unless_strict(self) -> None:
        checks = [preflight.Check("x", preflight.WARN, "cold model")]
        self.assertTrue(preflight._summary(checks, strict=False)["ready"])
        self.assertFalse(preflight._summary(checks, strict=True)["ready"])

    def test_failure_is_never_ready(self) -> None:
        checks = [preflight.Check("x", preflight.FAIL, "broken")]
        self.assertFalse(preflight._summary(checks, strict=False)["ready"])

    def test_format_includes_command(self) -> None:
        check = preflight.Check("Demo", preflight.FAIL, "nope", "fix-it")
        rendered = preflight._format_check(check, 8, color=False)
        self.assertIn("FAIL", rendered)
        self.assertIn("-> fix-it", rendered)

    def test_summary_json_parser_skips_log_prefix(self) -> None:
        payload = preflight._parse_run_summary(
            'log line\n{"frames": 30, "elapsed": 1.0}\n'
        )
        self.assertEqual(payload, {"frames": 30, "elapsed": 1.0})


class PreflightCliTest(unittest.TestCase):
    def test_json_mode_with_all_checks_skipped(self) -> None:
        argv = [
            "--json",
            "--skip-python",
            "--skip-data-dir",
            "--skip-dashboard",
            "--skip-videos",
            "--skip-sqlite",
            "--skip-backends",
            "--skip-evidence",
            "--skip-llm",
            "--skip-tests",
            "--skip-run",
        ]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = preflight.main(argv)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(payload["summary"]["ready"])
        self.assertEqual(payload["checks"], [])


if __name__ == "__main__":
    unittest.main()

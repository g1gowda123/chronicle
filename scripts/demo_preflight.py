#!/usr/bin/env python3
"""One-command demo preflight for Chronicle.

The ten-minute demo fails on stage for boring reasons: a missing clip, a
locked SQLite WAL, a detector that silently fell back to colour, a cold Qwen
model, or a test suite that broke after the last commit.  This script checks
all of them and prints a short PASS/WARN/FAIL table with the exact next
command for anything that needs attention.

Usage::

    ./chronicle.sh preflight
    ./chronicle.sh preflight --source tabletop --warm-llm --strict
    .venv/bin/python scripts/demo_preflight.py --json

The default run is model-aware but bounded: it checks every bundled video,
loads the configured production backends once, probes Ollama, runs the test
suite, and ingests 30 synthetic frames into a throwaway database.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep the PASS/WARN/FAIL table readable.  Backend libraries write progress and
# model-loader chatter to stderr; those messages are noise in a preflight
# report and are not actionable for the presenter.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.semantic.config import SemanticConfig  # noqa: E402
from chronicle.sources import (  # noqa: E402
    SOURCES,
    ClipSource,
    default_source,
    source_for_key,
)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
_STATUS_ORDER = {FAIL: 0, WARN: 1, PASS: 2, SKIP: 3}
_STATUS_COLOR = {
    PASS: "\033[32m",
    WARN: "\033[33m",
    FAIL: "\033[31m",
    SKIP: "\033[2m",
}
_RESET = "\033[0m"


@dataclass(frozen=True)
class Check:
    """One preflight result."""

    name: str
    status: str
    detail: str
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "command": self.command,
        }


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    frame: Any | None = None


@contextlib.contextmanager
def _quiet_stderr(enabled: bool = True) -> Iterator[None]:
    """Hide TensorFlow/MediaPipe C++ warnings during backend loading."""

    if not enabled:
        yield
        return
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _tail(text: str, *, lines: int = 4, limit: int = 400) -> str:
    rows = [row.strip() for row in (text or "").splitlines() if row.strip()]
    if not rows:
        return ""
    return " | ".join(rows[-lines:])[-limit:]


def _format_seconds(value: float) -> str:
    return f"{value:.1f}s"



def _source_for_key(key: str, data_dir: Path) -> ClipSource:
    try:
        return source_for_key(key, data_dir)
    except KeyError as exc:
        known = ", ".join(source.key for source in SOURCES)
        raise ValueError(f"unknown source {key!r}; choose one of: {known}") from exc


def _selected_sources(
    keys: Sequence[str] | None,
    data_dir: Path,
) -> tuple[list[ClipSource], bool]:
    """Return sources plus whether the caller named them explicitly."""

    if keys:
        return [_source_for_key(key, data_dir) for key in keys], True
    available = [source for source in SOURCES if source.available(data_dir)]
    if not available:
        available = [default_source(data_dir)]
    return available, False


def probe_video(path: Path) -> VideoInfo:
    """Decode the first and middle frame and return stream metadata."""

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV cannot open {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if frames > 0 and fps > 0 else 0.0
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("first frame did not decode")
        if frames > 2:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames // 2))
            ok_mid, middle = capture.read()
            if not ok_mid or middle is None:
                raise RuntimeError("middle frame did not decode")
        return VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            frames=frames,
            duration=duration,
            frame=frame,
        )
    finally:
        capture.release()


def check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    return Check("Python", PASS, f"{version} · {sys.executable}")


def check_data_dir(data_dir: Path) -> Check:
    if not data_dir.is_dir():
        return Check(
            "Data dir",
            FAIL,
            f"{data_dir} does not exist",
            "mkdir -p data",
        )
    probe = data_dir / ".preflight-write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        return Check(
            "Data dir",
            FAIL,
            f"{data_dir} is not writable ({exc})",
            "chmod u+w data",
        )
    return Check("Data dir", PASS, f"{data_dir} is writable")


def check_dashboard_deps() -> Check:
    missing = [
        module
        for module in ("nicegui", "fastapi")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        return Check(
            "Dashboard",
            FAIL,
            f"missing {', '.join(missing)}",
            "uv sync --extra ui",
        )
    return Check("Dashboard", PASS, "NiceGUI + FastAPI importable")


def check_videos(
    sources: Sequence[ClipSource],
    data_dir: Path,
    *,
    explicit: bool,
) -> tuple[list[Check], dict[str, VideoInfo]]:
    checks: list[Check] = []
    infos: dict[str, VideoInfo] = {}
    for source in sources:
        path = source.video_path(data_dir)
        label = f"Video: {source.key}"
        if not path.is_file():
            status = FAIL if explicit else WARN
            checks.append(
                Check(
                    label,
                    status,
                    f"missing {path}",
                    "git lfs pull",
                )
            )
            continue
        try:
            info = probe_video(path)
        except Exception as exc:  # noqa: BLE001 - preflight must never crash
            checks.append(
                Check(
                    label,
                    FAIL,
                    f"{path.name}: {type(exc).__name__}: {exc}",
                    "git lfs pull",
                )
            )
            continue
        infos[source.key] = info
        detail = (
            f"{path.name} · {info.width}×{info.height} · "
            f"{info.fps:.2f} fps · {info.duration:.1f}s · first+middle frame ok"
        )
        checks.append(Check(label, PASS, detail))
    return checks, infos


def check_sqlite(
    sources: Sequence[ClipSource],
    data_dir: Path,
    temp_dir: Path,
) -> Check:
    problems: list[str] = []
    created = 0
    for source in sources:
        target = source.db_path(data_dir)
        if not target.is_file():
            target = temp_dir / f"{source.key}.db"
            created += 1
        try:
            connection = sqlite3.connect(target, timeout=3.0)
            try:
                mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                quick = connection.execute("PRAGMA quick_check").fetchone()[0]
                checkpoint = connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
            finally:
                connection.close()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{source.key}: {type(exc).__name__}: {exc}")
            continue
        if str(mode).lower() != "wal":
            problems.append(f"{source.key}: journal_mode={mode}")
        if str(quick).lower() != "ok":
            problems.append(f"{source.key}: quick_check={quick}")
        if checkpoint and int(checkpoint[0]) != 0:
            problems.append(
                f"{source.key}: WAL checkpoint busy (a dashboard may be running)"
            )
    if problems:
        return Check(
            "SQLite",
            WARN,
            "; ".join(problems[:3]),
            "./chronicle.sh stop  # then rerun preflight",
        )
    note = f"{len(sources)} database(s) · WAL ok · checkpoint ok"
    if created:
        note += f" · {created} will be created on first run"
    return Check("SQLite", PASS, note)


def _resolved_fallback(
    requested: str,
    actual: str,
    source: ClipSource,
) -> bool:
    return requested == "auto" and source.real and actual in {
        "color-palette",
        "color-skin",
        "color-512",
    }


def check_backends(
    source: ClipSource,
    info: VideoInfo | None,
    data_dir: Path,
    *,
    detector_backend: str,
    hands_backend: str,
    embedder_backend: str,
) -> list[Check]:
    if info is None or info.frame is None:
        return [
            Check(
                "Backends",
                SKIP,
                "no decoded frame; video check failed",
                "git lfs pull",
            )
        ]

    from chronicle.perception.detector import build_detector
    from chronicle.perception.embedder import build_embedder
    from chronicle.perception.hands import build_hands

    config = PerceptionConfig()
    config.model_dir = ROOT / "models"
    config.vocabulary = source.vocabulary
    config.detector.backend = detector_backend
    config.hands.backend = hands_backend
    config.embedder.backend = embedder_backend

    checks: list[Check] = []
    frame = info.frame

    detector = None
    try:
        detector = build_detector(config, source_name=info.path.name)
        started = time.perf_counter()
        detections = detector.detect(frame, 0, 0.0)
        elapsed = time.perf_counter() - started
        actual = getattr(detector, "name", type(detector).__name__)
        status = WARN if _resolved_fallback(detector_backend, actual, source) else PASS
        detail = f"{actual} · {len(detections)} detection(s) · {_format_seconds(elapsed)}"
        command = (
            "uv sync && python scripts/fetch_assets.py"
            if status == WARN
            else None
        )
        checks.append(Check("Detector", status, detail, command))
    except Exception as exc:  # noqa: BLE001
        checks.append(
            Check(
                "Detector",
                FAIL,
                f"{detector_backend}: {type(exc).__name__}: {exc}",
                "uv sync && python scripts/fetch_assets.py",
            )
        )
    finally:
        if detector is not None:
            detector.close()

    hands = None
    try:
        hands = build_hands(config, source_name=info.path.name)
        started = time.perf_counter()
        observations = hands.detect(frame, 0, 0.0)
        elapsed = time.perf_counter() - started
        actual = getattr(hands, "name", type(hands).__name__)
        status = WARN if _resolved_fallback(hands_backend, actual, source) else PASS
        detail = f"{actual} · {len(observations)} hand(s) · {_format_seconds(elapsed)}"
        command = (
            "uv sync && python scripts/fetch_assets.py"
            if status == WARN
            else None
        )
        checks.append(Check("Hands", status, detail, command))
    except Exception as exc:  # noqa: BLE001
        checks.append(
            Check(
                "Hands",
                FAIL,
                f"{hands_backend}: {type(exc).__name__}: {exc}",
                "uv sync && python scripts/fetch_assets.py",
            )
        )
    finally:
        if hands is not None:
            hands.close()

    embedder = None
    try:
        embedder = build_embedder(config, source_name=info.path.name)
        started = time.perf_counter()
        vector = embedder.embed_image(frame)
        elapsed = time.perf_counter() - started
        actual = getattr(embedder, "backend", type(embedder).__name__)
        status = WARN if _resolved_fallback(embedder_backend, actual, source) else PASS
        detail = (
            f"{actual} · shape {getattr(vector, 'shape', '?')} · "
            f"{_format_seconds(elapsed)}"
        )
        command = (
            "uv sync && python scripts/fetch_assets.py"
            if status == WARN
            else None
        )
        checks.append(Check("Embedder", status, detail, command))
    except Exception as exc:  # noqa: BLE001
        checks.append(
            Check(
                "Embedder",
                FAIL,
                f"{embedder_backend}: {type(exc).__name__}: {exc}",
                "uv sync",
            )
        )
    finally:
        if embedder is not None:
            embedder.close()
    return checks


def _count_evidence(episode_dir: Path) -> tuple[int, int, int]:
    index = episode_dir / "evidence" / "index.jsonl"
    if not index.is_file():
        return 0, 0, 0
    indexed = 0
    files = 0
    total = 0
    for line in index.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        indexed += 1
        candidate = index.parent / str(row.get("file") or "")
        if candidate.is_file():
            files += 1
            total += candidate.stat().st_size
    return indexed, files, total


def check_evidence(source: ClipSource, data_dir: Path) -> Check:
    label = f"Evidence: {source.key}"
    episode_dir = source.db_path(data_dir).parent
    indexed, files, total = _count_evidence(episode_dir)
    if indexed and files:
        return Check(
            label,
            PASS,
            f"{files}/{indexed} stored frames · {total / 1e6:.2f} MB · {episode_dir.name}/evidence",
        )
    video = source.video_path(data_dir)
    if not video.is_file():
        return Check(label, SKIP, "video missing", "git lfs pull")
    try:
        from chronicle.frames import extract_video_frame_data_url

        duration = 10.0
        try:
            info = probe_video(video)
            duration = max(1.0, info.duration)
        except Exception:  # noqa: BLE001 - extraction itself is the real check
            pass
        extracted = extract_video_frame_data_url(
            video, duration / 2.0, max_width=320, quality=75
        )
    except Exception as exc:  # noqa: BLE001
        return Check(
            label,
            FAIL,
            f"frame extraction failed: {type(exc).__name__}: {exc}",
            f"python -m chronicle.cli run --source {source.key} --max-frames 1",
        )
    if not extracted:
        return Check(
            label,
            FAIL,
            "video-frame extraction returned no image",
            f"python -m chronicle.cli run --source {source.key} --max-frames 1",
        )
    url, actual_t = extracted
    approx_kb = max(1, len(url) * 3 // 4 // 1024)
    return Check(
        label,
        WARN,
        f"no stored evidence; fallback extraction ok at t={actual_t:.1f}s ({approx_kb} KB)",
        None,
    )


def check_evidence_all(
    sources: Sequence[ClipSource],
    data_dir: Path,
) -> Check:
    """Summarise stored evidence across sources in one readable row."""

    results = [check_evidence(source, data_dir) for source in sources]
    failures = [check for check in results if check.status == FAIL]
    skipped = [check for check in results if check.status == SKIP]
    stored = [check for check in results if check.status == PASS]
    missing = [check for check in results if check.status == WARN]
    if failures:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failures[:3])
        return Check(
            "Evidence",
            FAIL,
            detail,
            "python -m chronicle.cli run --source tabletop --max-frames 120",
        )
    if skipped:
        detail = f"{len(skipped)} source(s) skipped (missing video)"
        return Check("Evidence", WARN, detail, "git lfs pull")
    if missing:
        names = [check.name.split(": ", 1)[-1] for check in missing]
        shown = names if len(names) <= 4 else [*names[:3], f"+{len(names) - 3} more"]
        return Check(
            "Evidence",
            WARN,
            f"{len(stored)}/{len(results)} with stored frames; "
            f"video-frame fallback ok for {', '.join(shown)}",
            "python -m chronicle.cli run --source tabletop --max-frames 120",
        )
    return Check(
        "Evidence",
        PASS,
        f"{len(results)}/{len(results)} sources have stored evidence frames",
    )


def check_llm(
    config: SemanticConfig,
    *,
    warm: bool,
    require: bool,
) -> Check:
    backend = str(config.llm_backend).lower()
    if backend in {"template", "none"}:
        return Check("LLM", PASS, f"{backend} (deterministic answers; no server needed)")
    if backend == "transformers":
        if importlib.util.find_spec("transformers") is None:
            return Check(
                "LLM",
                FAIL,
                "transformers backend selected but the package is not installed",
                "uv sync --extra semantic-models",
            )
        return Check(
            "LLM",
            PASS,
            f"transformers · {config.hf_llm_model} (loaded on first query)",
        )

    from chronicle.semantic.llm import OllamaChatClient, probe_ollama_runtime

    status = probe_ollama_runtime(config.ollama_url, config.llm_model, timeout=2.0)
    if not status.get("server"):
        return Check(
            "LLM",
            FAIL if require else WARN,
            f"Ollama offline at {config.ollama_url}; answers fall back to templates",
            "ollama serve",
        )
    if not status.get("model_present"):
        return Check(
            "LLM",
            FAIL if require else WARN,
            f"{config.llm_model} is not pulled; answers fall back to templates",
            f"ollama pull {config.llm_model}",
        )

    loaded = {str(item.get("name") or "") for item in status.get("loaded", [])}
    is_loaded = config.llm_model in loaded or any(
        name.split(":")[0] == config.llm_model.split(":")[0] for name in loaded
    )
    version = f" · Ollama {status['version']}" if status.get("version") else ""
    if warm:
        client = OllamaChatClient(
            model=config.llm_model,
            base_url=config.ollama_url,
            timeout_s=min(float(config.llm_timeout_s), 120.0),
            keep_alive=config.llm_keep_alive,
            num_ctx=config.llm_num_ctx,
            num_predict=16,
            temperature=0.0,
        )
        started = time.perf_counter()
        try:
            client.complete("Reply with OK.", "ping")
        except Exception as exc:  # noqa: BLE001
            return Check(
                "LLM",
                FAIL,
                f"{config.llm_model} warm-up failed: {type(exc).__name__}: {exc}",
                f"ollama run {config.llm_model} ping",
            )
        elapsed = time.perf_counter() - started
        return Check(
            "LLM",
            PASS,
            f"{config.llm_model} warm in {_format_seconds(elapsed)}{version}",
        )
    if is_loaded:
        return Check(
            "LLM",
            PASS,
            f"{config.llm_model} loaded and warm{version}",
        )
    return Check(
        "LLM",
        WARN,
        f"{config.llm_model} present but cold; first answer may take a few seconds{version}",
        f'ollama run {config.llm_model} "ping"',
    )


def check_tests(root: Path, *, timeout: float = 300.0) -> Check:
    command = [sys.executable, "-m", "pytest", "-q"]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Check(
            "Tests",
            FAIL,
            f"pytest exceeded {_format_seconds(timeout)}",
            ".venv/bin/pytest -q",
        )
    elapsed = time.perf_counter() - started
    if completed.returncode == 0:
        return Check("Tests", PASS, f"pytest exit 0 · {_format_seconds(elapsed)}")
    tail = _tail(completed.stderr or completed.stdout)
    return Check(
        "Tests",
        FAIL,
        f"pytest exit {completed.returncode} · {tail or 'no output'}",
        ".venv/bin/pytest -q",
    )


def _parse_run_summary(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def check_smoke_run(
    root: Path,
    data_dir: Path,
    source: ClipSource,
    *,
    max_frames: int,
    temp_dir: Path,
    timeout: float = 300.0,
) -> Check:
    database = temp_dir / f"smoke-{source.key}.db"
    database.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "chronicle.cli",
        "--data-dir",
        str(data_dir),
        "run",
        "--source",
        source.key,
        "--db",
        str(database),
        "--max-frames",
        str(max_frames),
        "--reset",
        "--llm",
        "template",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Check(
            "Smoke run",
            FAIL,
            f"{source.key} exceeded {_format_seconds(timeout)}",
            f"python -m chronicle.cli run --source {source.key} --max-frames {max_frames}",
        )
    elapsed = time.perf_counter() - started
    summary = _parse_run_summary(completed.stdout)
    if completed.returncode != 0:
        tail = _tail(completed.stderr or completed.stdout)
        return Check(
            "Smoke run",
            FAIL,
            f"exit {completed.returncode} · {tail or 'no output'}",
            f"python -m chronicle.cli run --source {source.key} --max-frames {max_frames}",
        )
    if not summary:
        return Check(
            "Smoke run",
            WARN,
            f"exit 0 in {_format_seconds(elapsed)} but no summary JSON",
            f"python -m chronicle.cli run --source {source.key} --max-frames {max_frames}",
        )
    frames = int(summary.get("frames") or 0)
    run_elapsed = float(summary.get("elapsed") or 0.0)
    fps = frames / run_elapsed if run_elapsed > 0 else 0.0
    entities = int(summary.get("entities") or 0)
    transitions = int(summary.get("transitions") or 0)
    status = PASS if frames > 0 else FAIL
    detail = (
        f"{source.key} · {frames} frames · {fps:.1f} fps · "
        f"{entities} entities · {transitions} transitions · "
        f"{_format_seconds(elapsed)} · throwaway DB"
    )
    return Check(
        "Smoke run",
        status,
        detail,
        None
        if status == PASS
        else f"python -m chronicle.cli run --source {source.key} --max-frames {max_frames}",
    )


def _colorize(status: str, text: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_STATUS_COLOR.get(status, '')}{text}{_RESET}"


def _format_check(check: Check, width: int, *, color: bool) -> str:
    label = f"{check.status:<4}  {check.name:<{width}}  {check.detail}"
    line = _colorize(check.status, label, color)
    if check.command:
        line += f"\n        -> {check.command}"
    return line


def _summary(checks: Sequence[Check], *, strict: bool) -> dict[str, Any]:
    counts = {status: 0 for status in (PASS, WARN, FAIL, SKIP)}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    ready = counts[FAIL] == 0 and (not strict or counts[WARN] == 0)
    return {
        "ready": ready,
        "strict": strict,
        "counts": counts,
        "result": "READY" if ready else "NOT READY",
    }


def _next_commands(checks: Sequence[Check]) -> list[str]:
    commands: list[str] = []
    for check in checks:
        if check.command and check.status in {FAIL, WARN}:
            if check.command not in commands:
                commands.append(check.command)
    if not commands:
        commands.append("./chronicle.sh start tabletop")
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo_preflight",
        description="Check the Chronicle demo environment and print PASS/WARN/FAIL.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="source key to check; repeatable (default: all available sources)",
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--detector", default="auto", help="auto|color|yolo-world")
    parser.add_argument("--hands", default="auto", help="auto|mediapipe|hybrid|color|off")
    parser.add_argument("--embedder", default="auto", help="auto|color|openclip")
    parser.add_argument(
        "--llm",
        default=None,
        help="auto|ollama|transformers|template|none (default: SemanticConfig)",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--warm-llm",
        action="store_true",
        help="send a tiny prompt to load the model before the demo",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="treat an offline/missing Ollama model as FAIL instead of WARN",
    )
    parser.add_argument(
        "--smoke-source",
        default="synthetic",
        help="source used for the 30-frame ingestion smoke test (default: synthetic)",
    )
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-data-dir", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-sqlite", action="store_true")
    parser.add_argument("--skip-backends", action="store_true")
    parser.add_argument("--skip-evidence", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero when any WARN is present",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show third-party model loader output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    color = sys.stdout.isatty() and not args.no_color and not args.json
    checks: list[Check] = []

    def record(check: Check) -> None:
        checks.append(check)
        if not args.json:
            print(_format_check(check, width, color=color), flush=True)

    width = 22
    if not args.json:
        print("Chronicle demo preflight")
        print(f"Data dir: {data_dir}")
        print()

    # Environment
    if not args.skip_python:
        record(check_python())
    if not args.skip_data_dir:
        record(check_data_dir(data_dir))
    if not args.skip_dashboard:
        record(check_dashboard_deps())

    # Sources and video decode
    try:
        sources, explicit = _selected_sources(args.source, data_dir)
    except ValueError as exc:
        record(Check("Sources", FAIL, str(exc)))
        sources, explicit = [], True
    video_infos: dict[str, VideoInfo] = {}
    if sources and not args.skip_videos:
        video_checks, video_infos = check_videos(
            sources, data_dir, explicit=explicit
        )
        for check in video_checks:
            record(check)

    # SQLite / WAL
    temp_dir = Path(tempfile.mkdtemp(prefix="chronicle-preflight-"))
    try:
        if sources and not args.skip_sqlite:
            record(check_sqlite(sources, data_dir, temp_dir))

        # Backends use one real source (or the first selected source).
        primary = None
        for source in sources:
            if source.real and video_infos.get(source.key):
                primary = source
                break
        if primary is None:
            primary = next(
                (source for source in sources if video_infos.get(source.key)),
                sources[0] if sources else None,
            )
        if primary is not None and not args.skip_backends:
            with _quiet_stderr(not args.verbose):
                backend_checks = check_backends(
                    primary,
                    video_infos.get(primary.key),
                    data_dir,
                    detector_backend=args.detector,
                    hands_backend=args.hands,
                    embedder_backend=args.embedder,
                )
            for check in backend_checks:
                record(check)
            if not args.json:
                print(f"        (backends checked on source: {primary.key})")

        # Evidence images / fallback extraction
        if sources and not args.skip_evidence:
            record(check_evidence_all(sources, data_dir))

        # Local LLM
        if not args.skip_llm:
            try:
                semantic = SemanticConfig.from_env()
                changes: dict[str, Any] = {}
                if args.llm:
                    changes["llm_backend"] = args.llm
                if args.llm_model:
                    changes["llm_model"] = args.llm_model
                if changes:
                    semantic = semantic.replace(**changes)
                semantic.validate()
            except Exception as exc:  # noqa: BLE001
                record(
                    Check(
                        "LLM",
                        FAIL,
                        f"configuration error: {type(exc).__name__}: {exc}",
                        "check CHRONICLE_* environment variables",
                    )
                )
            else:
                record(
                    check_llm(
                        semantic,
                        warm=args.warm_llm,
                        require=args.require_llm,
                    )
                )

        # Test suite
        if not args.skip_tests:
            record(check_tests(ROOT))

        # Bounded ingestion smoke test
        if not args.skip_run:
            try:
                smoke_source = _source_for_key(args.smoke_source, data_dir)
            except ValueError as exc:
                record(Check("Smoke run", FAIL, str(exc)))
            else:
                # Make sure the smoke source has a video even when the caller
                # limited --source to a different clip.
                if smoke_source.video_path(data_dir).is_file():
                    record(
                        check_smoke_run(
                            ROOT,
                            data_dir,
                            smoke_source,
                            max_frames=args.max_frames,
                            temp_dir=temp_dir,
                        )
                    )
                else:
                    record(
                        Check(
                            "Smoke run",
                            FAIL,
                            f"missing {smoke_source.video_path(data_dir)}",
                            "git lfs pull",
                        )
                    )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    summary = _summary(checks, strict=args.strict)
    next_commands = _next_commands(checks)
    if args.json:
        print(
            json.dumps(
                {
                    "data_dir": str(data_dir),
                    "checks": [check.to_dict() for check in checks],
                    "summary": summary,
                    "next": next_commands,
                },
                indent=2,
            )
        )
    else:
        counts = summary["counts"]
        print()
        print(
            f"Summary: {counts[PASS]} PASS · {counts[WARN]} WARN · "
            f"{counts[FAIL]} FAIL · {counts[SKIP]} SKIP"
        )
        print(f"Result: {summary['result']}")
        if args.strict and counts[WARN]:
            print("Strict mode: warnings count as failures.")
        print("Next:")
        for command in next_commands:
            print(f"  {command}")
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible runtime benchmark for the bundled real clips.

Role A's acceptance bar includes running the tracked HoloLens/Aria clips through
the integrated A -> B -> C pipeline and reporting throughput plus the state
transitions/actions that came out.  This module is deliberately backend-neutral:
it measures whatever detector/hands/embedder the caller selects, records the
backends that were actually used, and checks the resulting snapshot for
structural invariants.

It does **not** claim perception accuracy on clips without ground-truth labels.
Use :mod:`chronicle.semantic.timeline` with a hand-labelled file when one is
available; the bundled HoloAssist JSON is the only labelled real clip.

Usage::

    python -m chronicle.benchmark --sources cook,play,walking
    python -m chronicle.benchmark --sources tabletop --detector yolo-world

The JSON report is written to ``runs/benchmark/sources.json`` by default.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .integration import ChronicleRuntime
from .perception.config import PerceptionConfig
from .semantic.config import SemanticConfig
from .sources import SOURCES, ClipSource, source_for_key
from .types import ObjectState

DEFAULT_SOURCES = ("cook", "play", "walking")
STATE_VALUES = frozenset(state.value for state in ObjectState)


def video_metadata(path: str | Path) -> dict[str, Any]:
    """Read width/height/fps/frame count without decoding the whole clip."""

    try:
        import cv2  # local import keeps the module importable without OpenCV
    except ImportError:  # pragma: no cover - core dependency is always installed
        return {"width": 0, "height": 0, "fps": 0.0, "frames": 0, "duration": 0.0}

    capture = cv2.VideoCapture(str(path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    duration = frames / fps if fps > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "frames": frames,
        "duration": round(duration, 3),
    }


def check_snapshot(
    summary: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    answers: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    """Structural invariants that must hold for every benchmark run."""

    entities = list(snapshot.get("entities") or [])
    actions = list(snapshot.get("actions") or [])
    transitions = list(snapshot.get("transitions") or [])
    entity_ids = {str(row.get("id")) for row in entities}

    try:
        json.dumps(snapshot)
        serializable = True
    except (TypeError, ValueError):
        serializable = False

    ordered = True
    last_timestamp: dict[str, float] = {}
    for row in transitions:
        entity_id = str(row.get("entity_id"))
        timestamp = float(row.get("timestamp") or 0.0)
        if entity_id in last_timestamp and timestamp < last_timestamp[entity_id] - 1e-9:
            ordered = False
            break
        last_timestamp[entity_id] = timestamp

    stats = snapshot.get("stats") or {}
    return {
        "snapshot_serializable": serializable,
        "entities_unique": len(entity_ids) == len(entities),
        "actions_reference_entities": all(
            str(row.get("entity_id")) in entity_ids for row in actions
        ),
        "transitions_reference_entities": all(
            str(row.get("entity_id")) in entity_ids for row in transitions
        ),
        "transition_states_known": all(
            str(row.get("from_state")) in STATE_VALUES
            and str(row.get("to_state")) in STATE_VALUES
            for row in transitions
        ),
        "transition_times_ordered": ordered,
        "summary_matches_snapshot": int(summary.get("frames") or 0)
        == int(stats.get("frames") or 0),
        "questions_answered": all(str(row.get("answer") or "").strip() for row in answers),
    }


@dataclass(frozen=True)
class SourceBenchmark:
    source: str
    label: str
    video: str
    metadata: Mapping[str, Any]
    summary: Mapping[str, Any]
    backends: Mapping[str, str]
    wall_time: float
    answers: tuple[Mapping[str, Any], ...]
    checks: Mapping[str, bool]
    error: str | None = None

    @property
    def frames(self) -> int:
        return int(self.summary.get("frames") or 0)

    @property
    def throughput(self) -> float:
        return self.frames / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.wall_time / self.frames if self.frames else 0.0

    @property
    def grounded(self) -> int:
        return sum(1 for row in self.answers if row.get("grounded"))

    @property
    def passed(self) -> bool:
        return self.error is None and all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "label": self.label,
            "video": self.video,
            "metadata": dict(self.metadata),
            "summary": dict(self.summary),
            "backends": dict(self.backends),
            "wall_time": round(self.wall_time, 3),
            "throughput_fps": round(self.throughput, 2),
            "latency_ms_per_frame": round(self.latency_ms, 2),
            "questions": [dict(row) for row in self.answers],
            "grounded_answers": self.grounded,
            "checks": dict(self.checks),
            "passed": self.passed,
            "error": self.error,
        }


def run_source(
    source: ClipSource,
    data_dir: Path | str,
    work_dir: Path | str,
    *,
    detector: str = "auto",
    hands: str = "auto",
    embedder: str = "auto",
    max_frames: int | None = None,
    stride: int = 1,
) -> SourceBenchmark:
    """Ingest one clip and collect throughput + snapshot checks."""

    video = source.video_path(data_dir)
    metadata = video_metadata(video)
    work = Path(work_dir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / f"{source.key}.db"

    config = PerceptionConfig(vocabulary=source.vocabulary)
    config.detector.backend = detector
    config.hands.backend = hands
    config.embedder.backend = embedder
    config.frame_stride = max(1, int(stride))

    started = time.perf_counter()
    error: str | None = None
    summary_payload: Mapping[str, Any] = {}
    snapshot: Mapping[str, Any] = {}
    answers: tuple[Mapping[str, Any], ...] = ()
    backends: Mapping[str, str] = {}
    runtime: ChronicleRuntime | None = None
    try:
        runtime = ChronicleRuntime(
            db_path,
            perception_config=config,
            semantic_config=SemanticConfig(llm_backend="template"),
            zone_label=source.zone_label,
            reset=True,
        )
        summary = runtime.ingest_video(video, max_frames=max_frames)
        snapshot = runtime.snapshot()
        answers = tuple(runtime.ask(question).to_dict() for question in source.questions)
        backends = runtime.backends()
        summary_payload = summary.as_dict()
    except Exception as exc:  # a failed clip should be reported, not abort the suite
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if runtime is not None:
            runtime.close()
    wall_time = time.perf_counter() - started

    checks = {"run_completed": error is None}
    if error is None:
        checks.update(check_snapshot(summary_payload, snapshot, answers))
    return SourceBenchmark(
        source=source.key,
        label=source.label,
        video=str(video),
        metadata=metadata,
        summary=summary_payload,
        backends=backends,
        wall_time=wall_time,
        answers=answers,
        checks=checks,
        error=error,
    )


def _selected_sources(keys: Sequence[str]) -> tuple[ClipSource, ...]:
    if not keys or list(keys) == ["all"]:
        return SOURCES
    return tuple(source_for_key(key, ".") for key in keys)


def format_markdown(results: Sequence[SourceBenchmark]) -> str:
    """Render the table pasted into ``docs/PERCEPTION.md``."""

    lines = [
        "| Source | Clip | Frames | Wall time | Throughput | Detections | Entities | Transitions | Actions | Clips | Grounded queries | Checks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        summary = result.summary
        resolution = (
            f"{result.metadata.get('width')}x{result.metadata.get('height')}"
            f" @ {result.metadata.get('fps')} fps"
        )
        passed = sum(1 for value in result.checks.values() if value)
        total = len(result.checks)
        checks = f"{passed}/{total}" + ("" if result.passed else " FAIL")
        lines.append(
            "| {source} | {resolution} | {frames} | {wall:.2f}s | {fps:.1f} fps | "
            "{detections} | {entities} | {transitions} | {actions} | {clips} | "
            "{grounded}/{asked} | {checks} |".format(
                source=result.source,
                resolution=resolution,
                frames=summary.get("frames", 0),
                wall=result.wall_time,
                fps=result.throughput,
                detections=summary.get("detections", 0),
                entities=summary.get("entities", 0),
                transitions=summary.get("transitions", 0),
                actions=summary.get("actions", 0),
                clips=summary.get("clips", 0),
                grounded=result.grounded,
                asked=len(result.answers),
                checks=checks,
            )
        )
    return "\n".join(lines)


def _write_report(results: Sequence[SourceBenchmark], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [result.to_dict() for result in results],
        "passed": all(result.passed for result in results),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help="comma-separated source keys, or 'all' (default: cook,play,walking)",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--work-dir", default="runs/benchmark")
    parser.add_argument(
        "--out",
        default="runs/benchmark/sources.json",
        help="JSON report path (default: runs/benchmark/sources.json)",
    )
    parser.add_argument("--detector", default="auto", help="auto|color|yolo-world")
    parser.add_argument("--hands", default="auto", help="auto|color|mediapipe")
    parser.add_argument("--embedder", default="auto", help="auto|color|openclip")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="print the JSON report")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    sources = _selected_sources([part.strip() for part in args.sources.split(",")])
    missing = [source.key for source in sources if not source.available(data_dir)]
    if missing:
        print("skipping missing clips: " + ", ".join(missing))

    results = [
        run_source(
            source,
            data_dir,
            work_dir,
            detector=args.detector,
            hands=args.hands,
            embedder=args.embedder,
            max_frames=args.max_frames,
            stride=args.stride,
        )
        for source in sources
        if source.available(data_dir)
    ]
    if not results:
        print("no benchmark clips were available")
        return 1

    _write_report(results, Path(args.out).expanduser())
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        print(format_markdown(results))
        for result in results:
            if result.error:
                print(f"{result.source}: ERROR {result.error}")
    return 0 if all(result.passed for result in results) else 1


__all__ = [
    "DEFAULT_SOURCES",
    "SourceBenchmark",
    "check_snapshot",
    "format_markdown",
    "main",
    "run_source",
    "video_metadata",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

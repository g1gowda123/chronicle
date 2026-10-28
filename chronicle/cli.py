"""Command line entry points for the integrated Chronicle runtime.

Usage::

    chronicle sources
    chronicle run data/demo/chronicle_demo.mp4 --max-frames 120
    chronicle ask "where are my keys?" --db data/demo/chronicle.db
    chronicle serve --source synthetic

The dashboard is optional; ``run``/``ask``/``sources`` only need the core
dependencies and work offline with the deterministic backends.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from chronicle.sources import DEFAULT_DATA_DIR

if TYPE_CHECKING:
    from chronicle.perception.config import PerceptionConfig
    from chronicle.semantic.config import SemanticConfig
    from chronicle.sources import ClipSource


def _runtime_class():
    """Resolve the runtime lazily, while leaving it patchable in tests."""

    runtime = globals().get("ChronicleRuntime")
    if runtime is not None:
        return runtime
    from chronicle.integration import ChronicleRuntime

    return ChronicleRuntime


def __getattr__(name: str):
    if name == "ChronicleRuntime":
        from chronicle.integration import ChronicleRuntime

        return ChronicleRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_source(
    data_dir: Path,
    *,
    source_key: str | None = None,
    video: str | None = None,
) -> tuple[ClipSource, Path]:
    """Resolve a source key or explicit video path to a source + path."""

    from chronicle.perception.config import PerceptionConfig
    from chronicle.sources import (
        ClipSource,
        default_source,
        source_for_key,
        source_for_video,
    )

    if video:
        path = Path(video).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"video not found: {path}")
        matched = source_for_video(path, data_dir)
        if matched is not None:
            return matched, path
        source = ClipSource(
            key="custom",
            label=path.stem,
            badge="CUSTOM",
            caption=path.name,
            description="Explicit video passed on the command line.",
            relative_path=str(path),
            relative_db="custom/chronicle.db",
            zone_label="Custom Scene",
            vocabulary=PerceptionConfig().vocabulary,
            questions=("what did I see?",),
        )
        return source, path

    source = source_for_key(source_key, data_dir) if source_key else default_source(data_dir)
    path = source.video_path(data_dir)
    if not path.is_file():
        raise SystemExit(
            f"video not found for source {source.key!r}: {path}\n"
            f"Run `chronicle sources` to list available clips."
        )
    return source, path


def _perception_config(
    args: argparse.Namespace, source: ClipSource | None = None
) -> PerceptionConfig:
    from chronicle.perception.config import PerceptionConfig

    config = PerceptionConfig()
    config.detector.backend = args.detector
    config.hands.backend = args.hands
    config.embedder.backend = args.embedder
    if source is not None and source.vocabulary:
        config.vocabulary = tuple(source.vocabulary)
    if getattr(args, "conf", None) is not None:
        config.detector.confidence = float(args.conf)
    if getattr(args, "imgsz", None) is not None:
        config.detector.imgsz = max(32, int(args.imgsz))
    if getattr(args, "max_frames", None) is not None:
        config.max_frames = int(args.max_frames)
    return config


def _semantic_config(
    args: argparse.Namespace, *, default_llm: str | None = None
) -> SemanticConfig:
    from chronicle.semantic.config import SemanticConfig

    config = SemanticConfig.from_env()
    changes = {}
    if getattr(args, "llm", None) is not None:
        changes["llm_backend"] = args.llm
    elif default_llm is not None:
        changes["llm_backend"] = default_llm
    if getattr(args, "llm_model", None) is not None:
        changes["llm_model"] = args.llm_model
    return config.replace(**changes) if changes else config


def _ensure_ollama(config: SemanticConfig) -> None:
    """Start Ollama only when the selected answer path can actually use it."""

    if getattr(config, "llm_backend", "auto") in {"auto", "ollama"}:
        from chronicle.semantic.llm import ensure_local_ollama

        ensure_local_ollama(config)


def cmd_sources(args: argparse.Namespace) -> int:
    from chronicle.sources import SOURCES

    data_dir = Path(args.data_dir).expanduser().resolve()
    rows = []
    for source in SOURCES:
        path = source.video_path(data_dir)
        rows.append(
            {
                "key": source.key,
                "label": source.label,
                "available": path.is_file(),
                "video": str(path),
                "db": str(source.db_path(data_dir)),
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    runtime_cls = _runtime_class()

    data_dir = Path(args.data_dir).expanduser().resolve()
    source, video = _resolve_source(
        data_dir,
        source_key=args.source,
        video=args.video,
    )
    db_path = Path(args.db).expanduser() if args.db else source.db_path(data_dir)
    sem_cfg = _semantic_config(args)
    _ensure_ollama(sem_cfg)
    runtime = runtime_cls(
        db_path,
        perception_config=_perception_config(args, source),
        semantic_config=sem_cfg,
        zone_label=args.zone or source.zone_label,
        reset=args.reset,
    )
    try:
        summary = runtime.ingest_video(
            video,
            max_frames=args.max_frames,
            fps_override=args.fps,
        )
        print(json.dumps(summary.as_dict(), indent=2))
        if args.snapshot_out:
            snapshot_path = Path(args.snapshot_out).expanduser()
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(runtime.export_snapshot_json(), encoding="utf-8")
            print(json.dumps({"snapshot": str(snapshot_path)}, indent=2))
        if args.ask:
            for question in args.ask:
                print(json.dumps(runtime.ask(question).to_dict(), indent=2))
        return 0
    finally:
        runtime.close()


def cmd_ask(args: argparse.Namespace) -> int:
    from chronicle.demo_data import demo_db_path

    runtime_cls = _runtime_class()

    if args.db:
        db_path = Path(args.db).expanduser()
        if not db_path.is_file():
            raise SystemExit(
                f"memory database not found: {db_path}\n"
                "Run `chronicle demo` first or pass --db to an existing memory."
            )
    else:
        db_path = demo_db_path()
        print(f"using bundled demo memory: {db_path}", file=sys.stderr)

    sem_cfg = _semantic_config(args, default_llm="template")
    _ensure_ollama(sem_cfg)
    runtime = runtime_cls(
        db_path,
        semantic_config=sem_cfg,
        load=True,
        read_only=True,
    )
    try:
        print(json.dumps(runtime.ask(args.question).to_dict(), indent=2))
        return 0
    finally:
        runtime.close()


def cmd_eval(args: argparse.Namespace) -> int:
    from chronicle.evaluation import format_report, run_evaluation

    try:
        report = run_evaluation(args.db, args.suite)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0 if report["failed"] == 0 else 1


def cmd_demo(args: argparse.Namespace) -> int:
    from chronicle.demo import main as demo_main

    argv: list[str] = []
    if args.db:
        argv.extend(["--db", args.db])
    if args.force:
        argv.append("--force")
    if args.json:
        argv.append("--json")
    return demo_main(argv)


def cmd_doctor(args: argparse.Namespace) -> int:
    from chronicle.doctor import main as doctor_main

    return doctor_main(["--json"] if args.json else [])


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from chronicle.ui import create_app
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "The dashboard needs the optional UI dependencies.\n"
            "Install them with: uv sync --extra ui  (or pip install 'nicegui>=3.0')"
        ) from exc

    sem_cfg = _semantic_config(args)
    _ensure_ollama(sem_cfg)

    create_app(
        data_dir=args.data_dir,
        video=args.video,
        source_key=args.source,
        auto_run=args.auto_run,
        port=args.port,
        host=args.host,
        show=args.show,
        max_frames=args.max_frames,
        detector_backend=args.detector,
        hands_backend=args.hands,
        embedder_backend=args.embedder,
        semantic_config=sem_cfg,
        qwen=args.qwen,
        qwen_model=args.qwen_model,
        vfm_enabled=not args.no_vfm,
        vfm_window_s=args.vfm_window_s,
        vfm_frames_per_window=args.vfm_frames_per_window,
        vfm_max_snapshots=args.vfm_max_snapshots,
        vfm_delete_video=args.vfm_delete_video,
    )
    return 0


def _add_backend_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", default="auto", help="auto|color|yolo-world")
    parser.add_argument("--hands", default="auto", help="auto|color|mediapipe")
    parser.add_argument("--embedder", default="auto", help="auto|color|deterministic|openclip")
    parser.add_argument("--llm", default=None, help="auto|ollama|transformers|template")
    parser.add_argument("--llm-model", default=None, help="Ollama / HF model name (default: qwen2.5:7b-instruct)")


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("chronicle-memory")
    except Exception:  # pragma: no cover - source tree without installed metadata
        from chronicle import __version__

        return __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronicle",
        description="Chronicle egocentric episodic memory: ingest, query, dashboard.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="run the zero-model synthetic demo (no video or weights needed)"
    )
    demo.add_argument("--db", help="persist the demo memory to this SQLite path")
    demo.add_argument(
        "--force", action="store_true", help="overwrite --db if it already exists"
    )
    demo.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    demo.set_defaults(func=cmd_demo)

    evaluation = subparsers.add_parser(
        "eval",
        help="run the deterministic model-free product smoke suite",
    )
    evaluation.add_argument(
        "--db",
        help="SQLite memory path (default: the bundled demo memory)",
    )
    evaluation.add_argument(
        "--suite",
        help="JSONL question suite (default: the bundled 45-case suite)",
    )
    evaluation.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    evaluation.set_defaults(func=cmd_eval)

    doctor = subparsers.add_parser(
        "doctor", help="check core, optional, CUDA and demo-memory availability"
    )
    doctor.add_argument("--json", action="store_true", help="emit a machine-readable report")
    doctor.set_defaults(func=cmd_doctor)

    sources = subparsers.add_parser("sources", help="list bundled/demo video sources")
    sources.set_defaults(func=cmd_sources)

    run = subparsers.add_parser("run", help="ingest a video through perception, memory and semantics")
    run.add_argument("video", nargs="?", help="explicit video path (defaults to --source)")
    run.add_argument("--source", help="source key from `chronicle sources`")
    run.add_argument("--db", help="SQLite database path (defaults to the source database)")
    run.add_argument("--zone", help="zone label override")
    run.add_argument("--max-frames", type=int)
    run.add_argument("--fps", type=float, help="override video FPS")
    run.add_argument("--conf", type=float, help="detector confidence threshold override")
    run.add_argument("--imgsz", type=int, help="detector input size override")
    run.add_argument("--reset", action="store_true", help="recreate the database before ingesting")
    run.add_argument("--ask", action="append", help="question to answer after ingestion")
    run.add_argument(
        "--snapshot-out",
        help="write the post-ingestion ChronicleRuntime snapshot JSON to this path",
    )
    _add_backend_flags(run)
    run.set_defaults(func=cmd_run)

    ask = subparsers.add_parser("ask", help="ask a persisted memory a question")
    ask.add_argument("question")
    ask.add_argument(
        "--db",
        help="SQLite memory path (default: the bundled demo memory)",
    )
    _add_backend_flags(ask)
    ask.set_defaults(func=cmd_ask)

    serve = subparsers.add_parser("serve", help="launch the NiceGUI dashboard")
    serve.add_argument("--source", help="source key from `chronicle sources`")
    serve.add_argument("--video", help="explicit video path")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--show", action="store_true", help="open a browser window")
    serve.add_argument("--auto-run", action="store_true", help="start ingestion immediately")
    serve.add_argument("--max-frames", type=int)
    serve.add_argument(
        "--qwen",
        action="store_true",
        help="enable Qwen answers over the compiled VFM memory pack (default: off)",
    )
    serve.add_argument(
        "--qwen-model",
        default=None,
        help="Ollama model for VFM answers (default: semantic config llm_model)",
    )
    serve.add_argument(
        "--no-vfm",
        action="store_true",
        help="disable VFM compiles/asks (the Memory tab stays visible but inert)",
    )
    serve.add_argument(
        "--vfm-window-s",
        type=float,
        default=30.0,
        help="VLM caption window in seconds (default: 30)",
    )
    serve.add_argument(
        "--vfm-frames-per-window",
        type=int,
        default=3,
        help="frames captioned per VLM window (default: 3)",
    )
    serve.add_argument(
        "--vfm-max-snapshots",
        type=int,
        default=40,
        help="maximum WebP evidence snapshots (default: 40)",
    )
    serve.add_argument(
        "--vfm-delete-video",
        action="store_true",
        help="delete uploaded source videos after compiling (opt-in; playback is lost)",
    )
    _add_backend_flags(serve)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]

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

from chronicle.integration import ChronicleRuntime
from chronicle.perception.config import PerceptionConfig
from chronicle.semantic.config import SemanticConfig
from chronicle.sources import (
    SOURCES,
    ClipSource,
    default_source,
    source_for_key,
    source_for_video,
)

DEFAULT_DATA_DIR = Path("data")


def _resolve_source(
    data_dir: Path,
    *,
    source_key: str | None = None,
    video: str | None = None,
) -> tuple[ClipSource, Path]:
    """Resolve a source key or explicit video path to a source + path."""

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


def _perception_config(args: argparse.Namespace) -> PerceptionConfig:
    config = PerceptionConfig()
    config.detector.backend = args.detector
    config.hands.backend = args.hands
    config.embedder.backend = args.embedder
    if getattr(args, "max_frames", None) is not None:
        config.max_frames = int(args.max_frames)
    return config


def cmd_sources(args: argparse.Namespace) -> int:
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
    data_dir = Path(args.data_dir).expanduser().resolve()
    source, video = _resolve_source(
        data_dir,
        source_key=args.source,
        video=args.video,
    )
    db_path = Path(args.db).expanduser() if args.db else source.db_path(data_dir)
    runtime = ChronicleRuntime(
        db_path,
        perception_config=_perception_config(args),
        semantic_config=SemanticConfig.from_env(),
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
        if args.ask:
            for question in args.ask:
                print(json.dumps(runtime.ask(question).to_dict(), indent=2))
        return 0
    finally:
        runtime.close()


def cmd_ask(args: argparse.Namespace) -> int:
    runtime = ChronicleRuntime(
        args.db,
        semantic_config=SemanticConfig.from_env(),
        load=True,
    )
    try:
        print(json.dumps(runtime.ask(args.question).to_dict(), indent=2))
        return 0
    finally:
        runtime.close()


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from chronicle.ui import create_app
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "The dashboard needs the optional UI dependencies.\n"
            "Install them with: uv sync --extra ui  (or pip install 'nicegui>=3.0')"
        ) from exc

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
    )
    return 0


def _add_backend_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", default="auto", help="auto|color|yolo-world")
    parser.add_argument("--hands", default="auto", help="auto|color|mediapipe")
    parser.add_argument("--embedder", default="auto", help="auto|color|deterministic|openclip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronicle",
        description="Chronicle egocentric episodic memory: ingest, query, dashboard.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    subparsers = parser.add_subparsers(dest="command", required=True)

    sources = subparsers.add_parser("sources", help="list bundled/demo video sources")
    sources.set_defaults(func=cmd_sources)

    run = subparsers.add_parser("run", help="ingest a video through perception, memory and semantics")
    run.add_argument("video", nargs="?", help="explicit video path (defaults to --source)")
    run.add_argument("--source", help="source key from `chronicle sources`")
    run.add_argument("--db", help="SQLite database path (defaults to the source database)")
    run.add_argument("--zone", help="zone label override")
    run.add_argument("--max-frames", type=int)
    run.add_argument("--fps", type=float, help="override video FPS")
    run.add_argument("--reset", action="store_true", help="recreate the database before ingesting")
    run.add_argument("--ask", action="append", help="question to answer after ingestion")
    _add_backend_flags(run)
    run.set_defaults(func=cmd_run)

    ask = subparsers.add_parser("ask", help="ask a persisted memory a question")
    ask.add_argument("question")
    ask.add_argument("--db", required=True)
    ask.set_defaults(func=cmd_ask)

    serve = subparsers.add_parser("serve", help="launch the NiceGUI dashboard")
    serve.add_argument("--source", help="source key from `chronicle sources`")
    serve.add_argument("--video", help="explicit video path")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--show", action="store_true", help="open a browser window")
    serve.add_argument("--auto-run", action="store_true", help="start ingestion immediately")
    serve.add_argument("--max-frames", type=int)
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

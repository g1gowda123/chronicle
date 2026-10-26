"""Launch the Chronicle Role D dashboard.

Usage::

    python run_dashboard.py                 # demo replay on :8081
    python run_dashboard.py --port 9000
    python run_dashboard.py --source live   # wait for POST /api/ingest
    python run_dashboard.py --source jsonl --events path/to/events.jsonl

The dashboard is standalone: with no arguments it replays the bundled synthetic
episode so judges can use it without the perception pipeline running.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chronicle dashboard (Role D)")
    parser.add_argument("--host", default=os.getenv("CHRONICLE_HOST", "0.0.0.0"), help="bind address")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("CHRONICLE_PORT", "8081")), help="HTTP port"
    )
    parser.add_argument(
        "--source",
        choices=("demo", "live", "jsonl", "sqlite"),
        default=os.getenv("CHRONICLE_SOURCE", "demo"),
        help="demo replay, live HTTP ingest, or a producer adapter",
    )
    parser.add_argument("--data-dir", default=os.getenv("CHRONICLE_DATA_DIR"), help="asset directory")
    parser.add_argument("--events", default=os.getenv("CHRONICLE_EVENTS_JSONL"), help="JSONL event file")
    parser.add_argument(
        "--replay",
        default=os.getenv("CHRONICLE_REPLAY"),
        help="replay a recorded .events.jsonl timeline (real clips) instead of the bundled demo",
    )
    parser.add_argument("--db", default=os.getenv("CHRONICLE_DB"), help="producer SQLite database")
    parser.add_argument("--video", default=os.getenv("CHRONICLE_VIDEO"), help="override video URL")
    parser.add_argument(
        "--start", type=float, default=None, help="initial playhead in seconds (demo only)"
    )
    parser.add_argument("--show", action="store_true", help="open a browser window")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    os.environ["CHRONICLE_SOURCE"] = "demo" if args.replay else args.source
    if args.data_dir:
        os.environ["CHRONICLE_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())
    if args.replay:
        os.environ["CHRONICLE_REPLAY"] = str(Path(args.replay).expanduser().resolve())
    if args.events:
        os.environ["CHRONICLE_EVENTS_JSONL"] = str(Path(args.events).expanduser().resolve())
    if args.db:
        os.environ["CHRONICLE_DB"] = str(Path(args.db).expanduser().resolve())
    if args.video:
        os.environ["CHRONICLE_VIDEO"] = args.video
    if args.start is not None:
        os.environ["CHRONICLE_DEMO_START"] = str(args.start)

    # Import after env is set: the shared Runtime is built at import time.
    from dashboard.app import run
    from dashboard.runtime import default_data_dir

    data_dir = default_data_dir()
    print("=" * 68)
    print(" CHRONICLE · Role D mission control")
    print(f" source     : {args.source}")
    print(f" data dir   : {data_dir}")
    print(f" dashboard  : http://localhost:{args.port}")
    print(f" api docs   : http://localhost:{args.port}/docs")
    print("=" * 68)
    sys.stdout.flush()

    run(host=args.host, port=args.port, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

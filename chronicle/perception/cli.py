"""Command-line entry point for the Role A perception pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from chronicle.perception.config import DEFAULT_VOCABULARY, PerceptionConfig
from chronicle.perception.pipeline import PerceptionPipeline
from chronicle.perception.telemetry import TelemetryPrinter
from chronicle.perception.visualize import render_overlay


def _source(value: str) -> str | int:
    if value.isdigit():
        return int(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronicle-perception",
        description="Chronicle Role A: video -> tracks, camera gating and interaction clips.",
    )
    parser.add_argument("video", help="video path, or camera index (e.g. 0)")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_VOCABULARY),
        help="comma-separated open-vocabulary targets",
    )
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--hand-model", default="hand_landmarker.task")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    parser.add_argument(
        "--detector", choices=("auto", "yolo-world", "color"), default="auto"
    )
    parser.add_argument(
        "--hands", choices=("auto", "mediapipe", "hybrid", "color", "off"), default="auto"
    )
    parser.add_argument(
        "--embedder", choices=("auto", "openclip", "color"), default="auto"
    )
    parser.add_argument("--conf", type=float, default=None, help="detection confidence")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--min-hits", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="override source FPS")
    parser.add_argument("--jsonl", default=None, help="write per-frame JSONL here")
    parser.add_argument("--clips-dir", default=None, help="save interaction composites here")
    parser.add_argument("--save-clip-frames", action="store_true")
    parser.add_argument("--save-video", default=None, help="write an annotated overlay video")
    parser.add_argument("--show", action="store_true", help="preview the overlay")
    parser.add_argument("--no-motion", action="store_true")
    parser.add_argument("--no-clips", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    if not labels:
        print("error: --labels must contain at least one target", file=sys.stderr)
        return 2

    config = PerceptionConfig(vocabulary=labels, model_dir=Path(args.model_dir), device=args.device)
    config.detector.backend = args.detector
    config.detector.model = args.model
    config.hands.backend = args.hands
    config.hands.model = args.hand_model
    config.embedder.backend = args.embedder
    config.frame_stride = max(1, args.stride)
    config.max_frames = args.max_frames
    if args.conf is not None:
        config.detector.confidence = args.conf
    if args.imgsz is not None:
        config.detector.imgsz = max(32, args.imgsz)
    if args.min_hits is not None:
        config.tracker.min_hits = max(1, args.min_hits)
    if args.no_motion:
        config.motion.enabled = False
    if args.no_clips:
        config.clip.enabled = False
    if args.save_clip_frames:
        config.clip.save_frames = True

    sink = None if args.quiet else TelemetryPrinter()
    pipeline = PerceptionPipeline(
        config,
        telemetry_sink=sink,
        output_jsonl=args.jsonl,
        clips_dir=args.clips_dir,
    )
    source = _source(args.video)
    writer: cv2.VideoWriter | None = None
    results = None
    frame_count = 0
    show = args.show
    try:
        results = pipeline.iter_results(source, fps_override=args.fps)
        for result in results:
            frame_count += 1
            overlay = None
            if (args.save_video or show) and result.image is not None:
                overlay = render_overlay(result.image, result)
            if args.save_video and overlay is not None:
                if writer is None:
                    height, width = overlay.shape[:2]
                    writer = cv2.VideoWriter(
                        args.save_video,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(pipeline.fps),
                        (width, height),
                    )
                    if not writer.isOpened():
                        print(f"warning: could not open video writer {args.save_video}", file=sys.stderr)
                        writer = None
                if writer is not None:
                    writer.write(overlay)
            if show and overlay is not None:
                try:
                    cv2.imshow("Chronicle perception", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("warning: OpenCV GUI is unavailable; continuing headless", file=sys.stderr)
                    show = False
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if results is not None:
            results.close()
        if writer is not None:
            writer.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        pipeline.close()

    summary = pipeline.summary()
    if not args.quiet:
        print()
        print(json.dumps(summary.as_dict(), indent=2))
    if frame_count == 0:
        print("error: no frames were processed", file=sys.stderr)
        return 1
    return 0

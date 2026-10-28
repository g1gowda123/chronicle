"""Run Chronicle on the EgoTempo clips and collect predictions.

Each EgoTempo clip is ingested once; all questions for that clip are answered
from the final graph (EgoTempo questions are about the whole clip, not a
mid-clip timestamp).  Predictions are written in the format consumed by
``scripts/evaluate_egotempo.py``.

This is an inference-only benchmark: no model training is required.

Example
-------
    # Smoke test on 3 clips
    python scripts/run_egotempo.py --limit-clips 3

    # Full benchmark
    python scripts/run_egotempo.py --out runs/egotempo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_pipeline import run_evaluation  # noqa: E402


def _clip_qa_payload(clip: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": clip["clip_id"],
        "questions": [
            {
                "question_id": question["question_id"],
                "query": question["question"],
                "answer": question["answer"],
                "category": question["question_type"],
            }
            for question in clip["questions"]
        ],
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/benchmarks/egotempo/manifest.json"),
    )
    parser.add_argument(
        "--clips-index",
        type=Path,
        default=Path("data/benchmarks/egotempo/clips.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/egotempo"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--clip-id", action="append", default=None)
    parser.add_argument(
        "--vocabulary-preset",
        choices=("default", "broad", "none"),
        default="broad",
        help="fixed generic detector vocabulary; never derived from questions",
    )
    parser.add_argument("--caption-stride", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        raise SystemExit(
            f"missing {args.manifest}; run scripts/egotempo_manifest.py first"
        )
    if not args.clips_index.is_file():
        raise SystemExit(
            f"missing {args.clips_index}; run scripts/prepare_egotempo.py first"
        )

    manifest = _load_json(args.manifest)
    clips_index = _load_json(args.clips_index)
    clips = list(manifest.get("clips") or ())
    if args.clip_id:
        wanted = set(args.clip_id)
        clips = [clip for clip in clips if clip["clip_id"] in wanted]
    if args.limit_clips is not None:
        clips = clips[: max(0, args.limit_clips)]

    args.out.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out / "predictions.json"
    clip_metrics_path = args.out / "clip_metrics.json"
    predictions: dict[str, str] = {}
    if args.skip_existing and predictions_path.is_file():
        predictions = {
            str(key): str(value)
            for key, value in _load_json(predictions_path).items()
        }
    metrics: list[dict[str, Any]] = []

    print(f"Running {len(clips)} EgoTempo clips")
    for index, clip in enumerate(clips, start=1):
        clip_id = str(clip["clip_id"])
        video_path = Path(str(clips_index.get(clip_id, "")))
        if not video_path.is_file():
            print(f"[{index}/{len(clips)}] missing clip file: {clip_id}")
            continue
        if args.skip_existing and all(
            question["question_id"] in predictions for question in clip["questions"]
        ):
            print(f"[{index}/{len(clips)}] skip existing {clip_id}")
            continue

        qa_path = args.out / "qa" / f"{clip_id}.json"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        qa_path.write_text(
            json.dumps(_clip_qa_payload(clip), indent=2), encoding="utf-8"
        )
        clip_out = args.out / "clips" / clip_id
        overrides: dict[str, Any] = {
            "paths": {"db_path": str(clip_out / "chronicle.db")},
            "perception": {"vocabulary_preset": args.vocabulary_preset},
        }
        if args.caption_stride is not None:
            overrides.setdefault("semantic", {})["caption_stride"] = int(
                args.caption_stride
            )

        print(f"[{index}/{len(clips)}] {clip_id} ({len(clip['questions'])} questions)")
        if args.dry_run:
            continue
        started = time.perf_counter()
        try:
            result = run_evaluation(
                video_path,
                qa_path,
                config_path=args.config,
                out_dir=clip_out,
                overrides=overrides,
                show_progress=False,
            )
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}")
            for question in clip["questions"]:
                predictions[question["question_id"]] = ""
            continue

        rows = list(result.get("rows") or ())
        # All EgoTempo questions are untimed, so the evaluator preserves the
        # input order.  Fall back to a query lookup if a run returns fewer rows.
        by_query = {str(row.get("query")): row for row in rows}
        for position, question in enumerate(clip["questions"]):
            if position < len(rows):
                row = rows[position]
            else:
                row = by_query.get(str(question["question"]), {})
            predictions[question["question_id"]] = str(row.get("predicted", ""))
        elapsed = time.perf_counter() - started
        clip_metrics = {
            "clip_id": clip_id,
            "questions": len(clip["questions"]),
            "elapsed_s": round(elapsed, 3),
            "vlm_calls": (result.get("metrics") or {}).get("vlm_calls"),
            "peak_vram_mb": (result.get("metrics") or {}).get("peak_vram_mb"),
            "avg_query_latency_ms": result.get("avg_query_latency_ms"),
            "error": (result.get("metrics") or {}).get("error"),
        }
        metrics.append(clip_metrics)
        predictions_path.write_text(
            json.dumps(predictions, indent=2), encoding="utf-8"
        )
        clip_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(
            f"  done in {elapsed:.1f}s; vlm={clip_metrics['vlm_calls']} "
            f"vram={clip_metrics['peak_vram_mb']} MB"
        )

    if not args.dry_run:
        predictions_path.write_text(
            json.dumps(predictions, indent=2), encoding="utf-8"
        )
        clip_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"predictions -> {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

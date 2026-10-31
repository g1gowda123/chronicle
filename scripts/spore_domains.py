#!/usr/bin/env python3
"""Run SPORE's self-supervised object-location evaluation across domains.

Each manifest entry is a short egocentric clip with a domain-specific
vocabulary.  The heavy lifting is done by ``spore_autoeval.evaluate``: it
builds questions from dense YOLO-World tracks, so no hand-written QA is needed
and the questions cannot be tuned to SPORE.

This wrapper adds a small, reproducible domain aggregate:

    python scripts/spore_domains.py --dry-run
    python scripts/spore_domains.py --allow-missing
    python scripts/spore_domains.py --sequence eat_0 --out-dir runs/spore_domains

Missing clips are skipped with ``--allow-missing``; otherwise the runner stops
with the fetch command.  Downloaded videos are local-only; see
``scripts/fetch_aria_clip.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spore_autoeval import DEFAULT_VOCABULARY, evaluate  # noqa: E402

DEFAULT_MANIFEST = ROOT / "data/aria_gen2pilot.json"


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"{path}: expected a non-empty 'sequences' list")
    return payload


def select(manifest: dict, names: list[str] | None) -> list[dict]:
    if not names:
        return list(manifest["sequences"])
    wanted = set(names)
    chosen = [item for item in manifest["sequences"] if item["name"] in wanted]
    missing = wanted - {item["name"] for item in chosen}
    if missing:
        raise KeyError(f"unknown sequence(s): {', '.join(sorted(missing))}")
    return chosen


def _as_args(item: dict, args) -> Namespace:
    vocabulary = item.get("vocabulary") or ",".join(DEFAULT_VOCABULARY)
    return Namespace(
        clip=str((ROOT / item["clip"]).resolve()),
        store=None,
        out="",
        mode=args.mode,
        sample_fps=args.sample_fps,
        budget=args.budget,
        max_questions=args.max_questions,
        min_track_frames=args.min_track_frames,
        min_confidence=args.min_confidence,
        no_context=False,
        vocabulary=vocabulary,
        context_fusion_box=args.context_fusion_box,
    )


def aggregate(summaries: list[dict]) -> dict:
    by_domain: dict[str, dict] = {}
    for summary in summaries:
        domain = summary["domain"]
        bucket = by_domain.setdefault(
            domain,
            {"clips": 0, "questions": 0, "semantic": 0.0, "loc30": 0.0, "iou": 0.0, "calls": 0.0, "dense": 0.0},
        )
        n = max(1, int(summary["questions"]))
        bucket["clips"] += 1
        bucket["questions"] += n
        bucket["semantic"] += float(summary["semantic_accuracy"]) * n
        bucket["loc30"] += float(summary["localization_30"]) * n
        bucket["iou"] += float(summary["mean_iou"]) * n
        bucket["calls"] += float(summary["mean_calls"]) * n
        bucket["dense"] += float(summary["dense_calls"]) * n

    rows = []
    for domain, bucket in sorted(by_domain.items()):
        n = max(1, bucket["questions"])
        rows.append(
            {
                "domain": domain,
                "clips": bucket["clips"],
                "questions": bucket["questions"],
                "semantic_accuracy": round(bucket["semantic"] / n, 3),
                "localization_30": round(bucket["loc30"] / n, 3),
                "mean_iou": round(bucket["iou"] / n, 3),
                "mean_calls": round(bucket["calls"] / n, 2),
                "dense_calls": int(round(bucket["dense"] / max(1, bucket["clips"]))),
            }
        )
    return {"domains": rows, "clips": summaries}


def render_markdown(aggregate_payload: dict, manifest: dict) -> str:
    lines = [
        "# SPORE multi-domain evaluation",
        "",
        f"Source: {manifest.get('dataset', '')}. License: {manifest.get('license', '')}.",
        "Questions are generated from dense YOLO-World tracks; see `scripts/spore_autoeval.py`.",
        "",
        "| domain | clips | questions | semantic | loc@.30 | mean IoU | calls/query | dense calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_payload["domains"]:
        lines.append(
            f"| {row['domain']} | {row['clips']} | {row['questions']} | "
            f"{row['semantic_accuracy']:.3f} | {row['localization_30']:.3f} | "
            f"{row['mean_iou']:.3f} | {row['mean_calls']:.2f} | {row['dense_calls']} |"
        )
    lines.extend(["", "## Per clip", "", "| clip | domain | questions | semantic | loc@.30 | mean IoU | calls/query |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for item in aggregate_payload["clips"]:
        lines.append(
            f"| {Path(item['clip']).name} | {item['domain']} | {item['questions']} | "
            f"{item['semantic_accuracy']:.3f} | {item['localization_30']:.3f} | "
            f"{item['mean_iou']:.3f} | {item['mean_calls']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sequence", action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/spore_domains")
    parser.add_argument("--mode", choices=("clip", "detector", "hybrid"), default="hybrid")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--max-questions", type=int, default=12)
    parser.add_argument("--min-track-frames", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--context-fusion-box", default="adaptive", choices=("mean", "representative", "adaptive"))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    chosen = select(manifest, args.sequence)

    if args.dry_run:
        print(f"{'name':<10} {'domain':<10} {'window':<14} clip")
        for item in chosen:
            print(
                f"{item['name']:<10} {item['domain']:<10} "
                f"{item['start_s']:>5.1f}+{item['duration_s']:<5.1f}  {item['clip']}"
            )
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in chosen:
        clip = (ROOT / item["clip"]).resolve()
        if not clip.exists():
            message = f"missing clip: {clip}"
            if args.allow_missing:
                print(f"[skip] {item['name']}: {message}", flush=True)
                continue
            parser.error(f"{message}; run scripts/fetch_aria_clip.py --sequence {item['sequence']}")
        run_args = _as_args(item, args)
        print(f"[domain] {item['name']} ({item['domain']}) {clip}", flush=True)
        summary = evaluate(run_args)
        summary["name"] = item["name"]
        summary["domain"] = item["domain"]
        summary["clip"] = str(clip)
        summaries.append(summary)
        (args.out_dir / f"{item['name']}.json").write_text(json.dumps(summary, indent=2))

    if not summaries:
        print("no clips evaluated", flush=True)
        return

    payload = aggregate(summaries)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    markdown = render_markdown(payload, manifest)
    (args.out_dir / "summary.md").write_text(markdown)
    print(markdown)
    print(f"[report] {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

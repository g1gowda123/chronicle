"""Detector-free scene recall probe on a long egocentric clip.

The object-centric SPORE index answers questions about things it can bind.
This probe asks a complementary question: can the scene-capsule CLIP index
retrieve *places* (stairs, buildings, signs) without loading a detector?

The eight prompts and their time windows were chosen from a contact sheet of
``aria_walk_60s.mp4`` before scoring.  They are a diagnostic, not a benchmark:
the sample is tiny and several places are visible for most of the clip.  The
useful result is the contrast between large scene structure (stairs, building)
and small text/signs (parking sign, door), not a single accuracy number.

Run::

    python scripts/spore_scene_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.perception.embedder import build_embedder  # noqa: E402
from chronicle.perception.spore import (  # noqa: E402
    SporeConfig,
    SporePerceiver,
    SporeQueryEngine,
)

# (query, start_s, end_s, note)
# Terms that are absent from the clip, used to test abstention.  A useful
# visibility operator must score these below the terms that are present.
NEGATIVE_QUESTIONS: tuple[str, ...] = (
    "elephant",
    "kitchen",
    "swimming pool",
    "toaster",
    "laptop",
    "bicycle",
    "dog",
    "beach",
    "office",
    "car",
    "train station",
    "basketball court",
)


SCENE_QUESTIONS: tuple[tuple[str, float, float, str], ...] = (
    ("stairs", 44.0, 60.0, "stairs dominate the last third"),
    ("parking sign", 28.0, 38.0, "NO PARKING painted on the curb"),
    ("person", 36.0, 60.0, "person in plaid walks ahead"),
    ("building", 0.0, 28.0, "campus building along the path"),
    ("trees", 0.0, 60.0, "scene-wide, not discriminative"),
    ("path", 0.0, 60.0, "scene-wide, not discriminative"),
    ("cup", 38.0, 60.0, "person carries a cup"),
    ("door", 8.0, 20.0, "glass entrance doors in the building"),
)


@dataclass
class SceneResult:
    query: str
    start_s: float
    end_s: float
    note: str
    top_times: list[float]
    top_scores: list[float]
    top1_hit: bool
    top3_hit: bool
    elapsed_ms: float


def evaluate(
    clip: Path,
    *,
    top_k: int = 3,
    store_path: Path | None = None,
) -> dict[str, object]:
    torch.set_grad_enabled(False)
    perception = PerceptionConfig(model_dir=str(ROOT / "models"))
    embedder = build_embedder(perception, source_name="tabletop")
    perceiver = SporePerceiver(
        SporeConfig(use_hands=False),
        perception,
        embedder=embedder,
        hands=None,
    )
    started = time.perf_counter()
    store = perceiver.ingest(str(clip), source_name="tabletop")
    ingest_s = time.perf_counter() - started
    if store_path is not None:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(store_path, include_view_images=False)

    engine = SporeQueryEngine(
        store,
        SporeConfig(use_hands=False),
        perception,
        embedder=embedder,
    )
    results: list[SceneResult] = []
    for query, start, end, note in SCENE_QUESTIONS:
        query_started = time.perf_counter()
        hits = engine.scene_recall(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - query_started) * 1000.0
        times = [round(float(scene.timestamp), 1) for scene, _ in hits]
        scores = [round(float(score), 3) for _, score in hits]
        results.append(
            SceneResult(
                query=query,
                start_s=start,
                end_s=end,
                note=note,
                top_times=times,
                top_scores=scores,
                top1_hit=bool(times and start <= times[0] <= end),
                top3_hit=any(start <= timestamp <= end for timestamp in times),
                elapsed_ms=elapsed_ms,
            )
        )
    negative_scores: dict[str, float] = {}
    for query in NEGATIVE_QUESTIONS:
        hits = engine.scene_recall(query, top_k=1)
        negative_scores[query] = round(float(hits[0][1]), 3) if hits else 0.0
    positive_scores = [float(result.top_scores[0]) for result in results]
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positive_scores
        for negative in negative_scores.values()
    )
    auc = wins / max(1, len(positive_scores) * len(negative_scores))
    threshold = SporeConfig().scene_recall_min_score
    return {
        "clip": str(clip),
        "ingest_s": round(ingest_s, 2),
        "scenes": len(store.scenes),
        "capsules": len(store.capsules),
        "contexts": len(store.contexts),
        "detector_calls": 0,
        "top1": sum(result.top1_hit for result in results),
        "top3": sum(result.top3_hit for result in results),
        "n": len(results),
        "positive_scores": [round(score, 3) for score in positive_scores],
        "negative_scores": negative_scores,
        "negative_above_threshold": sum(
            score >= threshold for score in negative_scores.values()
        ),
        "negative_n": len(negative_scores),
        "auc_present_vs_absent": round(auc, 3),
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=str(ROOT / "data/real/aria_walk_60s.mp4"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--store", default="")
    parser.add_argument(
        "--out",
        default=str(ROOT / "research/spore/results/aria_scene_recall.json"),
    )
    args = parser.parse_args()
    summary = evaluate(
        Path(args.clip),
        top_k=args.top_k,
        store_path=Path(args.store) if args.store else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(
        "[summary] "
        + json.dumps({key: value for key, value in summary.items() if key != "results"})
    )
    for result in summary["results"]:
        print(
            f"{result['query']:14s} window={result['start_s']:.0f}-{result['end_s']:.0f} "
            f"top1={result['top1_hit']} top3={result['top3_hit']} "
            f"times={result['top_times']} scores={result['top_scores']}"
        )
    print(
        "[calibration] negatives above threshold: "
        f"{summary['negative_above_threshold']}/{summary['negative_n']} "
        f"auc={summary['auc_present_vs_absent']}"
    )
    print(f"[report] {out}")


if __name__ == "__main__":
    main()

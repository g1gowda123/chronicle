"""Synthetic benchmark for SPORE: evidence retention and late binding.

Run::

    python scripts/spore_bench.py --mode synth
    python scripts/spore_bench.py --mode synth --ablations

The benchmark asks two separate questions:

1. **Did the perceiver keep evidence?**  For each query, route the question to
   capsules and check whether any retained view overlaps the ground-truth box.
   This is independent of naming, and is the mechanism SPORE actually adds.
2. **Can it name the evidence after the fact?**  Bind the query to a capsule
   view with YOLO-World, with a CLIP prompt ensemble, or with both, and measure
   localisation (IoU with ground truth) plus abstention.

The dense YOLO-World baseline sees the same frames at the same sampling rate
but pays for a detector call on every one, which is the cost SPORE is designed
to avoid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
from spore_world import SynthEvent, SynthWorld  # noqa: E402

from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.perception.embedder import build_embedder  # noqa: E402
from chronicle.perception.hands import build_hands  # noqa: E402
from chronicle.perception.spore import (  # noqa: E402
    SporeAnswer,
    SporeConfig,
    SporePerceiver,
    SporeQueryEngine,
)
from chronicle.types import BBox  # noqa: E402


@dataclass
class EventResult:
    name: str
    time: float
    question: str
    gt: list[float]
    retained_top1: bool
    retained_topk: bool
    retained_store: bool
    route_rank: int
    top1_iou: float
    answer_found: bool
    answer_iou: float
    answer_confidence: float
    answer_label: str
    abstained: bool
    best_capsule: str
    detector_calls: int


def _box_list(box: BBox) -> list[float]:
    return [float(box.x1), float(box.y1), float(box.x2), float(box.y2)]


def _visible_events(world: SynthWorld) -> list[SynthEvent]:
    events = []
    for event in world.events:
        if event.bbox.area >= 150 and event.bbox.width >= 12 and event.bbox.height >= 12:
            events.append(event)
    return events


def _event_eval(
    engine: SporeQueryEngine,
    config: SporeConfig,
    world: SynthWorld,
    event: SynthEvent,
    *,
    budget: int,
) -> EventResult:
    gt_cache: dict[int, BBox | None] = {}

    def gt_at(timestamp: float) -> BBox | None:
        key = int(round(timestamp * world.fps))
        if key not in gt_cache:
            _, frame_gt = world.render(key / world.fps)
            gt_cache[key] = frame_gt.get(event.name)
        return gt_cache[key]

    capsules = engine.route([event.name], budget=budget)
    best_iou = 0.0
    best_capsule = ""
    retained_top1 = False
    retained_topk = False
    route_rank = -1
    for rank, capsule in enumerate(capsules):
        capsule_iou = 0.0
        for view in capsule.best_views:
            target = gt_at(view.timestamp)
            if target is not None:
                capsule_iou = max(capsule_iou, view.bbox.iou(target))
        if rank == 0 and capsule_iou >= 0.30:
            retained_top1 = True
        if capsule_iou >= 0.30:
            retained_topk = True
            if route_rank < 0:
                route_rank = rank
        if capsule_iou > best_iou:
            best_iou = capsule_iou
            best_capsule = capsule.capsule_id

    retained_store = False
    for capsule in engine.store.capsules.values():
        for view in capsule.best_views:
            target = gt_at(view.timestamp)
            if target is not None and view.bbox.iou(target) >= 0.30:
                retained_store = True
                break
        if retained_store:
            break

    calls_before = engine.detector_calls
    answer: SporeAnswer = engine.answer(event.question, budget=budget)
    calls = engine.detector_calls - calls_before
    best = answer.best
    answer_iou = 0.0
    answer_label = ""
    if best is not None:
        target = gt_at(best.timestamp)
        if target is not None:
            answer_iou = best.bbox.iou(target)
        answer_label = best.label
    return EventResult(
        name=event.name,
        time=round(event.time, 2),
        question=event.question,
        gt=_box_list(event.bbox),
        retained_top1=retained_top1,
        retained_topk=retained_topk,
        retained_store=retained_store,
        route_rank=route_rank,
        top1_iou=round(float(best_iou), 3),
        answer_found=best is not None,
        answer_iou=round(float(answer_iou), 3),
        answer_confidence=round(float(best.confidence), 3) if best is not None else 0.0,
        answer_label=answer_label,
        abstained=answer.abstained,
        best_capsule=best_capsule,
        detector_calls=calls,
    )


def summarize(results: list[EventResult], *, label: str) -> dict[str, float | int | str]:
    count = max(1, len(results))
    return {
        "label": label,
        "queries": len(results),
        "retention_top1": round(sum(r.retained_top1 for r in results) / count, 3),
        "retention_topk": round(sum(r.retained_topk for r in results) / count, 3),
        "retention_store": round(sum(r.retained_store for r in results) / count, 3),
        "localization_30": round(sum(r.answer_iou >= 0.30 for r in results) / count, 3),
        "localization_50": round(sum(r.answer_iou >= 0.50 for r in results) / count, 3),
        "abstention": round(sum(r.abstained for r in results) / count, 3),
        "mean_top1_iou": round(float(np.mean([r.top1_iou for r in results])), 3),
        "mean_answer_iou": round(float(np.mean([r.answer_iou for r in results])), 3),
        "mean_confidence": round(float(np.mean([r.answer_confidence for r in results])), 3),
        "detector_calls_per_query": round(float(np.mean([r.detector_calls for r in results])), 2),
    }


def evaluate_mode(
    store,
    perception: PerceptionConfig,
    config: SporeConfig,
    world: SynthWorld,
    events: list[SynthEvent],
    *,
    mode: str,
    budget: int,
    embedder,
    detector,
    label: str,
) -> tuple[list[EventResult], dict, float]:
    mode_config = replace(config, binding_mode=mode)
    engine = SporeQueryEngine(
        store,
        mode_config,
        perception,
        embedder=embedder,
        detector=detector if mode in {"detector", "hybrid"} else None,
    )
    started = time.perf_counter()
    results = [
        _event_eval(engine, mode_config, world, event, budget=budget) for event in events
    ]
    elapsed = time.perf_counter() - started
    summary = summarize(results, label=label)
    summary["seconds"] = round(elapsed, 2)
    summary["detector_calls_total"] = engine.detector_calls
    for item in results:
        print(
            f"  [{label}] {item.name:8s} store={int(item.retained_store)} "
            f"top1={int(item.retained_top1)} topk={int(item.retained_topk)} "
            f"rank={item.route_rank:2d} iou={item.top1_iou:.2f} "
            f"conf={item.answer_confidence:.2f}",
            flush=True,
        )
    return results, summary, elapsed


def dense_detector_baseline(
    perception: PerceptionConfig,
    config: SporeConfig,
    world: SynthWorld,
    events: list[SynthEvent],
    *,
    detector,
    rate: float,
    labels: list[str],
) -> tuple[list[dict], dict]:
    detector.set_vocabulary(labels + list(config.support_labels))
    interval = 1.0 / rate
    count = int(round(world.duration * rate))
    observations: list[tuple[float, list]] = []
    started = time.perf_counter()
    for index in range(count):
        timestamp = index * interval
        frame, _ = world.render(timestamp)
        detections = detector.detect(frame, frame_index=index, timestamp=timestamp)
        observations.append((timestamp, detections))
    elapsed = time.perf_counter() - started
    rows: list[dict] = []
    for event in events:
        nearest = min(observations, key=lambda item: abs(item[0] - event.time))
        if abs(nearest[0] - event.time) > max(0.3, interval):
            rows.append({"name": event.name, "found": False, "iou": 0.0, "confidence": 0.0})
            continue
        candidates = [d for d in nearest[1] if d.label.strip().lower() == event.name]
        if not candidates:
            rows.append({"name": event.name, "found": False, "iou": 0.0, "confidence": 0.0})
            continue
        best = max(candidates, key=lambda d: d.confidence)
        rows.append(
            {
                "name": event.name,
                "found": True,
                "iou": round(float(best.bbox.iou(event.bbox)), 3),
                "confidence": round(float(best.confidence), 3),
            }
        )
    n = max(1, len(rows))
    summary = {
        "label": f"dense-yolo@{rate:g}fps",
        "queries": len(rows),
        "retention_topk": 1.0,
        "retention_note": "ground truth is in the frame stream by construction",
        "localization_30": round(sum(row["iou"] >= 0.30 for row in rows) / n, 3),
        "localization_50": round(sum(row["iou"] >= 0.50 for row in rows) / n, 3),
        "found": round(sum(row["found"] for row in rows) / n, 3),
        "mean_answer_iou": round(float(np.mean([row["iou"] for row in rows])), 3),
        "detector_calls_total": len(observations),
        "seconds": round(elapsed, 2),
    }
    return rows, summary


def run_synth(args: argparse.Namespace) -> dict:
    world = SynthWorld(seed=args.seed)
    events = _visible_events(world)
    labels = [obj.name for obj in world.objects]
    perception = PerceptionConfig(model_dir=str(ROOT / "models"))
    config = SporeConfig(
        sample_fps=args.sample_fps,
        max_width=640,
        binding_mode="hybrid",
        query_capsules=args.budget,
        query_views=2,
    )
    print(f"[world] seed={args.seed} objects={len(world.objects)} queries={len(events)}", flush=True)

    embedder = build_embedder(perception, source_name="tabletop")
    hands = build_hands(perception, source_name="tabletop")
    perceiver = SporePerceiver(config, perception, embedder=embedder, hands=hands)
    frames = (frame for _, frame in world.frames(fps=world.fps))
    started = time.perf_counter()
    store = perceiver.ingest(
        frames,
        fps_override=world.fps,
        source_name="tabletop",
    )
    ingest_seconds = time.perf_counter() - started
    stats = perceiver.stats.as_dict()
    print(f"[ingest] {json.dumps(stats)}", flush=True)
    print(
        f"[store] capsules={len(store.capsules)} bytes={store.memory_bytes()} "
        f"seconds={ingest_seconds:.2f}",
        flush=True,
    )

    detector = None
    if args.detector:
        from chronicle.perception.detector import build_detector

        detector = build_detector(perception, source_name="tabletop")
        detector.set_vocabulary(labels + list(config.support_labels))

    report: dict = {
        "mode": "synth",
        "seed": args.seed,
        "ingest": stats,
        "ingest_seconds": round(ingest_seconds, 2),
        "store_capsules": len(store.capsules),
        "store_bytes": store.memory_bytes(),
        "queries": len(events),
        "events": [],
        "modes": {},
        "dense": {},
        "config": asdict(config),
    }

    if detector is not None:
        for mode, budget in (("clip", args.budget), ("detector", args.budget), ("hybrid", args.budget)):
            results, summary, _ = evaluate_mode(
                store,
                perception,
                config,
                world,
                events,
                mode=mode,
                budget=budget,
                embedder=embedder,
                detector=detector,
                label=f"spore-{mode}",
            )
            report["modes"][mode] = summary
            if mode == "hybrid":
                report["events"] = [asdict(item) for item in results]
            print(f"[query] {json.dumps(summary)}", flush=True)

        for rate in args.dense_rates:
            rows, summary = dense_detector_baseline(
                perception,
                config,
                world,
                events,
                detector=detector,
                rate=rate,
                labels=labels,
            )
            report["dense"][f"{rate:g}fps"] = summary
            print(f"[dense] {json.dumps(summary)}", flush=True)

    # Persist a few qualitative frames for the paper/handoff notes.
    out_dir = ROOT / "research" / "spore" / "results" / f"synth_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for event in events[:3]:
        frame, gt = world.render(event.time)
        for _name, box in gt.items():
            cv2.rectangle(
                frame,
                (int(box.x1), int(box.y1)),
                (int(box.x2), int(box.y2)),
                (0, 255, 0),
                1,
            )
        cv2.imwrite(str(out_dir / f"event_{event.name}_{event.time:.1f}.jpg"), frame)
    with (out_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(f"[report] {out_dir / 'report.json'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synth",), default="synth")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--dense-rates", type=float, nargs="*", default=[4.0])
    parser.add_argument("--no-detector", dest="detector", action="store_false")
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    report = run_synth(args)
    if torch.cuda.is_available():
        print(f"[vram] peak={torch.cuda.max_memory_allocated() / 1e6:.1f}MB", flush=True)
        report["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)


if __name__ == "__main__":
    main()

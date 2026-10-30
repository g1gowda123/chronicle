"""Real-clip benchmark for SPORE evidence retention and late binding.

The clips do not ship with boxes, so this benchmark uses a dense YOLO-World
pass as *pseudo ground truth*: the same detector, run on every sampled frame,
is the upper bound on what a late-binding query could recover.  SPORE is then
scored on whether the capsule store retained a view that overlaps that dense
detection, and whether the query-time binding agrees with it.  This separates
two questions that the synthetic benchmark keeps together:

* did cheap perception keep the evidence?  (retention)
* can the query engine name and localise it?  (binding)

Run::

    python scripts/spore_real_bench.py --clip data/real/aria_cook_45s.mp4 \
        --qa data/qa/cook_45s.json --budget 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

from chronicle.perception.capsules import CapsuleStore  # noqa: E402
from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.perception.embedder import build_embedder  # noqa: E402
from chronicle.perception.hands import build_hands  # noqa: E402
from chronicle.perception.sources import VideoSource  # noqa: E402
from chronicle.perception.spore import (  # noqa: E402
    SporeAnswer,
    SporeConfig,
    SporePerceiver,
    SporeQueryEngine,
    _resize_max_width,
    extract_terms,
)
from chronicle.types import BBox  # noqa: E402


@dataclass
class QueryResult:
    query: str
    category: str
    timestamp: float
    expected: str
    terms: list[str]
    target_time: float | None
    target_confidence: float
    store_hit: bool
    routed_hit: bool
    route_rank: int
    routed_any: bool
    route_rank_any: int
    answer_text: str
    answer_label: str
    answer_time: float
    answer_iou: float
    answer_confidence: float
    answer_provenance: str
    answer_bbox: list[float]
    target_bbox: list[float]
    expected_answers: list[str]
    answer_match: bool | None
    abstained: bool
    detector_calls: int
    evidence_bytes: int
    object_question: bool


def _box_list(box: BBox) -> list[float]:
    return [float(box.x1), float(box.y1), float(box.x2), float(box.y2)]


def _match_label(label: str, term: str) -> bool:
    label = label.strip().lower()
    term = term.strip().lower()
    if not label or not term:
        return False
    if label == term:
        return True
    return len(term) >= 4 and (term in label or label in term)


def _nearest_detection(
    detections: list[tuple[float, list]],
    timestamp: float,
    term: str,
    *,
    tolerance: float = 1.2,
) -> tuple[float, object] | None:
    best: tuple[float, object] | None = None
    for det_time, items in detections:
        if abs(det_time - timestamp) > tolerance:
            continue
        for det in items:
            if not _match_label(det.label, term):
                continue
            score = det.confidence - 0.20 * abs(det_time - timestamp)
            if best is None or score > best[0]:
                best = (score, det, det_time)  # type: ignore[assignment]
    if best is None:
        return None
    return best[2], best[1]


def _semantic_match(question: dict, answer: SporeAnswer) -> bool | None:
    """Lenient text-level QA check against the annotated answer.

    The QA annotations name a region or a count, not a pixel box.  A separate
    semantic score keeps the text answer honest while the IoU metrics measure
    localisation quality against dense pseudo-ground-truth.
    """
    raw_expected = question.get("answers") or question.get("answer") or []
    if isinstance(raw_expected, str):
        raw_expected = [raw_expected]
    expected = [
        str(value).strip().lower()
        for value in raw_expected
        if str(value).strip()
    ]
    if not expected:
        return None
    if any("don't have" in value or "do not have" in value for value in expected):
        return answer.abstained
    if question.get("category") == "temporal":
        if answer.abstained:
            return False
        seconds = []
        for value in expected:
            match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b", value)
            if match:
                seconds.append(float(match.group(1)))
        if not seconds:
            return None
        best = answer.best
        if best is None:
            return False
        return any(
            abs(best.timestamp - value) <= max(1.0, 0.2 * value)
            for value in seconds
        )
    if answer.abstained:
        return False
    text = answer.answer.lower()
    if question.get("category") == "counting":
        digits = set(re.findall(r"\d+", " ".join(expected)))
        return bool(digits and any(digit in text for digit in digits))
    return any(value in text for value in expected)


def _covers(view_box: BBox, target_box: BBox) -> bool:
    """True when a capsule crop contains enough of the target box.

    A capsule view is deliberately a *region* around an object, often two or
    three times its size.  Plain IoU punishes that generosity; containment is
    the right test for evidence retention, while IoU is kept for the final
    tight binding box.
    """
    if target_box.area <= 0:
        return False
    intersection = view_box.intersection_area(target_box)
    return intersection / target_box.area >= 0.50 or view_box.iou(target_box) >= 0.30




def evaluate(args: argparse.Namespace) -> dict:
    qa = json.loads(Path(args.qa).read_text())
    questions = qa["questions"]
    perception = PerceptionConfig(model_dir=str(ROOT / "models"))
    config = SporeConfig(
        sample_fps=args.sample_fps,
        max_width=640,
        binding_mode=args.mode,
        context_binding=not args.no_context,
        context_hand_only=args.context_hand_only,
        context_adaptive_gap=args.adaptive_context,
        context_sweep_frames=args.context_sweep_frames,
        max_capsules=args.max_capsules,
        query_capsules=args.budget,
        query_views=2,
        rescan=args.rescan,
    )
    if args.context_gap is not None:
        config.context_min_gap = args.context_gap
    if args.clip_center is not None:
        config.clip_margin_center = args.clip_center
    if args.abstain is not None:
        config.query_abstain_confidence = args.abstain
    if args.binding_confidence is not None:
        config.query_detector_confidence = args.binding_confidence

    terms: list[str] = []
    for question in questions:
        for term in extract_terms(question["query"]):
            if term not in terms:
                terms.append(term)
    vocabulary = list(dict.fromkeys(terms + list(config.support_labels)))
    print(f"[labels] {vocabulary}", flush=True)

    # Dense pseudo-ground-truth pass on the same 640px frames SPORE sees.
    from chronicle.perception.detector import build_detector

    detector = build_detector(perception, source_name="tabletop")
    detector.set_vocabulary(vocabulary)
    dense: list[tuple[float, list]] = []
    dense_started = time.perf_counter()
    interval = 1.0 / args.sample_fps
    last_time = -1e9
    for frame, timestamp, _ in VideoSource(args.clip):
        if timestamp - last_time < interval - 1e-6:
            continue
        last_time = timestamp
        dense_frame, _ = _resize_max_width(frame, 640)
        dense.append(
            (
                timestamp,
                detector.detect(dense_frame, frame_index=len(dense), timestamp=timestamp),
            )
        )
    dense_seconds = time.perf_counter() - dense_started
    print(
        f"[dense] frames={len(dense)} calls={len(dense)} seconds={dense_seconds:.2f}",
        flush=True,
    )

    query_detector = detector
    if args.query_detector_model:
        query_detector_config = replace(
            perception.detector,
            model=args.query_detector_model,
            confidence=(
                args.query_detector_confidence
                if args.query_detector_confidence is not None
                else perception.detector.confidence
            ),
            imgsz=(
                args.query_detector_imgsz
                if args.query_detector_imgsz is not None
                else perception.detector.imgsz
            ),
        )
        query_perception = replace(perception, detector=query_detector_config)
        query_detector = build_detector(query_perception, source_name="tabletop")
        query_detector.set_vocabulary(vocabulary)
        print(
            f"[query-detector] {args.query_detector_model} "
            f"conf={query_detector_config.confidence} imgsz={query_detector_config.imgsz}",
            flush=True,
        )

    embedder = build_embedder(perception, source_name="tabletop")
    hands = build_hands(perception, source_name="tabletop")
    perceiver = None
    if args.store:
        store = CapsuleStore.load(args.store)
        ingest_seconds = 0.0
        print(f"[store] {len(store.capsules)} capsules {store.memory_bytes()} bytes", flush=True)
    else:
        perceiver = SporePerceiver(config, perception, embedder=embedder, hands=hands)
        started = time.perf_counter()
        store = perceiver.ingest(
            args.clip,
            max_frames=args.max_frames,
            source_name="tabletop",
        )
        ingest_seconds = time.perf_counter() - started
        print(
            f"[ingest] {json.dumps(perceiver.stats.as_dict())} seconds={ingest_seconds:.2f}",
            flush=True,
        )

    hybrid_results: list[QueryResult] = []
    if args.dump:
        dump_dir = Path(args.dump)
        dump_dir.mkdir(parents=True, exist_ok=True)
        store.save(dump_dir / "store.json")
        (dump_dir / "dense.json").write_text(
            json.dumps(
                [
                    {
                        "timestamp": timestamp,
                        "detections": [
                            {
                                "label": det.label,
                                "confidence": float(det.confidence),
                                "bbox": _box_list(det.bbox),
                            }
                            for det in items
                        ],
                    }
                    for timestamp, items in dense
                ]
            )
        )
        print(f"[dump] {dump_dir}", flush=True)
    for mode in (args.mode,):
        mode_config = replace(config, binding_mode=mode)
        mode_results: list[QueryResult] = []
        engine = SporeQueryEngine(
            store,
            mode_config,
            perception,
            embedder=embedder,
            detector=query_detector if mode in {"detector", "hybrid"} else None,
            source=args.clip,
        )
        for question in questions:
            query_terms = extract_terms(question["query"])
            object_question = question.get("category") in {
                "object_location",
                "inventory",
                "attribute",
                "counting",
            }
            calls_before = engine.detector_calls
            answer = engine.answer(
                question["query"],
                budget=args.budget,
                at_time=question.get("timestamp"),
            )
            calls = engine.detector_calls - calls_before
            target = None
            if argument := query_terms:
                target = _nearest_detection(dense, question["timestamp"], argument[0])
            if target is None:
                # Try the annotated answer as a second chance (e.g. "plate").
                expected = str(question.get("answer", "")).lower()
                if expected and expected not in {"countertop", "work table", "table"}:
                    target = _nearest_detection(dense, question["timestamp"], expected)

            store_hit = False
            routed_hit = False
            routed_any = False
            route_rank = -1
            route_rank_any = -1
            target_time = None
            target_confidence = 0.0
            target_bbox: list[float] = []
            answer_iou = 0.0
            if target is not None:
                target_time, target_det = target
                target_confidence = float(target_det.confidence)
                target_box = target_det.bbox
                target_bbox = _box_list(target_box)
                for capsule in store.capsules.values():
                    for candidate in capsule.best_views:
                        if (
                            abs(candidate.timestamp - target_time) <= 0.35
                            and _covers(candidate.bbox, target_box)
                        ):
                            store_hit = True
                            break
                    if store_hit:
                        break
                routed = engine.route(query_terms, budget=args.budget)
                for rank, capsule in enumerate(routed):
                    for candidate in capsule.best_views:
                        covered = _covers(candidate.bbox, target_box)
                        if covered and not routed_any:
                            routed_any = True
                            route_rank_any = rank
                        if (
                            covered
                            and abs(candidate.timestamp - target_time) <= 0.35
                        ):
                            routed_hit = True
                            if route_rank < 0:
                                route_rank = rank
                            break
                    if routed_hit:
                        break
                if not answer.abstained and answer.best is not None:
                    near = _nearest_detection(
                        dense,
                        answer.best.timestamp,
                        query_terms[0] if query_terms else "",
                        tolerance=0.35,
                    )
                    if near is not None:
                        answer_iou = answer.best.bbox.iou(near[1].bbox)

            expected_answers = [
                str(value).strip()
                for value in (question.get("answers") or [question.get("answer", "")])
                if str(value).strip()
            ]

            result = QueryResult(
                query=question["query"],
                category=question.get("category", ""),
                timestamp=float(question["timestamp"]),
                expected=str(question.get("answer", "")),
                terms=query_terms,
                target_time=round(float(target_time), 2) if target_time is not None else None,
                target_confidence=round(target_confidence, 3),
                store_hit=bool(store_hit),
                routed_hit=bool(routed_hit),
                route_rank=route_rank,
                routed_any=bool(routed_any),
                route_rank_any=route_rank_any,
                answer_text=answer.answer,
                answer_label=answer.best.label if answer.best is not None else "",
                answer_time=round(answer.best.timestamp, 2) if answer.best is not None else -1.0,
                answer_iou=round(float(answer_iou), 3),
                answer_confidence=round(
                    float(answer.best.confidence) if answer.best is not None else 0.0, 3
                ),
                answer_provenance=answer.best.provenance if answer.best is not None else "",
                answer_bbox=_box_list(answer.best.bbox) if answer.best is not None else [],
                target_bbox=target_bbox,
                expected_answers=expected_answers,
                answer_match=_semantic_match(question, answer),
                abstained=answer.abstained,
                detector_calls=calls,
                evidence_bytes=answer.evidence_bytes,
                object_question=object_question,
            )
            mode_results.append(result)
            if mode == args.mode:
                print(
                    f"  {question['query'][:42]:42s} "
                    f"t={result.target_time} store={int(store_hit)} "
                    f"routed={int(routed_hit)} any={int(routed_any)} "
                    f"iou={answer_iou:.2f} "
                    f"match={result.answer_match} "
                    f"conf={result.answer_confidence:.2f} calls={calls}",
                    flush=True,
                )
        if mode == args.mode:
            hybrid_results = mode_results

    # Summary for the hybrid mode (the default), plus a dense cost figure.
    object_results = [item for item in hybrid_results if item.object_question]
    detected = [item for item in object_results if item.target_time is not None]
    scored_answers = [item for item in hybrid_results if item.answer_match is not None]
    unknown = [item for item in hybrid_results if item.category == "unknown"]
    summary = {
        "clip": str(args.clip),
        "mode": args.mode,
        "context_binding": not args.no_context,
        "queries": len(hybrid_results),
        "object_queries": len(object_results),
        "dense_detected": len(detected),
        "store_retention": round(
            sum(item.store_hit for item in detected) / max(1, len(detected)), 3
        ),
        "routed_retention": round(
            sum(item.routed_hit for item in detected) / max(1, len(detected)), 3
        ),
        "routed_any_retention": round(
            sum(item.routed_any for item in detected) / max(1, len(detected)), 3
        ),
        "localization_30": round(
            sum(item.answer_iou >= 0.30 for item in detected) / max(1, len(detected)), 3
        ),
        "localization_50": round(
            sum(item.answer_iou >= 0.50 for item in detected) / max(1, len(detected)), 3
        ),
        "mean_iou": round(
            float(np.mean([item.answer_iou for item in detected])) if detected else 0.0, 3
        ),
        "mean_time_error": round(
            float(np.mean([abs(item.answer_time - item.timestamp) for item in detected]))
            if detected
            else 0.0,
            2,
        ),
        "abstention": round(
            sum(item.abstained for item in hybrid_results) / max(1, len(hybrid_results)), 3
        ),
        "semantic_accuracy": round(
            sum(bool(item.answer_match) for item in scored_answers)
            / max(1, len(scored_answers)),
            3,
        ),
        "semantic_scored": len(scored_answers),
        "unknown_abstention": round(
            sum(item.abstained for item in unknown) / max(1, len(unknown)), 3
        ),
        "mean_detector_calls": round(
            float(np.mean([item.detector_calls for item in hybrid_results])), 2
        ),
        "mean_evidence_bytes": round(
            float(np.mean([item.evidence_bytes for item in hybrid_results])), 0
        ),
        "dense_calls": len(dense),
        "dense_seconds": round(dense_seconds, 2),
        "spore_ingest_seconds": round(ingest_seconds, 2),
        "store_capsules": len(store.capsules),
        "store_bytes": store.memory_bytes(),
        "ingest": perceiver.stats.as_dict() if perceiver is not None else {},
        "config": asdict(config),
        "results": [asdict(item) for item in hybrid_results],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--out", default="")
    parser.add_argument("--dump", default="")
    parser.add_argument("--store", default="")
    parser.add_argument("--mode", default="hybrid", choices=("clip", "detector", "hybrid"))
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--context-hand-only", action="store_true")
    parser.add_argument("--adaptive-context", action="store_true")
    parser.add_argument("--context-sweep-frames", type=int, default=5)
    parser.add_argument("--max-capsules", type=int, default=300)
    parser.add_argument("--context-gap", type=float, default=None)
    parser.add_argument("--clip-center", type=float, default=None)
    parser.add_argument("--abstain", type=float, default=None)
    parser.add_argument("--binding-confidence", type=float, default=None)
    parser.add_argument("--query-detector-model", default="")
    parser.add_argument("--query-detector-confidence", type=float, default=None)
    parser.add_argument("--query-detector-imgsz", type=int, default=None)
    parser.add_argument("--rescan", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    summary = evaluate(args)
    out = Path(args.out) if args.out else (
        ROOT
        / "research"
        / "spore"
        / "results"
        / f"real_{Path(args.clip).stem}_{args.budget}"
        / "report.json"
    )
    if out.exists() and out.is_dir():
        out = out / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[summary] {json.dumps({k: v for k, v in summary.items() if k != 'results'})}")
    print(f"[report] {out}")
    if torch.cuda.is_available():
        print(f"[vram] peak={torch.cuda.max_memory_allocated() / 1e6:.1f}MB")


if __name__ == "__main__":
    main()

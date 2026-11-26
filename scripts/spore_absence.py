#!/usr/bin/env python3
"""Calibrate "not found in the store" as evidence of absence.

The SPORE paper reports an absence *gap*: a zero count is not verified
absence, and CLIP scene scores do not separate present from absent terms.
This script measures the operational version of that gap on a pre-registered
human-annotated probe set:

* scan every retained context frame once per probe term at a low detector
  floor and record the best confidence per context (thresholds are applied
  post hoc, so one scan serves every floor);
* compute per-context sensitivity/specificity against human presence
  intervals;
* compute how the false-negative rate of "no detection in k sampled
  contexts" changes with k;
* ask the shipped query path the same presence questions and record whether
  it abstains;
* answer a few state-at-time questions (is the object on a surface, or held?)
  against human labels.

The probe file is written and committed before the first ingest; see
``data/qa/eat_1_absence.json``.

Usage::

    python scripts/spore_absence.py \
        --clip data/footage/aria_eat_1_45s.mp4 \
        --probes data/qa/eat_1_absence.json \
        --dump research/spore/results/eat1_absence/store.json \
        --out research/spore/results/eat1_absence/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chronicle.perception.capsules import CapsuleStore  # noqa: E402
from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.perception.spore import (  # noqa: E402
    SporeConfig,
    SporePerceiver,
    SporeQueryEngine,
    _label_matches,
)

FLOORS = (0.22, 0.30, 0.40, 0.50)
K_VALUES = (4, 8, 12, 16, 24, 32, 48)


def load_probes(path: Path) -> dict:
    payload = json.loads(path.read_text())
    probes = payload.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError(f"{path}: expected a non-empty 'probes' list")
    for probe in probes:
        if not probe.get("term"):
            raise ValueError(f"{path}: probe without a term: {probe!r}")
        probe.setdefault("present", False)
        probe.setdefault("intervals", [])
    return payload


def present_in_intervals(timestamp: float, intervals: list[list[float]]) -> bool:
    return any(float(start) <= timestamp <= float(end) for start, end in intervals)


def best_confidence(detections, term: str) -> float | None:
    scores = [
        float(det.confidence)
        for det in detections
        if _label_matches(det.label, term)
    ]
    return max(scores) if scores else None


def rank_auc(positives: list[float], negatives: list[float]) -> float | None:
    """Probability a random positive scores above a random negative."""
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def threshold_hits(
    per_context: dict[int, float | None], floor: float
) -> dict[int, bool]:
    return {
        index: score is not None and score >= floor
        for index, score in per_context.items()
    }


def sample_curve(
    term_rows: list[dict],
    *,
    floor: float,
    k_values=K_VALUES,
) -> list[dict]:
    """False-negative/false-positive rate of sampling k contexts.

    Term-level: an object present anywhere in the clip but never detected in
    the sampled contexts is a false negative.  Context-level: score only the
    contexts that were actually sampled and where the human annotation says
    the term is present/absent.
    """
    rows = []
    for k in k_values:
        fn_terms = 0
        tp_terms = 0
        fp_terms = 0
        tn_terms = 0
        present_contexts = 0
        present_hits = 0
        absent_contexts = 0
        absent_hits = 0
        phases = 0
        for row in term_rows:
            contexts = row["contexts"]
            n = len(contexts)
            if n == 0:
                continue
            take = min(k, n)
            step = n / take
            phase_count = max(1, n // take)
            term_present = bool(row["present"])
            for phase in range(phase_count):
                indices = []
                for j in range(take):
                    index = int(math.floor(phase + j * step + 1e-9))
                    indices.append(min(n - 1, index))
                indices = list(dict.fromkeys(indices))
                phases += 1
                any_hit = any(
                    (contexts[index]["score"] or 0.0) >= floor
                    for index in indices
                )
                if term_present:
                    if any_hit:
                        tp_terms += 1
                    else:
                        fn_terms += 1
                else:
                    if any_hit:
                        fp_terms += 1
                    else:
                        tn_terms += 1
                for index in indices:
                    context = contexts[index]
                    hit = (context["score"] or 0.0) >= floor
                    if context["present_at"]:
                        present_contexts += 1
                        present_hits += int(hit)
                    else:
                        absent_contexts += 1
                        absent_hits += int(hit)
        rows.append(
            {
                "k": k,
                "phases": phases,
                "term_false_negative": (
                    round(fn_terms / (fn_terms + tp_terms), 3)
                    if (fn_terms + tp_terms)
                    else None
                ),
                "term_false_positive": (
                    round(fp_terms / (fp_terms + tn_terms), 3)
                    if (fp_terms + tn_terms)
                    else None
                ),
                "context_sensitivity": (
                    round(present_hits / present_contexts, 3)
                    if present_contexts
                    else None
                ),
                "context_false_positive": (
                    round(absent_hits / absent_contexts, 3)
                    if absent_contexts
                    else None
                ),
            }
        )
    return rows


def occupancy(row: dict) -> str:
    intervals = row.get("intervals") or []
    if not intervals:
        return "absent"
    total = sum(float(end) - float(start) for start, end in intervals)
    return f"{total:.1f}s"


def run(args: argparse.Namespace) -> dict:
    payload = load_probes(Path(args.probes))
    probes = payload["probes"]
    terms = [probe["term"] for probe in probes]
    perception = PerceptionConfig(model_dir=ROOT / "models")
    config = SporeConfig(sample_fps=args.sample_fps, max_width=640)
    if args.context_gap is not None:
        config.context_min_gap = args.context_gap

    ingest_seconds = 0.0
    if args.store:
        store = CapsuleStore.load(args.store)
        print(f"[store] {len(store.capsules)} capsules, {len(store.contexts)} contexts")
    else:
        perceiver = SporePerceiver(config, perception)
        started = time.perf_counter()
        store = perceiver.ingest(args.clip, source_name="tabletop")
        ingest_seconds = time.perf_counter() - started
        print(f"[ingest] {json.dumps(perceiver.stats.as_dict())} seconds={ingest_seconds:.1f}")
    if args.dump:
        dump = Path(args.dump)
        dump.parent.mkdir(parents=True, exist_ok=True)
        store.save(dump, include_view_images=False, include_scene_images=False, split=True)
        print(f"[dump] {dump}")

    contexts = list(store.contexts)
    if args.max_contexts and len(contexts) > args.max_contexts:
        indices = np.linspace(0, len(contexts) - 1, args.max_contexts, dtype=int)
        contexts = [contexts[int(index)] for index in indices]
    print(f"[scan] {len(contexts)} contexts x {len(terms)} terms at floor {args.floor}")

    engine = SporeQueryEngine(store, config, perception, source="tabletop")
    engine._ensure_backends()
    assert engine._detector is not None
    scan_started = time.perf_counter()
    per_term: dict[str, list[dict]] = {term: [] for term in terms}
    for position, context in enumerate(contexts):
        frame = context.image()
        if frame is None:
            continue
        for term in terms:
            detections = engine._detect_cached(
                frame,
                [term],
                frame_index=context.frame_index,
                timestamp=context.timestamp,
                confidence=args.floor,
            )
            score = best_confidence(detections, term)
            probe = next(item for item in probes if item["term"] == term)
            per_term[term].append(
                {
                    "frame_index": int(context.frame_index),
                    "timestamp": round(float(context.timestamp), 2),
                    "score": round(score, 4) if score is not None else None,
                    "present_at": present_in_intervals(
                        float(context.timestamp), probe.get("intervals") or []
                    ),
                }
            )
        if (position + 1) % 10 == 0:
            print(f"[scan] {position + 1}/{len(contexts)} contexts", flush=True)
    scan_seconds = time.perf_counter() - scan_started
    scan_calls = engine.detector_calls

    term_rows = []
    for probe in probes:
        term = probe["term"]
        contexts_rows = per_term[term]
        scores = [row["score"] for row in contexts_rows if row["score"] is not None]
        row = {
            "term": term,
            "present": bool(probe["present"]),
            "intervals": probe.get("intervals") or [],
            "occupancy": occupancy(probe),
            "note": probe.get("note", ""),
            "contexts": contexts_rows,
            "max_confidence": round(max(scores), 4) if scores else None,
            "hits": {
                str(floor): sum(
                    1
                    for context in contexts_rows
                    if (context["score"] or 0.0) >= floor
                )
                for floor in FLOORS
            },
            "first_hit": next(
                (
                    context["timestamp"]
                    for context in contexts_rows
                    if (context["score"] or 0.0) >= 0.22
                ),
                None,
            ),
        }
        term_rows.append(row)

    positives = [
        row["max_confidence"] or 0.0 for row in term_rows if row["present"]
    ]
    negatives = [
        row["max_confidence"] or 0.0 for row in term_rows if not row["present"]
    ]
    auc = rank_auc(positives, negatives)

    curves = {
        str(floor): sample_curve(term_rows, floor=floor) for floor in (0.22, 0.30)
    }

    # Shipped query path on the same terms.
    query_results = {}
    for mode in args.modes:
        query_config = SporeConfig(
            sample_fps=args.sample_fps,
            max_width=640,
            binding_mode=mode,
        )
        with SporeQueryEngine(store, query_config, perception, source="tabletop") as qe:
            rows = []
            for probe in probes:
                term = probe["term"]
                answer = qe.answer(f"where is the {term}?", budget=args.budget)
                top = answer.bindings[0] if answer.bindings else None
                rows.append(
                    {
                        "term": term,
                        "present": bool(probe["present"]),
                        "abstained": bool(answer.abstained),
                        "answer": answer.answer,
                        "bindings": len(answer.bindings),
                        "top_label": top.label if top else None,
                        "top_confidence": round(float(top.confidence), 4) if top else None,
                        "top_support": top.support if top else None,
                        "top_timestamp": round(float(top.timestamp), 2) if top else None,
                        "calls": int(answer.detector_calls),
                    }
                )
            query_results[mode] = rows
        print(f"[query:{mode}] {len(rows)} terms", flush=True)

    state_results = []
    if payload.get("state_questions"):
        with SporeQueryEngine(
            store,
            SporeConfig(sample_fps=args.sample_fps, max_width=640),
            perception,
            source="tabletop",
        ) as qe:
            for item in payload["state_questions"]:
                answer = qe.answer(
                    item["question"], budget=args.budget, at_time=float(item["at_time"])
                )
                top = answer.bindings[0] if answer.bindings else None
                found = bool(answer.bindings) and not answer.abstained
                state_results.append(
                    {
                        "question": item["question"],
                        "term": item.get("term"),
                        "at_time": float(item["at_time"]),
                        "expected_found": bool(item.get("expected_found", False)),
                        "expected_support": item.get("expected_support"),
                        "found": found,
                        "support": top.support if top else None,
                        "answer": answer.answer,
                        "top_confidence": round(float(top.confidence), 4) if top else None,
                        "top_timestamp": round(float(top.timestamp), 2) if top else None,
                        "calls": int(answer.detector_calls),
                        "found_match": found == bool(item.get("expected_found", False)),
                        "support_match": (
                            (top.support if top else None)
                            == item.get("expected_support")
                        ),
                    }
                )

    summary = {
        "probe_file": str(Path(args.probes)),
        "clip": args.clip,
        "store": args.store,
        "ingest_seconds": round(ingest_seconds, 2),
        "contexts_scanned": len(contexts),
        "terms": len(terms),
        "floor": args.floor,
        "scan_seconds": round(scan_seconds, 2),
        "scan_calls": int(scan_calls),
        "auc": round(auc, 3) if auc is not None else None,
        "present_terms": sum(1 for row in term_rows if row["present"]),
        "absent_terms": sum(1 for row in term_rows if not row["present"]),
        "terms_found_at_022": sum(
            1
            for row in term_rows
            if row["present"] and row["hits"]["0.22"] > 0
        ),
        "absent_clean_at_022": sum(
            1
            for row in term_rows
            if not row["present"] and row["hits"]["0.22"] == 0
        ),
        "term_rows": [
            {key: value for key, value in row.items() if key != "contexts"}
            for row in term_rows
        ],
        "curves": curves,
        "query": query_results,
        "state": state_results,
        "contexts": [
            {"frame_index": int(context.frame_index), "timestamp": round(float(context.timestamp), 2)}
            for context in contexts
        ],
    }
    return summary


def render_markdown(summary: dict) -> str:
    lines = ["# SPORE presence/absence calibration", ""]
    lines.append(
        f"Probe set: `{summary['probe_file']}`. Clip: `{summary['clip']}`. "
        f"{summary['contexts_scanned']} retained contexts x {summary['terms']} terms, "
        f"floor {summary['floor']}. Scan: {summary['scan_calls']} detector calls in "
        f"{summary['scan_seconds']:.0f}s."
    )
    lines.append("")
    lines.append(
        f"Term-level at floor .22: {summary['terms_found_at_022']}/"
        f"{summary['present_terms']} present terms found anywhere; "
        f"{summary['absent_clean_at_022']}/{summary['absent_terms']} absent terms "
        f"never detected. Max-confidence AUC {summary['auc']}."
    )
    lines.append("")
    lines.append("## Per term")
    lines.append("")
    lines.append("| term | human | occupancy | hits@.22 | hits@.30 | hits@.50 | max conf | first hit |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in summary["term_rows"]:
        lines.append(
            f"| {row['term']} | {'present' if row['present'] else 'absent'} | "
            f"{row['occupancy']} | {row['hits']['0.22']} | {row['hits']['0.3']} | "
            f"{row['hits']['0.5']} | {row['max_confidence']} | {row['first_hit']} |"
        )
    lines.append("")
    lines.append("## Sampling curve (term level)")
    lines.append("")
    lines.append("| k | phases | false negative (present, no hit) | false positive (absent, hit) | context sensitivity | context false positive |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in summary["curves"]["0.22"]:
        lines.append(
            f"| {row['k']} | {row['phases']} | {row['term_false_negative']} | "
            f"{row['term_false_positive']} | {row['context_sensitivity']} | "
            f"{row['context_false_positive']} |"
        )
    lines.append("")
    lines.append("## Shipped query path")
    lines.append("")
    lines.append("| mode | present terms abstained | absent terms answered |")
    lines.append("|---|---:|---:|")
    for mode, rows in summary["query"].items():
        present_abstained = sum(
            1 for row in rows if row["present"] and row["abstained"]
        )
        absent_answered = sum(
            1 for row in rows if not row["present"] and not row["abstained"]
        )
        present_total = sum(1 for row in rows if row["present"])
        absent_total = sum(1 for row in rows if not row["present"])
        lines.append(
            f"| {mode} | {present_abstained}/{present_total} | "
            f"{absent_answered}/{absent_total} |"
        )
    lines.append("")
    if summary["state"]:
        lines.append("## State-at-time probes")
        lines.append("")
        lines.append("| question | expected | found | support | found ok | support ok |")
        lines.append("|---|---|---|---|---|---|")
        for row in summary["state"]:
            expected = (
                f"found={row['expected_found']}, support={row['expected_support']}"
            )
            lines.append(
                f"| {row['question']} | {expected} | {row['found']} | "
                f"{row['support']} | {row['found_match']} | {row['support_match']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="", help="Ingest this clip when --store is absent.")
    parser.add_argument("--store", default="", help="Use an existing SPORE store.")
    parser.add_argument("--probes", required=True)
    parser.add_argument("--dump", default="", help="Optional store dump path.")
    parser.add_argument("--out", default="")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--context-gap", type=float, default=None)
    parser.add_argument("--max-contexts", type=int, default=0)
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument(
        "--modes",
        default="detector,hybrid",
        help="Comma-separated binding modes for the shipped query path.",
    )
    args = parser.parse_args()
    args.modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not args.clip and not args.store:
        parser.error("pass --clip or --store")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    summary = run(args)
    out = Path(args.out) if args.out else (
        ROOT / "research" / "spore" / "results" / "absence" / "report.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    out.with_suffix(".md").write_text(render_markdown(summary))
    print(f"[report] {out}")
    print(render_markdown(summary))
    if torch.cuda.is_available():
        print(f"[vram] peak={torch.cuda.max_memory_allocated() / 1e6:.1f}MB")


if __name__ == "__main__":
    main()

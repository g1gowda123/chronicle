#!/usr/bin/env python3
"""Evaluate the fixed-label presence/absence verifier on a probe set.

The verifier scans every retained context once with a closed-label detector
(COCO YOLOv8 by default) and answers term queries from the cache.  This harness
scores it against human presence labels and optional at-time questions.

Usage::

    python scripts/spore_absence_verify.py \
        --store research/spore/results/eat2_absence/store.json \
        --probes data/qa/eat_2_absence.json \
        --out research/spore/results/eat2_absence/verify.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chronicle.perception.absence import FixedLabelVerifier  # noqa: E402
from chronicle.perception.capsules import CapsuleStore  # noqa: E402


def load_probes(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("probes"), list) or not payload["probes"]:
        raise ValueError(f"{path}: expected a non-empty 'probes' list")
    return payload


def evaluate(args: argparse.Namespace) -> dict:
    payload = load_probes(Path(args.probes))
    probes = payload["probes"]
    store = CapsuleStore.load(args.store)
    verifier = FixedLabelVerifier(
        store,
        model=args.model,
        threshold=args.threshold,
        floor=args.floor,
    )
    started = time.perf_counter()
    calls = verifier.scan()
    scan_seconds = time.perf_counter() - started
    print(
        f"[scan] {calls} contexts in {scan_seconds:.1f}s "
        f"({verifier.scan_seconds:.1f}s detect), threshold {args.threshold}",
        flush=True,
    )

    rows = []
    for probe in probes:
        term = probe["term"]
        verdict = verifier.check(term)
        rows.append(
            {
                "term": term,
                "human_present": bool(probe.get("present", False)),
                "verdict": verdict.verdict,
                "supported": verdict.supported,
                "labels": list(verdict.labels),
                "hits": verdict.hits,
                "confidence": verdict.confidence,
                "timestamp": verdict.timestamp,
                "contexts_scanned": verdict.contexts_scanned,
            }
        )

    supported = [row for row in rows if row["supported"]]
    tp = sum(
        1 for row in supported if row["human_present"] and row["verdict"] == "present"
    )
    fp = sum(
        1 for row in supported if not row["human_present"] and row["verdict"] == "present"
    )
    fn = sum(
        1
        for row in supported
        if row["human_present"] and row["verdict"] != "present"
    )
    tn = sum(
        1
        for row in supported
        if not row["human_present"] and row["verdict"] != "present"
    )
    unsupported = [row["term"] for row in rows if not row["supported"]]

    time_rows = []
    for item in payload.get("time_questions", []):
        verdict = verifier.check(
            item["term"],
            at_time=float(item["at_time"]),
            window=args.window,
        )
        expected = bool(item.get("expected_present", False))
        time_rows.append(
            {
                "question": item.get("question", ""),
                "term": item["term"],
                "at_time": float(item["at_time"]),
                "expected_present": expected,
                "verdict": verdict.verdict,
                "correct": (verdict.verdict == "present") == expected,
                "supported": verdict.supported,
                "confidence": verdict.confidence,
                "timestamp": verdict.timestamp,
                "hits": verdict.hits,
                "contexts_scanned": verdict.contexts_scanned,
            }
        )

    summary = {
        "probe_file": str(Path(args.probes)),
        "store": args.store,
        "model": args.model,
        "threshold": args.threshold,
        "floor": args.floor,
        "contexts": len(store.contexts),
        "scan_calls": calls,
        "scan_seconds": round(scan_seconds, 2),
        "supported_terms": len(supported),
        "unsupported_terms": unsupported,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "positive_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "positive_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "false_presence_rate": round(fp / (fp + tn), 3) if fp + tn else None,
        "silence_rate_on_absent": round(tn / (tn + fp), 3) if tn + fp else None,
        "terms": rows,
        "time_questions": time_rows,
    }
    return summary


def render_markdown(summary: dict) -> str:
    lines = ["# Fixed-label absence verifier", ""]
    confusion = summary["confusion"]
    lines.append(
        f"Probe set `{summary['probe_file']}`, {summary['contexts']} contexts, "
        f"model `{summary['model']}`, threshold {summary['threshold']}. "
        f"Scan: {summary['scan_calls']} calls in {summary['scan_seconds']:.1f}s."
    )
    lines.append("")
    lines.append(
        f"Covered {summary['supported_terms']}/{len(summary['terms'])} terms; "
        f"unsupported: {', '.join(summary['unsupported_terms']) or 'none'}."
    )
    lines.append("")
    lines.append(
        f"Confusion: TP {confusion['tp']}, FP {confusion['fp']}, "
        f"FN {confusion['fn']}, TN {confusion['tn']}. "
        f"Positive precision {summary['positive_precision']}, recall "
        f"{summary['positive_recall']}, false-presence rate "
        f"{summary['false_presence_rate']}, silence on absent "
        f"{summary['silence_rate_on_absent']}."
    )
    lines.append("")
    lines.append("| term | human | verdict | hits | confidence | time |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in summary["terms"]:
        lines.append(
            f"| {row['term']} | {'present' if row['human_present'] else 'absent'} | "
            f"{row['verdict']} | {row['hits']} | {row['confidence']} | "
            f"{row['timestamp']} |"
        )
    if summary["time_questions"]:
        lines.append("")
        lines.append("## At-time questions")
        lines.append("")
        lines.append("| question | expected | verdict | correct | confidence |")
        lines.append("|---|---|---|---|---:|")
        for row in summary["time_questions"]:
            lines.append(
                f"| {row['question']} | "
                f"{'present' if row['expected_present'] else 'not present'} | "
                f"{row['verdict']} | {row['correct']} | {row['confidence']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--window", type=float, default=2.5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    summary = evaluate(args)
    out = Path(args.out) if args.out else Path(args.store).parent / "verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    out.with_suffix(".md").write_text(render_markdown(summary))
    print(render_markdown(summary))


if __name__ == "__main__":
    main()

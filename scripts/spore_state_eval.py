#!/usr/bin/env python3
"""Evaluate grounded presence/absence and state questions.

SPORE normally answers ``Where is X?`` by binding X to a capsule and reading a
support surface.  It has no explicit answer for ``Is X on the table?``,
``Is X still there?`` or ``Did X move?``: an empty binding is reported as
"not enough evidence", which conflates *not observed* with *absent*.

This harness evaluates the opt-in ``state_queries`` path against a
human-labelled question set.  It separates four outcomes:

* **present**  - the object was detected on the asked surface;
* **moved**    - it was detected on a different surface;
* **stayed**   - movement was asked and the surface did not change;
* **not observed** / **unknown** - no positive evidence.

The key metric is **false presence** on questions whose human label is "no":
a grounded state answer must not turn a detector miss into a positive claim.
A negative verdict is never scored as proof of absence.

Usage::

    python scripts/spore_state_eval.py \\
        --clip data/footage/aria_cook_0_45s.mp4 \\
        --qa data/qa/cook_0_state_45s.json \\
        --dump research/spore/results/cook0_state/store.json \\
        --out research/spore/results/cook0_state/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
from spore_real_bench import GENERAL_INVENTORY_LABELS  # noqa: E402

from chronicle.perception.capsules import CapsuleStore  # noqa: E402
from chronicle.perception.config import PerceptionConfig  # noqa: E402
from chronicle.perception.detector import build_detector  # noqa: E402
from chronicle.perception.embedder import build_embedder  # noqa: E402
from chronicle.perception.hands import build_hands  # noqa: E402
from chronicle.perception.spore import (  # noqa: E402
    SporeConfig,
    SporePerceiver,
    SporeQueryEngine,
    _normalise_surface,
    _support_aliases,
)


def support_matches(actual: str | None, expected: str | None) -> bool:
    if expected is None:
        return actual is None
    if actual is None:
        return False
    aliases = {_normalise_surface(alias) for alias in _support_aliases(expected)}
    return _normalise_surface(actual) in aliases


def score_state(question: dict, answer) -> tuple[bool | None, str]:
    expected = str(question.get("expected", "")).lower()
    verdict = answer.verdict
    support = (answer.stats.get("state") or {}).get("support")
    expected_support = question.get("expected_support")
    if question.get("kind") == "presence":
        if expected == "yes":
            if verdict != "present":
                return False, "missed_presence"
            if not support_matches(support, expected_support):
                return False, "wrong_support"
            return True, ""
        if expected == "no":
            if verdict in {"present", "moved", "stayed"}:
                return False, "false_presence"
            if verdict == "not_observed":
                return True, ""
            return False, "unexpected_verdict"
        return None, "unknown_label"
    if question.get("kind") == "still":
        if expected == "yes":
            if verdict != "present":
                return False, "missed_presence"
            if not support_matches(support, expected_support):
                return False, "wrong_support"
            return True, ""
        if expected == "no":
            if verdict == "present" and support_matches(support, expected_support):
                return True, ""
            if verdict == "not_observed":
                return True, ""
            return False, "wrong_state"
        return None, "unknown_label"
    if question.get("kind") == "moved":
        if expected == "unknown":
            return (verdict == "unknown"), "" if verdict == "unknown" else "false_presence"
        if expected in {"stayed", "no_move"}:
            if verdict == "moved":
                return False, "false_move"
            if verdict == "stayed":
                if not support_matches(support, expected_support):
                    return False, "wrong_support"
                return True, ""
            return False, "unexpected_verdict"
        if expected in {"moved", "yes"}:
            if verdict != "moved":
                return False, "missed_move"
            if expected_support and not support_matches(support, expected_support):
                return False, "wrong_support"
            return True, ""
        return None, "unknown_label"
    return None, "unknown_kind"


def score_default(question: dict, answer) -> tuple[bool | None, str]:
    """Baseline: the ordinary binding path has no state verdict."""
    expected = str(question.get("expected", "")).lower()
    if question.get("kind") != "presence":
        return None, "not_applicable"
    support = answer.best.support if answer.best is not None else None
    if expected == "yes":
        if answer.abstained:
            return False, "missed_presence"
        if not support_matches(support, question.get("expected_support")):
            return False, "wrong_support"
        return True, ""
    if expected == "no":
        return (answer.abstained), "" if answer.abstained else "false_presence"
    return None, "unknown_label"


def run_questions(engine, questions: list[dict], scorer) -> list[dict]:
    rows: list[dict] = []
    for question in questions:
        started = time.perf_counter()
        calls_before = engine.detector_calls
        answer = engine.answer(
            question["query"],
            budget=question.get("budget", 8),
            at_time=question.get("timestamp"),
        )
        calls = engine.detector_calls - calls_before
        correct, error = scorer(question, answer)
        rows.append(
            {
                "id": question.get("id"),
                "query": question["query"],
                "kind": question.get("kind"),
                "timestamp": question.get("timestamp"),
                "expected": question.get("expected"),
                "expected_support": question.get("expected_support"),
                "answer": answer.answer,
                "abstained": bool(answer.abstained),
                "verdict": answer.verdict,
                "support": (answer.stats.get("state") or {}).get("support"),
                "state": answer.stats.get("state", {}),
                "bindings": len(answer.bindings),
                "correct": correct,
                "error": error,
                "detector_calls": calls,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "evidence_bytes": answer.evidence_bytes,
            }
        )
        print(
            f"  {question.get('id', '')[:28]:28s} "
            f"exp={question.get('expected'):7s} verdict={str(answer.verdict):13s} "
            f"support={str((answer.stats.get('state') or {}).get('support')):10s} "
            f"correct={correct} {error} calls={calls}",
            flush=True,
        )
    return rows


def summarise(rows: list[dict], *, label: str) -> dict:
    scored = [row for row in rows if row["correct"] is not None]
    errors: dict[str, int] = {}
    for row in scored:
        if row["correct"]:
            continue
        errors[row["error"] or "wrong"] = errors.get(row["error"] or "wrong", 0) + 1
    by_kind: dict[str, dict] = {}
    for row in scored:
        bucket = by_kind.setdefault(
            row["kind"], {"questions": 0, "correct": 0, "errors": {}}
        )
        bucket["questions"] += 1
        bucket["correct"] += int(bool(row["correct"]))
        if not row["correct"]:
            edge = row["error"] or "wrong"
            bucket["errors"][edge] = bucket["errors"].get(edge, 0) + 1
    absence = [
        row
        for row in scored
        if row["kind"] == "presence" and str(row["expected"]).lower() == "no"
    ]
    positives = [
        row
        for row in scored
        if str(row["expected"]).lower() == "yes"
    ]
    return {
        "label": label,
        "questions": len(rows),
        "scored": len(scored),
        "correct": sum(bool(row["correct"]) for row in scored),
        "accuracy": round(
            sum(bool(row["correct"]) for row in scored) / max(1, len(scored)), 3
        ),
        "errors": errors,
        "false_presence": sum(
            1
            for row in absence
            if row["verdict"] in {"present", "moved", "stayed"}
            or (row["error"] == "false_presence")
        ),
        "absence_questions": len(absence),
        "missed_presence": sum(
            1 for row in positives if row["error"] == "missed_presence"
        ),
        "positive_questions": len(positives),
        "by_kind": by_kind,
        "mean_detector_calls": round(
            float(np.mean([row["detector_calls"] for row in rows])), 2
        ),
        "total_detector_calls": sum(row["detector_calls"] for row in rows),
        "mean_elapsed_s": round(
            float(np.mean([row["elapsed_s"] for row in rows])), 3
        ),
    }


def markdown(summary: dict, rows: list[dict], *, clip: str) -> str:
    lines = [
        f"# State/absence eval: `{clip}`",
        "",
        f"Mode: **{summary['label']}**",
        "",
        f"- questions: {summary['questions']} ({summary['scored']} scored)",
        f"- accuracy: **{summary['accuracy']}** ({summary['correct']}/{summary['scored']})",
        f"- false presence on absent probes: **{summary['false_presence']}/{summary['absence_questions']}**",
        f"- missed presence on positive probes: **{summary['missed_presence']}/{summary['positive_questions']}**",
        f"- detector calls: {summary['total_detector_calls']} "
        f"({summary['mean_detector_calls']}/question)",
        f"- mean latency: {summary['mean_elapsed_s']} s/question",
        "",
        "| id | expected | verdict | support | correct | answer |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        correct = "n/a" if row["correct"] is None else ("yes" if row["correct"] else f"no ({row['error']})")
        answer = str(row["answer"]).replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['expected']} | {row['verdict'] or '-'} | "
            f"{row['support'] or '-'} | {correct} | {answer} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--store", default="")
    parser.add_argument("--dump", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--max-capsules", type=int, default=300)
    parser.add_argument("--state-contexts", type=int, default=32)
    parser.add_argument("--state-window", type=float, default=2.5)
    parser.add_argument("--state-still-window", type=float, default=1.5)
    parser.add_argument("--state-min-confidence", type=float, default=0.30)
    parser.add_argument("--state-min-observations", type=int, default=2)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable the competing-label check for positive state answers.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run the ordinary (state_queries=False) binding path.",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.qa).read_text())
    questions = payload["questions"]
    perception = PerceptionConfig(model_dir=ROOT / "models")
    config = SporeConfig(
        sample_fps=args.sample_fps,
        max_width=640,
        max_capsules=args.max_capsules,
        query_capsules=args.budget,
        state_queries=True,
        state_contexts=args.state_contexts,
        state_time_window=args.state_window,
        state_still_window=args.state_still_window,
        state_min_confidence=args.state_min_confidence,
        state_min_observations=args.state_min_observations,
        state_verify_labels=() if args.no_verify else GENERAL_INVENTORY_LABELS,
    )

    embedder = build_embedder(perception, source_name="tabletop")
    hands = build_hands(perception, source_name="tabletop")
    detector = build_detector(perception, source_name="tabletop")
    if args.store:
        store = CapsuleStore.load(args.store)
        ingest_seconds = 0.0
        print(
            f"[store] {len(store.capsules)} capsules, {len(store.contexts)} contexts, "
            f"{store.memory_bytes()} bytes",
            flush=True,
        )
    else:
        perceiver = SporePerceiver(config, perception, embedder=embedder, hands=hands)
        started = time.perf_counter()
        store = perceiver.ingest(args.clip, source_name="tabletop")
        ingest_seconds = time.perf_counter() - started
        print(
            f"[ingest] {json.dumps(perceiver.stats.as_dict())} seconds={ingest_seconds:.2f}",
            flush=True,
        )
        perceiver.close()
        if args.dump:
            dump = Path(args.dump)
            dump.parent.mkdir(parents=True, exist_ok=True)
            store.save(
                dump,
                include_view_images=False,
                include_scene_images=False,
                split=True,
            )
            print(f"[dump] {dump}", flush=True)

    reports: dict[str, dict] = {}
    all_rows: dict[str, list[dict]] = {}
    modes = [("state", True)]
    if args.compare:
        modes.append(("default", False))
    for label, enabled in modes:
        mode_config = SporeConfig(
            sample_fps=config.sample_fps,
            max_width=config.max_width,
            max_capsules=config.max_capsules,
            query_capsules=config.query_capsules,
            state_queries=enabled,
            state_contexts=config.state_contexts,
            state_time_window=config.state_time_window,
            state_still_window=config.state_still_window,
            state_min_confidence=config.state_min_confidence,
            state_min_observations=config.state_min_observations,
            state_verify_labels=config.state_verify_labels,
        )
        engine = SporeQueryEngine(
            store,
            mode_config,
            perception,
            embedder=embedder,
            detector=detector,
            source=args.clip,
        )
        print(f"[mode] {label} state_queries={enabled}", flush=True)
        rows = run_questions(
            engine, questions, score_state if enabled else score_default
        )
        summary = summarise(rows, label=label)
        reports[label] = summary
        all_rows[label] = rows
        print(f"[summary] {json.dumps(summary)}", flush=True)
        engine.close()
    detector.close()
    embedder.close()
    hands.close()

    out = Path(args.out) if args.out else (
        ROOT
        / "research"
        / "spore"
        / "results"
        / f"state_{Path(args.clip).stem}"
        / "report.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "clip": args.clip,
        "qa": args.qa,
        "window": payload.get("window"),
        "protocol": payload.get("protocol"),
        "ingest_seconds": round(ingest_seconds, 2),
        "store_capsules": len(store.capsules),
        "store_contexts": len(store.contexts),
        "store_bytes": store.memory_bytes(),
        "config": {
            "budget": args.budget,
            "sample_fps": args.sample_fps,
            "state_contexts": args.state_contexts,
            "state_window": args.state_window,
            "state_still_window": args.state_still_window,
            "state_min_confidence": args.state_min_confidence,
            "state_min_observations": args.state_min_observations,
            "state_verify": not args.no_verify,
        },
        "summaries": reports,
        "results": all_rows,
    }
    out.write_text(json.dumps(combined, indent=2))
    md = out.with_suffix(".md")
    md.write_text(
        "\n".join(
            markdown(reports[label], all_rows[label], clip=args.clip)
            for label in reports
        )
    )
    print(f"[report] {out}", flush=True)
    print(f"[markdown] {md}", flush=True)
    if torch.cuda.is_available():
        print(
            f"[vram] peak={torch.cuda.max_memory_allocated() / 1e6:.1f}MB",
            flush=True,
        )


if __name__ == "__main__":
    main()

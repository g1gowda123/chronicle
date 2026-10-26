#!/usr/bin/env python3
"""Vault size / latency / accuracy report for the VFM vault.

Two modes:

1. Curated multi-episode question set (recommended):
    python prototype/benchmark.py \
      --questions benchmark/questions.json \
      --vault-dir /tmp/vfm \
      --source-root /home/g1/projects/chronicle/data/real

2. Legacy single-episode smoke test:
    python prototype/benchmark.py --source-db path/chronicle.db --vault vault.db

The curated set checks answer content, timestamps, evidence and cache reuse.
It is self-defined and graph-derived, not a public VideoQA benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from query import Answer, answer, load_store, stats

QUESTIONS = [
    ("where is the cup?", "answer"),
    ("where are my keys?", "unknown"),
    ("when did I last touch the cup?", "answer"),
    ("what did I see?", "answer"),
    ("what happened around 20 seconds?", "answer"),
    ("what was my friend doing with the cup?", "answer"),
    ("what happened before the cup was placed?", "answer"),
    ("how many entities were tracked?", "answer"),
    ("where was the cup last found?", "answer"),
    ("what is not in this memory?", "unknown"),
]

_TIME_IN_TEXT = re.compile(r"t\s*=\s*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _size(path: Path) -> int:
    total = 0
    for suffix in ("", "-wal"):
        candidate = Path(str(path) + suffix)
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def _video_bytes(folder: Path) -> int:
    candidates = (
        list(folder.glob("*.mp4"))
        + list(folder.glob("*.webm"))
        + list(folder.glob("*.mkv"))
    )
    return sum(path.stat().st_size for path in candidates if path.is_file())


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[index]


def _full_text(result: Answer) -> str:
    parts = [result.answer]
    for citation in result.citations:
        parts.append(str(citation.get("caption") or ""))
    return " ".join(parts).lower()


def score_answer(result: Answer, expected: dict[str, Any]) -> tuple[bool, str]:
    """Check content, timestamp, citation and evidence requirements."""

    text = _full_text(result)
    answer_text = result.answer.lower()
    if expected.get("kind") == "unknown":
        ok = (
            result.method.startswith("no-match")
            or "cache:no-match" in result.method
            or "don't have" in answer_text
            or "no record" in answer_text
        )
        return ok, "unknown handled" if ok else "expected an explicit unknown"

    if result.method.startswith("no-match") or "cache:no-match" in result.method:
        return False, "no answer produced"

    for term in expected.get("must_contain", []):
        if str(term).lower() not in text:
            return False, f"missing term: {term}"

    if expected.get("t_near") is not None:
        target = float(expected["t_near"])
        tolerance = float(expected.get("t_tol", 1.0))
        times = [float(value) for value in _TIME_IN_TEXT.findall(text)]
        times.extend(float(cite["t"]) for cite in result.citations if "t" in cite)
        if not any(abs(value - target) <= tolerance for value in times):
            return False, f"no timestamp within {tolerance}s of {target}s"

    if expected.get("cite_near") is not None:
        target = float(expected["cite_near"])
        tolerance = float(expected.get("cite_tol", 1.0))
        times = [float(cite["t"]) for cite in result.citations if "t" in cite]
        if not any(abs(value - target) <= tolerance for value in times):
            return False, f"no citation within {tolerance}s of {target}s"

    if expected.get("evidence", False) and not result.evidence:
        return False, "answer has no evidence JPEG"

    return True, "ok"


def _load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list) or not questions:
        raise SystemExit(f"no questions in {path}")
    return questions


def run_curated(
    questions_path: Path,
    vault_dir: Path,
    source_root: Path,
    *,
    repeats: int = 3,
    reset_cache: bool = True,
    json_output: bool = False,
) -> dict[str, Any]:
    questions = _load_questions(questions_path)
    episodes = sorted({str(item["episode"]) for item in questions})
    stores: dict[str, Any] = {}
    connections: dict[str, sqlite3.Connection] = {}
    size_rows: dict[str, dict[str, Any]] = {}
    try:
        for episode in episodes:
            vault = vault_dir / f"{episode}.db"
            if not vault.is_file():
                raise SystemExit(f"vault not found: {vault}")
            con = sqlite3.connect(str(vault))
            connections[episode] = con
            stores[episode] = load_store(con)
            if reset_cache:
                try:
                    con.execute("delete from qa_cache")
                    con.commit()
                    con.execute("VACUUM")
                    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
            info = stats(stores[episode], vault)
            folder = source_root / episode
            source_db = folder / "chronicle.db"
            video_bytes = _video_bytes(folder)
            if not video_bytes:
                prefix = episode.split("_")[0][:4].lower()
                for candidate in source_root.glob("*.mp4"):
                    if prefix and prefix in candidate.name.lower():
                        video_bytes += candidate.stat().st_size
            vault_bytes = _size(vault)
            size_rows[episode] = {
                "vault_bytes": vault_bytes,
                "evidence_bytes": int(info.get("evidence_bytes", 0)),
                "memory_bytes": vault_bytes + int(info.get("evidence_bytes", 0)),
                "video_bytes": video_bytes,
                "vault_pct_of_video": round(100.0 * vault_bytes / video_bytes, 3) if video_bytes else None,
                "memory_pct_of_video": round(
                    100.0 * (vault_bytes + int(info.get("evidence_bytes", 0))) / video_bytes, 3
                ) if video_bytes else None,
                "source_db_bytes": _size(source_db),
                "events": info.get("events", 0),
                "entities": info.get("entities", 0),
                "ledger_windows": info.get("ledger_windows", 0),
                "ledger_blob_bytes": int(
                    stores[episode].con.execute(
                        "select coalesce(sum(length(blob)),0) from caption_ledger"
                    ).fetchone()[0]
                ),
                "evidence_count": info.get("evidence", 0),
            }

        cold_latencies: list[float] = []
        warm_latencies: list[float] = []
        rows: list[dict[str, Any]] = []
        category_totals: dict[str, list[bool]] = defaultdict(list)
        per_episode: dict[str, list[bool]] = defaultdict(list)

        # Cold pass: no cache reads, no cache writes.
        for item in questions:
            store = stores[str(item["episode"])]
            result: Answer | None = None
            times: list[float] = []
            for _ in range(max(1, repeats)):
                started = time.perf_counter()
                result = answer(store, str(item["question"]), use_cache=False)
                times.append((time.perf_counter() - started) * 1000.0)
            assert result is not None
            cold_latencies.extend(times)
            ok, reason = score_answer(result, item.get("expected") or {})
            category_totals[str(item["category"])].append(ok)
            per_episode[str(item["episode"])].append(ok)
            rows.append(
                {
                    "id": item["id"],
                    "episode": item["episode"],
                    "category": item["category"],
                    "question": item["question"],
                    "method": result.method,
                    "cached": False,
                    "correct": ok,
                    "reason": reason,
                    "confidence": round(result.confidence, 2),
                    "p50_ms": round(statistics.median(times), 3),
                    "answer": result.answer.splitlines()[0][:220],
                    "citations": [cite.get("t") for cite in result.citations[:6]],
                    "evidence": bool(result.evidence),
                }
            )

        # Warm pass in question order: base questions populate the cache first,
        # paraphrase rows later in the file should hit it.
        warm_rows: list[dict[str, Any]] = []
        for item in questions:
            store = stores[str(item["episode"])]
            started = time.perf_counter()
            result = answer(store, str(item["question"]), use_cache=True)
            elapsed = (time.perf_counter() - started) * 1000.0
            warm_latencies.append(elapsed)
            ok, reason = score_answer(result, item.get("expected") or {})
            warm_rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "cached": bool(result.cached),
                    "method": result.method,
                    "correct": ok,
                    "reason": reason,
                    "elapsed_ms": round(elapsed, 3),
                }
            )

        # A second warm pass measures steady-state cache latency.
        steady_latencies: list[float] = []
        for item in questions:
            store = stores[str(item["episode"])]
            started = time.perf_counter()
            answer(store, str(item["question"]), use_cache=True)
            steady_latencies.append((time.perf_counter() - started) * 1000.0)

        paraphrase_rows = [row for row in warm_rows if row["category"] == "paraphrase"]
        cached_paraphrases = [row for row in paraphrase_rows if row["cached"]]
        reused_correct = sum(1 for row in cached_paraphrases if row["correct"])
        cache_entries = sum(
            int(
                store.con.execute("select count(*) from qa_cache").fetchone()[0]
            )
            for store in stores.values()
        )
        cache_hits = sum(
            int(
                store.con.execute("select coalesce(sum(hits),0) from qa_cache").fetchone()[0]
            )
            for store in stores.values()
        )
    finally:
        for con in connections.values():
            con.close()

    cold_correct = sum(1 for row in rows if row["correct"])
    result = {
        "questions": len(questions),
        "episodes": size_rows,
        "cold": {
            "correct": cold_correct,
            "accuracy": round(cold_correct / len(questions), 3),
            "p50_ms": round(statistics.median(cold_latencies), 3),
            "p95_ms": round(_percentile(cold_latencies, 95), 3),
        },
        "warm": {
            "p50_ms": round(statistics.median(warm_latencies), 3),
            "p95_ms": round(_percentile(warm_latencies, 95), 3),
            "steady_p50_ms": round(statistics.median(steady_latencies), 3),
            "steady_p95_ms": round(_percentile(steady_latencies, 95), 3),
        },
        "cache": {
            "entries": cache_entries,
            "hits": cache_hits,
            "paraphrases": len(paraphrase_rows),
            "paraphrase_hits": len(cached_paraphrases),
            "hit_rate": round(len(cached_paraphrases) / len(paraphrase_rows), 3)
            if paraphrase_rows else 0.0,
            "reused_accuracy": round(reused_correct / len(cached_paraphrases), 3)
            if cached_paraphrases else None,
        },
        "categories": {
            category: {
                "correct": sum(1 for value in values if value),
                "total": len(values),
                "accuracy": round(sum(1 for value in values if value) / len(values), 3),
            }
            for category, values in sorted(category_totals.items())
        },
        "episode_accuracy": {
            episode: {
                "correct": sum(1 for value in values if value),
                "total": len(values),
                "accuracy": round(sum(1 for value in values if value) / len(values), 3),
            }
            for episode, values in sorted(per_episode.items())
        },
        "rows": rows,
        "warm_rows": warm_rows,
    }
    if json_output:
        print(json.dumps(result, indent=2))
        return result

    print(f"questions   : {result['questions']}")
    print(f"cold accuracy: {result['cold']['correct']}/{result['questions']} "
          f"= {result['cold']['accuracy']:.1%}")
    print(f"latency      : cold p50 {result['cold']['p50_ms']} ms · "
          f"p95 {result['cold']['p95_ms']} ms")
    print(f"warm latency : p50 {result['warm']['steady_p50_ms']} ms · "
          f"p95 {result['warm']['steady_p95_ms']} ms")
    print(f"cache        : {result['cache']['paraphrase_hits']}/{result['cache']['paraphrases']} "
          f"paraphrases hit = {result['cache']['hit_rate']:.0%}; "
          f"reused accuracy = {result['cache']['reused_accuracy']}")
    print()
    print("| Category | Correct | Total | Accuracy |")
    print("|---|---:|---:|---:|")
    for category, data in result["categories"].items():
        print(f"| {category} | {data['correct']} | {data['total']} | {data['accuracy']:.0%} |")
    print()
    print("| Episode | Vault | Evidence | Memory | Video | Memory % | Events | Ledger windows |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for episode, data in result["episodes"].items():
        print(
            f"| {episode} | {data['vault_bytes']/1024:.1f} KB | "
            f"{data['evidence_bytes']/1024:.1f} KB | "
            f"{data['memory_bytes']/1024:.1f} KB | "
            f"{data['video_bytes']/1048576:.2f} MB | "
            f"{data['memory_pct_of_video']}% | {data['events']} | "
            f"{data['ledger_windows']} |"
        )
    print()
    print("| ID | Category | Question | Correct | Method | Answer |")
    print("|---|---|---|---:|---|---|")
    for row in rows:
        print(
            f"| {row['id']} | {row['category']} | {row['question']} | "
            f"{'yes' if row['correct'] else 'no'} | {row['method']} | "
            f"{row['answer'].replace('|', '/')} |"
        )
    failures = [row for row in rows if not row["correct"]]
    if failures:
        print()
        print("Failures:")
        for row in failures:
            print(f"- {row['id']} {row['question']} -> {row['reason']}")
    print()
    print("| Paraphrase | Cached | Method | Correct |")
    print("|---|---:|---|---:|")
    for row in warm_rows:
        if row["category"] == "paraphrase":
            print(
                f"| {row['id']} {row['question']} | {'yes' if row['cached'] else 'no'} | "
                f"{row['method']} | {'yes' if row['correct'] else 'no'} |"
            )
    return result


def run(source_db: Path, vault: Path, repeats: int = 5) -> dict[str, Any]:
    """Legacy single-episode smoke test."""

    con = sqlite3.connect(str(vault))
    con.row_factory = sqlite3.Row
    store = load_store(con)
    rows = []
    latencies: list[float] = []
    correct = 0
    try:
        for question, expected in QUESTIONS:
            times: list[float] = []
            result = None
            for _ in range(repeats):
                started = time.perf_counter()
                result = answer(store, question, use_cache=False)
                times.append((time.perf_counter() - started) * 1000.0)
            assert result is not None
            latencies.extend(times)
            answered = not result.method.startswith("no-match")
            ok = answered == (expected == "answer")
            correct += int(ok)
            rows.append(
                {
                    "question": question,
                    "expected": expected,
                    "method": result.method,
                    "answered": answered,
                    "correct": ok,
                    "p50_ms": round(statistics.median(times), 2),
                    "confidence": round(result.confidence, 2),
                    "answer": result.answer.splitlines()[0][:160],
                }
            )
        vault_stats = stats(store, vault)
    finally:
        con.close()

    vault_bytes = _size(vault)
    source_bytes = _size(source_db)
    video_bytes = _video_bytes(source_db.parent)
    return {
        "source_db": str(source_db),
        "vault": str(vault),
        "source_db_bytes": source_bytes,
        "video_bytes": video_bytes,
        "vault_bytes": vault_bytes,
        "vault_as_pct_of_video": round(100.0 * vault_bytes / video_bytes, 3) if video_bytes else None,
        "entities": vault_stats.get("entities", 0),
        "events": vault_stats.get("events", 0),
        "captions": vault_stats.get("captions", 0),
        "event_blob_bytes": vault_stats.get("event_blob_bytes", 0),
        "caption_blob_bytes": vault_stats.get("caption_blob_bytes", 0),
        "bytes_per_event": round(
            vault_stats.get("event_blob_bytes", 0) / max(1, vault_stats.get("events", 1)), 2
        ),
        "questions": len(QUESTIONS),
        "correct": correct,
        "accuracy": round(correct / len(QUESTIONS), 3),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(_percentile(latencies, 95), 2),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--vault-dir", type=Path, default=Path("/tmp/vfm"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/g1/projects/chronicle/data/real"),
    )
    parser.add_argument("--source-db", type=Path, default=None)
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.questions is not None:
        run_curated(
            args.questions.expanduser().resolve(),
            args.vault_dir.expanduser().resolve(),
            args.source_root.expanduser().resolve(),
            repeats=max(1, args.repeats),
            reset_cache=not args.keep_cache,
            json_output=args.json,
        )
        return 0

    if args.source_db is None or args.vault is None:
        parser.error("provide --questions, or both --source-db and --vault")
    report = run(
        args.source_db.expanduser().resolve(),
        args.vault.expanduser().resolve(),
        repeats=max(1, args.repeats),
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"source DB   : {report['source_db_bytes']/1024:.1f} KB")
    print(f"source video: {report['video_bytes']/1048576:.2f} MB")
    print(f"vault       : {report['vault_bytes']/1024:.1f} KB "
          f"({report['vault_as_pct_of_video']}% of video)")
    print(f"entities / events / captions: "
          f"{report['entities']} / {report['events']} / {report['captions']}")
    print(f"event blob  : {report['event_blob_bytes']} B "
          f"({report['bytes_per_event']} B/event)")
    print(f"caption blob: {report['caption_blob_bytes']} B (zlib)")
    print(f"accuracy    : {report['correct']}/{report['questions']} = {report['accuracy']:.0%}")
    print(f"latency     : p50 {report['p50_ms']} ms · p95 {report['p95_ms']} ms")
    print()
    print("| Question | Expected | Method | Correct | p50 ms | Answer |")
    print("|---|---|---|---:|---:|---|")
    for row in report["rows"]:
        print(
            f"| {row['question']} | {row['expected']} | {row['method']} | "
            f"{'yes' if row['correct'] else 'no'} | {row['p50_ms']} | {row['answer']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

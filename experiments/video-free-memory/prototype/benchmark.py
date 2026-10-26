#!/usr/bin/env python3
"""Vault size / latency / coverage report.

Usage:
    python prototype/benchmark.py \
      --source-db /path/to/chronicle.db \
      --vault /tmp/vault.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path

from query import answer

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


def _size(path: Path) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def _video_bytes(source_db: Path) -> int:
    folder = source_db.parent
    candidates = list(folder.glob("*.mp4")) + list(folder.glob("*.webm")) + list(folder.glob("*.mkv"))
    return sum(path.stat().st_size for path in candidates if path.is_file())


def _count(con: sqlite3.Connection, table: str) -> int:
    try:
        return int(con.execute(f"select count(*) from {table}").fetchone()[0])
    except sqlite3.Error:
        return 0


def run(source_db: Path, vault: Path, repeats: int = 5) -> dict:
    con = sqlite3.connect(str(vault))
    con.row_factory = sqlite3.Row
    rows = []
    latencies: list[float] = []
    correct = 0
    try:
        for question, expected in QUESTIONS:
            times: list[float] = []
            result = None
            for _ in range(repeats):
                started = time.perf_counter()
                result = answer(con, question)
                times.append((time.perf_counter() - started) * 1000.0)
            assert result is not None
            latencies.extend(times)
            answered = result.method != "no-match"
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
    finally:
        con.close()

    vault_bytes = _size(vault)
    source_bytes = _size(source_db)
    video_bytes = _video_bytes(source_db)
    return {
        "source_db": str(source_db),
        "vault": str(vault),
        "source_db_bytes": source_bytes,
        "video_bytes": video_bytes,
        "vault_bytes": vault_bytes,
        "vault_as_pct_of_video": round(100.0 * vault_bytes / video_bytes, 3) if video_bytes else None,
        "events": _count(sqlite3.connect(str(vault)), "events"),
        "captions": _count(sqlite3.connect(str(vault)), "captions"),
        "entities": _count(sqlite3.connect(str(vault)), "entities"),
        "questions": len(QUESTIONS),
        "correct": correct,
        "accuracy": round(correct / len(QUESTIONS), 3),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        args.source_db.expanduser().resolve(),
        args.vault.expanduser().resolve(),
        repeats=max(1, args.repeats),
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"source DB : {report['source_db_bytes']/1024:.1f} KB")
    print(f"source video: {report['video_bytes']/1048576:.2f} MB")
    print(f"vault      : {report['vault_bytes']/1024:.1f} KB "
          f"({report['vault_as_pct_of_video']}% of video)")
    print(f"entities / events / captions: "
          f"{report['entities']} / {report['events']} / {report['captions']}")
    print(f"accuracy: {report['correct']}/{report['questions']} = {report['accuracy']:.0%}")
    print(f"latency: p50 {report['p50_ms']} ms · p95 {report['p95_ms']} ms")
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

#!/usr/bin/env python3
"""Benchmark the original Chronicle runtime (graph + Qwen 7B) on the 30 questions.

This runs the same `benchmark/questions.json` against the original query path so
we can compare answer quality, latency and at-rest footprint with the VFM vault.

The original runtime is used read-only; no files in the main repo are modified.

Usage:
    /home/g1/projects/chronicle/.venv/bin/python prototype/original_baseline.py \
      --questions benchmark/questions.json \
      --source-root /home/g1/projects/chronicle/data/real \
      --model qwen2.5:7b-instruct
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import time
from pathlib import Path
from typing import Any

import benchmark
from query import Answer


class _Timeout(Exception):
    pass


def _alarm(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise _Timeout()


def _video(source_root: Path, episode: str) -> Path | None:
    folder = source_root / episode
    for pattern in ("*.mp4", "*.webm", "*.mkv"):
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    prefix = episode.split("_")[0][:4].lower()
    for pattern in ("*.mp4", "*.webm", "*.mkv"):
        for candidate in sorted(source_root.glob(pattern)):
            if prefix and prefix in candidate.name.lower():
                return candidate
    return None


def _convert(result: Any) -> Answer:
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    citations = []
    for citation in payload.get("citations") or []:
        timestamp = citation.get("timestamp")
        if timestamp is None:
            continue
        citations.append(
            {
                "t": round(float(timestamp), 3),
                "kind": str(citation.get("relation") or "fact"),
                "caption": f"{citation.get('relation', 'fact')} at t={float(timestamp):.1f}s",
            }
        )
    frame = payload.get("frame")
    evidence = []
    if isinstance(frame, dict) and frame:
        evidence.append({"t": float(frame.get("t") or payload.get("last_seen") or 0.0)})
    answer_text = str(payload.get("answer") or "")
    method = str(payload.get("method") or payload.get("source") or "graphrag")
    if "don't have" in answer_text.lower() or "don't remember" in answer_text.lower():
        method = "no-match"
    return Answer(
        question=str(payload.get("query") or ""),
        answer=answer_text,
        citations=citations,
        evidence=evidence,
        confidence=1.0 if payload.get("grounded") else 0.4,
        method=method,
    )


def run(
    questions_path: Path,
    source_root: Path,
    *,
    model: str,
    limit: int = 0,
    timeout: float = 90.0,
    json_output: bool = False,
) -> dict[str, Any]:
    from chronicle.integration import ChronicleRuntime
    from chronicle.semantic.config import SemanticConfig
    from chronicle.semantic.llm import ensure_local_ollama

    questions = benchmark._load_questions(questions_path)
    if limit:
        questions = questions[:limit]
    episodes = sorted({str(item["episode"]) for item in questions})

    config = SemanticConfig.from_env().replace(llm_backend="ollama", llm_model=model)
    ensure_local_ollama(config)

    runtimes: dict[str, Any] = {}
    load_ms: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    categories: dict[str, dict[str, int]] = {}
    try:
        for episode in episodes:
            db_path = source_root / episode / "chronicle.db"
            started = time.perf_counter()
            runtimes[episode] = ChronicleRuntime(
                db_path,
                semantic_config=config,
                load=True,
            )
            load_ms[episode] = round((time.perf_counter() - started) * 1000.0, 1)

        for item in questions:
            episode = str(item["episode"])
            expected = item.get("expected") or {}
            runtime = runtimes[episode]
            started = time.perf_counter()
            timed_out = False
            try:
                signal.signal(signal.SIGALRM, _alarm)
                signal.setitimer(signal.ITIMER_REAL, max(5.0, timeout))
                result = runtime.ask(str(item["question"]))
                converted = _convert(result)
            except _Timeout:
                timed_out = True
                converted = Answer(
                    question=str(item["question"]),
                    answer="",
                    method="timeout",
                    confidence=0.0,
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            elapsed = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed)
            ok, reason = benchmark.score_answer(converted, expected)
            if timed_out:
                ok, reason = False, f"timeout after {timeout:g}s"
            category = str(item["category"])
            bucket = categories.setdefault(category, {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(ok)
            rows.append(
                {
                    "id": item["id"],
                    "episode": episode,
                    "category": category,
                    "question": item["question"],
                    "correct": ok,
                    "reason": reason,
                    "method": converted.method,
                    "elapsed_ms": round(elapsed, 1),
                    "answer": converted.answer[:300],
                    "citations": [cite["t"] for cite in converted.citations[:6]],
                    "evidence": bool(converted.evidence),
                }
            )
    finally:
        for runtime in runtimes.values():
            runtime.close()

    correct = sum(1 for row in rows if row["correct"])
    sizes = {}
    for episode in episodes:
        folder = source_root / episode
        video = _video(source_root, episode)
        sizes[episode] = {
            "db_bytes": benchmark._size(folder / "chronicle.db"),
            "video_bytes": video.stat().st_size if video else 0,
        }
    report = {
        "questions": len(rows),
        "model": model,
        "load_ms": load_ms,
        "correct": correct,
        "accuracy": round(correct / len(rows), 3) if rows else 0.0,
        "p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95_ms": round(benchmark._percentile(latencies, 95), 1) if latencies else 0.0,
        "categories": {
            name: {
                **data,
                "accuracy": round(data["correct"] / data["total"], 3) if data["total"] else 0.0,
            }
            for name, data in sorted(categories.items())
        },
        "sizes": sizes,
        "rows": rows,
    }
    if json_output:
        print(json.dumps(report, indent=2))
        return report

    print(f"questions : {report['questions']}")
    print(
        f"original  : {correct}/{len(rows)} = {report['accuracy']:.1%} · "
        f"p50 {report['p50_ms']:.0f} ms · p95 {report['p95_ms']:.0f} ms"
    )
    print(f"model     : {model}")
    print("load ms   : " + ", ".join(f"{ep}={value}" for ep, value in load_ms.items()))
    print()
    print("| Category | Correct | Total | Accuracy |")
    print("|---|---:|---:|---:|")
    for name, data in report["categories"].items():
        print(f"| {name} | {data['correct']} | {data['total']} | {data['accuracy']:.0%} |")
    print()
    print("| ID | Category | Correct | Method | Answer |")
    print("|---|---|---:|---|---|")
    for row in rows:
        print(
            f"| {row['id']} | {row['category']} | {'yes' if row['correct'] else 'no'} | "
            f"{row['method']} | {row['answer'].replace('|', '/')[:180]} |"
        )
    failures = [row for row in rows if not row["correct"]]
    if failures:
        print()
        print("Failures:")
        for row in failures:
            print(f"- {row['id']} {row['question']} -> {row['reason']}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    run(
        args.questions.expanduser().resolve(),
        args.source_root.expanduser().resolve(),
        model=args.model,
        limit=max(0, args.limit),
        timeout=max(5.0, args.timeout),
        json_output=args.json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

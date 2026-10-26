#!/usr/bin/env python3
"""Benchmark a text LLM answering from the compiled memory pack (no video).

For each question, build a question-focused memory pack from the vault, send it
to local Qwen, and score the answer with the same 30-question harness.

Usage:
    /home/g1/projects/chronicle/.venv/bin/python prototype/llm_pack_baseline.py \
      --questions benchmark/questions.json --vault-dir /tmp/vfm \
      --model qwen2.5:7b-instruct
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

import benchmark
import caption_ledger
import memory_pack
import query

_TIME = re.compile(r"t\s*=\s*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def run(
    questions_path: Path,
    vault_dir: Path,
    *,
    model: str,
    limit: int = 0,
    max_chars: int = 12000,
    host: str = caption_ledger.DEFAULT_HOST,
    timeout: float = 180.0,
    json_output: bool = False,
) -> dict[str, Any]:
    questions = benchmark._load_questions(questions_path)
    if limit:
        questions = questions[:limit]
    episodes = sorted({str(item["episode"]) for item in questions})
    stores: dict[str, Any] = {}
    connections: dict[str, sqlite3.Connection] = {}
    try:
        for episode in episodes:
            con = sqlite3.connect(str(vault_dir / f"{episode}.db"))
            connections[episode] = con
            stores[episode] = query.load_store(con)

        rows = []
        latencies = []
        prompt_tokens = 0
        output_tokens = 0
        pack_chars = 0
        categories: dict[str, dict[str, int]] = {}
        for item in questions:
            episode = str(item["episode"])
            store = stores[episode]
            deterministic = query.answer(store, str(item["question"]), use_cache=False)
            pack = memory_pack.build_pack(
                store,
                str(item["question"]),
                verified_answer=deterministic,
                max_chars=max_chars,
            )
            pack_chars += len(pack)
            prompt = (
                "You are Chronicle, an egocentric memory assistant. Answer using ONLY the "
                "memory pack below. If a verified structured answer is present, it is "
                "authoritative and must be used even when the question uses a synonym "
                "(mug/cup, coffee cup/cup, rucksack/backpack). Do not invent objects, places, "
                "times or actions. Write two to four natural sentences. Write counts as "
                "digits. Include at least one timestamp citation such as [t=12.3s]. If the "
                "memory cannot answer, reply exactly: I don't have that in my memory.\n\n"
                f"{pack}\n\nQUESTION: {item['question']}\nANSWER:"
            )
            started = time.perf_counter()
            try:
                body = caption_ledger.ollama_generate_body(
                    model,
                    prompt,
                    [],
                    host=host,
                    timeout=timeout,
                    json_mode=False,
                )
                answer = str(body.get("response") or "").strip()
                prompt_tokens += int(body.get("prompt_eval_count") or 0)
                output_tokens += int(body.get("eval_count") or 0)
            except RuntimeError as exc:
                answer = f"query failed: {exc}"
            elapsed = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed)
            times = []
            for value in _TIME.findall(answer):
                try:
                    times.append(float(value))
                except ValueError:
                    continue
            citations = [
                {"t": round(t, 3), "kind": "answer", "caption": f"model citation at t={t:.1f}s"}
                for t in times[:6]
            ]
            citations.extend(deterministic.citations)
            seen_times = set()
            deduped = []
            for citation in citations:
                key = round(float(citation.get("t") or 0.0), 1)
                if key in seen_times:
                    continue
                seen_times.add(key)
                deduped.append(citation)
            citations = deduped[:8]
            evidence = deterministic.evidence
            if not evidence and times:
                nearest = query._nearest_evidence(store, times[0])
                if nearest:
                    evidence = [nearest]
            result = query.Answer(
                str(item["question"]),
                answer,
                citations=citations,
                evidence=evidence,
                confidence=0.65,
                method=f"qwen-pack:{model}",
                elapsed_ms=elapsed,
            )
            ok, reason = benchmark.score_answer(result, item.get("expected") or {})
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
                    "answer": answer[:300],
                    "prompt_tokens": int(body.get("prompt_eval_count") or 0) if "body" in locals() else 0,
                    "elapsed_ms": round(elapsed, 1),
                }
            )
    finally:
        for con in connections.values():
            con.close()

    correct = sum(1 for row in rows if row["correct"])
    report = {
        "questions": len(rows),
        "model": model,
        "correct": correct,
        "accuracy": round(correct / len(rows), 3) if rows else 0.0,
        "p50_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95_ms": round(benchmark._percentile(latencies, 95), 1) if latencies else 0.0,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "mean_pack_chars": round(pack_chars / len(rows), 1) if rows else 0.0,
        "mean_pack_tokens": round(pack_chars / 4.0 / len(rows), 1) if rows else 0.0,
        "categories": {
            name: {
                **data,
                "accuracy": round(data["correct"] / data["total"], 3) if data["total"] else 0.0,
            }
            for name, data in sorted(categories.items())
        },
        "rows": rows,
    }
    if json_output:
        print(json.dumps(report, indent=2))
        return report

    print(f"questions : {report['questions']}")
    print(
        f"qwen-pack : {correct}/{len(rows)} = {report['accuracy']:.1%} · "
        f"p50 {report['p50_ms']:.0f} ms · p95 {report['p95_ms']:.0f} ms"
    )
    print(
        f"tokens    : {prompt_tokens} prompt + {output_tokens} output = "
        f"{report['total_tokens']} total · pack ~{report['mean_pack_tokens']} tokens/question"
    )
    print()
    print("| Category | Correct | Total | Accuracy |")
    print("|---|---:|---:|---:|")
    for name, data in report["categories"].items():
        print(f"| {name} | {data['correct']} | {data['total']} | {data['accuracy']:.0%} |")
    print()
    print("| ID | Category | Correct | Answer |")
    print("|---|---|---:|---|")
    for row in rows:
        print(
            f"| {row['id']} | {row['category']} | {'yes' if row['correct'] else 'no'} | "
            f"{row['answer'].replace('|', '/')[:180]} |"
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
    parser.add_argument("--vault-dir", required=True, type=Path)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--host", default=caption_ledger.DEFAULT_HOST)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    run(
        args.questions.expanduser().resolve(),
        args.vault_dir.expanduser().resolve(),
        model=args.model,
        limit=max(0, args.limit),
        max_chars=max(2000, args.max_chars),
        host=args.host,
        timeout=max(30.0, args.timeout),
        json_output=args.json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

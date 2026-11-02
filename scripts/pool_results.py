"""Pool per-dataset evaluation runs into one combined result file.

The paper reports per-dataset Recall@1 and a pooled 18-question comparison.
Pooling only makes sense when the runs were produced by the same system and
configuration; this helper concatenates rows, recomputes aggregate counts and
leaves the per-row evidence untouched so ``scripts/analyze_results.py`` can run
paired tests over the combined suite.

Example
-------
    python scripts/pool_results.py \
        --run runs/research/final_cook/results.json \
        --run runs/research/final_holo/results.json \
        --out runs/research/pooled/final.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def pool(results: Sequence[dict[str, Any]], *, name: str = "pooled") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(result.get("rows") or ())

    categories: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = categories.setdefault(
            str(row.get("category", "unknown")), {"correct": 0, "total": 0}
        )
        bucket["total"] += 1
        if row.get("correct"):
            bucket["correct"] += 1
    for bucket in categories.values():
        bucket["recall_at_1"] = (
            bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
        )

    correct = sum(1 for row in rows if row.get("correct"))
    total = len(rows)
    latency = [
        float(row["query_latency_ms"])
        for row in rows
        if row.get("query_latency_ms") is not None
    ]
    return {
        "method": "chronicle",
        "dataset": name,
        "video": name,
        "qa_file": name,
        "evaluated_at": max(
            (str(result.get("evaluated_at", "")) for result in results), default=""
        ),
        "total": total,
        "answered": sum(1 for row in rows if not row.get("error")),
        "unreachable": sum(1 for row in rows if row.get("unreachable_timestamp")),
        "correct": correct,
        "recall_at_1": correct / total if total else 0.0,
        "avg_query_latency_ms": (
            round(sum(latency) / len(latency), 4) if latency else None
        ),
        "categories": categories,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="pooled")
    args = parser.parse_args(argv)

    payload = pool(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.run],
        name=args.name,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"{args.name}: {payload['correct']}/{payload['total']} "
        f"({100.0 * payload['recall_at_1']:.1f}%) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

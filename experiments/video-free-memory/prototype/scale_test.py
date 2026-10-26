#!/usr/bin/env python3
"""Long-horizon scaling test: 10/30/60 minute synthetic episodes.

Cycles the real compiled episodes (cook, walking, hololens) to the requested
duration, then measures vault growth, event density and query latency. This is a
synthetic concatenation test, not a new recording.

Usage:
    python prototype/scale_test.py \
      --vault /tmp/vfm/cook.db --vault /tmp/vfm/walking.db \
      --vault /tmp/vfm/hololens.db \
      --out-dir /tmp/vfm/scale --minutes 10 30 60
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

from concat_vaults import merge
from query import answer, load_store


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[index]


def _queries(duration: float) -> list[str]:
    fractions = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    queries = [
        "where is the cup?",
        "where is the joy-con?",
        "when did I touch the cup?",
        "how many cups?",
        "what did I see?",
        "what did I do with the tomato?",
        "where is the backpack?",
    ]
    for fraction in fractions:
        t = max(1.0, duration * fraction)
        queries.append(f"what happened around {t:.0f} seconds?")
    return queries


def run_one(vault: Path, *, repeats: int = 3) -> dict[str, Any]:
    vault = vault.resolve()
    started = time.perf_counter()
    con = sqlite3.connect(str(vault))
    con.row_factory = sqlite3.Row
    store = load_store(con)
    load_ms = (time.perf_counter() - started) * 1000.0
    latencies: list[float] = []
    late_window_errors: list[float] = []
    try:
        queries = _queries(store.duration)
        for query in queries:
            for _ in range(repeats):
                begin = time.perf_counter()
                result = answer(store, query, use_cache=False)
                latencies.append((time.perf_counter() - begin) * 1000.0)
            if query.startswith("what happened around"):
                target = float(query.split("around ")[1].split(" seconds")[0])
                cites = [float(cite["t"]) for cite in result.citations if "t" in cite]
                if cites:
                    late_window_errors.append(min(abs(t - target) for t in cites))
        info = {
            "vault_bytes": vault.stat().st_size,
            "events": len(store.events),
            "entities": len(store.entities),
            "ledger_windows": len(store.ledger_windows),
            "evidence_rows": len(store.evidence),
            "unique_evidence_files": len({row["rel_path"] for row in store.evidence}),
            "unique_evidence_bytes": sum(
                {row["rel_path"]: row["bytes"] for row in store.evidence}.values()
            ),
        }
    finally:
        con.close()
    info.update(
        {
            "duration_s": round(store.duration, 3),
            "minutes": round(store.duration / 60.0, 2),
            "events_per_min": round(len(store.events) / max(1e-9, store.duration / 60.0), 2),
            "bytes_per_event": round(info["vault_bytes"] / max(1, info["events"]), 2),
            "load_ms": round(load_ms, 3),
            "queries": len(latencies),
            "p50_ms": round(statistics.median(latencies), 3),
            "p95_ms": round(_percentile(latencies, 95), 3),
            "p99_ms": round(_percentile(latencies, 99), 3),
            "worst_window_error_s": round(max(late_window_errors), 3)
            if late_window_errors
            else None,
        }
    )
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--minutes", nargs="+", type=float, default=[10, 30, 60])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-evidence", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    vaults = [path.expanduser().resolve() for path in args.vault]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for minutes in args.minutes:
        out = out_dir / f"synth_{minutes:g}m.db"
        summary = merge(vaults, out, minutes, max_evidence=args.max_evidence)
        metrics = run_one(out, repeats=max(1, args.repeats))
        rows.append({**summary, **metrics, "out": str(out)})
        print(
            f"{minutes:g} min: {metrics['events']} events, "
            f"vault {metrics['vault_bytes']/1024:.1f} KB, "
            f"load {metrics['load_ms']:.1f} ms, "
            f"p50 {metrics['p50_ms']:.3f} ms, p95 {metrics['p95_ms']:.3f} ms, "
            f"worst window error {metrics['worst_window_error_s']} s"
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print()
    print(
        "| Minutes | Segments | Events | Ev/min | Vault | B/event | Load ms | "
        "p50 ms | p95 ms | Evidence rows | Unique images | Unique bytes |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['minutes']:g} | {row['segments']} | {row['events']} | "
            f"{row['events_per_min']} | {row['vault_bytes']/1024:.1f} KB | "
            f"{row['bytes_per_event']} | {row['load_ms']} | {row['p50_ms']} | "
            f"{row['p95_ms']} | {row['evidence_rows']} | "
            f"{row['unique_evidence_files']} | "
            f"{row['unique_evidence_bytes']/1024:.1f} KB |"
        )
    print()
    print("Synthetic concatenation: evidence images repeat across cycles; unique bytes are real, "
          "row counts are not. A real 45–60 min episode would store up to 40 unique snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Retention budget and evidence pruning for a VFM vault.

The event ledger is cheap (~17 bytes/event). Evidence JPEG/WebP files dominate,
so retention is mostly about how many snapshots per hour we keep and how large
each one is.

Modes:
  --report                 show bytes/hour and 7/30/365-day projections
  --sweep 40,12,4          simulate snapshot budgets on temporary vault copies
  --apply --keep-per-hour N prune evidence rows in place
  --delete-files           with --apply, delete unreferenced image files

Usage:
  python prototype/retention.py --vault /tmp/vfm/scale/synth_60m.db --report
  python prototype/retention.py --vault /tmp/vfm/cook.db --sweep 40,12,4
  python prototype/retention.py --vault /tmp/vfm/cook.db \
    --apply --keep-per-hour 12 --drop-cache --vacuum
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema_v2.sql"
SCHEMA_V3 = ROOT / "schema_v3.sql"

KIND_SCORE = {"interaction": 4, "anchor": 3, "state": 2, "keyframe": 1}


def _size(path: Path) -> int:
    """Bytes at rest: main database plus an un-checkpointed WAL, excluding -shm."""
    total = 0
    for suffix in ("", "-wal"):
        candidate = Path(str(path) + suffix)
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def _checkpoint(con: sqlite3.Connection) -> None:
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass


def _duration(con: sqlite3.Connection) -> float:
    try:
        row = con.execute("select duration_s from episodes limit 1").fetchone()
        if row and row[0]:
            return float(row[0])
    except sqlite3.Error:
        pass
    return 0.0


def _evidence(con: sqlite3.Connection) -> list[dict[str, Any]]:
    con.row_factory = sqlite3.Row
    try:
        rows = list(con.execute("select * from evidence order by t"))
    except sqlite3.Error:
        return []
    return [
        {
            "id": str(row["id"]),
            "t": float(row["t"]),
            "kind": str(row["kind"]),
            "label": str(row["label"] or ""),
            "rel_path": str(row["rel_path"]),
            "bytes": int(row["bytes"] or 0),
        }
        for row in rows
    ]


def _score(row: dict[str, Any]) -> float:
    score = float(KIND_SCORE.get(str(row["kind"]), 1))
    if str(row["label"]).lower() in {"person", "cup", "plate", "bowl", "knife", "joy-con"}:
        score += 0.2
    return score


def _select(rows: list[dict[str, Any]], keep_per_hour: int, duration: float) -> list[dict[str, Any]]:
    """Select at most keep_per_hour rows per clock hour, spread across the hour."""

    if keep_per_hour <= 0:
        return []
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[int(float(row["t"]) // 3600.0)].append(row)

    selected: list[dict[str, Any]] = []
    for _, bucket in sorted(buckets.items()):
        if len(bucket) <= keep_per_hour:
            selected.extend(bucket)
            continue
        bucket.sort(key=lambda row: float(row["t"]))
        slots: list[list[dict[str, Any]]] = [[] for _ in range(keep_per_hour)]
        width = 3600.0 / keep_per_hour
        for row in bucket:
            slot = min(keep_per_hour - 1, int((float(row["t"]) % 3600.0) // width))
            slots[slot].append(row)
        chosen: list[dict[str, Any]] = []
        for slot in slots:
            if slot:
                chosen.append(max(slot, key=_score))
        if len(chosen) < keep_per_hour:
            chosen_ids = {id(row) for row in chosen}
            leftovers = [row for row in bucket if id(row) not in chosen_ids]
            leftovers.sort(key=_score, reverse=True)
            chosen.extend(leftovers[: keep_per_hour - len(chosen)])
        selected.extend(chosen[:keep_per_hour])
    return sorted(selected, key=lambda row: float(row["t"]))


def _unique_bytes(rows: list[dict[str, Any]]) -> tuple[int, int]:
    files = {row["rel_path"]: int(row["bytes"]) for row in rows}
    return len(files), sum(files.values())


def _breakdown(con: sqlite3.Connection, vault: Path) -> dict[str, int]:
    try:
        rows = list(
            con.execute(
                "select name, sum(pgsize) from dbstat group by name order by sum(pgsize) desc"
            )
        )
        return {str(row[0]): int(row[1]) for row in rows}
    except sqlite3.Error:
        return {"vault": _size(vault)}


def _apply_prune(
    con: sqlite3.Connection,
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    vacuum: bool,
    drop_cache: bool,
) -> None:
    keep_ids = {row["id"] for row in selected}
    delete_ids = [row["id"] for row in rows if row["id"] not in keep_ids]
    if delete_ids:
        con.executemany("delete from evidence where id=?", [(value,) for value in delete_ids])
    if drop_cache:
        try:
            con.execute("delete from qa_cache")
        except sqlite3.Error:
            pass
    con.commit()
    if vacuum:
        con.execute("VACUUM")
    _checkpoint(con)


def _copy_and_measure(
    vault: Path, selected_ids: set[str], *, vacuum: bool = True
) -> int:
    with tempfile.TemporaryDirectory(prefix="vfm-retention-") as tmp:
        target = Path(tmp) / "vault.db"
        shutil.copy2(vault, target)
        con = sqlite3.connect(str(target))
        try:
            con.executescript(SCHEMA.read_text(encoding="utf-8"))
            con.executescript(SCHEMA_V3.read_text(encoding="utf-8"))
            rows = _evidence(con)
            keep = set(selected_ids)
            delete_ids = [row["id"] for row in rows if row["id"] not in keep]
            con.executemany("delete from evidence where id=?", [(value,) for value in delete_ids])
            try:
                con.execute("delete from qa_cache")
            except sqlite3.Error:
                pass
            con.commit()
            if vacuum:
                con.execute("VACUUM")
            _checkpoint(con)
        finally:
            con.close()
        return _size(target)


def _project_per_hour(
    vault_bytes: int,
    selected_rows: list[dict[str, Any]],
    duration: float,
    *,
    max_snapshots_per_hour: int = 40,
) -> dict[str, Any]:
    unique_files, unique_bytes = _unique_bytes(selected_rows)
    avg_image = unique_bytes / unique_files if unique_files else 0.0
    hours = max(duration / 3600.0, 1.0 / 60.0)
    if duration >= 3000:
        vault_per_hour = vault_bytes / hours
        snapshots_per_hour = len(selected_rows) / hours
    else:
        # Short clips have fixed schema overhead; project only the variable parts.
        vault_per_hour = vault_bytes
        snapshots_per_hour = len(selected_rows)
    snapshots_per_hour = min(float(max_snapshots_per_hour), snapshots_per_hour)
    evidence_per_hour = snapshots_per_hour * avg_image
    total = vault_per_hour + evidence_per_hour
    return {
        "avg_image_bytes": round(avg_image, 1),
        "snapshots_per_hour": round(snapshots_per_hour, 2),
        "vault_bytes_per_hour": round(vault_per_hour, 1),
        "evidence_bytes_per_hour": round(evidence_per_hour, 1),
        "memory_bytes_per_hour": round(total, 1),
        "unique_evidence_files": unique_files,
        "unique_evidence_bytes": unique_bytes,
    }


def report(vault: Path, sweep: list[int], *, hours_per_day: float = 1.0) -> dict[str, Any]:
    con = sqlite3.connect(str(vault))
    con.row_factory = sqlite3.Row
    try:
        _checkpoint(con)
        duration = _duration(con)
        rows = _evidence(con)
        breakdown = _breakdown(con, vault)
        counts = {
            "events": 0,
            "entities": 0,
            "ledger_windows": 0,
            "qa_cache": 0,
        }
        try:
            counts["events"] = int(con.execute("select n from event_pack where id=1").fetchone()[0])
        except sqlite3.Error:
            pass
        try:
            counts["entities"] = int(con.execute("select count(*) from entities").fetchone()[0])
        except sqlite3.Error:
            pass
        try:
            counts["ledger_windows"] = int(
                con.execute("select n from caption_ledger where id=1").fetchone()[0]
            )
        except sqlite3.Error:
            pass
        try:
            counts["qa_cache"] = int(con.execute("select count(*) from qa_cache").fetchone()[0])
        except sqlite3.Error:
            pass
        vault_bytes = _size(vault)
        unique_files, unique_bytes = _unique_bytes(rows)
    finally:
        con.close()

    sweep_rows: list[dict[str, Any]] = []
    for keep in sweep:
        selected = _select(rows, keep, duration)
        selected_ids = {row["id"] for row in selected}
        pruned_bytes = _copy_and_measure(vault, selected_ids)
        selected_unique_files, selected_unique_bytes = _unique_bytes(selected)
        projection = _project_per_hour(pruned_bytes, selected, duration)
        sweep_rows.append(
            {
                "keep_per_hour": keep,
                "evidence_rows": len(selected),
                "unique_files": selected_unique_files,
                "unique_bytes": selected_unique_bytes,
                "vault_bytes_after": pruned_bytes,
                **projection,
                "projected_30d_bytes": round(
                    projection["memory_bytes_per_hour"] * hours_per_day * 30.0
                ),
                "projected_365d_bytes": round(
                    projection["memory_bytes_per_hour"] * hours_per_day * 365.0
                ),
            }
        )

    current = _project_per_hour(vault_bytes, rows, duration)
    return {
        "vault": str(vault),
        "duration_s": round(duration, 3),
        "counts": counts,
        "bytes": {
            "vault_total": vault_bytes,
            "unique_evidence_files": unique_files,
            "unique_evidence_bytes": unique_bytes,
            "memory_total": vault_bytes + unique_bytes,
        },
        "breakdown": breakdown,
        "current": current,
        "sweep": sweep_rows,
    }


def apply(
    vault: Path,
    keep_per_hour: int,
    *,
    delete_files: bool = False,
    data_root: Path | None = None,
    drop_cache: bool = False,
    vacuum: bool = True,
) -> dict[str, Any]:
    if delete_files and data_root is None:
        raise SystemExit("--delete-files requires --data-root")
    before = _size(vault)
    con = sqlite3.connect(str(vault))
    con.row_factory = sqlite3.Row
    try:
        _checkpoint(con)
        duration = _duration(con)
        rows = _evidence(con)
        selected = _select(rows, keep_per_hour, duration)
        selected_ids = {row["id"] for row in selected}
        _apply_prune(con, rows, selected, vacuum=vacuum, drop_cache=drop_cache)
        remaining = _evidence(con)
        referenced = {row["rel_path"] for row in remaining}
        reclaimed = 0
        if delete_files and data_root is not None:
            root = data_root.expanduser().resolve()
            for row in rows:
                rel_path = row["rel_path"]
                if rel_path in referenced:
                    continue
                candidate = Path(rel_path)
                if not candidate.is_absolute():
                    candidate = root / candidate
                try:
                    resolved = candidate.resolve()
                    resolved.relative_to(root)
                except (OSError, ValueError):
                    continue
                if resolved.is_file():
                    reclaimed += resolved.stat().st_size
                    resolved.unlink()
        _checkpoint(con)
    finally:
        con.close()
    return {
        "vault": str(vault),
        "keep_per_hour": keep_per_hour,
        "evidence_before": len(rows),
        "evidence_after": len(remaining),
        "vault_bytes_before": before,
        "vault_bytes_after": _size(vault),
        "files_deleted_bytes": reclaimed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--sweep", default="40,12,4")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-per-hour", type=int, default=40)
    parser.add_argument("--delete-files", action="store_true")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--drop-cache", action="store_true")
    parser.add_argument("--no-vacuum", action="store_true")
    parser.add_argument("--hours-per-day", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    vault = args.vault.expanduser().resolve()
    if not vault.is_file():
        raise SystemExit(f"vault not found: {vault}")

    if args.apply:
        result = apply(
            vault,
            max(0, args.keep_per_hour),
            delete_files=args.delete_files,
            data_root=args.data_root.expanduser().resolve() if args.data_root else None,
            drop_cache=args.drop_cache,
            vacuum=not args.no_vacuum,
        )
        print(json.dumps(result, indent=2))
        return 0

    sweep = sorted({int(value) for value in args.sweep.split(",") if value.strip()}, reverse=True)
    result = report(vault, sweep, hours_per_day=max(0.1, args.hours_per_day))
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"vault        : {vault}")
    print(f"duration     : {result['duration_s']/60:.1f} min")
    print(f"vault bytes  : {result['bytes']['vault_total']/1024:.1f} KB")
    print(
        f"evidence     : {result['bytes']['unique_evidence_files']} unique files, "
        f"{result['bytes']['unique_evidence_bytes']/1024:.1f} KB"
    )
    print(
        f"memory total : {result['bytes']['memory_total']/1024:.1f} KB "
        f"(vault + unique evidence)"
    )
    print(
        f"per hour     : {result['current']['memory_bytes_per_hour']/1024:.1f} KB "
        f"({result['current']['snapshots_per_hour']:.1f} snapshots/h)"
    )
    print()
    print("Breakdown (SQLite pages):")
    for name, size in list(result["breakdown"].items())[:10]:
        print(f"  {name:22} {size/1024:8.1f} KB")
    print()
    print("| Snapshots/h | Evidence rows | Unique files | Unique bytes | Vault after | Memory/h | 30 days | 365 days |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["sweep"]:
        print(
            f"| {row['keep_per_hour']} | {row['evidence_rows']} | {row['unique_files']} | "
            f"{row['unique_bytes']/1024:.1f} KB | {row['vault_bytes_after']/1024:.1f} KB | "
            f"{row['memory_bytes_per_hour']/1024:.1f} KB | "
            f"{row['projected_30d_bytes']/1048576:.1f} MB | "
            f"{row['projected_365d_bytes']/1048576:.1f} MB |"
        )
    print()
    print(
        "Projection assumes 1 hour recorded per day; multiply the 30/365-day figures "
        "for heavier use. The vault grows with events, evidence with snapshots."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Concatenate compiled vaults into a synthetic long-horizon episode.

This is a scaling test, not a new recording. It cycles the input episodes until
the requested duration, shifts timestamps, and merges events/captions/ledger
windows/evidence rows into one vault. Reused evidence images keep their original
paths, so `unique_evidence_bytes` is reported separately from row counts.

Usage:
    python prototype/concat_vaults.py \
      --vault /tmp/vfm/cook.db --vault /tmp/vfm/walking.db \
      --vault /tmp/vfm/hololens.db \
      --minutes 10 --out /tmp/vfm/synth_10m.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from compact import StringTable, compress_json, pack_events
from query import load_store

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema_v2.sql"
SCHEMA_V3 = ROOT / "schema_v3.sql"


def _segments(paths: list[Path], target_s: float) -> list[tuple[Path, float, Any]]:
    loaded = []
    for path in paths:
        con = sqlite3.connect(str(path))
        con.row_factory = sqlite3.Row
        store = load_store(con)
        loaded.append((path, store, con))
    segments: list[tuple[Path, float, Any]] = []
    offset = 0.0
    while offset < target_s:
        for path, store, _con in loaded:
            duration = max(1.0, float(store.duration))
            segments.append((path, offset, store))
            offset += duration
            if offset >= target_s:
                break
    for _, _, con in loaded:
        con.close()
    return segments


def merge(
    paths: list[Path], out: Path, minutes: float, *, max_evidence: int = 40
) -> dict[str, Any]:
    target_s = max(1.0, minutes * 60.0)
    segments = _segments(paths, target_s)
    if out.exists():
        out.unlink()

    table = StringTable()
    events: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    ledger_windows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    segment_meta: list[dict[str, Any]] = []

    for index, (path, offset, store) in enumerate(segments):
        segment_meta.append(
            {
                "index": index,
                "source": path.name,
                "offset": round(offset, 3),
                "duration": round(float(store.duration), 3),
            }
        )
        for event in store.events:
            events.append(
                {
                    **event,
                    "t": round(float(event["t"]) + offset, 3),
                    "meta": {"segment": index, "source": path.stem},
                }
            )
        for entity in store.entities:
            first = entity.get("first_seen")
            last = entity.get("last_seen")
            entities.append(
                {
                    "id": f"s{index:03d}:{entity['id']}",
                    "label": str(entity["label"]),
                    "state": str(entity["state"]),
                    "zone": str(entity["zone"]),
                    "first_seen": float(first) + offset if first is not None else None,
                    "last_seen": float(last) + offset if last is not None else None,
                    "attrs": "{}",
                }
            )
        for caption in store.captions:
            captions.append(
                {
                    **caption,
                    "t_start": round(float(caption["t_start"]) + offset, 3),
                    "t_end": round(float(caption["t_end"]) + offset, 3),
                }
            )
        for window in store.ledger_windows:
            shifted = dict(window)
            shifted["t_start"] = round(float(window.get("t_start", 0.0)) + offset, 3)
            shifted["t_end"] = round(float(window.get("t_end", 0.0)) + offset, 3)
            shifted["segment"] = index
            frames = []
            for frame in window.get("frames") or []:
                if isinstance(frame, dict):
                    frames.append({**frame, "t": round(float(frame.get("t", 0.0)) + offset, 3)})
            shifted["frames"] = frames
            ledger_windows.append(shifted)
        for evidence in store.evidence:
            evidence_rows.append(
                {
                    **evidence,
                    "id": f"s{index:03d}:{evidence['id']}",
                    "t": round(float(evidence["t"]) + offset, 3),
                }
            )

    events.sort(key=lambda event: float(event["t"]))
    all_evidence_rows = sorted(evidence_rows, key=lambda row: float(row["t"]))
    if max_evidence and len(all_evidence_rows) > max_evidence:
        step = len(all_evidence_rows) / float(max_evidence)
        evidence_rows = [
            all_evidence_rows[min(len(all_evidence_rows) - 1, int(index * step))]
            for index in range(max_evidence)
        ]
    else:
        evidence_rows = all_evidence_rows
    for event in events:
        table.add(event["subject"])
        table.add(event["relation"])
        table.add(event["object"])
        table.add(event["zone"])
    for entity in entities:
        table.add(entity["label"])
        table.add(entity["state"])
        table.add(entity["zone"])
    blob = pack_events(events, table)
    captions_blob = compress_json(captions)
    ledger = {
        "version": 1,
        "model": "synthetic-concat",
        "mode": "concat",
        "window_s": 30.0,
        "source": " + ".join(path.stem for path in paths),
        "duration_s": round(target_s, 3),
        "generated_at": time.time(),
        "windows": ledger_windows,
    }
    ledger_blob = compress_json(ledger)

    vault = sqlite3.connect(str(out))
    vault.row_factory = sqlite3.Row
    try:
        vault.executescript(SCHEMA.read_text(encoding="utf-8"))
        vault.executescript(SCHEMA_V3.read_text(encoding="utf-8"))
        now = time.time()
        zone_id = table.add(entities[0]["zone"]) if entities else 0
        vault.execute(
            """
            insert into episodes
                (id, title, source_path, zone_id, duration_s, fps, video_retained,
                 compiled_at, compiler)
            values (?, ?, ?, ?, ?, NULL, 0, ?, ?)
            """,
            (
                f"synthetic-{minutes:g}m",
                f"synthetic {minutes:g} minute episode",
                " + ".join(str(path) for path in paths),
                zone_id,
                target_s,
                now,
                "vfm-v3-concat",
            ),
        )
        vault.executemany(
            "insert or replace into strings(id, value) values (?, ?)", table.rows()
        )
        vault.execute(
            "insert or replace into event_pack(id, blob, n) values (1, ?, ?)",
            (blob, len(events)),
        )
        vault.execute(
            "insert or replace into captions_blob(id, blob) values (1, ?)", (captions_blob,)
        )
        vault.execute(
            "insert or replace into caption_ledger(id, model, window_s, blob, n, created_at) "
            "values (1, ?, ?, ?, ?, ?)",
            ("synthetic-concat", 30.0, ledger_blob, len(ledger_windows), now),
        )
        for entity in entities:
            vault.execute(
                """
                insert or replace into entities
                    (id, label_id, state_id, zone_id, first_seen, last_seen, attrs)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity["id"],
                    table.add(entity["label"]),
                    table.add(entity["state"]),
                    table.add(entity["zone"]),
                    entity["first_seen"],
                    entity["last_seen"],
                    entity["attrs"],
                ),
            )
        for row in evidence_rows:
            vault.execute(
                """
                insert or replace into evidence
                    (id, t, kind, label, file, rel_path, bytes, width, height, meta)
                values (?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
                """,
                (
                    row["id"],
                    row["t"],
                    row["kind"],
                    row["label"],
                    Path(row["rel_path"]).name,
                    row["rel_path"],
                    row["bytes"],
                    json.dumps({"segment": row["id"].split(":", 1)[0]}),
                ),
            )
        vault.executemany(
            "insert or replace into meta(key, value) values (?, ?)",
            [
                ("schema_version", "vfm-v3-concat"),
                ("compiled_at", str(now)),
                ("event_count", str(len(events))),
                ("caption_windows", str(len(captions))),
                ("ledger_windows", str(len(ledger_windows))),
                ("event_blob_bytes", str(len(blob))),
                ("caption_blob_bytes", str(len(captions_blob))),
                ("ledger_blob_bytes", str(len(ledger_blob))),
                ("segments", str(len(segments))),
                ("video_retained", "0"),
            ],
        )
        vault.commit()
        vault.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        vault.close()

    unique_files = {row["rel_path"]: int(row["bytes"]) for row in evidence_rows}
    return {
        "out": str(out),
        "minutes": minutes,
        "duration_s": round(target_s, 3),
        "segments": len(segments),
        "events": len(events),
        "entities": len(entities),
        "ledger_windows": len(ledger_windows),
        "evidence_rows": len(evidence_rows),
        "all_evidence_rows": len(all_evidence_rows),
        "max_evidence": max_evidence,
        "unique_evidence_files": len(unique_files),
        "unique_evidence_bytes": sum(unique_files.values()),
        "vault_bytes": out.stat().st_size,
        "event_blob_bytes": len(blob),
        "ledger_blob_bytes": len(ledger_blob),
        "segment_meta": segment_meta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--minutes", required=True, type=float)
    args = parser.parse_args(argv)
    summary = merge(
        [path.expanduser().resolve() for path in args.vault],
        args.out.expanduser().resolve(),
        args.minutes,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

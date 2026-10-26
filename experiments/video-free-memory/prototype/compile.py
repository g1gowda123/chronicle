#!/usr/bin/env python3
"""Compile a Chronicle SQLite episode into a video-free, compressed VFM vault.

v2 layout:
  * dictionary-encoded strings (uint16 ids)
  * events packed as a struct-of-arrays blob (~16 bytes/event)
  * window captions zlib-compressed JSON
  * no video reference; `video_retained=0`

Usage:
    python prototype/compile.py --source-db path/to/chronicle.db --out vault.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from compact import StringTable, compress_json, pack_events

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema_v2.sql"
WINDOW_S = 30.0

#: States worth a ledger row. Visibility flicker (OCCLUDED/OUT_OF_VIEW) is not.
STABLE_STATES = {
    "NEW",
    "VISIBLE_STATIC",
    "VISIBLE_MOVING",
    "INTERACTED",
    "HELD",
    "CONTAINED",
    "LOST",
}


def _rows(con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    try:
        return list(con.execute(sql, params))
    except sqlite3.Error:
        return []


def _node_label(raw: str, labels: dict[str, str]) -> str:
    value = str(raw or "").strip()
    if value in labels:
        return labels[value]
    for prefix in ("entity:", "anchor:", "zone:"):
        if value.startswith(prefix) and value[len(prefix) :] in labels:
            return labels[value[len(prefix) :]]
    return value.removeprefix("entity:").removeprefix("anchor:").removeprefix("zone:")


def _event(
    t: float,
    kind: str,
    subject: str,
    relation: str,
    obj: str,
    zone: str | None,
    confidence: float | None,
    source: str | None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "t": round(float(t), 3),
        "kind": kind,
        "subject": subject or "",
        "relation": relation or "",
        "object": obj or "",
        "zone": zone or "",
        "confidence": confidence,
        "source": source or "",
        "meta": meta or {},
    }


def _dedupe(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events.sort(key=lambda item: (item["t"], item["kind"], item["subject"]))
    seen: set[tuple[float, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for event in events:
        key = (
            round(float(event["t"]), 1),
            str(event["subject"]).lower(),
            str(event["relation"]).upper(),
            str(event["object"]).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


def build_events(source: sqlite3.Connection, duration: float) -> list[dict[str, Any]]:
    entities = _rows(source, "select id, label, state, zone_id, last_seen from entities")
    zones = _rows(source, "select id, label from zones")
    furniture = _rows(source, "select id, label from furniture")

    labels: dict[str, str] = {}
    for row in entities:
        labels[str(row["id"])] = str(row["label"])
        labels[f"entity:{row['id']}"] = str(row["label"])
    for row in zones:
        labels[str(row["id"])] = str(row["label"])
        labels[f"zone:{row['id']}"] = str(row["label"])
    for row in furniture:
        labels[str(row["id"])] = str(row["label"])
        labels[f"anchor:{row['id']}"] = str(row["label"])

    zone_labels = {str(row["id"]): str(row["label"]) for row in zones}
    entity_zone = {
        str(row["id"]): zone_labels.get(str(row["zone_id"]), "")
        for row in entities
    }

    events: list[dict[str, Any]] = []

    # 1. Stable lifecycle changes.
    transition_rows = _rows(
        source,
        """
        select s.entity_id, e.label, s.from_state, s.to_state, s.t_event, s.evidence
        from state_transitions s left join entities e on e.id = s.entity_id
        """,
    )
    by_entity: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in transition_rows:
        by_entity[str(row["entity_id"])].append(row)
    for entity_id, rows_for_entity in by_entity.items():
        rows_for_entity.sort(key=lambda row: float(row["t_event"]))
        label = str(rows_for_entity[0]["label"] or _node_label(entity_id, labels))
        zone = entity_zone.get(entity_id)
        last_stable: str | None = None
        for row in rows_for_entity:
            to_state = str(row["to_state"])
            if to_state not in STABLE_STATES or to_state == last_stable:
                continue
            last_stable = to_state
            events.append(
                _event(
                    row["t_event"],
                    "state",
                    label,
                    f"{row['from_state']}->{to_state}",
                    label,
                    zone,
                    None,
                    "lifecycle",
                    {"from": row["from_state"], "to": to_state},
                )
            )

    # 2. One anchor per (entity, relation, target): terminal/preferred latest.
    anchor_rows = _rows(
        source,
        """
        select a.entity_id, e.label, a.anchor_id, a.relation, a.t_start, a.is_terminal, a.confidence
        from anchors a left join entities e on e.id = a.entity_id
        order by a.is_terminal desc, a.t_start desc
        """,
    )
    seen_anchor: set[tuple[str, str, str]] = set()
    for row in anchor_rows:
        label = str(row["label"] or _node_label(row["entity_id"], labels))
        target = _node_label(row["anchor_id"], labels)
        key = (label.lower(), str(row["relation"]).upper(), target.lower())
        if key in seen_anchor:
            continue
        seen_anchor.add(key)
        events.append(
            _event(
                row["t_start"],
                "anchor",
                label,
                str(row["relation"]),
                target,
                entity_zone.get(str(row["entity_id"])),
                row["confidence"],
                "anchor",
                {"terminal": bool(row["is_terminal"])},
            )
        )

    # 3. Non-spatial graph relations (IN_ZONE is implied by the entity/zone table).
    for row in _rows(source, "select src, dst, relation, t_start from edges"):
        relation = str(row["relation"] or "")
        if relation.upper() in {"IN_ZONE", "RESTING_ON", "INSIDE", "ON", "CONTAINED_IN"}:
            continue
        t = row["t_start"]
        if t is None:
            continue
        events.append(
            _event(
                t,
                "relation",
                _node_label(row["src"], labels),
                relation,
                _node_label(row["dst"], labels),
                None,
                None,
                "graph",
            )
        )

    # 4. Periodic scene anchors so quiet periods are not invisible.
    total = max(0.0, float(duration))
    step = WINDOW_S
    tick = 0.0
    while tick < total:
        events.append(_event(tick, "keyframe", "", "ANCHOR", "", "", None, "periodic"))
        tick += step

    return _dedupe(events)


def build_captions(events: list[dict[str, Any]], window_s: float = WINDOW_S) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["kind"] == "keyframe":
            continue
        buckets[int(float(event["t"]) // window_s)].append(event)

    captions: list[dict[str, Any]] = []
    for bucket in sorted(buckets):
        rows = buckets[bucket]
        start = bucket * window_s
        end = start + window_s
        lines = [caption_for(event) for event in rows[:12]]
        if len(rows) > 12:
            lines.append(f"... {len(rows) - 12} more events.")
        tags = sorted(
            {str(event["subject"]).lower() for event in rows if event["subject"]}
            | {str(event["object"]).lower() for event in rows if event["object"]}
        )
        captions.append(
            {
                "t_start": start,
                "t_end": end,
                "text": f"From {start:.1f}s to {end:.1f}s:\n" + "\n".join(lines),
                "tags": tags[:20],
            }
        )
    return captions


def caption_for(event: dict[str, Any]) -> str:
    t = float(event["t"])
    subject = str(event["subject"]).lower()
    relation = str(event["relation"]).replace("_", " ").lower()
    obj = str(event["object"]).lower()
    zone = str(event["zone"])
    kind = event["kind"]
    if kind == "state":
        text = f"At t={t:.1f}s, the {subject} transitioned {relation}"
    elif kind == "anchor":
        text = f"At t={t:.1f}s, the {subject} was {relation} the {obj}"
    elif kind == "relation":
        text = f"At t={t:.1f}s, {subject} {relation} {obj}"
    elif kind == "keyframe":
        text = f"Scene anchor at t={t:.1f}s"
    else:
        text = f"At t={t:.1f}s, {subject} {obj}"
    if zone and zone.lower() not in {"", subject, obj}:
        text += f" in the {zone}"
    return text.rstrip(".") + "."


def compile_vault(source_db: Path, out_db: Path, *, title: str | None = None) -> dict[str, Any]:
    if not source_db.is_file():
        raise SystemExit(f"source database not found: {source_db}")
    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()

    source = sqlite3.connect(str(source_db))
    source.row_factory = sqlite3.Row
    try:
        entities = _rows(
            source,
            "select id, label, state, zone_id, first_seen, last_seen, attrs from entities",
        )
        zones = _rows(source, "select id, label from zones")
        zone_labels = {str(row["id"]): str(row["label"]) for row in zones}
        last_t = _rows(source, "select max(t) as last_t from observations")
        duration = float(last_t[0]["last_t"] or 0.0) if last_t else 0.0
    finally:
        source.close()

    # Re-open for event building (keeps the function readable).
    source = sqlite3.connect(str(source_db))
    source.row_factory = sqlite3.Row
    try:
        events = build_events(source, duration)
    finally:
        source.close()

    table = StringTable()
    for row in entities:
        table.add(str(row["label"]))
        table.add(str(row["state"]))
        table.add(zone_labels.get(str(row["zone_id"]), ""))
    for event in events:
        table.add(event["subject"])
        table.add(event["relation"])
        table.add(event["object"])
        table.add(event["zone"])
    blob = pack_events(events, table)

    captions = build_captions(events)
    captions_blob = compress_json(captions)

    vault = sqlite3.connect(str(out_db))
    vault.row_factory = sqlite3.Row
    try:
        vault.executescript(SCHEMA.read_text(encoding="utf-8"))
        now = time.time()
        episode_id = source_db.parent.name or source_db.stem
        zone_id = table.add(zone_labels.get(str(entities[0]["zone_id"]), "")) if entities else 0
        vault.execute(
            """
            insert or replace into episodes
                (id, title, source_path, zone_id, duration_s, fps, video_retained, compiled_at)
            values (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                episode_id,
                title or episode_id,
                str(source_db),
                zone_id,
                duration,
                None,
                now,
            ),
        )
        vault.executemany("insert or replace into strings(id, value) values (?, ?)", table.rows())
        vault.execute("insert or replace into event_pack(id, blob, n) values (1, ?, ?)", (blob, len(events)))
        vault.execute("insert or replace into captions_blob(id, blob) values (1, ?)", (captions_blob,))
        for row in entities:
            vault.execute(
                """
                insert or replace into entities
                    (id, label_id, state_id, zone_id, first_seen, last_seen, attrs)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["id"]),
                    table.add(str(row["label"])),
                    table.add(str(row["state"])),
                    table.add(zone_labels.get(str(row["zone_id"]), "")),
                    row["first_seen"],
                    row["last_seen"],
                    str(row["attrs"] or "{}"),
                ),
            )
        vault.executemany(
            "insert or replace into meta(key, value) values (?, ?)",
            [
                ("schema_version", "vfm-v2"),
                ("compiled_at", str(now)),
                ("event_count", str(len(events))),
                ("caption_windows", str(len(captions))),
                ("event_blob_bytes", str(len(blob))),
                ("caption_blob_bytes", str(len(captions_blob))),
                ("video_retained", "0"),
            ],
        )
        vault.commit()
        vault.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {
            "vault": str(out_db),
            "episode": episode_id,
            "events": len(events),
            "caption_windows": len(captions),
            "entities": len(entities),
            "event_blob_bytes": len(blob),
            "caption_blob_bytes": len(captions_blob),
            "vault_bytes": out_db.stat().st_size,
            "duration_s": round(duration, 3),
            "bytes_per_event": round(len(blob) / len(events), 2) if events else 0,
        }
    finally:
        vault.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    summary = compile_vault(
        args.source_db.expanduser().resolve(),
        args.out.expanduser().resolve(),
        title=args.title,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

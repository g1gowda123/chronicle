#!/usr/bin/env python3
"""Compile a Chronicle SQLite episode into a video-free VFM vault.

Input:  the graph database produced by the main Chronicle pipeline.
Output: one SQLite vault with entities, events, captions and optional evidence
        metadata. No video is read or referenced.

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

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema.sql"
WINDOW_S = 30.0


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return row is not None


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
    caption: str,
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
        "caption": caption,
        "meta": meta or {},
    }


def build_events(source: sqlite3.Connection) -> list[dict[str, Any]]:
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
    events: list[dict[str, Any]] = []

    # 1. State transitions.
    for row in _rows(
        source,
        """
        select s.entity_id, e.label, s.from_state, s.to_state, s.t_event, s.evidence
        from state_transitions s left join entities e on e.id = s.entity_id
        """,
    ):
        label = str(row["label"] or _node_label(row["entity_id"], labels))
        zone = _zone_for_entity(source, str(row["entity_id"]), zone_labels)
        caption = (
            f"At t={float(row['t_event']):.1f}s, the {label.lower()} transitioned "
            f"from {row['from_state']} to {row['to_state']}"
            + (f" in the {zone}" if zone else "")
            + "."
        )
        try:
            evidence = json.loads(row["evidence"] or "{}")
        except (TypeError, json.JSONDecodeError):
            evidence = {}
        events.append(
            _event(
                row["t_event"],
                "state",
                label,
                f"{row['from_state']}->{row['to_state']}",
                label,
                zone,
                evidence.get("confidence") if isinstance(evidence, dict) else None,
                "lifecycle",
                caption,
                {"evidence": evidence},
            )
        )

    # 2. Terminal and historical spatial anchors.
    for row in _rows(
        source,
        """
        select a.entity_id, e.label, a.anchor_id, a.relation, a.t_start, a.t_end,
               a.is_terminal, a.confidence
        from anchors a left join entities e on e.id = a.entity_id
        """,
    ):
        label = str(row["label"] or _node_label(row["entity_id"], labels))
        target = _node_label(row["anchor_id"], labels)
        zone = _zone_for_entity(source, str(row["entity_id"]), zone_labels)
        caption = (
            f"At t={float(row['t_start']):.1f}s, the {label.lower()} was "
            f"{str(row['relation']).replace('_', ' ').lower()} the {target}"
            + (f" in the {zone}" if zone else "")
            + "."
        )
        events.append(
            _event(
                row["t_start"],
                "anchor",
                label,
                str(row["relation"]),
                target,
                zone,
                row["confidence"],
                "anchor",
                caption,
                {"is_terminal": bool(row["is_terminal"]), "t_end": row["t_end"]},
            )
        )

    # 3. Relational edges (skip IN_ZONE duplicates; keep meaningful relations).
    for row in _rows(
        source,
        """
        select src, dst, relation, t_start, t_end, props from edges
        """,
    ):
        relation = str(row["relation"] or "")
        if relation.upper() in {"IN_ZONE"}:
            continue
        subject = _node_label(row["src"], labels)
        obj = _node_label(row["dst"], labels)
        t = row["t_start"]
        if t is None:
            continue
        caption = (
            f"At t={float(t):.1f}s, {subject.lower()} {relation.replace('_', ' ').lower()} "
            f"{obj.lower()}."
        )
        events.append(
            _event(
                t,
                "relation",
                subject,
                relation,
                obj,
                None,
                None,
                "graph",
                caption,
                {"t_end": row["t_end"]},
            )
        )

    events.sort(key=lambda item: (item["t"], item["kind"], item["subject"]))
    # Anchors and relation edges describe the same placement; keep one row.
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, str, str, str]] = set()
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
        deduped.append(event)
    return deduped


def _zone_for_entity(
    source: sqlite3.Connection, entity_id: str, zone_labels: dict[str, str]
) -> str | None:
    row = source.execute("select zone_id from entities where id=?", (entity_id,)).fetchone()
    if row is None or not row["zone_id"]:
        return None
    return zone_labels.get(str(row["zone_id"]), str(row["zone_id"]))


def build_captions(events: list[dict[str, Any]], window_s: float = WINDOW_S) -> list[dict[str, Any]]:
    captions: list[dict[str, Any]] = []
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        buckets[int(float(event["t"]) // window_s)].append(event)

    for bucket in sorted(buckets):
        rows = buckets[bucket]
        start = bucket * window_s
        end = start + window_s
        lines = [event["caption"] for event in rows[:12]]
        if len(rows) > 12:
            lines.append(f"... {len(rows) - 12} more events.")
        text = f"From {start:.1f}s to {end:.1f}s:\n" + "\n".join(lines)
        tags = sorted(
            {
                str(event["subject"]).lower()
                for event in rows
                if event["subject"]
            }
            | {str(event["object"]).lower() for event in rows if event["object"]}
        )
        captions.append(
            {
                "t_start": start,
                "t_end": end,
                "kind": "window",
                "text": text,
                "tags": json.dumps(tags[:20]),
                "evidence_id": None,
            }
        )

    for event in events:
        captions.append(
            {
                "t_start": float(event["t"]),
                "t_end": float(event["t"]),
                "kind": "event",
                "text": event["caption"],
                "tags": json.dumps(
                    [event["subject"], event["relation"], event["object"]]
                ),
                "evidence_id": event.get("evidence_id"),
            }
        )
    return captions


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
        events = build_events(source)
        captions = build_captions(events)
        observations = _rows(source, "select max(t) as last_t from observations")
        last_t = float(observations[0]["last_t"] or 0.0) if observations else 0.0
        duration = max(last_t, max((event["t"] for event in events), default=0.0))
    finally:
        source.close()

    vault = sqlite3.connect(str(out_db))
    vault.row_factory = sqlite3.Row
    try:
        vault.executescript(SCHEMA.read_text(encoding="utf-8"))
        now = time.time()
        episode_id = source_db.parent.name or source_db.stem
        vault.execute(
            """
            insert or replace into episodes
                (id, title, source_path, zone, duration_s, fps, video_retained, compiled_at)
            values (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                episode_id,
                title or episode_id,
                str(source_db),
                zone_labels.get(str(entities[0]["zone_id"]), "") if entities else "",
                duration,
                None,
                now,
            ),
        )
        for row in entities:
            vault.execute(
                """
                insert or replace into entities
                    (id, label, state, zone, first_seen, last_seen, attributes)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row["id"]),
                    str(row["label"]),
                    str(row["state"] or ""),
                    zone_labels.get(str(row["zone_id"]), str(row["zone_id"] or "")),
                    row["first_seen"],
                    row["last_seen"],
                    str(row["attrs"] or "{}"),
                ),
            )
        for event in events:
            vault.execute(
                """
                insert into events
                    (t, kind, subject, relation, object, zone, confidence, source,
                     caption, evidence_id, meta)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["t"],
                    event["kind"],
                    event["subject"],
                    event["relation"],
                    event["object"],
                    event["zone"],
                    event["confidence"],
                    event["source"],
                    event["caption"],
                    event.get("evidence_id"),
                    json.dumps(event["meta"], separators=(",", ":")),
                ),
            )
        for caption in captions:
            vault.execute(
                """
                insert into captions (t_start, t_end, kind, text, tags, evidence_id)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    caption["t_start"],
                    caption["t_end"],
                    caption["kind"],
                    caption["text"],
                    caption["tags"],
                    caption["evidence_id"],
                ),
            )
        vault.executemany(
            "insert or replace into meta(key, value) values (?, ?)",
            [
                ("schema_version", "vfm-0"),
                ("compiled_at", str(now)),
                ("event_count", str(len(events))),
                ("caption_count", str(len(captions))),
            ],
        )
        vault.commit()
        vault.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {
            "vault": str(out_db),
            "episode": episode_id,
            "events": len(events),
            "captions": len(captions),
            "entities": len(entities),
            "duration_s": round(duration, 3),
            "bytes": out_db.stat().st_size,
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

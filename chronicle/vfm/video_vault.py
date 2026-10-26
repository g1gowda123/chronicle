#!/usr/bin/env python3
"""Build a queryable VFM vault from a raw video + dense caption ledger.

This is the upload path: no Chronicle graph DB is required. Window captions and
their object/action tags become the event ledger; unique objects become coarse
entities. It is intentionally honest about being caption-first, not the full
perception graph.

Usage:
    python prototype/video_vault.py --video clip.mp4 --ledger ledger.json --out vault.db
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .compact import StringTable, compress_json, pack_events

PACKAGE_DIR = Path(__file__).resolve().parent
SCHEMA = PACKAGE_DIR / "schema_v2.sql"
SCHEMA_V3 = PACKAGE_DIR / "schema_v3.sql"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48] or "object"


def _window_objects(window: dict[str, Any]) -> list[str]:
    objects = []
    for value in window.get("objects") or []:
        text = str(value).strip().lower()
        if text and text not in objects:
            objects.append(text[:80])
    return objects[:16]


def _window_tags(window: dict[str, Any]) -> list[str]:
    tags = []
    for key in ("objects", "actions", "tags"):
        for value in window.get(key) or []:
            text = str(value).strip().lower()
            if text and text not in tags:
                tags.append(text[:80])
    return tags[:24]


def compile_caption_vault(
    video: Path,
    ledger_path: Path,
    out_db: Path,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    windows = ledger.get("windows") or []
    if not windows:
        raise ValueError("ledger has no windows")
    duration = float(ledger.get("duration_s") or 0.0)
    if duration <= 0:
        duration = max(float(window.get("t_end") or 0.0) for window in windows)
    if out_db.exists():
        out_db.unlink()

    table = StringTable()
    events: list[dict[str, Any]] = []
    entity_windows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for window in windows:
        start = float(window.get("t_start") or 0.0)
        end = float(window.get("t_end") or start + 30.0)
        zone = str(window.get("zone") or "").strip()
        frames = [frame for frame in window.get("frames") or [] if isinstance(frame, dict)]
        if frames:
            for frame in frames:
                frame_t = float(frame.get("t") or start)
                events.append(
                    {
                        "t": round(frame_t, 3),
                        "kind": "keyframe",
                        "subject": "",
                        "relation": "ANCHOR",
                        "object": "",
                        "zone": zone,
                        "confidence": 0.5,
                    }
                )
                for obj in _window_objects(frame):
                    entity_windows[obj].append({"window": window, "t": frame_t})
                    events.append(
                        {
                            "t": round(frame_t, 3),
                            "kind": "state",
                            "subject": obj,
                            "relation": "SEEN",
                            "object": obj,
                            "zone": zone,
                            "confidence": 0.55,
                        }
                    )
        else:
            events.append(
                {
                    "t": round(start, 3),
                    "kind": "keyframe",
                    "subject": "",
                    "relation": "ANCHOR",
                    "object": "",
                    "zone": zone,
                    "confidence": 0.5,
                }
            )
            for obj in _window_objects(window):
                entity_windows[obj].append({"window": window, "t": start})
                events.append(
                    {
                        "t": round(start, 3),
                        "kind": "state",
                        "subject": obj,
                        "relation": "SEEN",
                        "object": obj,
                        "zone": zone,
                        "confidence": 0.55,
                    }
                )

    # One coarse entity per unique object label mentioned by the captions.
    entities: list[dict[str, Any]] = []
    for label, mentions in sorted(entity_windows.items()):
        zones = [
            str(mention["window"].get("zone") or "")
            for mention in mentions
            if str(mention["window"].get("zone") or "")
        ]
        zone = Counter(zones).most_common(1)[0][0] if zones else ""
        first = min(float(mention["t"]) for mention in mentions)
        last = max(float(mention["t"]) for mention in mentions)
        entities.append(
            {
                "id": f"obj:{_slug(label)}",
                "label": label,
                "state": "SEEN",
                "zone": zone,
                "first_seen": first,
                "last_seen": last,
                "attrs": json.dumps({"source": "caption", "mentions": len(mentions)}),
            }
        )
    entities.sort(key=lambda row: (-float(row["last_seen"]), row["label"]))

    events.sort(key=lambda event: (float(event["t"]), str(event["subject"])))
    for event in events:
        for key in ("subject", "relation", "object", "zone"):
            table.add(event[key])
    for entity in entities:
        table.add(entity["label"])
        table.add(entity["state"])
        table.add(entity["zone"])
    blob = pack_events(events, table)

    captions = []
    for window in windows:
        start = float(window.get("t_start") or 0.0)
        end = float(window.get("t_end") or start + 30.0)
        caption = str(window.get("caption") or "").strip()
        tags = _window_tags(window)
        captions.append(
            {
                "t_start": round(start, 3),
                "t_end": round(end, 3),
                "text": caption,
                "tags": tags,
            }
        )
    captions_blob = compress_json(captions)
    ledger["mode"] = "caption-vault"
    ledger_blob = compress_json(ledger)

    out_db.parent.mkdir(parents=True, exist_ok=True)
    vault = sqlite3.connect(str(out_db))
    try:
        vault.executescript(SCHEMA.read_text(encoding="utf-8"))
        vault.executescript(SCHEMA_V3.read_text(encoding="utf-8"))
        now = time.time()
        episode_id = out_db.parent.name or out_db.stem
        zone_id = table.add(entities[0]["zone"] if entities else "")
        vault.execute(
            """
            insert into episodes
                (id, title, source_path, zone_id, duration_s, fps, video_retained,
                 compiled_at, compiler)
            values (?, ?, ?, ?, ?, NULL, 0, ?, ?)
            """,
            (
                episode_id,
                title or video.stem,
                video.name,
                zone_id,
                duration,
                now,
                "vfm-studio-captions",
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
            (
                str(ledger.get("model") or ""),
                float(ledger.get("window_s") or 30.0),
                ledger_blob,
                len(windows),
                now,
            ),
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
        vault.executemany(
            "insert or replace into meta(key, value) values (?, ?)",
            [
                ("schema_version", "vfm-studio-captions"),
                ("mode", "caption-vault"),
                ("compiled_at", str(now)),
                ("event_count", str(len(events))),
                ("caption_windows", str(len(captions))),
                ("event_blob_bytes", str(len(blob))),
                ("caption_blob_bytes", str(len(captions_blob))),
                ("ledger_blob_bytes", str(len(ledger_blob))),
                ("video_retained", "0"),
            ],
        )
        vault.commit()
        vault.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        vault.close()

    return {
        "vault": str(out_db),
        "mode": "caption-vault",
        "duration_s": round(duration, 3),
        "events": len(events),
        "entities": len(entities),
        "windows": len(windows),
        "vault_bytes": out_db.stat().st_size,
        "event_blob_bytes": len(blob),
        "ledger_blob_bytes": len(ledger_blob),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    summary = compile_caption_vault(
        args.video.expanduser().resolve(),
        args.ledger.expanduser().resolve(),
        args.out.expanduser().resolve(),
        title=args.title,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

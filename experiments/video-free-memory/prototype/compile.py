#!/usr/bin/env python3
"""Compile a Chronicle SQLite episode into a video-free, compressed VFM vault.

v3 layout (backwards-compatible with the v2 reader):
  * dictionary-encoded strings (uint16 ids)
  * events packed as a struct-of-arrays blob (~16 bytes/event)
  * window captions zlib-compressed JSON
  * salience coalescing driven by the source DB's physical `reason` labels
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
SCHEMA_V3 = ROOT / "schema_v3.sql"
WINDOW_S = 30.0
COMPILER = "vfm-v3"

#: Transition reasons that are visibility noise, not memory-worthy events.
NOISE_REASONS = {
    "camera_pan",
    "occluder_overlap",
    "line_of_sight_lost",
    "reacquired_without_support_geometry",
    "resting_no_surface_detector",
    "independent_motion",
    "independent_motion_relaxed",
}

#: Coalesce repeated touches of the same object closer than this.
INTERACTION_GAP_S = 6.0
#: Re-placements on the same support inside this window are not new events.
PLACEMENT_GAP_S = 8.0


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


def _transition_evidence(row: sqlite3.Row) -> dict[str, Any]:
    try:
        value = json.loads(str(row["evidence"] or "{}"))
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    try:
        box = json.loads(str(value)) if isinstance(value, str) else value
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        return tuple(float(item) for item in box)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def visibility_counts(
    source: sqlite3.Connection, entities: list[sqlite3.Row]
) -> dict[str, dict[str, Any]]:
    """Per-label track count, peak visibility and cross-label duplicates.

    A tracker can split one physical object into several tracks; the track count
    alone is therefore an upper bound. The peak number of same-label tracks seen
    in a single observation frame is a lower bound when the tracker is clean.

    Some episodes also contain the *same* box emitted under two labels (cook:
    the two plates appear as both ``bowl`` and ``plate`` with IoU 0.91–0.99).
    Those tracks are counted once under the label with more observations; the
    losing label reports ``physical`` (before transient filtering) and how many
    of its tracks were merged. All of this is computed at compile time, so the
    query engine still never touches raw observations or video.
    """

    labels = {str(row["id"]): str(row["label"]) for row in entities}
    if not labels:
        return {}
    track_totals: dict[str, int] = defaultdict(int)
    for label in labels.values():
        track_totals[label] += 1

    # Round to the observation cadence (10 fps) so rows from the same frame
    # collapse even when float timestamps differ in the last decimals.
    frames: dict[float, dict[str, tuple[float, float, float, float] | None]] = defaultdict(dict)
    samples: dict[str, list[tuple[float, tuple[float, float, float, float] | None]]] = defaultdict(list)
    for row in _rows(source, "select entity_id, t, bbox from observations"):
        entity_id = str(row["entity_id"])
        if entity_id not in labels:
            continue
        timestamp = round(float(row["t"]), 2)
        box = _bbox(row["bbox"])
        frames[timestamp][entity_id] = box
        samples[entity_id].append((timestamp, box))

    # A "strong" duplicate is the same pixels under two labels: IoU >= 0.90
    # and centres within 5 px. This is deliberately tighter than the merge
    # gate for same-label tracking, because a real second object can overlap.
    strong: dict[tuple[str, str], int] = defaultdict(int)
    for boxes in frames.values():
        ids = list(boxes)
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                if labels[left] == labels[right]:
                    continue
                left_box, right_box = boxes[left], boxes[right]
                if left_box is None or right_box is None:
                    continue
                if _iou(left_box, right_box) >= 0.90 and _center_distance(left_box, right_box) <= 5.0:
                    strong[tuple(sorted((left, right)))] += 1

    observations = {entity_id: len(rows) for entity_id, rows in samples.items()}
    # Decide which label is canonical at the label level, not the track level.
    # A single long bowl track can outlive one plate track; comparing total
    # observations per label avoids a chain where that plate yields to the bowl.
    totals: dict[str, int] = defaultdict(int)
    for entity_id, count in observations.items():
        totals[labels[entity_id]] += count
    loser_to_winners: dict[str, set[str]] = defaultdict(set)
    for left, right in strong:
        left_label, right_label = labels[left], labels[right]
        if totals[left_label] == totals[right_label]:
            continue
        winner, loser = (
            (left_label, right_label)
            if totals[left_label] > totals[right_label]
            else (right_label, left_label)
        )
        loser_to_winners[loser].add(winner)

    duplicate_of: dict[str, str] = {}
    for entity_id in labels:
        winners = loser_to_winners.get(labels[entity_id])
        if not winners:
            continue
        best: str | None = None
        best_score: tuple[int, int] | None = None
        for (left, right), hits in strong.items():
            if left != entity_id and right != entity_id:
                continue
            other = right if left == entity_id else left
            if labels[other] not in winners:
                continue
            score = (hits, observations.get(other, 0))
            if best_score is None or score > best_score:
                best, best_score = other, score
        if best is not None:
            duplicate_of[entity_id] = labels[best]

    # Peak visibility is per label: count distinct same-label ids per frame.
    per_label: dict[str, dict[float, set[str]]] = defaultdict(lambda: defaultdict(set))
    for timestamp, boxes in frames.items():
        for entity_id in boxes:
            per_label[labels[entity_id]][timestamp].add(entity_id)

    # On a tiny fixture a single observation may be the only evidence; only
    # treat one-frame tracks as transient when the episode has enough data.
    total_observations = sum(observations.values())
    last_timestamp = max((t for rows in samples.values() for t, _ in rows), default=0.0)
    transient_min = 2 if total_observations >= 10 and last_timestamp >= 5.0 else 1

    counts: dict[str, dict[str, Any]] = {}
    for label, per_time in per_label.items():
        track_ids = [entity_id for entity_id in labels if labels[entity_id] == label]
        peak = max((len(ids) for ids in per_time.values()), default=0)
        duplicates = sum(1 for entity_id in track_ids if entity_id in duplicate_of)
        transient = sum(
            1
            for entity_id in track_ids
            if observations.get(entity_id, 0) < transient_min and entity_id not in duplicate_of
        )
        physical = sum(
            1
            for entity_id in track_ids
            if entity_id not in duplicate_of and observations.get(entity_id, 0) >= transient_min
        )
        merged_into: dict[str, int] = defaultdict(int)
        for entity_id in track_ids:
            target = duplicate_of.get(entity_id)
            if target is not None:
                merged_into[target] += 1
        counts[label] = {
            "tracks": track_totals[label],
            "peak": peak,
            "physical": physical,
            "duplicates": duplicates,
            "transient": transient,
            "merged_into": dict(sorted(merged_into.items())),
        }
    return counts


def build_events(source: sqlite3.Connection, duration: float) -> list[dict[str, Any]]:
    entities = _rows(source, "select id, label, state, zone_id, first_seen, last_seen from entities")
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

    # 1. First appearance, one per tracked entity. Track fragmentation means several
    #    rows can share a label; keep them as distinct tracks and say so in the answer.
    for row in entities:
        first_seen = row["first_seen"]
        if first_seen is None:
            continue
        events.append(
            _event(
                first_seen,
                "state",
                str(row["label"]),
                "APPEARED",
                str(row["label"]),
                entity_zone.get(str(row["id"])),
                None,
                "lifecycle",
                {"track": str(row["id"])},
            )
        )

    # 2. Salient lifecycle changes. The source DB tags every transition with a
    #    physical reason; drop camera/occlusion flicker and keep contact events.
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
        last_state: str | None = None
        last_emit_t: float | None = None
        last_container = ""
        last_placement = ""
        last_placement_t = 0.0
        for row in rows_for_entity:
            to_state = str(row["to_state"])
            t = float(row["t_event"])
            evidence = _transition_evidence(row)
            reason = str(evidence.get("reason") or "")

            if reason in NOISE_REASONS:
                continue

            kind = "state"
            relation = to_state
            obj = label
            if reason == "hand_contact" and to_state == "INTERACTED":
                if label.lower() in {"person", "agent", "user"}:
                    continue
                if (
                    last_state in {"INTERACTED", "HELD"}
                    and last_emit_t is not None
                    and t - last_emit_t < INTERACTION_GAP_S
                ):
                    continue
                kind, relation, obj = "interaction", "TOUCHED", "person"
            elif reason == "lockstep_hand_motion" and to_state == "HELD":
                if last_state == "HELD" or label.lower() in {"person", "agent", "user"}:
                    continue
                kind, relation, obj = "interaction", "HELD", "person"
            elif reason == "enclosed_by_container" and to_state == "CONTAINED":
                container = _node_label(str(evidence.get("container_id") or ""), labels)
                if last_state == "CONTAINED" and container == last_container:
                    continue
                kind, relation, obj = "anchor", "PLACED_IN", container or "container"
                last_container = container
            elif reason == "resting_on_support" and to_state == "VISIBLE_STATIC":
                if last_state not in {"INTERACTED", "HELD", "CONTAINED"}:
                    continue
                support = _node_label(str(evidence.get("support_id") or ""), labels)
                if support == last_placement and t - last_placement_t < PLACEMENT_GAP_S:
                    continue
                kind, relation, obj = "anchor", "PLACED_ON", support or "surface"
                last_placement = support
                last_placement_t = t
            elif to_state == "LOST":
                kind, relation, obj = "state", "LOST", label
            else:
                continue

            events.append(
                _event(
                    t,
                    kind,
                    label,
                    relation,
                    obj,
                    zone,
                    None,
                    "lifecycle",
                    {
                        "from": row["from_state"],
                        "to": to_state,
                        "reason": reason,
                        "track": entity_id,
                    },
                )
            )
            last_state = to_state
            last_emit_t = t

    # 3. One anchor per (entity, relation, target): terminal/preferred latest.
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

    # 4. Non-spatial graph relations (IN_ZONE is implied by the entity/zone table).
    #    ACTION edges duplicate hand-contact lifecycle events, so they are skipped.
    for row in _rows(source, "select src, dst, relation, t_start from edges"):
        relation = str(row["relation"] or "")
        if relation.upper() in {"IN_ZONE", "RESTING_ON", "INSIDE", "ON", "CONTAINED_IN", "ACTION"}:
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

    # 5. Periodic scene anchors so quiet periods are not invisible.
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
    if kind == "interaction":
        text = f"At t={t:.1f}s, the {subject} was {relation}"
    elif kind == "anchor":
        if relation == "placed in":
            text = f"At t={t:.1f}s, the {subject} was placed in the {obj}"
        elif relation == "placed on":
            text = f"At t={t:.1f}s, the {subject} was placed on the {obj}"
        else:
            text = f"At t={t:.1f}s, the {subject} was {relation} the {obj}"
    elif kind == "state":
        if relation == "appeared":
            text = f"At t={t:.1f}s, the {subject} appeared"
        elif relation == "lost":
            text = f"At t={t:.1f}s, the {subject} was lost from view"
        else:
            text = f"At t={t:.1f}s, the {subject} was {relation}"
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
        counts = visibility_counts(source, entities)
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
        vault.executescript(SCHEMA_V3.read_text(encoding="utf-8"))
        now = time.time()
        episode_id = source_db.parent.name or source_db.stem
        zone_id = table.add(zone_labels.get(str(entities[0]["zone_id"]), "")) if entities else 0
        vault.execute(
            """
            insert or replace into episodes
                (id, title, source_path, zone_id, duration_s, fps, video_retained, compiled_at, compiler)
            values (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                episode_id,
                title or episode_id,
                str(source_db),
                zone_id,
                duration,
                None,
                now,
                COMPILER,
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
                ("schema_version", COMPILER),
                ("event_policy", "salience-coalesce-v1"),
                ("compiled_at", str(now)),
                ("event_count", str(len(events))),
                ("caption_windows", str(len(captions))),
                ("event_blob_bytes", str(len(blob))),
                ("caption_blob_bytes", str(len(captions_blob))),
                ("counts", json.dumps(counts, sort_keys=True, separators=(",", ":"))),
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
            "counts": counts,
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

#!/usr/bin/env python3
"""Query a compressed VFM vault with no video and no language model.

Reads the packed event blob, the string dictionary and compressed captions.
Supports both the v2 vault and the older row-per-event vault as a fallback.

Usage:
    python prototype/query.py --db vault.db "where is the cup?"
    python prototype/query.py --db vault.db --json "what happened around 20s?"
    python prototype/query.py --db vault.db --stats
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compact import StringTable, decompress_json, unpack_events

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"(?:at|around|near|t\s*=)?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\b")


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "vault"
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class EventStore:
    events: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    captions: list[dict[str, Any]]
    con: sqlite3.Connection

    @property
    def duration(self) -> float:
        return max((float(event["t"]) for event in self.events), default=0.0)


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(str(text).lower()) if token}


def load_store(con: sqlite3.Connection) -> EventStore:
    con.row_factory = sqlite3.Row
    table = StringTable()
    try:
        table.load((int(row["id"]), str(row["value"])) for row in con.execute("select id, value from strings"))
    except sqlite3.Error:
        pass

    events: list[dict[str, Any]] = []
    try:
        row = con.execute("select blob from event_pack where id=1").fetchone()
        if row is not None:
            events = unpack_events(bytes(row["blob"]), table)
    except sqlite3.Error:
        events = []
    if not events:
        # Legacy row-per-event vault.
        try:
            for row in con.execute("select * from events order by t"):
                events.append(
                    {
                        "t": float(row["t"]),
                        "kind": str(row["kind"]),
                        "subject": str(row["subject"] or ""),
                        "relation": str(row["relation"] or ""),
                        "object": str(row["object"] or ""),
                        "zone": str(row["zone"] or ""),
                        "confidence": float(row["confidence"] or 0.0),
                        "evidence_id": row["evidence_id"],
                    }
                )
        except sqlite3.Error:
            events = []

    captions: list[dict[str, Any]] = []
    try:
        row = con.execute("select blob from captions_blob where id=1").fetchone()
        if row is not None:
            captions = decompress_json(bytes(row["blob"])) or []
    except sqlite3.Error:
        captions = []
    if not captions:
        try:
            captions = [
                {
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "text": str(row["text"]),
                    "tags": json.loads(row["tags"] or "[]"),
                }
                for row in con.execute("select * from captions order by t_start")
            ]
        except sqlite3.Error:
            captions = []

    entities: list[dict[str, Any]] = []
    try:
        rows = list(con.execute("select * from entities"))
        for row in rows:
            entities.append(
                {
                    "id": str(row["id"]),
                    "label": table.get(row["label_id"]) if "label_id" in row.keys() else str(row["label"]),
                    "state": table.get(row["state_id"]) if "state_id" in row.keys() else str(row["state"] or ""),
                    "zone": table.get(row["zone_id"]) if "zone_id" in row.keys() else str(row["zone"] or ""),
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                }
            )
    except sqlite3.Error:
        entities = []

    return EventStore(events=events, entities=entities, captions=captions, con=con)


def _citation(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "t": round(float(event["t"]), 3),
        "kind": event["kind"],
        "subject": event["subject"],
        "relation": event["relation"],
        "object": event["object"],
        "zone": event["zone"],
        "caption": caption_for(event),
    }


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


def _match_entity(store: EventStore, query: str) -> dict[str, Any] | None:
    tokens = _tokens(query)
    best: tuple[float, dict[str, Any]] | None = None
    for entity in store.entities:
        label = str(entity["label"]).lower()
        label_tokens = _tokens(label)
        score = 0.0
        if label and label in query.lower():
            score += 3.0
        score += len(tokens & label_tokens) * 2.0
        if not score:
            singular = label[:-1] if label.endswith("s") and len(label) > 3 else label
            if singular in tokens or any(token[:-1] == label for token in tokens if token.endswith("s")):
                score += 1.5
        if score <= 0:
            continue
        if entity["state"] in {"VISIBLE_STATIC", "HELD", "CONTAINED"}:
            score += 0.1
        if best is None or score > best[0]:
            best = (score, entity)
    return best[1] if best else None


def _events_for(store: EventStore, label: str, *, before: float | None = None) -> list[dict[str, Any]]:
    rows = [event for event in store.events if str(event["subject"]).lower() == label.lower()]
    if before is not None:
        rows = [event for event in rows if float(event["t"]) <= before]
    return sorted(rows, key=lambda event: float(event["t"]))


def _nearest_evidence(con: sqlite3.Connection, t: float) -> dict[str, Any] | None:
    try:
        row = con.execute(
            "select * from evidence order by abs(t - ?) limit 1", (float(t),)
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {
        "id": row["id"],
        "t": round(float(row["t"]), 3),
        "kind": row["kind"],
        "label": row["label"],
        "rel_path": row["rel_path"],
        "bytes": row["bytes"],
    }


def answer_where(store: EventStore, query: str) -> Answer:
    entity = _match_entity(store, query)
    if entity is None:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    label = str(entity["label"])
    history = _events_for(store, label)
    anchors = [event for event in history if event["kind"] == "anchor"]
    latest_anchor = anchors[-1] if anchors else None
    latest = history[-1] if history else None
    state = str(entity["state"] or "unknown").replace("_", " ").lower()
    zone = str(entity["zone"] or "")
    parts = [f"The {label.lower()} is currently {state}" + (f" in the {zone}" if zone else "") + "."]
    if latest_anchor is not None:
        parts.append(
            f"It was last {str(latest_anchor['relation']).replace('_', ' ').lower()} "
            f"the {str(latest_anchor['object']).lower()} at [t={float(latest_anchor['t']):.1f}s]."
        )
    elif latest is not None:
        parts.append(caption_for(latest))
    citations = [_citation(latest_anchor or latest)] if (latest_anchor or latest) else []
    evidence = _nearest_evidence(store.con, float((latest_anchor or latest or {"t": 0})["t"]))
    return Answer(
        query,
        " ".join(parts),
        citations=citations,
        evidence=[evidence] if evidence else [],
        confidence=0.85 if (latest_anchor or latest) else 0.6,
        method="vault-where",
    )


def answer_when(store: EventStore, query: str) -> Answer:
    entity = _match_entity(store, query)
    if entity is None:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    label = str(entity["label"])
    rows = _events_for(store, label)
    if not rows:
        return Answer(query, f"No events recorded for the {label.lower()}.", method="no-match")
    rows = rows[-8:]
    lines = [f"Here is the recorded history for the {label.lower()}:"]
    for event in rows:
        lines.append(f"- {caption_for(event)}")
    evidence = _nearest_evidence(store.con, float(rows[-1]["t"]))
    return Answer(
        query,
        "\n".join(lines),
        citations=[_citation(event) for event in rows],
        evidence=[evidence] if evidence else [],
        confidence=0.8,
        method="vault-when",
    )


def answer_before(store: EventStore, query: str, t: float) -> Answer:
    rows = [event for event in store.events if float(event["t"]) <= t][-20:]
    if not rows:
        return Answer(query, f"No events recorded before t={t:.1f}s.", method="no-match")
    lines = [f"Before t={t:.1f}s:"]
    lines.extend(f"- {caption_for(event)}" for event in rows)
    evidence = _nearest_evidence(store.con, float(rows[-1]["t"]))
    return Answer(
        query,
        "\n".join(lines),
        citations=[_citation(event) for event in rows],
        evidence=[evidence] if evidence else [],
        confidence=0.7,
        method="vault-before",
    )


def answer_window(store: EventStore, query: str, t: float) -> Answer:
    rows = [event for event in store.events if t - 10.0 <= float(event["t"]) <= t + 10.0]
    rows.sort(key=lambda event: float(event["t"]))
    if not rows:
        return Answer(query, f"No events recorded around t={t:.1f}s.", method="no-match")
    lines = [f"Around t={t:.1f}s:"]
    lines.extend(f"- {caption_for(event)}" for event in rows[:20])
    evidence = _nearest_evidence(store.con, t)
    return Answer(
        query,
        "\n".join(lines),
        citations=[_citation(event) for event in rows[:20]],
        evidence=[evidence] if evidence else [],
        confidence=0.75,
        method="vault-window",
    )


def answer_inventory(store: EventStore, query: str) -> Answer:
    counts: dict[str, int] = {}
    for entity in store.entities:
        label = str(entity["label"])
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return Answer(query, "The memory has no entities yet.", method="no-match")
    total = sum(counts.values())
    parts = [f"{count} {label}{'s' if count > 1 else ''}" for label, count in sorted(counts.items())]
    return Answer(
        query,
        f"The memory tracked {total} entities: " + ", ".join(parts) + ".",
        confidence=0.9,
        method="vault-inventory",
    )


def answer_keyword(store: EventStore, query: str) -> Answer:
    tokens = sorted(
        _tokens(query)
        - {"the", "a", "an", "is", "was", "did", "what", "where", "when", "how", "many"}
    )
    if not tokens:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    scored: list[tuple[int, dict[str, Any]]] = []
    for caption in store.captions:
        text = str(caption["text"]).lower()
        hits = sum(1 for token in tokens if token in text)
        if hits:
            scored.append((hits, caption))
    if not scored:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    scored.sort(key=lambda item: (item[0], float(item[1]["t_start"])), reverse=True)
    rows = scored[:5]
    lines = ["Here is what the memory contains:"]
    for _, caption in rows:
        lines.append(f"- {str(caption['text']).splitlines()[0]}")
    evidence = _nearest_evidence(store.con, float(rows[0][1]["t_start"]))
    return Answer(
        query,
        "\n".join(lines),
        citations=[
            {
                "t": round(float(caption["t_start"]), 3),
                "kind": "caption",
                "caption": str(caption["text"])[:500],
            }
            for _, caption in rows
        ],
        evidence=[evidence] if evidence else [],
        confidence=0.55,
        method="vault-keyword",
    )


def _time_hint(query: str) -> float | None:
    lower = query.lower()
    if not re.search(r"\b(s|sec|secs|seconds?)\b|t\s*=", lower):
        return None
    match = _NUMBER.search(lower)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def answer(store: EventStore, query: str) -> Answer:
    started = time.perf_counter()
    lower = query.lower().strip()
    if not lower:
        return Answer(query, "Ask a question about the episode.", method="empty")
    entity = _match_entity(store, query)
    if re.search(r"\b(what did i see|what did i observe|inventory|tracked|overview)\b", lower):
        result = answer_inventory(store, query)
    elif re.search(r"\bnot in (this )?memory\b|\bunknown\b|\bno such\b", lower):
        result = Answer(query, "I don't have that in my memory.", method="no-match")
    elif "before" in lower and _time_hint(query) is not None:
        result = answer_before(store, query, float(_time_hint(query) or 0.0))
    elif "before" in lower and entity is not None:
        rows = _events_for(store, str(entity["label"]))
        anchors = [event for event in rows if event["kind"] == "anchor"]
        result = answer_before(store, query, float(anchors[-1]["t"])) if anchors else answer_when(store, query)
    elif re.search(r"\b(doing with|what was .* doing|what did .* do)\b", lower):
        result = answer_when(store, query)
    elif re.search(r"\b(what happened|around|at)\b", lower) and _time_hint(query) is not None:
        result = answer_window(store, query, float(_time_hint(query) or 0.0))
    elif lower.startswith("where") or " where " in lower:
        result = answer_where(store, query)
    elif lower.startswith("when") or " when " in lower:
        result = answer_when(store, query)
    elif lower.startswith("what did") and " do" in lower:
        result = answer_when(store, query)
    else:
        result = answer_keyword(store, query)
    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result


def stats(store: EventStore, vault: Path) -> dict[str, Any]:
    payload = {
        "vault": str(vault),
        "bytes": vault.stat().st_size if vault.is_file() else 0,
        "entities": len(store.entities),
        "events": len(store.events),
        "captions": len(store.captions),
        "duration_s": round(store.duration, 3),
    }
    try:
        row = store.con.execute("select value from meta where key='event_blob_bytes'").fetchone()
        if row:
            payload["event_blob_bytes"] = int(row["value"])
        row = store.con.execute("select value from meta where key='caption_blob_bytes'").fetchone()
        if row:
            payload["caption_blob_bytes"] = int(row["value"])
        payload["evidence"] = int(
            store.con.execute("select count(*) from evidence").fetchone()[0]
        )
    except sqlite3.Error:
        pass
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args(argv)
    if not args.db.is_file():
        raise SystemExit(f"vault not found: {args.db}")
    con = sqlite3.connect(str(args.db))
    try:
        store = load_store(con)
        if args.stats:
            print(json.dumps(stats(store, args.db), indent=2))
            return 0
        question = args.question or input("question> ").strip()
        result = answer(store, question)
    finally:
        con.close()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.answer)
        if result.citations:
            print("\nEvidence:")
            for cite in result.citations[:6]:
                print(f"  [t={cite.get('t')}s] {str(cite.get('caption', cite.get('kind', '')))[:160]}")
        print(f"\nmethod={result.method} · {result.elapsed_ms:.1f} ms · confidence={result.confidence:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

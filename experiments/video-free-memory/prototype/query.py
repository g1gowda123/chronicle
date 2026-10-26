#!/usr/bin/env python3
"""Query a compressed VFM vault with no video and no language model.

Reads the packed event blob, the string dictionary, compressed window captions
and (when present) the dense VLM caption ledger. Supports both the v3 vault and
the older row-per-event vault as a fallback.

Query answers are cached inside the vault in `qa_cache`, keyed by a canonical
(intent, entity, time, action) tuple so paraphrases hit the same entry.

Usage:
    python prototype/query.py --db vault.db "where is the cup?"
    python prototype/query.py --db vault.db --json "what happened around 20s?"
    python prototype/query.py --db vault.db --stats
    python prototype/query.py --db vault.db --no-cache "when did I touch the cup?"
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compact import StringTable, decompress_json, unpack_events

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(
    r"(?:at|around|near|after|before|t\s*=)?\s*(\d+(?:\.\d+)?)\s*"
    r"(?:s|sec|secs|seconds?|m|min|mins|minutes?)?\b"
)
_INVENTORY = re.compile(
    r"\b(what did i see|what did i observe|inventory|tracked|overview|"
    r"what objects|what is in (this )?memory)\b"
)
_NEGATIVE = re.compile(r"\bnot in (this )?memory\b|\bunknown\b|\bno such\b")
_HOLDING = re.compile(
    r"\b(?:holding|carrying|holds|carries|has)\s+(?:a|an|the|his|her|their|my)?\s*"
    r"([a-z0-9][a-z0-9' \-]{1,60})",
    re.IGNORECASE,
)
_ACTION_WORDS = {
    "touch": {"touch", "touched", "touching", "pick", "picked", "grab", "grabbed",
              "hold", "held", "handle", "handled", "use", "used", "grasp", "grasped"},
    "place": {"place", "placed", "put", "putting", "set", "drop", "dropped",
              "insert", "inserted", "return", "returned"},
    "see": {"see", "saw", "seen", "appear", "appeared", "spot", "spotted"},
    "lost": {"lose", "lost", "disappear", "disappeared"},
}
_QUERY_SYNONYMS = {
    "mug": "cup",
    "glasses": "cup",
    "glass": "cup",
    "dish": "plate",
    "dishes": "plate",
    "utensil": "knife",
    "utensils": "knife",
    "silverware": "knife",
    "knives": "knife",
    "knifes": "knife",
    "people": "person",
    "persons": "person",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "friend": "person",
    "user": "person",
    "someone": "person",
    "rucksack": "backpack",
    "bag": "backpack",
}
_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "am", "did", "do", "does",
    "what", "where", "when", "which", "who", "how", "many", "much", "my",
    "i", "me", "in", "on", "at", "to", "of", "for", "with", "and", "or",
    "this", "that", "it", "there", "be", "been", "being", "have", "has",
    "had", "can", "could", "would", "should", "about", "around", "last",
    "first", "before", "after", "then", "tell", "show", "memory",
}
#: Bump when answer wording or canonicalisation changes, so stale answers miss.
CACHE_VERSION = "v3.2"


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "vault"
    elapsed_ms: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "cached": self.cached,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class EventStore:
    events: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    captions: list[dict[str, Any]]
    con: sqlite3.Connection
    evidence: list[dict[str, Any]] = field(default_factory=list)
    #: Per-label counts from compile-time observations. ``tracks`` is the raw
    #: tracker count (upper bound), ``peak`` is the most same-label tracks
    #: visible in one frame, and ``physical`` removes cross-label duplicate
    #: detections and one-frame transients (a conservative lower estimate).
    counts: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    by_subject: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    times: list[float] = field(default_factory=list)
    ledger_text: str = ""
    duration_override: float = 0.0

    @property
    def duration(self) -> float:
        events = max((float(event["t"]) for event in self.events), default=0.0)
        return max(events, float(self.duration_override or 0.0))

    @property
    def ledger_windows(self) -> list[dict[str, Any]]:
        windows = self.ledger.get("windows") if isinstance(self.ledger, dict) else None
        return windows if isinstance(windows, list) else []


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(str(text).lower()) if token}


def _normal_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _tokens(text):
        token = _QUERY_SYNONYMS.get(token, token)
        tokens.add(token)
        if token.endswith("s") and len(token) > 3:
            tokens.add(token.rstrip("s"))
    return tokens


def _singular(token: str) -> str:
    token = token.lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token.rstrip("s")
    return token


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

    ledger: dict[str, Any] = {}
    try:
        row = con.execute("select blob from caption_ledger where id=1").fetchone()
        if row is not None:
            ledger = decompress_json(bytes(row["blob"])) or {}
    except sqlite3.Error:
        ledger = {}

    entities: list[dict[str, Any]] = []
    try:
        rows = list(con.execute("select * from entities"))
        for row in rows:
            keys = row.keys()
            entities.append(
                {
                    "id": str(row["id"]),
                    "label": table.get(row["label_id"]) if "label_id" in keys else str(row["label"]),
                    "state": table.get(row["state_id"]) if "state_id" in keys else str(row["state"] or ""),
                    "zone": table.get(row["zone_id"]) if "zone_id" in keys else str(row["zone"] or ""),
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                }
            )
    except sqlite3.Error:
        entities = []

    evidence: list[dict[str, Any]] = []
    try:
        for row in con.execute("select * from evidence order by t"):
            evidence.append(
                {
                    "id": str(row["id"]),
                    "t": round(float(row["t"]), 3),
                    "kind": str(row["kind"]),
                    "label": str(row["label"] or ""),
                    "rel_path": str(row["rel_path"]),
                    "bytes": int(row["bytes"] or 0),
                }
            )
    except sqlite3.Error:
        evidence = []

    duration_override = 0.0
    try:
        row = con.execute("select duration_s from episodes limit 1").fetchone()
        if row is not None and row[0] is not None:
            duration_override = float(row[0])
    except sqlite3.Error:
        pass

    counts: dict[str, dict[str, Any]] = {}
    try:
        row = con.execute("select value from meta where key='counts'").fetchone()
        if row is not None:
            raw = json.loads(str(row[0]) or "{}")
            if isinstance(raw, dict):
                for label, value in raw.items():
                    if not isinstance(value, dict):
                        continue
                    clean: dict[str, Any] = {
                        "tracks": int(value.get("tracks") or 0),
                        "peak": int(value.get("peak") or 0),
                    }
                    for key in ("physical", "duplicates", "transient"):
                        if key in value:
                            clean[key] = int(value.get(key) or 0)
                    merged = value.get("merged_into")
                    if isinstance(merged, dict):
                        clean["merged_into"] = {
                            str(name): int(number)
                            for name, number in merged.items()
                            if int(number or 0)
                        }
                    if clean["tracks"]:
                        counts[str(label)] = clean
    except (sqlite3.Error, TypeError, ValueError):
        counts = {}

    events.sort(key=lambda event: float(event["t"]))
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_subject.setdefault(str(event["subject"]).lower(), []).append(event)
    times = [float(event["t"]) for event in events]
    ledger_text_parts: list[str] = []
    for window in ledger.get("windows", []) if isinstance(ledger, dict) else []:
        if not isinstance(window, dict):
            continue
        ledger_text_parts.append(str(window.get("caption") or window.get("text") or ""))
        for key in ("objects", "actions", "tags"):
            ledger_text_parts.append(" ".join(str(item) for item in window.get(key) or []))
        for frame in window.get("frames") or []:
            if isinstance(frame, dict):
                ledger_text_parts.append(str(frame.get("caption") or ""))

    return EventStore(
        events=events,
        entities=entities,
        captions=captions,
        con=con,
        evidence=evidence,
        counts=counts,
        ledger=ledger,
        by_subject=by_subject,
        times=times,
        ledger_text=" ".join(ledger_text_parts).lower(),
        duration_override=duration_override,
    )


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


def _match_entity(store: EventStore, query: str) -> dict[str, Any] | None:
    tokens = _normal_tokens(query)
    lower = query.lower()
    best: tuple[float, float, dict[str, Any]] | None = None
    for entity in store.entities:
        label = str(entity["label"]).lower()
        label_tokens = _normal_tokens(label)
        score = 0.0
        if label and re.search(rf"\b{re.escape(label)}\b", lower):
            score += 3.0
        score += len(tokens & label_tokens) * 2.0
        if not score and label:
            singular = _singular(label)
            if singular in tokens or any(_singular(token) == singular for token in tokens):
                score += 1.5
        if score <= 0:
            continue
        if str(entity["state"]) in {"VISIBLE_STATIC", "HELD", "CONTAINED"}:
            score += 0.1
        last_seen = float(entity.get("last_seen") or 0.0)
        rank = (score, last_seen)
        if best is None or rank > (best[0], best[1]):
            best = (score, last_seen, entity)
    return best[2] if best else None


def _events_for(store: EventStore, label: str, *, before: float | None = None) -> list[dict[str, Any]]:
    wanted = label.lower()
    rows = store.by_subject.get(wanted, [])
    if before is not None:
        rows = [event for event in rows if float(event["t"]) <= before]
    return rows


def _nearest_evidence(store: EventStore, t: float) -> dict[str, Any] | None:
    if not store.evidence:
        return None
    row = min(store.evidence, key=lambda item: abs(float(item["t"]) - float(t)))
    result = dict(row)
    result["distance_s"] = round(abs(float(row["t"]) - float(t)), 3)
    return result


def _evidence_list(store: EventStore, t: float | None) -> list[dict[str, Any]]:
    if t is None:
        return []
    item = _nearest_evidence(store, t)
    return [item] if item else []


def _latest(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(rows, key=lambda event: float(event["t"])) if rows else None


def _action(query: str) -> str:
    tokens = _tokens(query)
    for action, words in _ACTION_WORDS.items():
        if tokens & words:
            return action
    return ""


def _matches_action(event: dict[str, Any], action: str) -> bool:
    relation = str(event["relation"]).upper()
    kind = str(event["kind"])
    if not action:
        return True
    if action == "touch":
        return kind == "interaction" or relation in {"TOUCHED", "HELD"}
    if action == "place":
        return kind == "anchor" and relation in {"PLACED_ON", "PLACED_IN", "RESTING_ON", "INSIDE"}
    if action == "see":
        return relation in {"APPEARED", "VISIBLE_STATIC"} or kind == "relation"
    if action == "lost":
        return relation == "LOST"
    return True


def _window_caption(store: EventStore, t: float) -> dict[str, Any] | None:
    windows = store.ledger_windows
    if not windows:
        return None
    covering = [
        window
        for window in windows
        if float(window.get("t_start", 0.0)) <= t <= float(window.get("t_end", 0.0))
    ]
    if covering:
        return covering[0]
    return min(
        windows,
        key=lambda window: abs(
            (float(window.get("t_start", 0.0)) + float(window.get("t_end", 0.0))) / 2.0 - t
        ),
    )


def answer_where(store: EventStore, query: str, entity: dict[str, Any]) -> Answer:
    label = str(entity["label"])
    history = _events_for(store, label)
    latest = _latest(history)
    placements = [
        event
        for event in history
        if str(event["kind"]) == "anchor"
        and str(event["relation"]).upper() in {"PLACED_ON", "PLACED_IN", "RESTING_ON", "INSIDE"}
    ]
    last_place = _latest(placements)
    zone = str(entity["zone"] or "")
    last_seen = entity.get("last_seen")

    if latest is None:
        return Answer(query, f"I don't have a location for the {label.lower()}.", method="no-match")

    parts: list[str] = []
    latest_is_held = str(latest["kind"]) == "interaction" and str(latest["relation"]).upper() == "HELD"
    if latest_is_held:
        parts.append(
            f"The {label.lower()} was last held at t={float(latest['t']):.1f}s"
            + (f" in the {zone}" if zone else "")
            + "."
        )
    else:
        last_seen = entity.get("last_seen")
        seen_t = float(last_seen) if last_seen is not None else float(latest["t"])
        seen_t = max(seen_t, float(latest["t"]))
        parts.append(
            f"The {label.lower()} was last seen at t={seen_t:.1f}s"
            + (f" in the {zone}" if zone else "")
            + "."
        )
    if last_place is not None:
        parts.append(
            f"Last recorded placement: {caption_for(last_place).rstrip('.')}."
        )
    elif latest is not None:
        parts.append(f"Last recorded event: {caption_for(latest).rstrip('.')}.")

    citations = [
        cite
        for cite in (
            _citation(last_place) if last_place is not None else None,
            _citation(latest),
        )
        if cite
    ]
    return Answer(
        query,
        " ".join(parts),
        citations=citations,
        evidence=_evidence_list(store, float(latest["t"])),
        confidence=0.82,
        method="vault-where",
    )


def answer_when(store: EventStore, query: str, entity: dict[str, Any], action: str) -> Answer:
    label = str(entity["label"])
    rows = _events_for(store, label)
    if not rows:
        return Answer(query, f"No events recorded for the {label.lower()}.", method="no-match")
    filtered = [event for event in rows if _matches_action(event, action)]
    if not filtered:
        what = action or "events"
        return Answer(
            query,
            f"No {what} events are recorded for the {label.lower()}.",
            method="no-match",
        )
    first = "first" in query.lower()
    target = filtered[0] if first else filtered[-1]
    prefix = "first" if first else "last"
    action_text = {
        "touch": "touched or held",
        "place": "placed",
        "see": "seen",
        "lost": "lost from view",
    }.get(action, "recorded")
    summary = (
        f"The {label.lower()} was {prefix} {action_text} at "
        f"t={float(target['t']):.1f}s"
        + (f" in the {target['zone']}" if target.get("zone") else "")
        + "."
    )
    citations = [_citation(target)]
    context = [event for event in filtered if event is not target][-4:]
    if context and action:
        citations = [_citation(event) for event in [*context, target]]
    return Answer(
        query,
        summary + " " + caption_for(target),
        citations=citations,
        evidence=_evidence_list(store, float(target["t"])),
        confidence=0.8,
        method="vault-when",
    )


def answer_history(store: EventStore, query: str, entity: dict[str, Any]) -> Answer:
    label = str(entity["label"])
    rows = _events_for(store, label)[-6:]
    if not rows:
        return Answer(query, f"No events recorded for the {label.lower()}.", method="no-match")
    lines = [f"Here is the recorded history for the {label.lower()}:"]
    lines.extend(f"- {caption_for(event)}" for event in rows)
    evidence = _nearest_evidence(store, float(rows[-1]["t"]))
    return Answer(
        query,
        "\n".join(lines),
        citations=[_citation(event) for event in rows],
        evidence=[evidence] if evidence else [],
        confidence=0.75,
        method="vault-history",
    )


def _anchor_time(
    store: EventStore, entity: dict[str, Any] | None, action: str, t: float | None
) -> tuple[float | None, dict[str, Any] | None]:
    if t is not None:
        return t, None
    if entity is None:
        return None, None
    rows = _events_for(store, str(entity["label"]))
    filtered = [event for event in rows if _matches_action(event, action)]
    target = _latest(filtered or rows)
    return (float(target["t"]) if target else None), target


def answer_before(
    store: EventStore,
    query: str,
    entity: dict[str, Any] | None,
    action: str,
    t: float | None,
) -> Answer:
    anchor_t, anchor_event = _anchor_time(store, entity, action, t)
    if anchor_t is None:
        return Answer(query, "I need a time or an object to answer that.", method="no-match")
    lo = bisect.bisect_left(store.times, anchor_t - 15.0)
    hi = bisect.bisect_left(store.times, anchor_t)
    rows = store.events[lo:hi]
    if not rows:
        return Answer(query, f"No events recorded before t={anchor_t:.1f}s.", method="no-match")
    if anchor_event is not None:
        t_text = f"{float(anchor_event['t']):.1f}s"
        description = caption_for(anchor_event)
        description = description.removeprefix(f"At t={float(anchor_event['t']):.1f}s, ").rstrip(".")
        head = f"Before {description} at t={t_text}:"
    else:
        head = f"Before t={anchor_t:.1f}s:"
    lines = [head]
    lines.extend(f"- {caption_for(event)}" for event in rows)
    evidence = _nearest_evidence(store, float(rows[-1]["t"]))
    citations = ([_citation(anchor_event)] if anchor_event is not None else [])
    citations.extend(_citation(event) for event in rows)
    return Answer(
        query,
        "\n".join(lines),
        citations=citations,
        evidence=[evidence] if evidence else [],
        confidence=0.72,
        method="vault-before",
    )


def answer_after(
    store: EventStore,
    query: str,
    entity: dict[str, Any] | None,
    action: str,
    t: float | None,
) -> Answer:
    anchor_t, anchor_event = _anchor_time(store, entity, action, t)
    if anchor_t is None:
        return Answer(query, "I need a time or an object to answer that.", method="no-match")
    lo = bisect.bisect_right(store.times, anchor_t)
    hi = bisect.bisect_right(store.times, anchor_t + 15.0)
    rows = store.events[lo:hi]
    if not rows:
        return Answer(query, f"No events recorded after t={anchor_t:.1f}s.", method="no-match")
    if anchor_event is not None:
        description = caption_for(anchor_event)
        description = description.removeprefix(f"At t={float(anchor_event['t']):.1f}s, ").rstrip(".")
        head = f"After {description} at t={float(anchor_event['t']):.1f}s:"
    else:
        head = f"After t={anchor_t:.1f}s:"
    lines = [head]
    lines.extend(f"- {caption_for(event)}" for event in rows)
    evidence = _nearest_evidence(store, float(rows[0]["t"]))
    citations = ([_citation(anchor_event)] if anchor_event is not None else [])
    citations.extend(_citation(event) for event in rows)
    return Answer(
        query,
        "\n".join(lines),
        citations=citations,
        evidence=[evidence] if evidence else [],
        confidence=0.72,
        method="vault-after",
    )


def answer_window(store: EventStore, query: str, t: float) -> Answer:
    """Answer around a time, separating a tight event window from a caption window.

    A 30 s caption always starts before ``t``; listing those events as if they
    happened "around t" is misleading. Show the containing caption explicitly,
    list events within 5 s first, and label farther events as nearest/fallback.
    """

    tight = 5.0
    lo = bisect.bisect_left(store.times, t - tight)
    hi = bisect.bisect_right(store.times, t + tight)
    near = store.events[lo:hi]
    radius = tight
    if not near:
        radius = 20.0
        lo = bisect.bisect_left(store.times, t - radius)
        hi = bisect.bisect_right(store.times, t + radius)
    rows = store.events[lo:hi]
    caption = _window_caption(store, t)
    if not rows and caption is None:
        return Answer(query, f"No events recorded around t={t:.1f}s.", method="no-match")

    lines = [f"Around t={t:.1f}s:"]
    if caption is not None:
        start = float(caption.get("t_start") or 0.0)
        end = float(caption.get("t_end") or start + 30.0)
        lines.append(
            f"Caption [{start:.1f}-{end:.1f}s]: "
            f"{caption.get('caption') or caption.get('text') or ''}"
        )
    if near:
        lines.extend(f"- {caption_for(event)}" for event in near[:12])
    elif rows:
        lines.append(f"No state changes within {tight:g}s; nearest events:")
        lines.extend(f"- {caption_for(event)}" for event in rows[:12])

    evidence = _nearest_evidence(store, t)
    citations = [_citation(event) for event in (near or rows)[:12]]
    if caption is not None:
        citations.insert(
            0,
            {
                "t": round(float(caption.get("t_start", 0.0)), 3),
                "kind": "caption",
                "caption": str(caption.get("caption") or caption.get("text") or "")[:500],
            },
        )
    return Answer(
        query,
        "\n".join(lines),
        citations=citations,
        evidence=[evidence] if evidence else [],
        confidence=0.8 if near else (0.7 if caption else 0.6),
        method="vault-window",
    )


def answer_count(store: EventStore, query: str, entity: dict[str, Any] | None) -> Answer:
    if entity is None:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    label = str(entity["label"])
    count = sum(1 for row in store.entities if str(row["label"]).lower() == label.lower())
    if count == 0:
        return Answer(query, f"No {label.lower()} is in the memory.", method="no-match")
    history = _events_for(store, label)
    latest = _latest(history)
    if isinstance(store.ledger, dict) and store.ledger.get("mode") == "caption-vault":
        mentions = [
            window
            for window in store.ledger_windows
            if label.lower()
            in {str(value).lower() for value in (window.get("objects") or [])}
        ]
        windows = len(mentions)
        answer_text = (
            f"The captions mention the {label.lower()} in {windows} window"
            f"{'s' if windows != 1 else ''}. This is caption coverage, not a physical "
            "object count."
        )
        return Answer(
            query,
            answer_text,
            citations=[
                {
                    "t": round(float(window.get("t_start") or 0.0), 3),
                    "kind": "caption",
                    "caption": str(window.get("caption") or "")[:500],
                }
                for window in mentions[:4]
            ],
            evidence=_evidence_list(store, float(latest["t"])) if latest else [],
            confidence=0.6,
            method="vault-count-captions",
        )
    plural = "" if count == 1 else "s"
    summary = store.counts.get(label.lower()) or {}
    peak = int(summary.get("peak") or 0)
    physical = summary.get("physical")
    duplicates = int(summary.get("duplicates") or 0)
    transient = int(summary.get("transient") or 0)
    merged = summary.get("merged_into") or {}
    if physical is not None and int(physical) < count:
        merged_text = ", ".join(
            f"{int(number)} {name} track{'' if int(number) == 1 else 's'}"
            for name, number in merged.items()
        )
        parts = [f"The memory tracked {count} {label.lower()} track{plural}."]
        if duplicates:
            detail = (
                f"{duplicates} of them are near-identical duplicate detections of "
                f"{merged_text or 'another label'}"
            )
            if transient:
                detail += (
                    f" and {transient} one-frame track{'s' if transient != 1 else ''} "
                    f"{'are' if transient != 1 else 'is'} transient"
                )
            parts.append(
                f"{detail}, so the physical {label.lower()} count is {int(physical)}."
            )
        else:
            parts.append(
                f"{transient} one-frame track{'s' if transient != 1 else ''} "
                f"{'are' if transient != 1 else 'is'} transient, so the physical "
                f"{label.lower()} count is {int(physical)}."
            )
        return Answer(
            query,
            " ".join(parts),
            citations=[_citation(latest)] if latest else [],
            evidence=_evidence_list(store, float(latest["t"])) if latest else [],
            confidence=0.8,
            method="vault-count-dedup",
        )
    if peak and peak < count:
        answer_text = (
            f"The memory tracked {count} {label.lower()} track{plural}. "
            f"At most {peak} {label.lower()} track{'' if peak == 1 else 's'} "
            f"{'was' if peak == 1 else 'were'} visible at the same instant, so the "
            f"physical count is between {peak} and {count}: fragmentation or "
            "occlusion may inflate the track count."
        )
    elif peak == 1 and count == 1:
        answer_text = (
            f"The memory tracked 1 {label.lower()} track. It was visible in the "
            "observation window, but this is still a track count, not a verified "
            "physical count."
        )
    elif peak:
        answer_text = (
            f"The memory tracked {count} {label.lower()} track{plural}. "
            f"Up to {peak} were visible at the same instant, which is still a track "
            "count, not a verified physical count."
        )
    else:
        answer_text = (
            f"The memory tracked {count} {label.lower()} track{plural}. "
            "The tracker can split one physical object into more than one track, "
            "so this is a track count, not a verified object count."
        )
    return Answer(
        query,
        answer_text,
        citations=[_citation(latest)] if latest else [],
        evidence=_evidence_list(store, float(latest["t"])) if latest else [],
        confidence=0.75 if peak else 0.7,
        method="vault-count-visibility" if peak else "vault-count",
    )


def answer_inventory(store: EventStore, query: str) -> Answer:
    counts: dict[str, int] = {}
    for entity in store.entities:
        label = str(entity["label"])
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return Answer(query, "The memory has no entities yet.", method="no-match")
    total = sum(counts.values())
    latest = _latest(store.events)
    if isinstance(store.ledger, dict) and store.ledger.get("mode") == "caption-vault":
        parts = [
            f"{count} {label}" for label, count in sorted(counts.items())
        ]
        text = (
            f"The captions mention {len(counts)} object labels across "
            f"{len(store.ledger_windows)} window"
            f"{'s' if len(store.ledger_windows) != 1 else ''}: "
            + ", ".join(parts)
            + "."
        )
        return Answer(
            query,
            text,
            citations=[_citation(latest)] if latest else [],
            evidence=_evidence_list(store, float(latest["t"])) if latest else [],
            confidence=0.75,
            method="vault-inventory-captions",
        )
    parts = [
        f"{count} {label} track{'s' if count > 1 else ''}"
        for label, count in sorted(counts.items())
    ]
    text = (
        f"The memory tracked {total} object tracks across {len(counts)} labels: "
        + ", ".join(parts)
        + "."
    )
    if store.ledger_windows:
        text += f" The dense caption ledger covers {len(store.ledger_windows)} window(s)."
    return Answer(
        query,
        text,
        citations=[_citation(latest)] if latest else [],
        evidence=_evidence_list(store, float(latest["t"])) if latest else [],
        confidence=0.88,
        method="vault-inventory",
    )


def answer_yesno(
    store: EventStore,
    query: str,
    entity: dict[str, Any] | None,
    action: str,
) -> Answer:
    if entity is None:
        return Answer(query, "No, I don't have that in my memory.", method="no-match")
    label = str(entity["label"])
    rows = [event for event in _events_for(store, label) if _matches_action(event, action)]
    if not rows:
        return Answer(
            query,
            f"No record of the {label.lower()} being "
            + {"touch": "touched", "place": "placed", "see": "seen", "lost": "lost"}.get(
                action, "in an event"
            )
            + ".",
            method="no-match",
        )
    target = rows[-1]
    verb = {"touch": "touched", "place": "placed", "see": "seen", "lost": "lost"}.get(
        action, "recorded"
    )
    return Answer(
        query,
        f"Yes. The {label.lower()} was {verb} at t={float(target['t']):.1f}s. "
        + caption_for(target),
        citations=[_citation(target)],
        evidence=_evidence_list(store, float(target["t"])),
        confidence=0.85,
        method="vault-yesno",
    )


def _clean_holding(value: str) -> str:
    text = re.split(r"[.;,:]|\band\b|\bwhile\b|\bwho\b|\bwith\b", value, maxsplit=1)[0]
    text = " ".join(text.split()).strip(" -")
    if not text:
        return ""
    if not re.match(r"^(a|an|the|his|her|their|my|some)\b", text, re.IGNORECASE):
        text = ("an " if text[0].lower() in "aeiou" else "a ") + text
    return text


def answer_holding(
    store: EventStore, query: str, entity: dict[str, Any] | None
) -> Answer:
    """Answer "what is X holding?" from explicit caption phrases."""

    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for window in store.ledger_windows:
        text = str(window.get("caption") or window.get("text") or "")
        for match in _HOLDING.finditer(text):
            value = _clean_holding(match.group(1))
            if value:
                candidates.append((float(window.get("t_start") or 0.0), value, window))
    if not candidates:
        return Answer(
            query,
            "The captions do not say what was being held.",
            method="no-match",
        )

    seen: set[str] = set()
    unique: list[tuple[float, str, dict[str, Any]]] = []
    for t, value, window in candidates:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((t, value, window))
    # Prefer the first explicit phrase: "what is the other person holding?"
    # should be one direct answer, not a dump of every caption clause.
    first_t, first_value, first_window = unique[0]
    answer_text = first_value[:1].upper() + first_value[1:] + "."
    return Answer(
        query,
        answer_text,
        citations=[
            {
                "t": round(first_t, 3),
                "kind": "caption",
                "caption": str(first_window.get("caption") or "")[:500],
            }
        ],
        evidence=_nearest_evidence(store, first_t),
        confidence=0.75,
        method="vault-holding",
    )


def answer_keyword(store: EventStore, query: str) -> Answer:
    tokens = sorted(
        _normal_tokens(query) - _STOPWORDS
    )
    if not tokens:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    scored: list[tuple[int, dict[str, Any]]] = []
    for caption in store.ledger_windows:
        text = " ".join(
            [
                str(caption.get("caption") or caption.get("text") or ""),
                " ".join(str(tag) for tag in caption.get("tags") or []),
                " ".join(str(tag) for tag in caption.get("objects") or []),
                " ".join(str(tag) for tag in caption.get("actions") or []),
            ]
        ).lower()
        hits = sum(1 for token in tokens if token in text)
        if hits:
            scored.append((hits * 2, caption))
    for caption in store.captions:
        text = str(caption.get("text") or "").lower()
        hits = sum(1 for token in tokens if token in text)
        if hits:
            scored.append((hits, caption))
    if not scored:
        # Fall back to structured events only when the dense ledger has no hit.
        for event in store.events:
            text = caption_for(event).lower()
            hits = sum(1 for token in tokens if token in text)
            if hits:
                scored.append((hits, {**event, "caption": caption_for(event)}))
    if not scored:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    scored.sort(key=lambda item: (item[0], float(item[1].get("t_start", item[1].get("t", 0.0)))), reverse=True)
    rows = scored[:5]
    lines = ["Here is what the memory contains:"]
    citations: list[dict[str, Any]] = []
    for _, row in rows:
        text = str(row.get("caption") or row.get("text") or caption_for(row))
        t = float(row.get("t_start", row.get("t", 0.0)))
        lines.append(f"- [t={t:.1f}s] {text.splitlines()[0][:240]}")
        citations.append(
            {
                "t": round(t, 3),
                "kind": str(row.get("kind") or "caption"),
                "caption": text[:500],
            }
        )
    evidence = _nearest_evidence(store, float(citations[0]["t"]))
    return Answer(
        query,
        "\n".join(lines),
        citations=citations,
        evidence=[evidence] if evidence else [],
        confidence=0.6,
        method="vault-keyword",
    )


def _time_hint(query: str) -> float | None:
    lower = query.lower()
    if not re.search(
        r"\d+\s*(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes)\b"
        r"|t\s*=|\b(?:at|around|near|time)\s+\d",
        lower,
    ):
        return None
    match = _NUMBER.search(lower)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if re.search(rf"{re.escape(match.group(1))}\s*(?:m|min|mins|minutes)\b", lower):
        value *= 60.0
    return value


def _ledger_text(store: EventStore) -> str:
    return store.ledger_text


def _ledger_has(store: EventStore, query: str) -> bool:
    tokens = _normal_tokens(query) - _STOPWORDS
    if not tokens:
        return False
    text = _ledger_text(store)
    return any(re.search(rf"\b{re.escape(token)}\b", text) for token in tokens)


def classify(
    store: EventStore, query: str
) -> tuple[str, dict[str, Any] | None, float | None, str]:
    lower = query.lower().strip()
    entity = _match_entity(store, query)
    t = _time_hint(query)
    action = _action(query)
    in_ledger = entity is None and _ledger_has(store, query)

    if re.search(r"\bhow many entities\b|\bhow many objects\b", lower):
        return "inventory", None, None, ""
    if re.search(r"\bhow many\b|\bcount\b", lower):
        return ("keyword" if in_ledger else "count"), entity, None, ""
    if _INVENTORY.search(lower):
        return "inventory", None, None, ""
    if _NEGATIVE.search(lower):
        return "unknown", None, None, ""
    if "before" in lower:
        return ("keyword" if in_ledger and t is None else "before"), entity, t, action
    if "after" in lower:
        return ("keyword" if in_ledger and t is None else "after"), entity, t, action
    if re.search(r"\b(what happened|what was happening|what occurred|what was going on)\b", lower):
        if t is not None:
            return "window", entity, t, ""
        return ("keyword" if in_ledger else "history"), entity, None, ""
    if re.search(r"\b(holding|carrying|holds|carries|in (?:his|her|their) hand)\b", lower):
        return "holding", entity, None, ""
    if re.search(r"\b(where|which room|what room|what zone)\b", lower):
        return ("keyword" if in_ledger else "where"), entity, None, ""
    if re.search(r"\b(when|what time|at what (second|time))\b", lower):
        return ("keyword" if in_ledger else "when"), entity, t, action
    if re.match(r"\b(did|do|does|was|were|is|are|have|has)\b", lower):
        return "yesno", entity, t, action
    if "doing with" in lower or "do with" in lower or "did i do" in lower:
        return ("keyword" if in_ledger else "history"), entity, None, ""
    if "see" in lower and t is not None:
        return "window", entity, t, ""
    return "keyword", entity, t, ""


def canonical_key(
    intent: str,
    entity: dict[str, Any] | None,
    t: float | None,
    action: str,
    store: EventStore,
    query: str,
) -> str:
    entity_key = str(entity["id"]) if entity else "-"
    if intent == "inventory":
        return f"{CACHE_VERSION}:inventory"
    if intent == "unknown":
        return f"{CACHE_VERSION}:unknown:" + "|".join(sorted(_normal_tokens(query) - _STOPWORDS))
    if intent in {"where", "count", "history"}:
        return f"{CACHE_VERSION}:{intent}:{entity_key}"
    if intent == "holding":
        return f"{CACHE_VERSION}:holding:" + "|".join(sorted(_normal_tokens(query) - _STOPWORDS))
    if intent in {"when", "yesno"}:
        return f"{CACHE_VERSION}:{intent}:{entity_key}:{action or 'any'}"
    if intent in {"before", "after"}:
        return f"{CACHE_VERSION}:{intent}:{entity_key}:{action or 'any'}:{t if t is not None else 'anchor'}"
    if intent == "window":
        return f"{CACHE_VERSION}:window:{round(float(t or 0.0), 1)}"
    if intent == "keyword":
        return f"{CACHE_VERSION}:keyword:" + "|".join(sorted(_normal_tokens(query) - _STOPWORDS))
    return f"{CACHE_VERSION}:{intent}:{entity_key}:{action or 'any'}"


def _cache_get(con: sqlite3.Connection, key: str, question: str) -> Answer | None:
    try:
        row = con.execute(
            "select answer, citations, evidence, confidence, method from qa_cache "
            "where question_hash=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        con.execute("update qa_cache set hits = hits + 1 where question_hash=?", (key,))
        con.commit()
        return Answer(
            question,
            str(row["answer"]),
            citations=json.loads(row["citations"] or "[]"),
            evidence=json.loads(row["evidence"] or "[]"),
            confidence=float(row["confidence"] or 0.0),
            method="cache:" + str(row["method"]),
            cached=True,
        )
    except (sqlite3.Error, ValueError, TypeError):
        return None


def _cache_put(con: sqlite3.Connection, key: str, question: str, result: Answer) -> None:
    try:
        con.execute(
            """
            insert into qa_cache
                (question, question_hash, answer, citations, evidence, confidence,
                 method, created_at, hits)
            values (?, ?, ?, ?, ?, ?, ?, ?, 0)
            on conflict(question_hash) do update set
                answer=excluded.answer,
                citations=excluded.citations,
                evidence=excluded.evidence,
                confidence=excluded.confidence,
                method=excluded.method
            """,
            (
                question,
                key,
                result.answer,
                json.dumps(result.citations, ensure_ascii=False),
                json.dumps(result.evidence, ensure_ascii=False),
                float(result.confidence),
                result.method,
                time.time(),
            ),
        )
        con.commit()
    except sqlite3.Error:
        pass


def answer(store: EventStore, query: str, *, use_cache: bool = True) -> Answer:
    started = time.perf_counter()
    lower = query.lower().strip()
    if not lower:
        return Answer(query, "Ask a question about the episode.", method="empty")

    intent, entity, t, action = classify(store, query)
    key = canonical_key(intent, entity, t, action, store, query)
    if use_cache:
        cached = _cache_get(store.con, key, query)
        if cached is not None:
            cached.elapsed_ms = (time.perf_counter() - started) * 1000.0
            return cached

    if intent == "inventory":
        result = answer_inventory(store, query)
    elif intent == "unknown":
        result = Answer(query, "I don't have that in my memory.", method="no-match")
    elif intent == "count":
        result = answer_count(store, query, entity)
    elif intent == "holding":
        result = answer_holding(store, query, entity)
    elif intent == "before":
        result = answer_before(store, query, entity, action, t)
    elif intent == "after":
        result = answer_after(store, query, entity, action, t)
    elif intent == "window":
        result = answer_window(store, query, float(t or 0.0))
    elif intent == "where":
        if entity is None:
            result = Answer(query, "I don't have that in my memory.", method="no-match")
        else:
            result = answer_where(store, query, entity)
    elif intent == "when":
        if entity is None:
            result = Answer(query, "I don't have that in my memory.", method="no-match")
        else:
            result = answer_when(store, query, entity, action)
    elif intent == "yesno":
        result = answer_yesno(store, query, entity, action)
    elif intent == "history":
        if entity is None:
            result = Answer(query, "I don't have that in my memory.", method="no-match")
        else:
            result = answer_history(store, query, entity)
    else:
        result = answer_keyword(store, query)

    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    if use_cache and result.method != "empty":
        _cache_put(store.con, key, query, result)
    return result


def stats(store: EventStore, vault: Path) -> dict[str, Any]:
    payload = {
        "vault": str(vault),
        "bytes": vault.stat().st_size if vault.is_file() else 0,
        "entities": len(store.entities),
        "events": len(store.events),
        "captions": len(store.captions),
        "ledger_windows": len(store.ledger_windows),
        "ledger_model": str(store.ledger.get("model") or "") if isinstance(store.ledger, dict) else "",
        "duration_s": round(store.duration, 3),
        # Per-label track count and peak simultaneous visibility. Empty for
        # caption-first vaults, which have no per-frame observations.
        "counts": store.counts,
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
        payload["evidence_bytes"] = int(
            store.con.execute("select coalesce(sum(bytes),0) from evidence").fetchone()[0]
        )
        payload["cache_entries"] = int(
            store.con.execute("select count(*) from qa_cache").fetchone()[0]
        )
        payload["cache_hits"] = int(
            store.con.execute("select coalesce(sum(hits),0) from qa_cache").fetchone()[0]
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
    parser.add_argument("--no-cache", action="store_true")
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
        result = answer(store, question, use_cache=not args.no_cache)
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
        cache_label = " · cached" if result.cached else ""
        print(
            f"\nmethod={result.method} · {result.elapsed_ms:.1f} ms · "
            f"confidence={result.confidence:.2f}{cache_label}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

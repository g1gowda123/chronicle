#!/usr/bin/env python3
"""Build an LLM-ready memory pack from a VFM vault.

The pack is the contract between the compiler and any text LLM: dense timestamped
captions, salient events, entities and evidence references — no video path, no
raw frames. It can be exported for RAG or passed directly to Qwen.

Usage:
    python prototype/memory_pack.py --db vault.db --out memory_pack.md
    python prototype/memory_pack.py --db vault.db --question "what happened?" --json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import query

_TOKEN = re.compile(r"[a-z0-9]+")
_TIME = re.compile(r"t\s*=\s*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_STOP = {
    "the", "a", "an", "is", "was", "were", "are", "am", "did", "do", "does",
    "what", "where", "when", "which", "who", "how", "many", "much", "my",
    "i", "me", "in", "on", "at", "to", "of", "for", "with", "and", "or",
    "this", "that", "it", "there", "be", "been", "being", "have", "has",
    "had", "can", "could", "would", "should", "about", "around", "last",
    "first", "before", "after", "then", "tell", "show", "memory",
}
_KIND_WEIGHT = {"interaction": 4, "anchor": 3, "state": 2, "keyframe": 1}


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(str(text).lower()) if token}


def _window_score(window: dict[str, Any], tokens: set[str]) -> int:
    text = " ".join(
        [
            str(window.get("caption") or window.get("text") or ""),
            " ".join(str(value) for value in window.get("objects") or []),
            " ".join(str(value) for value in window.get("actions") or []),
            " ".join(str(value) for value in window.get("tags") or []),
        ]
    ).lower()
    return sum(1 for token in tokens if token in text)


def _select_windows(
    store: query.EventStore, question: str | None, max_windows: int
) -> list[dict[str, Any]]:
    windows = list(store.ledger_windows)
    if not windows:
        return []
    if not question:
        return windows[:max_windows]
    tokens = _tokens(question) - _STOP
    if not tokens:
        return windows[:max_windows]
    ranked = sorted(
        windows,
        key=lambda window: (
            _window_score(window, tokens),
            float(window.get("t_start") or 0.0),
        ),
        reverse=True,
    )
    selected = [window for window in ranked[:max_windows] if _window_score(window, tokens)]
    if not selected:
        selected = windows[:max_windows]
    return sorted(selected, key=lambda window: float(window.get("t_start") or 0.0))


def _select_events(
    store: query.EventStore, question: str | None, max_events: int
) -> list[dict[str, Any]]:
    if not question:
        selected = sorted(
            store.events,
            key=lambda event: (
                _KIND_WEIGHT.get(str(event["kind"]), 1),
                -float(event["t"]),
            ),
            reverse=True,
        )[:max_events]
        return sorted(selected, key=lambda event: float(event["t"]))
    entity = query._match_entity(store, question)
    rows = query._events_for(store, str(entity["label"])) if entity else store.events
    time_hint = query._time_hint(question)
    if time_hint is not None:
        rows = sorted(rows, key=lambda event: abs(float(event["t"]) - time_hint))[:max_events]
        return sorted(rows, key=lambda event: float(event["t"]))
    return rows[-max_events:]


def build_pack(
    store: query.EventStore,
    question: str | None = None,
    *,
    verified_answer: query.Answer | None = None,
    max_chars: int = 14000,
    max_windows: int = 12,
    max_events: int = 60,
) -> str:
    windows = _select_windows(store, question, max_windows)
    events = _select_events(store, question, max_events)
    counts: dict[str, int] = {}
    for entity in store.entities:
        label = str(entity["label"])
        counts[label] = counts.get(label, 0) + 1

    lines: list[str] = [
        "# Chronicle memory pack",
        f"Duration: {store.duration:.1f}s",
        f"Caption windows: {len(store.ledger_windows)} total, {len(windows)} shown",
        "Counts (tracker tracks, not verified physical objects): "
        + ", ".join(f"{label}: {count}" for label, count in sorted(counts.items())),
        "",
    ]
    if question:
        entity = query._match_entity(store, question)
        if entity is not None:
            lines.append(
                f"Question entity: {entity['label']} "
                "(treat synonyms such as mug/cup or rucksack/backpack as this entity)."
            )
            lines.append("")
    if verified_answer is not None and verified_answer.method not in {"no-match", "empty"}:
        lines.extend(
            [
                "## Verified structured answer",
                "Use this as the primary answer; use the captions below for extra detail.",
                verified_answer.answer,
            ]
        )
        for citation in verified_answer.citations[:4]:
            lines.append(
                f"- citation [t={float(citation.get('t') or 0):.1f}s] "
                f"{str(citation.get('caption') or '')[:240]}"
            )
        lines.append("")

    lines.extend(["## Timestamped caption windows"])
    for window in windows:
        start = float(window.get("t_start") or 0.0)
        end = float(window.get("t_end") or start + 30.0)
        caption = str(window.get("caption") or window.get("text") or "").strip()
        lines.append(f"[{start:.1f}-{end:.1f}s] {caption}")
        objects = ", ".join(str(value) for value in window.get("objects") or [])
        actions = ", ".join(str(value) for value in window.get("actions") or [])
        zone = str(window.get("zone") or "")
        if objects:
            lines.append(f"  objects: {objects}")
        if actions:
            lines.append(f"  actions: {actions}")
        if zone and zone != "unknown":
            lines.append(f"  zone: {zone}")
        for frame in window.get("frames") or []:
            if isinstance(frame, dict) and frame.get("caption"):
                lines.append(f"  frame t={float(frame.get('t') or 0):.1f}s: {frame['caption']}")

    lines.extend(["", "## Event ledger"])
    for event in events:
        lines.append(f"- [t={float(event['t']):.1f}s] {query.caption_for(event)}")

    lines.extend(["", "## Evidence references"])
    evidence = sorted(
        store.evidence,
        key=lambda row: (
            min(
                (abs(float(row["t"]) - float(event["t"])) for event in events),
                default=9999.0,
            )
        ),
    )[:24]
    for row in evidence:
        lines.append(f"- [t={float(row['t']):.1f}s] {row.get('label') or row.get('kind')}")

    lines.extend(
        [
            "",
            "## Answering rules",
            "- The verified structured answer, when present, is authoritative.",
            "- Question terms may be synonyms (mug/cup, coffee cup/cup, rucksack/backpack); do not",
            "  answer unknown when a verified answer exists.",
            "- Answer only from this memory. Do not invent objects, places, times or actions.",
            "- Write counts as digits, e.g. '2 cup tracks', not 'two cups'.",
            "- Cite timestamps as [t=12.3s].",
            "- If the memory cannot answer, reply exactly: I don't have that in my memory.",
        ]
    )
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n... (memory pack truncated)"
    return text


def build_pack_data(
    store: query.EventStore,
    *,
    max_windows: int = 200,
    max_events: int = 400,
) -> dict[str, Any]:
    windows = store.ledger_windows[:max_windows]
    events = sorted(store.events, key=lambda event: float(event["t"]))[:max_events]
    return {
        "version": 1,
        "duration_s": round(store.duration, 3),
        "entities": [
            {
                "id": entity["id"],
                "label": entity["label"],
                "state": entity["state"],
                "zone": entity["zone"],
                "first_seen": entity.get("first_seen"),
                "last_seen": entity.get("last_seen"),
            }
            for entity in store.entities
        ],
        "windows": [
            {
                "t_start": window.get("t_start"),
                "t_end": window.get("t_end"),
                "caption": window.get("caption") or window.get("text") or "",
                "objects": window.get("objects") or [],
                "actions": window.get("actions") or [],
                "zone": window.get("zone") or "",
                "tags": window.get("tags") or [],
            }
            for window in windows
        ],
        "events": [
            {
                "t": round(float(event["t"]), 3),
                "kind": event["kind"],
                "subject": event["subject"],
                "relation": event["relation"],
                "object": event["object"],
                "zone": event["zone"],
                "caption": query.caption_for(event),
            }
            for event in events
        ],
        "evidence": [
            {
                "t": round(float(row["t"]), 3),
                "kind": row["kind"],
                "label": row["label"],
                "id": row["id"],
            }
            for row in sorted(store.evidence, key=lambda row: float(row["t"]))
        ],
        "stats": {
            "chars": len(build_pack(store)),
            "approx_tokens": round(len(build_pack(store)) / 4.0),
            "windows": len(store.ledger_windows),
            "events": len(store.events),
            "evidence": len(store.evidence),
        },
    }


def load_store(db: Path) -> query.EventStore:
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    return query.load_store(con)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--max-chars", type=int, default=14000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    store = load_store(args.db.expanduser().resolve())
    try:
        if args.json:
            payload = build_pack_data(store)
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        else:
            text = build_pack(store, args.question, max_chars=max(2000, args.max_chars))
    finally:
        store.con.close()
    if args.out:
        args.out.expanduser().resolve().write_text(text, encoding="utf-8")
        print(json.dumps({"out": str(args.out), "chars": len(text), "approx_tokens": round(len(text) / 4)}, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

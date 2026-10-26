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

#: Rough character/token ratio used for budget sizing. The pack is plain English
#: with short numbers and timestamps; four characters per token is the same
#: approximation used by the benchmark and Studio.
CHARS_PER_TOKEN = 4.0

#: Suggested budgets. Any positive integer is accepted; these are the defaults
#: exposed by the CLI and Studio selector.
TOKEN_BUDGETS = (2000, 4000, 8000)


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


_RULES = [
    "## Answering rules",
    "- The verified structured answer, when present, is authoritative.",
    "- Question terms may be synonyms (mug/cup, coffee cup/cup, rucksack/backpack); do not",
    "  answer unknown when a verified answer exists.",
    "- Answer only from this memory. Do not invent objects, places, times or actions.",
    "- Write counts as digits, e.g. '2 cup tracks', not 'two cups'.",
    "- Cite timestamps as [t=12.3s].",
    "- If the memory cannot answer, reply exactly: I don't have that in my memory.",
]


def _approx_tokens(text: str) -> int:
    return int(round(len(text) / CHARS_PER_TOKEN))


def _budget_chars(token_budget: int | None, max_chars: int) -> int:
    """Translate a token budget into a character ceiling.

    ``max_chars`` is kept for backwards compatibility. When ``token_budget`` is
    given it wins, because that it is what the CLI/Studio selector exposes.
    """

    if token_budget is None:
        return max(2000, int(max_chars))
    return max(800, int(round(float(token_budget) * CHARS_PER_TOKEN)))


def _window_block(window: dict[str, Any], *, include_frames: bool = True) -> list[str]:
    start = float(window.get("t_start") or 0.0)
    end = float(window.get("t_end") or start + 30.0)
    caption = str(window.get("caption") or window.get("text") or "").strip()
    lines = [f"[{start:.1f}-{end:.1f}s] {caption}"]
    objects = ", ".join(str(value) for value in window.get("objects") or [])
    actions = ", ".join(str(value) for value in window.get("actions") or [])
    zone = str(window.get("zone") or "")
    if objects:
        lines.append(f"  objects: {objects}")
    if actions:
        lines.append(f"  actions: {actions}")
    if zone and zone != "unknown":
        lines.append(f"  zone: {zone}")
    if include_frames:
        for frame in window.get("frames") or []:
            if isinstance(frame, dict) and frame.get("caption"):
                lines.append(f"  frame t={float(frame.get('t') or 0):.1f}s: {frame['caption']}")
    return lines


def _event_line(event: dict[str, Any]) -> str:
    return f"- [t={float(event['t']):.1f}s] {query.caption_for(event)}"


def _evidence_line(row: dict[str, Any]) -> str:
    return f"- [t={float(row['t']):.1f}s] {row.get('label') or row.get('kind')}"


def _header_lines(
    store: query.EventStore,
    question: str | None,
    window_count: int,
    event_count: int,
    evidence_count: int,
    *,
    token_budget: int | None = None,
) -> list[str]:
    counts: dict[str, int] = {}
    for entity in store.entities:
        label = str(entity["label"])
        counts[label] = counts.get(label, 0) + 1
    lines = [
        "# Chronicle memory pack",
        f"Duration: {store.duration:.1f}s",
        f"Caption windows: {len(store.ledger_windows)} total, {window_count} shown",
        "Counts (tracker tracks, not verified physical objects): "
        + ", ".join(f"{label}: {count}" for label, count in sorted(counts.items())),
    ]
    if token_budget is not None:
        lines.append(f"Budget: ~{int(token_budget)} tokens")
    lines.append(f"Events shown: {event_count} · evidence refs: {evidence_count}")
    lines.append("")
    if question:
        entity = query._match_entity(store, question)
        if entity is not None:
            lines.append(
                f"Question entity: {entity['label']} "
                "(treat synonyms such as mug/cup or rucksack/backpack as this entity)."
            )
            lines.append("")
    return lines


def _verified_lines(
    verified_answer: query.Answer | None, *, max_answer_chars: int | None = None
) -> list[str]:
    if verified_answer is None or verified_answer.method in {"no-match", "empty"}:
        return []
    answer = str(verified_answer.answer or "")
    if max_answer_chars is not None and len(answer) > max_answer_chars:
        answer = answer[: max(40, max_answer_chars)].rstrip() + " …"
    lines = [
        "## Verified structured answer",
        "Use this as the primary answer; use the captions below for extra detail.",
        answer,
    ]
    for citation in verified_answer.citations[:4]:
        lines.append(
            f"- citation [t={float(citation.get('t') or 0):.1f}s] "
            f"{str(citation.get('caption') or '')[:240]}"
        )
    lines.append("")
    return lines


def build_pack(
    store: query.EventStore,
    question: str | None = None,
    *,
    verified_answer: query.Answer | None = None,
    max_chars: int = 14000,
    max_windows: int = 12,
    max_events: int = 60,
    token_budget: int | None = None,
) -> str:
    """Render a memory pack that fits ``token_budget`` (or ``max_chars``).

    A budget never cuts a line in half. It keeps the verified answer and the
    answering rules, then adds caption windows, events and evidence references
    until the budget is full. Smaller budgets therefore lose detail, not
    instructions.
    """

    limit = _budget_chars(token_budget, max_chars)
    # A small reserve lets us append "(N omitted ...)" notes without crossing
    # the ceiling. Without a token budget the legacy max_chars path is used.
    reserve = 220 if token_budget is not None else 120
    working_limit = max(400, limit - reserve)

    if token_budget is not None:
        # Let larger budgets actually use their headroom instead of being
        # capped by the small default retrieval window.
        max_windows = max(max_windows, max(6, int(token_budget // 300)))
        max_events = max(max_events, max(20, int(token_budget // 50)))

    windows = _select_windows(store, question, max_windows)
    events = _select_events(store, question, max_events)
    evidence = sorted(
        store.evidence,
        key=lambda row: (
            min(
                (abs(float(row["t"]) - float(event["t"])) for event in events),
                default=9999.0,
            )
        ),
    )[:24]

    chosen_windows: list[list[str]] = []
    chosen_events: list[str] = []
    chosen_evidence: list[str] = []
    omitted = {"windows": 0, "events": 0, "evidence": 0}
    verified_text_limit: int | None = None

    def render() -> str:
        lines = _header_lines(
            store,
            question,
            len(chosen_windows),
            len(chosen_events),
            len(chosen_evidence),
            token_budget=token_budget,
        )
        lines.extend(_verified_lines(verified_answer, max_answer_chars=verified_text_limit))
        lines.append("## Timestamped caption windows")
        if chosen_windows:
            for block in chosen_windows:
                lines.extend(block)
        else:
            lines.append("(no caption windows in this vault)")
        lines.extend(["", "## Event ledger"])
        if chosen_events:
            lines.extend(chosen_events)
        else:
            lines.append("(no events in this vault)")
        lines.extend(["", "## Evidence references"])
        if chosen_evidence:
            lines.extend(chosen_evidence)
        else:
            lines.append("(no evidence snapshots)")
        lines.extend([""] + _RULES)
        if any(omitted.values()):
            notes = [
                f"{count} more {name} omitted for the "
                f"{int(token_budget)}-token budget"
                if token_budget is not None
                else f"{count} more {name} omitted for the size limit"
                for name, count in omitted.items()
                if count
            ]
            lines.extend(["", "… " + "; ".join(notes) + "."])
        return "\n".join(lines)

    # If even the fixed part (header + verified answer + rules) is over budget,
    # shorten only the verified answer text; never drop the instructions.
    if len(render()) > working_limit and verified_answer is not None:
        fixed_without_answer = len(render())
        answer_len = len(str(verified_answer.answer or ""))
        overflow = fixed_without_answer - working_limit
        verified_text_limit = max(80, answer_len - overflow - 80)

    for window in windows:
        block = _window_block(window, include_frames=True)
        chosen_windows.append(block)
        if len(render()) <= working_limit:
            continue
        chosen_windows.pop()
        lean = _window_block(window, include_frames=False)
        chosen_windows.append(lean)
        if len(render()) <= working_limit:
            continue
        chosen_windows.pop()
        omitted["windows"] += 1

    for event in events:
        chosen_events.append(_event_line(event))
        if len(render()) <= working_limit:
            continue
        chosen_events.pop()
        omitted["events"] += 1

    for row in evidence:
        chosen_evidence.append(_evidence_line(row))
        if len(render()) <= working_limit:
            continue
        chosen_evidence.pop()
        omitted["evidence"] += 1

    text = render()
    if len(text) > limit:
        # Last-resort guard for pathological strings (e.g. an enormous verified
        # answer). Trim on a line boundary so the pack stays valid text.
        text = text[:limit].rsplit("\n", 1)[0] + "\n... (memory pack truncated)"
    return text


def build_pack_data(
    store: query.EventStore,
    *,
    max_windows: int = 200,
    max_events: int = 400,
    token_budget: int | None = None,
) -> dict[str, Any]:
    windows = store.ledger_windows[:max_windows]
    events = sorted(store.events, key=lambda event: float(event["t"]))[:max_events]
    pack = build_pack(store, token_budget=token_budget)
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
            "chars": len(pack),
            "approx_tokens": _approx_tokens(pack),
            "token_budget": token_budget,
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
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="target pack size in tokens, e.g. 2000, 4000 or 8000 (overrides --max-chars)",
    )
    parser.add_argument(
        "--list-budgets",
        action="store_true",
        help="print pack size at each suggested budget and exit",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    store = load_store(args.db.expanduser().resolve())
    try:
        if args.list_budgets:
            rows = []
            for budget in TOKEN_BUDGETS:
                text = build_pack(
                    store,
                    args.question,
                    token_budget=budget,
                    max_chars=max(2000, args.max_chars),
                )
                rows.append(
                    {
                        "token_budget": budget,
                        "chars": len(text),
                        "approx_tokens": _approx_tokens(text),
                    }
                )
            text = json.dumps(rows, indent=2)
            if not args.out:
                print(text)
                return 0
        elif args.json:
            payload = build_pack_data(store, token_budget=args.token_budget)
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        else:
            text = build_pack(
                store,
                args.question,
                token_budget=args.token_budget,
                max_chars=max(2000, args.max_chars),
            )
    finally:
        store.con.close()
    if args.out:
        args.out.expanduser().resolve().write_text(text, encoding="utf-8")
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "chars": len(text),
                    "approx_tokens": _approx_tokens(text),
                    "token_budget": args.token_budget,
                },
                indent=2,
            )
        )
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

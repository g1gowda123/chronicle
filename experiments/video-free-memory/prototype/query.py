#!/usr/bin/env python3
"""Query a compiled VFM vault with no video and no language model.

This is the v0 query engine: intents + SQL. It is intentionally boring. If a
question cannot be answered here, the vault is missing structure, not
intelligence.

Usage:
    python prototype/query.py --db vault.db "where is the cup?"
    python prototype/query.py --db vault.db --json "what happened around 20s?"
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


def _rows(con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    try:
        return list(con.execute(sql, params))
    except sqlite3.Error:
        return []


def _citation(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "t": round(float(row["t"]), 3),
        "kind": row["kind"],
        "subject": row["subject"],
        "relation": row["relation"],
        "object": row["object"],
        "zone": row["zone"],
        "caption": row["caption"],
    }


def _evidence(con: sqlite3.Connection, evidence_id: str | None) -> dict[str, Any] | None:
    if not evidence_id:
        return None
    rows = _rows(con, "select * from evidence where id=?", (evidence_id,))
    if not rows:
        return None
    row = rows[0]
    return {
        "id": row["id"],
        "t": round(float(row["t"]), 3),
        "kind": row["kind"],
        "label": row["label"],
        "rel_path": row["rel_path"],
        "bytes": row["bytes"],
    }


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(str(text).lower()) if token}


def _match_entity(con: sqlite3.Connection, query: str) -> sqlite3.Row | None:
    tokens = _tokens(query)
    candidates = _rows(con, "select * from entities")
    best: tuple[float, sqlite3.Row] | None = None
    for row in candidates:
        label = str(row["label"]).lower()
        label_tokens = _tokens(label)
        query_lower = query.lower()
        score = 0.0
        if label in query_lower:
            score += 3.0
        score += len(tokens & label_tokens) * 2.0
        if not score:
            singular = label[:-1] if label.endswith("s") and len(label) > 3 else label
            if singular in tokens or any(token[:-1] == label for token in tokens if token.endswith("s")):
                score += 1.5
        if score <= 0:
            continue
        if row["state"] in {"VISIBLE_STATIC", "HELD", "CONTAINED"}:
            score += 0.1
        if best is None or score > best[0]:
            best = (score, row)
    return best[1] if best else None


def _time_hint(query: str) -> float | None:
    match = _NUMBER.search(query.lower())
    if not match:
        return None
    # Avoid treating "3 objects" as a timestamp when the word second is absent.
    if not re.search(r"\b(s|sec|secs|seconds?)\b|t\s*=", query.lower()):
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _latest_event_for(con: sqlite3.Connection, label: str, *, before: float | None = None) -> sqlite3.Row | None:
    sql = "select * from events where lower(subject)=lower(?)"
    params: list[Any] = [label]
    if before is not None:
        sql += " and t <= ?"
        params.append(before)
    sql += " order by t desc limit 1"
    rows = _rows(con, sql, tuple(params))
    return rows[0] if rows else None


def _events_for(con: sqlite3.Connection, label: str, limit: int = 8) -> list[sqlite3.Row]:
    return _rows(
        con,
        "select * from events where lower(subject)=lower(?) order by t desc limit ?",
        (label, limit),
    )


def answer_where(con: sqlite3.Connection, query: str) -> Answer:
    entity = _match_entity(con, query)
    if entity is None:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    label = str(entity["label"])
    latest = _latest_event_for(con, label)
    anchor = _rows(
        con,
        """
        select * from events
        where lower(subject)=lower(?) and kind='anchor'
        order by t desc limit 1
        """,
        (label,),
    )
    state = str(entity["state"] or "unknown")
    zone = str(entity["zone"] or "")
    parts = [f"The {label.lower()} is currently {state.replace('_', ' ').lower()}"]
    if zone:
        parts[0] += f" in the {zone}"
    parts[0] += "."
    if anchor:
        row = anchor[0]
        parts.append(
            f"It was last {str(row['relation']).replace('_', ' ').lower()} "
            f"the {str(row['object']).lower()} at [t={float(row['t']):.1f}s]."
        )
    elif latest is not None:
        parts.append(str(latest["caption"]))
    citations = [(_citation(anchor[0]) if anchor else _citation(latest))] if (anchor or latest) else []
    evidence_id = None
    for row in ([anchor[0]] if anchor else []) + ([latest] if latest else []):
        if row is not None and row["evidence_id"]:
            evidence_id = row["evidence_id"]
            break
    evidence = _evidence(con, evidence_id)
    return Answer(
        query,
        " ".join(parts),
        citations=citations,
        evidence=[evidence] if evidence else [],
        confidence=0.85 if anchor or latest else 0.6,
        method="vault-where",
    )


def answer_when(con: sqlite3.Connection, query: str) -> Answer:
    entity = _match_entity(con, query)
    if entity is None:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    label = str(entity["label"])
    rows = _events_for(con, label, limit=6)
    if not rows:
        return Answer(query, f"No events recorded for the {label.lower()}.", method="no-match")
    lines = [f"Here is the recorded history for the {label.lower()}:"]
    citations: list[dict[str, Any]] = []
    for row in reversed(rows):
        lines.append(f"- {row['caption']}")
        citations.append(_citation(row))
    return Answer(
        query,
        "\n".join(lines),
        citations=citations,
        confidence=0.8,
        method="vault-when",
    )


def answer_window(con: sqlite3.Connection, query: str, t: float) -> Answer:
    rows = _rows(
        con,
        "select * from events where t between ? and ? order by t limit 20",
        (max(0.0, t - 10.0), t + 10.0),
    )
    if not rows:
        return Answer(query, f"No events recorded around t={t:.1f}s.", method="no-match")
    lines = [f"Around t={t:.1f}s:"]
    for row in rows:
        lines.append(f"- {row['caption']}")
    return Answer(
        query,
        "\n".join(lines),
        citations=[_citation(row) for row in rows],
        confidence=0.75,
        method="vault-window",
    )


def answer_inventory(con: sqlite3.Connection, query: str) -> Answer:
    rows = _rows(
        con,
        "select label, count(*) as n from entities group by label order by n desc, label",
    )
    if not rows:
        return Answer(query, "The memory has no entities yet.", method="no-match")
    total = sum(int(row["n"]) for row in rows)
    parts = [f"{row['n']} {row['label']}{'s' if int(row['n']) > 1 else ''}" for row in rows]
    return Answer(
        query,
        f"The memory tracked {total} entities: " + ", ".join(parts) + ".",
        confidence=0.9,
        method="vault-inventory",
    )


def answer_keyword(con: sqlite3.Connection, query: str) -> Answer:
    tokens = sorted(_tokens(query) - {"the", "a", "an", "is", "was", "did", "what", "where", "when"})
    if not tokens:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    clauses = " or ".join("lower(text) like ?" for _ in tokens)
    params = tuple(f"%{token}%" for token in tokens)
    rows = _rows(
        con,
        f"select * from captions where {clauses} order by t_start desc limit 5",
        params,
    )
    if not rows:
        return Answer(query, "I don't have that in my memory.", method="no-match")
    lines = ["Here is what the memory contains:"]
    citations = []
    for row in rows:
        lines.append(f"- {row['text'].splitlines()[0]}")
        citations.append(
            {
                "t": round(float(row["t_start"]), 3),
                "kind": row["kind"],
                "caption": row["text"][:300],
            }
        )
    return Answer(
        query,
        "\n".join(lines),
        citations=citations,
        confidence=0.55,
        method="vault-keyword",
    )


def answer(con: sqlite3.Connection, query: str) -> Answer:
    started = time.perf_counter()
    lower = query.lower().strip()
    if not lower:
        return Answer(query, "Ask a question about the episode.", method="empty")
    if re.search(r"\b(what did i see|what did i observe|inventory|list|tracked|overview)\b", lower):
        result = answer_inventory(con, query)
    elif re.search(r"\bnot in (this )?memory\b|\bunknown\b|\bno such\b", lower):
        result = Answer(query, "I don't have that in my memory.", method="no-match")
    elif re.search(r"\b(doing with|what was .* doing|what did .* do)\b", lower):
        result = answer_when(con, query)
    elif re.search(r"\b(what happened|around|at)\b", lower) and _time_hint(query) is not None:
        result = answer_window(con, query, float(_time_hint(query) or 0.0))
    elif lower.startswith("where") or " where " in lower:
        result = answer_where(con, query)
    elif lower.startswith("when") or " when " in lower:
        result = answer_when(con, query)
    elif lower.startswith("what did") and " do" in lower:
        result = answer_when(con, query)
    else:
        result = answer_keyword(con, query)
    result.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.db.is_file():
        raise SystemExit(f"vault not found: {args.db}")
    question = args.question
    if not question:
        question = input("question> ").strip()
    con = sqlite3.connect(str(args.db))
    con.row_factory = sqlite3.Row
    try:
        result = answer(con, question)
    finally:
        con.close()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.answer)
        if result.citations:
            print("\nEvidence:")
            for cite in result.citations[:6]:
                print(f"  [t={cite.get('t')}s] {cite.get('caption', cite.get('kind', ''))[:160]}")
        print(f"\nmethod={result.method} · {result.elapsed_ms:.1f} ms · confidence={result.confidence:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

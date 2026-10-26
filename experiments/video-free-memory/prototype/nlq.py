#!/usr/bin/env python3
"""Optional natural-language answers over a VFM vault.

The deterministic `query.py` remains the default. When `--qwen` is enabled, this
module retrieves dense caption windows/events and asks a local text model to
phrase the answer. It never sees video.

Usage:
    from nlq import answer_nl
    result = answer_nl(store, "what was I doing with the tomato?",
                       model="qwen2.5:7b-instruct")
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

import caption_ledger
import memory_pack
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


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(str(text).lower()) if token}


def _context(store: query.EventStore, question: str, deterministic: query.Answer) -> str:
    tokens = _tokens(question) - _STOP
    windows = []
    for window in store.ledger_windows:
        text = " ".join(
            [
                str(window.get("caption") or window.get("text") or ""),
                " ".join(str(value) for value in window.get("objects") or []),
                " ".join(str(value) for value in window.get("actions") or []),
                " ".join(str(value) for value in window.get("tags") or []),
            ]
        ).lower()
        hits = sum(1 for token in tokens if token in text)
        if hits:
            windows.append((hits, window))
    windows.sort(
        key=lambda item: (item[0], float(item[1].get("t_start") or 0.0)), reverse=True
    )
    selected = [window for _, window in windows[:8]]
    if not selected:
        selected = store.ledger_windows[-6:]

    lines = ["CAPTION MEMORY:"]
    for window in selected:
        start = float(window.get("t_start") or 0.0)
        end = float(window.get("t_end") or start + 30.0)
        caption = str(window.get("caption") or window.get("text") or "").strip()
        tags = ", ".join(str(value) for value in window.get("tags") or [])
        lines.append(f"[{start:.1f}-{end:.1f}s] {caption}")
        if tags:
            lines.append(f"  tags: {tags}")

    entity = query._match_entity(store, question)
    events = store.events
    if entity is not None:
        focused = query._events_for(store, str(entity["label"]))
        if focused:
            events = focused
    time_hint = query._time_hint(question)
    if time_hint is not None:
        events = sorted(events, key=lambda event: abs(float(event["t"]) - time_hint))[:12]
    else:
        events = sorted(events, key=lambda event: float(event["t"]))[-12:]
    lines.append("EVENT LEDGER:")
    for event in events:
        lines.append(f"- {query.caption_for(event)}")

    if deterministic.method != "no-match":
        lines.append("DETERMINISTIC ANSWER:")
        lines.append(deterministic.answer[:1200])
    return "\n".join(lines)


def _ollama(
    model: str, prompt: str, *, host: str, timeout: float
) -> str:
    body = caption_ledger.ollama_generate_body(
        model,
        prompt,
        [],
        host=host,
        timeout=timeout,
        json_mode=False,
    )
    return str(body.get("response") or "").strip()


def answer_nl(
    store: query.EventStore,
    question: str,
    *,
    model: str = "qwen2.5:7b-instruct",
    host: str = caption_ledger.DEFAULT_HOST,
    timeout: float = 180.0,
) -> query.Answer:
    started = time.perf_counter()
    deterministic = query.answer(store, question, use_cache=False)
    pack = memory_pack.build_pack(
        store, question, verified_answer=deterministic, max_chars=12000
    )
    prompt = (
        "You are Chronicle, an egocentric memory assistant. Answer using ONLY the memory "
        "pack below. If a verified structured answer is present, it is authoritative and "
        "must be used even when the question uses a synonym (mug/cup, coffee cup/cup, "
        "rucksack/backpack). Do not invent objects, places, times or actions. Write two to "
        "four natural sentences. Write counts as digits. Include at least one timestamp "
        "citation such as [t=12.3s]. If the memory cannot answer, reply exactly: "
        "I don't have that in my memory.\n\n"
        f"{pack}\n\nQUESTION: {question}\nANSWER:"
    )
    answer_text = ""
    method = f"qwen-pack:{model}"
    try:
        answer_text = _ollama(model, prompt, host=host, timeout=timeout)
    except RuntimeError:
        answer_text = ""
    if not answer_text:
        fallback = deterministic
        fallback.elapsed_ms = (time.perf_counter() - started) * 1000.0
        fallback.method = f"qwen-fallback:{fallback.method}"
        fallback.cached = False
        return fallback

    times = []
    for value in _TIME.findall(answer_text):
        try:
            times.append(float(value))
        except ValueError:
            continue
    citations = []
    for timestamp in times[:6]:
        citations.append(
            {
                "t": round(timestamp, 3),
                "kind": "answer",
                "caption": f"model citation at t={timestamp:.1f}s",
            }
        )
    if not citations:
        citations = list(deterministic.citations)
    if not citations:
        for window in store.ledger_windows[-3:]:
            citations.append(
                {
                    "t": round(float(window.get("t_start") or 0.0), 3),
                    "kind": "caption",
                    "caption": str(window.get("caption") or window.get("text") or "")[:500],
                }
            )
    evidence = deterministic.evidence
    if not evidence and citations:
        nearest = query._nearest_evidence(store, float(citations[0].get("t") or 0.0))
        evidence = [nearest] if nearest else []
    return query.Answer(
        question,
        answer_text,
        citations=citations[:6],
        evidence=evidence,
        confidence=0.65,
        method=method,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        cached=False,
    )

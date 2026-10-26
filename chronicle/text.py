"""Small text helpers for entity resolution in the query extractor."""

from __future__ import annotations

import re
from collections.abc import Iterable

_PUNCT = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "did",
        "do",
        "does",
        "find",
        "for",
        "from",
        "go",
        "had",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "last",
        "left",
        "locate",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "put",
        "remember",
        "see",
        "tell",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "were",
        "where",
        "which",
        "with",
        "you",
        "your",
    }
)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""

    cleaned = _PUNCT.sub(" ", (text or "").lower())
    return _WS.sub(" ", cleaned).strip()


def singular(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_tokens(text: str) -> set[str]:
    tokens = {singular(tok) for tok in normalize(text).split()}
    return {tok for tok in tokens if tok and tok not in STOPWORDS}


def match_score(query: str, label: str) -> float:
    """Heuristic score in ``[0, 1]`` for a free-text query against a label."""

    q_norm = normalize(query)
    l_norm = normalize(label)
    if not q_norm or not l_norm:
        return 0.0
    if q_norm == l_norm:
        return 1.0

    q_tokens = content_tokens(query)
    l_tokens = content_tokens(label)
    if not q_tokens or not l_tokens:
        return 0.0

    if l_norm in q_norm or q_norm in l_norm:
        return 0.85
    if l_tokens.issubset(q_tokens):
        return 0.9
    if q_tokens.issubset(l_tokens):
        return 0.8
    intersection = q_tokens & l_tokens
    if intersection:
        return 0.5 + 0.3 * (len(intersection) / len(q_tokens))
    # token-level fuzzy prefix match ("keychain" vs "keys")
    for q_tok in q_tokens:
        for l_tok in l_tokens:
            short, long = sorted((q_tok, l_tok), key=len)
            if len(short) >= 3 and long.startswith(short):
                return 0.6
    return 0.0


def best_label(query: str, labels: Iterable[str]) -> tuple[str | None, float]:
    best: str | None = None
    best_score = 0.0
    for label in labels:
        score = match_score(query, label)
        if score > best_score:
            best, best_score = label, score
    return best, best_score

"""Small text helpers for entity resolution in the query extractor."""

from __future__ import annotations

import re
from collections.abc import Iterable

_PUNCT = re.compile(r"[^a-z0-9\s-]+")
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


_IRREGULAR_PLURALS = frozenset(
    {"children", "feet", "geese", "men", "mice", "people", "teeth", "women"}
)


def is_plural(label: str) -> bool:
    """Best-effort agreement for detector labels such as ``keys`` or ``glass``.

    Entity labels are open vocabulary, so this intentionally stays a heuristic:
    labels ending in ``ss``/``us``/``is`` are treated as singular (glass, bus,
    analysis) while known irregular plurals and ordinary ``-s`` labels are not.
    """

    word = (label or "").strip().lower()
    if not word:
        return False
    if word in _IRREGULAR_PLURALS:
        return True
    if word.endswith(("ss", "us", "is")):
        return False
    return word.endswith("s")


def content_tokens(text: str) -> set[str]:
    tokens = {singular(tok) for tok in normalize(text).split()}
    return {tok for tok in tokens if tok and tok not in STOPWORDS}


# Open-vocabulary detectors and human questions rarely agree on one label.  Keep
# the alias set deliberately small and symmetric: broadening it (for example
# ``cup -> mug``) would make the existing retrieval contract ambiguous.
_LABEL_ALIASES: dict[str, frozenset[str]] = {
    "backpack": frozenset({"bag", "rucksack", "tote"}),
    "bag": frozenset({"backpack", "rucksack", "tote"}),
    "bottle": frozenset({"flask", "drink"}),
    "box": frozenset({"container"}),
    "car": frozenset({"vehicle", "automobile"}),
    "companion": frozenset({"person", "friend", "people"}),
    "controller": frozenset({"joycon", "joy-con"}),
    "couch": frozenset({"sofa"}),
    "flask": frozenset({"bottle"}),
    "fridge": frozenset({"refrigerator"}),
    "friend": frozenset({"person", "people", "companion", "colleague"}),
    "joy-con": frozenset({"controller", "joycon"}),
    "joycon": frozenset({"controller", "joy-con"}),
    "key": frozenset({"keychain", "keyring", "keys"}),
    "keychain": frozenset({"key", "keyring", "keys"}),
    "keyring": frozenset({"key", "keychain", "keys"}),
    "keys": frozenset({"key", "keychain", "keyring"}),
    "mobile": frozenset({"phone", "smartphone", "cellphone"}),
    "pedestrian": frozenset({"person", "people"}),
    "person": frozenset({"friend", "people", "companion", "pedestrian", "someone", "colleague", "man", "woman"}),
    "people": frozenset({"person", "friend", "companion", "someone"}),
    "phone": frozenset({"mobile", "smartphone", "cellphone"}),
    "purse": frozenset({"wallet"}),
    "refrigerator": frozenset({"fridge"}),
    "smartphone": frozenset({"mobile", "phone", "cellphone"}),
    "sofa": frozenset({"couch"}),
    "someone": frozenset({"person", "friend"}),
    "table": frozenset({"desk", "surface", "counter", "countertop"}),
    "television": frozenset({"tv"}),
    "tv": frozenset({"television"}),
    "vehicle": frozenset({"car"}),
    "wallet": frozenset({"purse"}),
}


def token_aliases(token: str) -> frozenset[str]:
    """Return the singular alias variants for one normalized token."""

    return _LABEL_ALIASES.get(singular(normalize(token)), frozenset())


def expand_token(token: str) -> set[str]:
    base = singular(normalize(token))
    return {base, *token_aliases(base)}


def canonical_token(token: str) -> str:
    """Collapse a token and its aliases to a deterministic representative."""

    base = singular(normalize(token))
    return min({base, *token_aliases(base)})


def canonical_label(label: str) -> str:
    """Canonical label key used to group duplicates and aliases."""

    return " ".join(canonical_token(tok) for tok in normalize(label).split())


def _match(query: str, label: str) -> tuple[float, str]:
    q_norm = normalize(query)
    l_norm = normalize(label)
    if not q_norm or not l_norm:
        return 0.0, "empty"
    if q_norm == l_norm:
        return 1.0, "exact label"

    q_tokens = content_tokens(query)
    l_tokens = content_tokens(label)
    if not q_tokens or not l_tokens:
        return 0.0, "no content tokens"

    if l_norm in q_norm or q_norm in l_norm:
        return 0.85, "label substring"
    if l_tokens.issubset(q_tokens):
        return 0.9, "plural or label subset"
    if q_tokens.issubset(l_tokens):
        return 0.8, "partial label"

    q_expanded = {variant for token in q_tokens for variant in expand_token(token)}
    l_expanded = {variant for token in l_tokens for variant in expand_token(token)}
    if q_expanded & l_expanded:
        return 0.85, "label alias"

    intersection = q_tokens & l_tokens
    if intersection:
        return 0.5 + 0.3 * (len(intersection) / len(q_tokens)), "token overlap"
    # token-level fuzzy prefix match ("keychain" vs "keys")
    for q_tok in q_tokens:
        for l_tok in l_tokens:
            short, long = sorted((q_tok, l_tok), key=len)
            if len(short) >= 3 and long.startswith(short):
                return 0.6, "fuzzy prefix"
    return 0.0, "miss"


def match_details(query: str, label: str) -> tuple[float, str]:
    """Score a query against a label and explain the winning heuristic."""

    return _match(query, label)


def match_score(query: str, label: str) -> float:
    """Heuristic score in ``[0, 1]`` for a free-text query against a label."""

    return _match(query, label)[0]


def best_label(query: str, labels: Iterable[str]) -> tuple[str | None, float]:
    best: str | None = None
    best_score = 0.0
    for label in labels:
        score = match_score(query, label)
        if score > best_score:
            best, best_score = label, score
    return best, best_score

"""Offline evaluation set for GraphRAG answers.

The template path is the reference answer: it is deterministic and only cites
timestamps that exist in the supplied subgraph.  This module scores that path
(and any LLM path) on six question families:

* exact location ("where is X?");
* last-seen time ("when did I last see X?");
* action ("what did I do with X?");
* temporal / multi-hop ("where was X before it was placed on Y?");
* appearance ("what colour is X?", "how big is X?", "describe X");
* unknown / missing detail ("where is my passport?" -> "I don't have that in my memory").

It also reports **citation precision**: every citation in an answer must map
back to a fact in that case's context.  A hallucinated timestamp is a failure,
even if the sentence sounds right.

Run it with::

    python -m chronicle.semantic.evaluation
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .graphrag import Citation, GraphContext, QueryAnswer

CITATION_TOLERANCE = 0.35


@dataclass(frozen=True)
class GraphRAGCase:
    name: str
    query: str
    context: Mapping[str, Any]
    expected_location: str | None = None
    expected_timestamp: float | None = None
    expected_relation: str | None = None
    expect_unknown: bool = False


@dataclass(frozen=True)
class CaseOutcome:
    name: str
    query: str
    passed: bool
    location_ok: bool | None
    timestamp_ok: bool | None
    relation_ok: bool | None
    unknown_ok: bool | None
    grounded: bool
    source: str
    citations: int
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "query": self.query,
            "passed": self.passed,
            "location_ok": self.location_ok,
            "timestamp_ok": self.timestamp_ok,
            "relation_ok": self.relation_ok,
            "unknown_ok": self.unknown_ok,
            "grounded": self.grounded,
            "source": self.source,
            "citations": self.citations,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class EvaluationReport:
    cases: tuple[CaseOutcome, ...]
    citation_total: int
    citation_valid: int
    tolerance: float = CITATION_TOLERANCE

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @staticmethod
    def _rate(values: Iterable[bool | None]) -> float | None:
        relevant = [value for value in values if value is not None]
        return sum(1 for value in relevant if value) / len(relevant) if relevant else None

    @property
    def location_accuracy(self) -> float | None:
        return self._rate(case.location_ok for case in self.cases)

    @property
    def timestamp_accuracy(self) -> float | None:
        return self._rate(case.timestamp_ok for case in self.cases)

    @property
    def relation_accuracy(self) -> float | None:
        return self._rate(case.relation_ok for case in self.cases)

    @property
    def unknown_accuracy(self) -> float | None:
        return self._rate(case.unknown_ok for case in self.cases)

    @property
    def citation_precision(self) -> float:
        return (
            self.citation_valid / self.citation_total if self.citation_total else 1.0
        )

    def failures(self) -> tuple[CaseOutcome, ...]:
        return tuple(case for case in self.cases if not case.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "location_accuracy": _round(self.location_accuracy),
            "timestamp_accuracy": _round(self.timestamp_accuracy),
            "relation_accuracy": _round(self.relation_accuracy),
            "unknown_accuracy": _round(self.unknown_accuracy),
            "citation_precision": round(self.citation_precision, 4),
            "citation_total": self.citation_total,
            "citation_valid": self.citation_valid,
            "cases": [case.to_dict() for case in self.cases],
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"GraphRAG evaluation: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.0%}), citation precision {self.citation_precision:.0%}"
        ]
        for label, value in (
            ("location", self.location_accuracy),
            ("timestamp", self.timestamp_accuracy),
            ("relation", self.relation_accuracy),
            ("unknown", self.unknown_accuracy),
        ):
            if value is not None:
                lines.append(f"  {label:<9} {value:.0%}")
        for case in self.failures():
            lines.append(
                f"  FAIL {case.name}: {case.answer!r} "
                f"(location={case.location_ok}, time={case.timestamp_ok}, "
                f"relation={case.relation_ok}, unknown={case.unknown_ok})"
            )
        return lines


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _context_of(case: GraphRAGCase) -> GraphContext:
    return GraphContext.from_mapping(case.context)


def _citation_is_valid(citation: Citation, context: GraphContext, tolerance: float) -> bool:
    if citation.timestamp is None:
        return False
    timestamp = float(citation.timestamp)
    if (
        context.last_seen is not None
        and abs(timestamp - float(context.last_seen)) <= tolerance
    ):
        return True
    return any(
        fact.timestamp is not None
        and abs(float(fact.timestamp) - timestamp) <= tolerance
        for fact in context.facts
    )


def _location_hit(answer: QueryAnswer, expected: str) -> bool:
    needle = expected.casefold()
    if needle in answer.answer.casefold():
        return True
    return any(
        citation.zone is not None and needle in citation.zone.casefold()
        for citation in answer.citations
    )


def _timestamp_hit(answer: QueryAnswer, expected: float, tolerance: float) -> bool:
    return any(
        abs(float(citation.timestamp) - float(expected)) <= tolerance
        for citation in answer.citations
        if citation.timestamp is not None
    )


def _relation_hit(answer: QueryAnswer, expected: str) -> bool:
    needle = expected.upper()
    if needle in answer.answer.upper():
        return True
    return any(
        citation.relation is not None and citation.relation.upper() == needle
        for citation in answer.citations
    )


def _unknown_hit(answer: QueryAnswer) -> bool:
    text = answer.answer.casefold()
    return not answer.grounded and (
        "don't have" in text or "do not have" in text or "no memory" in text
    )


def evaluate_graphrag(
    answerer: Any,
    cases: Sequence[GraphRAGCase] | None = None,
    *,
    tolerance: float = CITATION_TOLERANCE,
) -> EvaluationReport:
    """Run ``cases`` through a ``GraphRAG``/``SemanticEngine``/callable."""
    selected = tuple(cases or DEFAULT_GRAPHRAG_CASES)
    if hasattr(answerer, "answer"):
        answer_fn = answerer.answer
    elif hasattr(answerer, "graphrag") and hasattr(answerer.graphrag, "answer"):
        answer_fn = answerer.graphrag.answer
    elif callable(answerer):
        answer_fn = answerer
    else:
        raise TypeError("answerer must be GraphRAG, SemanticEngine or a callable")

    outcomes: list[CaseOutcome] = []
    citation_total = 0
    citation_valid = 0
    for case in selected:
        context = _context_of(case)
        answer: QueryAnswer = answer_fn(case.query, case.context)
        location_ok = (
            _location_hit(answer, case.expected_location)
            if case.expected_location
            else None
        )
        timestamp_ok = (
            _timestamp_hit(answer, case.expected_timestamp, tolerance)
            if case.expected_timestamp is not None
            else None
        )
        relation_ok = (
            _relation_hit(answer, case.expected_relation)
            if case.expected_relation
            else None
        )
        unknown_ok = _unknown_hit(answer) if case.expect_unknown else None

        citation_total += len(answer.citations)
        citation_valid += sum(
            1
            for citation in answer.citations
            if _citation_is_valid(citation, context, tolerance)
        )

        checks = [
            value
            for value in (location_ok, timestamp_ok, relation_ok, unknown_ok)
            if value is not None
        ]
        outcomes.append(
            CaseOutcome(
                name=case.name,
                query=case.query,
                passed=all(checks) if checks else True,
                location_ok=location_ok,
                timestamp_ok=timestamp_ok,
                relation_ok=relation_ok,
                unknown_ok=unknown_ok,
                grounded=answer.grounded,
                source=answer.source,
                citations=len(answer.citations),
                answer=answer.answer,
            )
        )

    return EvaluationReport(
        cases=tuple(outcomes),
        citation_total=citation_total,
        citation_valid=citation_valid,
        tolerance=tolerance,
    )


_KEYS_CONTEXT: Mapping[str, Any] = {
    "entity": {
        "id": "keys",
        "label": "keys",
        "state": "CONTAINED",
        "zone": "work desk",
        "last_seen": 16.4,
    },
    "facts": [
        {
            "subject": "keys",
            "relation": "RESTING_ON",
            "object": "work desk",
            "timestamp": 12.0,
            "zone": "work desk",
            "confidence": 0.91,
        },
        {
            "subject": "user",
            "relation": "PICK_UP",
            "object": "keys",
            "timestamp": 14.2,
            "zone": "work desk",
            "confidence": 0.88,
        },
        {
            "subject": "keys",
            "relation": "INSERT_INTO",
            "object": "drawer",
            "timestamp": 16.4,
            "zone": "work desk",
            "confidence": 0.86,
        },
    ],
}

_MUG_CONTEXT: Mapping[str, Any] = {
    "entity": {"id": "mug", "label": "mug", "zone": "work desk"},
    "facts": [
        {
            "subject": "mug",
            "relation": "RESTING_ON",
            "object": "work desk",
            "timestamp": 12.0,
            "zone": "work desk",
            "confidence": 0.9,
        },
        {
            "subject": "user",
            "relation": "PICK_UP",
            "object": "mug",
            "timestamp": 14.2,
            "zone": "work desk",
            "confidence": 0.87,
        },
    ],
}

_MUG_APPEARANCE_CONTEXT: Mapping[str, Any] = {
    "entity": {"id": "mug", "label": "mug", "zone": "work desk"},
    "attributes": {
        "colour": "red",
        "colour_hex": "#c0392b",
        "colour_confidence": 0.88,
        "shape": "round",
        "aspect_ratio": 1.0,
        "size": "medium",
        "relative_area": 0.08,
        "texture": "smooth",
        "brightness": "bright",
        "saturation": "vivid",
        "position": "center",
        "vertical_band": "middle",
    },
    "facts": [
        {
            "subject": "mug",
            "relation": "RESTING_ON",
            "object": "work desk",
            "timestamp": 12.0,
            "zone": "work desk",
            "confidence": 0.9,
        },
        {
            "subject": "user",
            "relation": "PICK_UP",
            "object": "mug",
            "timestamp": 14.2,
            "zone": "work desk",
            "confidence": 0.87,
        },
    ],
}

_CHARGER_CONTEXT: Mapping[str, Any] = {
    "entity": {"id": "charger", "label": "charger", "zone": "bedroom"},
    "facts": [
        {
            "subject": "charger",
            "relation": "REMOVE_FROM",
            "object": "drawer",
            "timestamp": 40.0,
            "zone": "bedroom",
            "confidence": 0.83,
        }
    ],
}

_KEYS_LOST_CONTEXT: Mapping[str, Any] = {
    "entity": {
        "id": "keys",
        "label": "keys",
        "state": "LOST",
        "zone": "work desk",
    },
    "facts": [
        {
            "subject": "keys",
            "relation": "INSERT_INTO",
            "object": "drawer",
            "timestamp": 16.4,
            "zone": "work desk",
            "confidence": 0.86,
        }
    ],
}

_ZONE_ONLY_CONTEXT: Mapping[str, Any] = {
    "entity": {
        "id": "charger",
        "label": "charger",
        "zone": "bedroom",
        "last_seen": 30.0,
    },
    "facts": [],
}

_CUP_CONTEXT: Mapping[str, Any] = {
    "entity": {"id": "cup", "label": "cup", "zone": "kitchen"},
    "facts": [
        {
            "subject": "cup",
            "relation": "RESTING_ON",
            "object": "work desk",
            "timestamp": 5.0,
            "zone": "kitchen",
            "confidence": 0.9,
        },
        {
            "subject": "user",
            "relation": "PICK_UP",
            "object": "cup",
            "timestamp": 8.0,
            "zone": "kitchen",
            "confidence": 0.88,
        },
        {
            "subject": "cup",
            "relation": "PLACED",
            "object": "table",
            "timestamp": 10.0,
            "zone": "kitchen",
            "confidence": 0.86,
        },
    ],
}

DEFAULT_GRAPHRAG_CASES: tuple[GraphRAGCase, ...] = (
    GraphRAGCase(
        name="exact-location",
        query="Where are my keys?",
        context=_KEYS_CONTEXT,
        expected_location="drawer",
        expected_timestamp=16.4,
        expected_relation="INSERT_INTO",
    ),
    GraphRAGCase(
        name="action",
        query="What did I do with the keys?",
        context=_KEYS_CONTEXT,
        expected_timestamp=16.4,
        expected_relation="INSERT_INTO",
    ),
    GraphRAGCase(
        name="last-seen-time",
        query="When did I last see the mug?",
        context=_MUG_CONTEXT,
        expected_timestamp=14.2,
        expected_relation="PICK_UP",
    ),
    GraphRAGCase(
        name="remove-from",
        query="Where is the charger?",
        context=_CHARGER_CONTEXT,
        expected_location="drawer",
        expected_timestamp=40.0,
        expected_relation="REMOVE_FROM",
    ),
    GraphRAGCase(
        name="lost-state",
        query="Where are my keys?",
        context=_KEYS_LOST_CONTEXT,
        expected_location="drawer",
        expected_timestamp=16.4,
        expected_relation="INSERT_INTO",
    ),
    GraphRAGCase(
        name="zone-only",
        query="Where is the charger?",
        context=_ZONE_ONLY_CONTEXT,
        expected_location="bedroom",
        expected_timestamp=30.0,
    ),
    GraphRAGCase(
        name="before-placement",
        query="Where was the cup before it was placed on the table?",
        context=_CUP_CONTEXT,
        expected_location="work desk",
        expected_timestamp=5.0,
        expected_relation="RESTING_ON",
    ),
    GraphRAGCase(
        name="appearance-colour",
        query="What colour is the mug?",
        context=_MUG_APPEARANCE_CONTEXT,
        expected_location="red",
        expected_timestamp=14.2,
        expected_relation="APPEARANCE",
    ),
    GraphRAGCase(
        name="appearance-size",
        query="How big is the mug?",
        context=_MUG_APPEARANCE_CONTEXT,
        expected_location="medium",
        expected_timestamp=14.2,
        expected_relation="APPEARANCE",
    ),
    GraphRAGCase(
        name="appearance-describe",
        query="Describe the mug.",
        context=_MUG_APPEARANCE_CONTEXT,
        expected_location="red",
        expected_timestamp=14.2,
        expected_relation="APPEARANCE",
    ),
    GraphRAGCase(
        name="appearance-missing",
        query="What colour is the passport?",
        context={"entity": {"id": "passport", "label": "passport"}},
        expect_unknown=True,
    ),
    GraphRAGCase(
        name="unknown-entity",
        query="Where is my passport?",
        context={"entity": {"id": "passport", "label": "passport"}},
        expect_unknown=True,
    ),
    GraphRAGCase(
        name="before-placement",
        query="Where were the keys before they were placed in the drawer?",
        context=_KEYS_CONTEXT,
        expected_location="work desk",
        expected_timestamp=12.0,
        expected_relation="RESTING_ON",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    from .config import SemanticConfig
    from .engine import SemanticEngine

    parser = argparse.ArgumentParser(description="Run the offline GraphRAG evaluation set")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument(
        "--cases", default=None, help="path to a JSON list of GraphRAGCase objects"
    )
    args = parser.parse_args(argv)

    cases: tuple[GraphRAGCase, ...] | None = None
    if args.cases:
        with open(args.cases, encoding="utf-8") as handle:
            payload = json.load(handle)
        cases = tuple(
            GraphRAGCase(
                name=str(item["name"]),
                query=str(item["query"]),
                context=item["context"],
                expected_location=item.get("expected_location"),
                expected_timestamp=item.get("expected_timestamp"),
                expected_relation=item.get("expected_relation"),
                expect_unknown=bool(item.get("expect_unknown", False)),
            )
            for item in payload
        )

    engine = SemanticEngine(
        config=SemanticConfig(llm_backend="template"),
        chat_client=None,
        embedder_backend="deterministic",
        captioner_backend="stub",
    )
    report = evaluate_graphrag(engine, cases)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for line in report.summary_lines():
            print(line)
    return 0 if report.pass_rate == 1.0 and report.citation_precision == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

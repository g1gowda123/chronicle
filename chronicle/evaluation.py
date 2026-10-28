"""Deterministic end-to-end evaluation of the model-free Chronicle path.

This is intentionally a small product smoke suite, not a research benchmark.
It answers fixed questions against the bundled demo memory and checks for
grounded phrases, timestamps, methods and honest abstentions.  No model, network
access, LLM judge, or GPU is involved, so the result is reproducible in CI.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chronicle.demo_data import demo_db_path


@dataclass(frozen=True)
class EvaluationCase:
    """One deterministic question and its expected answer properties."""

    id: str
    category: str
    question: str
    expect_all: tuple[str, ...] = ()
    expect_any: tuple[str, ...] = ()
    expect_none: tuple[str, ...] = ()
    expect_method: str | None = None
    tests: str = ""


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    answer: str
    method: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    expected_method: str | None = None
    tests: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return " ".join(text.split())


def load_cases(path: Path | str) -> list[EvaluationCase]:
    """Load a JSONL suite, validating required fields and unique ids."""

    suite_path = Path(path)
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    with suite_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{suite_path}:{line_number}: invalid JSON: {exc}") from exc
            try:
                case = EvaluationCase(
                    id=str(payload["id"]),
                    category=str(payload["category"]),
                    question=str(payload["question"]),
                    expect_all=tuple(payload.get("expect_all", ())),
                    expect_any=tuple(payload.get("expect_any", ())),
                    expect_none=tuple(payload.get("expect_none", ())),
                    expect_method=payload.get("expect_method"),
                    tests=str(payload.get("tests", "")),
                )
            except KeyError as exc:
                raise ValueError(
                    f"{suite_path}:{line_number}: missing field {exc.args[0]!r}"
                ) from exc
            if case.id in seen:
                raise ValueError(f"{suite_path}:{line_number}: duplicate id {case.id!r}")
            if not case.question.strip():
                raise ValueError(f"{suite_path}:{line_number}: question is empty")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"{suite_path}: suite is empty")
    return cases


def score_case(case: EvaluationCase, answer: str, method: str) -> CaseResult:
    """Score one answer using exact substring/method checks."""

    normalised = _normalise(answer)
    failures: list[str] = []
    for phrase in case.expect_all:
        if _normalise(phrase) not in normalised:
            failures.append(f"missing {phrase!r}")
    if case.expect_any and not any(
        _normalise(phrase) in normalised for phrase in case.expect_any
    ):
        failures.append("missing any of " + ", ".join(repr(p) for p in case.expect_any))
    for phrase in case.expect_none:
        if _normalise(phrase) in normalised:
            failures.append(f"unexpected {phrase!r}")
    if case.expect_method is not None and method != case.expect_method:
        failures.append(f"method {method!r} != {case.expect_method!r}")

    return CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        answer=answer,
        method=method,
        passed=not failures,
        failures=failures,
        expected_method=case.expect_method,
        tests=case.tests,
    )


def evaluate_cases(cases: Iterable[EvaluationCase], runtime: Any) -> dict[str, Any]:
    """Run cases against an already-constructed runtime."""

    results: list[CaseResult] = []
    for case in cases:
        runtime_answer = runtime.ask(case.question)
        results.append(
            score_case(case, runtime_answer.answer, getattr(runtime_answer, "method", ""))
        )

    categories: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = categories.setdefault(result.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)

    total = len(results)
    passed = sum(result.passed for result in results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "score": passed / total if total else 0.0,
        "by_category": categories,
        "cases": [result.to_dict() for result in results],
    }


def run_evaluation(
    db_path: Path | str | None = None,
    suite_path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate the bundled demo memory (or an explicit database)."""

    from chronicle.integration import ChronicleRuntime
    from chronicle.semantic.config import SemanticConfig

    if suite_path is None:
        suite_path = Path(__file__).with_name("evaluation_data") / "demo_questions.jsonl"
    memory_path = Path(db_path) if db_path is not None else demo_db_path()
    if not Path(memory_path).is_file():
        raise FileNotFoundError(f"memory database not found: {memory_path}")

    cases = load_cases(suite_path)
    runtime = ChronicleRuntime(
        memory_path,
        semantic_config=SemanticConfig(llm_backend="template"),
        load=True,
        read_only=True,
    )
    try:
        report = evaluate_cases(cases, runtime)
    finally:
        runtime.close()

    report["suite"] = str(Path(suite_path).resolve())
    report["db"] = str(Path(memory_path).resolve())
    report["kind"] = "deterministic product smoke suite"
    return report


def format_report(report: dict[str, Any]) -> str:
    """Human-readable report; the final line is always the score."""

    lines = [
        "Chronicle product evaluation",
        f"Suite: {report['suite']}",
        f"Memory: {report['db']}",
        "This is a deterministic smoke suite, not a research benchmark.",
        "",
    ]
    for result in report["cases"]:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"[{status}] {result['id']}: {result['question']}")
        lines.append(f"       {result['answer']}")
        if result["failures"]:
            lines.append("       " + "; ".join(result["failures"]))
    lines.append("")
    for category, counts in sorted(report["by_category"].items()):
        lines.append(
            f"  {category:<12} {counts['passed']}/{counts['total']}"
        )
    lines.append(
        f"Score: {report['passed']}/{report['total']} "
        f"({report['score']:.1%})"
    )
    return "\n".join(lines)


__all__ = [
    "CaseResult",
    "EvaluationCase",
    "evaluate_cases",
    "format_report",
    "load_cases",
    "run_evaluation",
    "score_case",
]

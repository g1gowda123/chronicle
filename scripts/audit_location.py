"""Audit whether location answers name the queried support or only the zone.

The official matcher accepts every alias listed in the QA file.  For Cook that
includes the zone ("Kitchen"), so a sentence such as "knife is in the Kitchen"
passes a question whose gold support is "countertop".  This audit keeps the
official outcome and adds a stricter check: when the gold answer is not the
zone itself, the prediction must contain the gold support noun.

Example
-------
    python scripts/audit_location.py \
        --run runs/research/final_cook/results.json \
        --qa data/qa/cook_45s.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def audit(
    rows: Sequence[dict[str, Any]],
    questions: Sequence[dict[str, Any]],
    *,
    zone_label: str | None,
) -> dict[str, Any]:
    by_query = {str(q.get("query", "")): q for q in questions}
    zone = (zone_label or "").strip().lower()
    records: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("category")) != "object_location":
            continue
        question = by_query.get(str(row.get("query", "")))
        if question is None:
            continue
        gold = str(question.get("answer", "")).strip()
        predicted = str(row.get("predicted", ""))
        gold_lower = gold.lower()
        # A zone-level gold answer is already the strict check.
        requires_support = bool(gold_lower) and gold_lower != zone
        strict = gold_lower in predicted.lower() if requires_support else bool(row.get("correct"))
        records.append(
            {
                "query": row.get("query"),
                "gold": gold,
                "predicted": predicted,
                "official": bool(row.get("correct")),
                "strict_support": strict,
                "requires_support": requires_support,
            }
        )
    return {
        "total": len(records),
        "official_correct": sum(1 for r in records if r["official"]),
        "strict_correct": sum(1 for r in records if r["strict_support"]),
        "records": records,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Location-answer audit",
        "",
        f"Official correct: {report['official_correct']}/{report['total']}",
        f"Strict support correct: {report['strict_correct']}/{report['total']}",
        "",
        "| Question | Gold | Official | Strict | Prediction |",
        "|---|---|:--:|:--:|---|",
    ]
    for row in report["records"]:
        prediction = str(row["predicted"]).replace("|", "\\|")
        lines.append(
            f"| {row['query']} | {row['gold']} | "
            f"{'yes' if row['official'] else 'no'} | "
            f"{'yes' if row['strict_support'] else 'no'} | {prediction} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = json.loads(args.run.read_text(encoding="utf-8"))
    qa = json.loads(args.qa.read_text(encoding="utf-8"))
    report = audit(
        result.get("rows") or (),
        qa.get("questions") or (),
        zone_label=qa.get("zone_label"),
    )
    text = markdown(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

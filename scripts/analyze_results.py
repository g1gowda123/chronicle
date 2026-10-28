"""Statistical analysis for Chronicle evaluation runs.

The evaluation suites in this repository are small (8-10 questions per
dataset), so a bare percentage hides enormous uncertainty.  This script adds
the reporting a paper needs:

* exact (Clopper-Pearson) and Wilson confidence intervals for Recall@1,
* paired bootstrap confidence intervals for the difference between two runs,
* an exact McNemar test over the questions both runs answered, and
* a per-category breakdown.

Runs are aligned by ``(video_id, query)`` so two systems are compared on the
same questions even if row order differs.  Missing questions count as
incorrect for the system that does not have them.

Example
-------
    python scripts/analyze_results.py \
        --run baseline=runs/research/leakfree_holo/results.json \
        --run planner=runs/research/planner_holo/results.json \
        --run prompts=runs/research/prompts_holo/results.json \
        --out runs/research/stats_holo
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # SciPy is available in the evaluation environment; keep a fallback.
    from scipy.stats import beta as _beta_dist
except Exception:  # pragma: no cover - optional dependency
    _beta_dist = None


@dataclass(frozen=True)
class Run:
    label: str
    path: Path
    rows: tuple[dict[str, Any], ...]
    dataset_name: str = ""

    @property
    def dataset(self) -> str:
        if self.dataset_name:
            return self.dataset_name
        video = str(self.rows[0].get("video_id", "")) if self.rows else ""
        return video or self.path.parent.name

    @property
    def correct(self) -> int:
        return sum(1 for row in self.rows if row.get("correct"))

    @property
    def total(self) -> int:
        return len(self.rows)


def load_run(label: str, path: Path) -> Run:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = tuple(payload.get("rows") or ())
    return Run(
        label=label,
        path=path,
        rows=rows,
        dataset_name=str(payload.get("dataset", "")),
    )


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("video_id", "")), str(row.get("query", "")))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval; falls back to Wilson without SciPy."""

    if n <= 0:
        return 0.0, 1.0
    if _beta_dist is None:  # pragma: no cover - SciPy is present in CI
        return wilson(k, n, alpha)
    lower = 0.0 if k == 0 else float(_beta_dist.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(_beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = 1.959963984540054 if alpha == 0.05 else _norm_ppf(1 - alpha / 2)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _norm_ppf(p: float) -> float:
    """Acklam's inverse normal CDF (only used if SciPy is absent)."""

    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    samples: int = 10000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Bootstrap CI for accuracy(right) - accuracy(left) over questions."""

    if not left or len(left) != len(right):
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(left)
    deltas: list[float] = []
    for _ in range(samples):
        indexes = [rng.randrange(n) for _ in range(n)]
        deltas.append(
            sum(right[i] for i in indexes) / n - sum(left[i] for i in indexes) / n
        )
    deltas.sort()
    observed = sum(right) / n - sum(left) / n
    low = deltas[int(0.025 * samples)]
    high = deltas[min(samples - 1, int(0.975 * samples))]
    return observed, low, high


def categories(run: Run) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in run.rows:
        buckets.setdefault(str(row.get("category", "unknown")), []).append(row)
    output: dict[str, dict[str, float | int]] = {}
    for name, rows in sorted(buckets.items()):
        correct = sum(1 for row in rows if row.get("correct"))
        output[name] = {
            "correct": correct,
            "total": len(rows),
            "recall_at_1": correct / len(rows) if rows else 0.0,
        }
    return output


def summarise(run: Run) -> dict[str, Any]:
    low, high = clopper_pearson(run.correct, run.total)
    wlow, whigh = wilson(run.correct, run.total)
    return {
        "label": run.label,
        "dataset": run.dataset,
        "correct": run.correct,
        "total": run.total,
        "recall_at_1": run.correct / run.total if run.total else 0.0,
        "cp95": [low, high],
        "wilson95": [wlow, whigh],
        "categories": categories(run),
    }


def compare(left: Run, right: Run) -> dict[str, Any]:
    left_rows = {_key(row): bool(row.get("correct")) for row in left.rows}
    right_rows = {_key(row): bool(row.get("correct")) for row in right.rows}
    keys = sorted(set(left_rows) | set(right_rows))
    left_vec = [left_rows.get(key, False) for key in keys]
    right_vec = [right_rows.get(key, False) for key in keys]
    only_right = sum(1 for key in keys if right_rows.get(key) and not left_rows.get(key))
    only_left = sum(1 for key in keys if left_rows.get(key) and not right_rows.get(key))
    observed, low, high = paired_bootstrap(left_vec, right_vec)
    return {
        "left": left.label,
        "right": right.label,
        "questions": len(keys),
        "right_only": only_right,
        "left_only": only_left,
        "accuracy_delta": observed,
        "bootstrap95": [low, high],
        "mcnemar_exact_p": exact_mcnemar(only_right, only_left),
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown(runs: Sequence[Run], comparisons: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Statistical analysis",
        "",
        "Exact (Clopper-Pearson) 95% intervals.  The paired bootstrap resamples",
        "questions; McNemar uses only questions where the two runs disagree.",
        "With 8-10 questions these intervals are wide by construction.",
        "",
        "## Recall@1 with uncertainty",
        "",
        "| Run | Dataset | Correct | Recall@1 | 95% CI (exact) | 95% CI (Wilson) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in runs:
        info = summarise(run)
        lines.append(
            f"| {run.label} | {run.dataset} | {run.correct}/{run.total} | "
            f"{_pct(info['recall_at_1'])} | "
            f"[{_pct(info['cp95'][0])}, {_pct(info['cp95'][1])}] | "
            f"[{_pct(info['wilson95'][0])}, {_pct(info['wilson95'][1])}] |"
        )
    if comparisons:
        lines += [
            "",
            "## Paired comparisons",
            "",
            "| Comparison | Delta | Bootstrap 95% CI | Discordant (right only / left only) | McNemar p |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in comparisons:
            lines.append(
                f"| {item['right']} - {item['left']} | "
                f"{_pct(item['accuracy_delta'])} | "
                f"[{_pct(item['bootstrap95'][0])}, {_pct(item['bootstrap95'][1])}] | "
                f"{item['right_only']} / {item['left_only']} | "
                f"{item['mcnemar_exact_p']:.3f} |"
            )
    lines += ["", "## Per-category Recall@1", ""]
    for run in runs:
        lines.append(f"### {run.label}")
        lines.append("")
        lines.append("| Category | Correct | Recall@1 |")
        lines.append("|---|---:|---:|")
        for name, info in categories(run).items():
            lines.append(
                f"| {name} | {info['correct']}/{info['total']} | {_pct(float(info['recall_at_1']))} |"
            )
        lines.append("")

    lines += [
        "## Per-question outcomes",
        "",
        "| Run | Question | Correct | Prediction |",
        "|---|---|---|---|",
    ]
    for run in runs:
        for row in run.rows:
            prediction = str(row.get("predicted", "")).replace("|", "\\|")
            question = str(row.get("query", "")).replace("|", "\\|")
            lines.append(
                f"| {run.label} | {question} | "
                f"{'yes' if row.get('correct') else 'no'} | {prediction} |"
            )
    lines.append("")
    return "\n".join(lines)


def analyze(runs: Sequence[Run], *, compare_to: str | None = None) -> dict[str, Any]:
    summaries = [summarise(run) for run in runs]
    comparisons: list[dict[str, Any]] = []
    baseline = next((run for run in runs if run.label == compare_to), runs[0] if runs else None)
    if baseline is not None:
        for run in runs:
            if run is baseline:
                continue
            if run.dataset != baseline.dataset:
                continue
            comparisons.append(compare(baseline, run))
    return {
        "runs": summaries,
        "comparisons": comparisons,
        "categories": {run.label: categories(run) for run in runs},
    }


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use --run label=path/to/results.json")
    label, path = value.split("=", 1)
    return label.strip(), Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument(
        "--compare-to",
        default=None,
        help="label of the baseline run (default: first --run)",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/research/stats"))
    args = parser.parse_args(argv)

    runs = [load_run(label, path) for label, path in args.run]
    payload = analyze(runs, compare_to=args.compare_to)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Rebuild the markdown from the loaded runs so the per-question table is present.
    (args.out / "stats.md").write_text(
        markdown(runs, payload["comparisons"]), encoding="utf-8"
    )
    print(markdown(runs, payload["comparisons"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

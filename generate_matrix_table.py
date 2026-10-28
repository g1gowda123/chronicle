#!/usr/bin/env python3
"""Build a supplementary matrix table across evaluation runs.

Walks one or more run directories and pairs every ``results.json`` with the
matching ``baseline_results.json``.  Configuration labels are decoded from the
run-name suffixes written by ``scripts/run_paper_eval.sh``:

    _ft   Florence-2 + template answers
    _fq   Florence-2 + Qwen2.5 answers
    _qt   Qwen2.5-VL + template answers
    _qq   Qwen2.5-VL + Qwen2.5 answers

Usage:
    python generate_matrix_table.py \
        --run-dir runs/bench \
        --run-dir runs/bench_holo \
        --out paper/table_matrix.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUFFIXES = {
    "_ft": ("Florence-2", "template"),
    "_fq": ("Florence-2", "Qwen2.5"),
    "_qt": ("Qwen2.5-VL", "template"),
    "_qq": ("Qwen2.5-VL", "Qwen2.5"),
}

DATASETS = {
    "aria_cook_45s": "Cook 45 s",
    "holoassist_switch_34s": "HoloAssist 34 s",
}


def _load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _split_name(stem: str) -> tuple[str, str, str, str]:
    """Return (dataset, condition, captioner, answerer) for a run stem."""

    base = stem.removeprefix("baseline_")
    dataset = base
    suffix = ""
    for candidate in SUFFIXES:
        if base.endswith(candidate):
            dataset = base[: -len(candidate)]
            suffix = candidate
            break
    captioner, answerer = SUFFIXES.get(suffix, ("Florence-2", "template"))
    condition = "Chronicle" if not stem.startswith("baseline_") else "Dense 1 FPS"
    return dataset, condition, captioner, answerer


def _metric(payload: dict[str, Any], *keys: str) -> Any:
    metrics = payload.get("metrics") or {}
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _row(dataset: str, payload: dict[str, Any], baseline: bool) -> dict[str, Any]:
    name = f"baseline_{dataset}" if baseline else dataset
    dataset_name, condition, captioner, answerer = _split_name(name)
    recall = payload.get("recall_at_1")
    if recall is None:
        recall = payload.get("recall")
    latency = _metric(payload, "avg_latency_steady_ms", "avg_latency_ms")
    vram = _metric(payload, "peak_gpu_used_mb", "peak_vram_mb")
    return {
        "dataset": DATASETS.get(dataset_name, dataset_name),
        "system": condition,
        "captioner": captioner,
        "answerer": answerer,
        "recall_at_1": float(recall) if recall is not None else None,
        "vlm_calls": _metric(payload, "vlm_calls"),
        "avg_latency_ms": float(latency) if latency is not None else None,
        "peak_vram_mb": float(vram) if vram is not None else None,
        "frames": _metric(payload, "frames_processed"),
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "---"
    return f"{float(value):.{digits}f}"


def _latex(rows: list[dict[str, Any]], caption: str, label: str) -> str:
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table*}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\small",
        "  \\begin{tabular}{llllcccr}",
        "    \\toprule",
        "    Dataset & System & Captioner & Answerer & Recall@1 & VLM Calls"
        " & Latency (ms) & VRAM (MB) \\\\",
        "    \\midrule",
    ]
    previous = None
    for row in rows:
        if previous is not None and row["dataset"] != previous:
            lines.append("    \\midrule")
        previous = row["dataset"]
        recall = row["recall_at_1"]
        lines.append(
            "    "
            + " & ".join(
                [
                    row["dataset"],
                    row["system"],
                    row["captioner"],
                    row["answerer"],
                    "---" if recall is None else f"{100.0 * recall:.1f}\\%",
                    _fmt(row["vlm_calls"], 0),
                    _fmt(row["avg_latency_ms"]),
                    _fmt(row["peak_vram_mb"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | System | Captioner | Answerer | Recall@1 | VLM Calls "
        "| Latency (ms) | VRAM (MB) |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        recall = row["recall_at_1"]
        lines.append(
            "| {dataset} | {system} | {captioner} | {answerer} | {recall} | "
            "{calls} | {latency} | {vram} |".format(
                recall="---" if recall is None else f"{100.0 * recall:.1f}%",
                calls=_fmt(row["vlm_calls"], 0),
                latency=_fmt(row["avg_latency_ms"]),
                vram=_fmt(row["peak_vram_mb"]),
                **row,
            )
        )
    return "\n".join(lines) + "\n"


def collect(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in run_dirs:
        for path in sorted(root.rglob("results.json")):
            if path in seen:
                continue
            seen.add(path)
            payload = _load(path)
            if payload is None or "recall_at_1" not in payload:
                continue
            rows.append(_row(path.parent.name, payload, baseline=False))
        for path in sorted(root.rglob("baseline_results.json")):
            payload = _load(path)
            if payload is None:
                continue
            rows.append(_row(path.parent.name.removeprefix("baseline_"), payload, True))
    rows.sort(key=lambda row: (row["dataset"], row["captioner"], row["answerer"], row["system"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Run directory to scan (repeatable). Defaults to runs/bench.",
    )
    parser.add_argument("--out", type=Path, default=Path("paper/table_matrix.tex"))
    parser.add_argument(
        "--caption",
        default="Model matrix: event-gated Chronicle versus dense 1 FPS captions.",
    )
    parser.add_argument("--label", default="tab:model-matrix")
    args = parser.parse_args()

    run_dirs = args.run_dir or [Path("runs/bench")]
    rows = collect(run_dirs)
    if not rows:
        raise SystemExit(f"no evaluation results found under {run_dirs}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_latex(rows, args.caption, args.label), encoding="utf-8")
    args.out.with_suffix(".md").write_text(_markdown(rows), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"caption": args.caption, "label": args.label, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(_markdown(rows))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

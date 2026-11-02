#!/usr/bin/env python3
"""Generate paper-ready comparison assets from evaluation JSON files.

Reads:
    runs/eval/cook/results.json              (Chronicle, Task 3)
    runs/eval/baseline_cook/baseline_results.json  (Dense 1 FPS, Task 4)

Writes:
    paper/table_comparison.tex
    paper/table_comparison.md
    paper/table_comparison.json

The LaTeX table matches the requested columns:

    | Method | Recall@1 | VLM Calls | Avg Latency (ms) | Peak VRAM (MB) |

Usage:
    python generate_paper_assets.py \
      --chronicle runs/eval/cook/results.json \
      --baseline runs/eval/baseline_cook/baseline_results.json \
      --out paper/table_comparison.tex
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_CHRONICLE = Path("runs/eval/cook/results.json")
DEFAULT_BASELINE = Path("runs/eval/baseline_cook/baseline_results.json")
DEFAULT_OUT = Path("paper/table_comparison.tex")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _metric(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _recall(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("recall_at_1")
    if value is None:
        value = _metric(payload, "metrics", "recall_at_1")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _vlm_calls(payload: Mapping[str, Any]) -> int | None:
    value = _metric(payload, "metrics", "vlm_calls")
    if value is None:
        value = payload.get("vlm_calls")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _latency(payload: Mapping[str, Any]) -> float | None:
    for key in ("avg_latency_steady_ms", "avg_latency_ms"):
        value = _metric(payload, "metrics", key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _vram(payload: Mapping[str, Any]) -> float | None:
    # Prefer device-wide usage: it also includes a Qwen captioner served by a
    # separate Ollama process. Fall back to torch's per-process peak.
    for key in ("peak_gpu_used_mb", "peak_vram_mb"):
        value = _metric(payload, "metrics", key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _fmt_pct(value: float | None) -> str:
    return "---" if value is None else f"{100.0 * value:.1f}\\%"


def _fmt_int(value: int | None) -> str:
    return "---" if value is None else str(value)


def _fmt_float(value: float | None, *, digits: int = 1) -> str:
    return "---" if value is None else f"{value:.{digits}f}"


def _row(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    return {
        "method": name,
        "recall_at_1": _recall(payload),
        "vlm_calls": _vlm_calls(payload),
        "avg_latency_ms": _latency(payload),
        "peak_vram_mb": _vram(payload),
        "frames_processed": metrics.get("frames_processed"),
        "vlm_model_calls": metrics.get("vlm_model_calls"),
        "vlm_call_reduction_pct": metrics.get("vlm_call_reduction_pct"),
        "source": payload.get("video"),
    }


def _latex(rows: Sequence[Mapping[str, Any]], *, caption: str, label: str) -> str:
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    Method & Recall@1 & VLM Calls & Avg Latency (ms) & Peak VRAM (MB) \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            + " & ".join(
                [
                    str(row["method"]),
                    _fmt_pct(row.get("recall_at_1")),
                    _fmt_int(row.get("vlm_calls")),
                    _fmt_float(row.get("avg_latency_ms")),
                    _fmt_float(row.get("peak_vram_mb")),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Method | Recall@1 | VLM Calls | Avg Latency (ms) | Peak VRAM (MB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        recall = row.get("recall_at_1")
        lines.append(
            f"| {row['method']} | "
            f"{'---' if recall is None else f'{100.0 * float(recall):.1f}%'} | "
            f"{_fmt_int(row.get('vlm_calls'))} | "
            f"{_fmt_float(row.get('avg_latency_ms'))} | "
            f"{_fmt_float(row.get('peak_vram_mb'))} |"
        )
    return "\n".join(lines) + "\n"


def build_assets(
    chronicle_path: Path,
    baseline_path: Path,
    *,
    out_path: Path = DEFAULT_OUT,
    chronicle_label: str = "Chronicle (event-gated)",
    baseline_label: str = "Dense 1 FPS (Florence-2)",
    caption: str = "Episodic memory QA under a fixed local model budget.",
    label: str = "tab:chronicle-vs-dense",
) -> dict[str, Any]:
    chronicle = _load(chronicle_path)
    baseline = _load(baseline_path)
    rows = [
        _row(chronicle_label, chronicle),
        _row(baseline_label, baseline),
    ]
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_latex(rows, caption=caption, label=label), encoding="utf-8")
    (out_path.with_suffix(".md")).write_text(_markdown(rows), encoding="utf-8")
    payload = {
        "caption": caption,
        "label": label,
        "rows": rows,
        "sources": {
            "chronicle": str(chronicle_path),
            "baseline": str(baseline_path),
        },
    }
    (out_path.with_suffix(".json")).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chronicle", type=Path, default=DEFAULT_CHRONICLE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chronicle-label", default="Chronicle (event-gated)")
    parser.add_argument("--baseline-label", default="Dense 1 FPS (Florence-2)")
    parser.add_argument(
        "--caption",
        default="Episodic memory QA under a fixed local model budget.",
    )
    parser.add_argument("--label", default="tab:chronicle-vs-dense")
    args = parser.parse_args(argv)

    payload = build_assets(
        args.chronicle.expanduser().resolve(),
        args.baseline.expanduser().resolve(),
        out_path=args.out,
        chronicle_label=args.chronicle_label,
        baseline_label=args.baseline_label,
        caption=args.caption,
        label=args.label,
    )
    print(_markdown(payload["rows"]))
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.md')}")
    print(f"wrote {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

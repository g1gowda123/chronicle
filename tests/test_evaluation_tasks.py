from __future__ import annotations

import json
from pathlib import Path

from baseline_dense import CaptionEntry, FlatTextMemory, run_baseline
from eval_pipeline import run_evaluation, score_answer
from generate_paper_assets import build_assets

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data/real/cook/aria_cook_45s.mp4"


def test_score_answer_variants() -> None:
    assert score_answer("The cup was last seen in the Kitchen.", "The cup is in the Kitchen.")[0]
    assert score_answer("At t=4.5s the cup was touched.", "4.5 seconds")[0]
    assert score_answer("I don't have that in my memory.", "unknown")[0]
    assert not score_answer("I don't have that in my memory.", "The cup is in the Kitchen.")[0]
    assert not score_answer("The cup is in the kitchen.", "4.5 seconds")[0]
    assert score_answer("There are 2 cups on the table.", "2")[0]
    assert not score_answer("The joy-con was picked up at [t=21.8s].", "2")[0]


def test_flat_text_memory() -> None:
    memory = FlatTextMemory(
        [
            CaptionEntry(4.0, "The cup is in the kitchen on the counter."),
            CaptionEntry(43.4, "The plate is placed in the bowl."),
        ]
    )
    answer, hits = memory.answer("where is the cup?", 5.0)
    assert "kitchen" in answer.lower()
    assert hits and hits[0].t == 4.0

    answer, _ = memory.answer("when was the plate placed?", 50.0)
    assert "43.4" in answer

    answer, _ = memory.answer("where is the cup?", 3.0)
    assert "don't have" in answer.lower()

    answer, _ = memory.answer("how many bowls?", 50.0)
    assert "caption" in answer.lower()


def test_evaluation_writes_reports(tmp_path: Path) -> None:
    qa = tmp_path / "qa.json"
    qa.write_text(
        json.dumps(
            {
                "video_id": "aria_cook_45s",
                "questions": [
                    {
                        "query": "where are my keys?",
                        "answer": "I don't have that in my memory.",
                        "timestamp": 0.2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = run_evaluation(
        VIDEO,
        qa,
        out_dir=tmp_path,
        overrides={
            "runtime": {"profile": "offline"},
            "perception": {"max_frames": 5},
            "paths": {"db_path": str(tmp_path / "eval.db")},
        },
        show_progress=False,
    )
    assert payload["total"] == 1
    assert payload["rows"][0]["correct"] is True
    assert (tmp_path / "results.json").is_file()
    assert (tmp_path / "summary_report.md").is_file()


def test_dense_baseline_writes_results(tmp_path: Path) -> None:
    qa = tmp_path / "qa.json"
    qa.write_text(
        json.dumps(
            {
                "video_id": "aria_cook_45s",
                "questions": [
                    {"query": "where are my keys?", "answer": "unknown", "timestamp": 1.0}
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = run_baseline(
        VIDEO,
        qa,
        out_dir=tmp_path,
        captioner_backend="stub",
        max_captions=2,
        show_progress=False,
    )
    assert payload["total"] == 1
    assert payload["metrics"]["vlm_calls"] == 2
    assert (tmp_path / "baseline_results.json").is_file()
    assert (tmp_path / "baseline_summary.md").is_file()


def test_paper_assets(tmp_path: Path) -> None:
    chronicle = tmp_path / "results.json"
    baseline = tmp_path / "baseline_results.json"
    chronicle.write_text(
        json.dumps(
            {
                "method": "chronicle",
                "recall_at_1": 0.8,
                "metrics": {
                    "vlm_calls": 12,
                    "avg_latency_steady_ms": 14.2,
                    "peak_vram_mb": 1200.0,
                },
            }
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            {
                "method": "dense-1fps",
                "recall_at_1": 0.2,
                "metrics": {
                    "vlm_calls": 45,
                    "avg_latency_steady_ms": 210.0,
                    "peak_vram_mb": 1500.0,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "table.tex"
    payload = build_assets(chronicle, baseline, out_path=out)
    text = out.read_text(encoding="utf-8")
    assert "Chronicle (event-gated)" in text
    assert "Dense 1 FPS (Florence-2)" in text
    assert "80.0\\%" in text
    assert "45" in text
    assert payload["rows"][0]["vlm_calls"] == 12
    assert out.with_suffix(".md").is_file()

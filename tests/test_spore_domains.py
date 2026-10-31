"""Tests for the multi-domain SPORE evaluation manifest and aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spore_domains import aggregate, load_manifest, render_markdown, select  # noqa: E402


def _summary(name: str, domain: str, questions: int, semantic: float, loc: float, iou: float) -> dict:
    return {
        "name": name,
        "domain": domain,
        "clip": f"/tmp/{name}.mp4",
        "questions": questions,
        "semantic_accuracy": semantic,
        "localization_30": loc,
        "mean_iou": iou,
        "mean_calls": 4.0,
        "dense_calls": 100,
    }


def test_manifest_has_distinct_held_out_domains() -> None:
    manifest = load_manifest(ROOT / "data/aria_gen2pilot.json")
    domains = {item["domain"] for item in manifest["sequences"]}
    assert {"eating", "cleaning", "social", "outdoor"} <= domains
    assert len(select(manifest, ["eat_0"])) == 1


def test_aggregate_weights_by_question_count() -> None:
    payload = aggregate(
        [
            _summary("a", "eating", 1, 1.0, 1.0, 1.0),
            _summary("b", "eating", 3, 0.0, 0.0, 0.0),
            _summary("c", "outdoor", 2, 0.5, 0.5, 0.25),
        ]
    )
    eating = next(row for row in payload["domains"] if row["domain"] == "eating")
    assert eating["questions"] == 4
    assert eating["semantic_accuracy"] == 0.25
    assert eating["localization_30"] == 0.25
    assert eating["mean_iou"] == 0.25


def test_markdown_mentions_every_clip() -> None:
    payload = aggregate([_summary("a", "eating", 2, 0.5, 0.5, 0.5)])
    markdown = render_markdown(payload, {"dataset": "Test", "license": "CC"})
    assert "| eating |" in markdown
    assert "a.mp4" in markdown

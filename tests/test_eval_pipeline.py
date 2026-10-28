from __future__ import annotations

import json
import sys
from pathlib import Path

from eval_pipeline import (
    ChronicleEngine,
    MetricsTracker,
    instrument_captioner,
    load_qa_pairs,
    parse_timestamp,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parse_timestamp_shapes() -> None:
    assert parse_timestamp(12.5) == 12.5
    assert parse_timestamp("12.5") == 12.5
    assert parse_timestamp("01:02") == 62.0
    assert parse_timestamp("01:02:03.5") == 3723.5
    assert parse_timestamp({"start": 7.25}) == 7.25
    assert parse_timestamp(None) is None


def test_load_ego4d_qa_aliases(tmp_path: Path) -> None:
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps(
            {
                "video_id": "clip-1",
                "questions": [
                    {"query": "where is the cup?", "answer": "kitchen", "timestamp": 4.5},
                    {
                        "question": "did I see a dog?",
                        "annotation": ["no"],
                        "time": "00:10",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    pairs = load_qa_pairs(path)
    assert [pair.query for pair in pairs] == ["where is the cup?", "did I see a dog?"]
    assert pairs[0].video_id == "clip-1"
    assert pairs[1].answer == "no"
    assert pairs[1].timestamp == 10.0


def test_metrics_reduction_and_graph_counts() -> None:
    class Graph:
        def __init__(self, nodes: int, edges: int) -> None:
            self._nodes = nodes
            self._edges = edges

        def number_of_nodes(self) -> int:
            return self._nodes

        def number_of_edges(self) -> int:
            return self._edges

    class Memory:
        foreground = type("F", (), {"graph": Graph(5, 8)})()
        background = type("B", (), {"graph": Graph(3, 2)})()

    metrics = MetricsTracker(track_vram=False, percentiles=(50, 95))
    metrics.start()
    for index in range(10):
        metrics.record_frame(4.0 + index, timestamp=index * 0.1)
    metrics.record_vlm_call(latency_ms=8.0, backend="FlorenceCaptioner")
    metrics.record_vlm_call(latency_ms=7.0, backend="StubCaptioner")
    metrics.finished_at = metrics.started_at + 2.0
    payload = metrics.to_dict(memory=Memory())

    assert payload["frames_processed"] == 10
    assert payload["vlm_calls"] == 2
    assert payload["vlm_model_calls"] == 1
    assert payload["vlm_stub_calls"] == 1
    assert payload["vlm_call_reduction_pct"] == 80.0
    assert payload["graph"]["foreground_nodes"] == 5
    assert payload["graph"]["background_edges"] == 2
    assert payload["p50_ms"] == 8.0


def test_offline_engine_smoke(tmp_path: Path) -> None:
    """The engine must ingest without NiceGUI or downloaded Florence weights."""

    engine = ChronicleEngine(
        video_path=ROOT / "data/real/cook/aria_cook_45s.mp4",
        qa_json_path=ROOT / "data/qa/ego4d_example.json",
        root=ROOT,
        overrides={
            "runtime": {"profile": "offline"},
            "perception": {"max_frames": 4},
            "paths": {"out_dir": str(tmp_path), "db_path": str(tmp_path / "run.db")},
        },
    )
    try:
        payload = engine.run_ingestion()
    finally:
        engine.close()

    assert payload["frames_processed"] == 4
    assert payload["error"] is None
    assert payload["vlm_call_reduction_pct"] == 100.0
    assert "foreground_nodes" in payload["graph"]
    assert (tmp_path / "run.db").is_file()
    # Task-3 contract: pairs are loaded and timestamp-ordered.
    assert [pair.query for pair in engine.qa_pairs][0] == "when did I first touch the cup?"


def test_headless_research_imports_do_not_touch_ui() -> None:
    """`chronicle.vfm` must import on a machine without the UI stack."""

    blocked = {"nicegui", "fastapi", "dashboard", "chronicle.ui", "uvicorn"}

    class Blocker:
        def find_spec(self, name: str, path: object = None, target: object = None):  # noqa: ANN001
            if name in blocked or any(name.startswith(f"{item}.") for item in blocked):
                raise ImportError(f"blocked UI import: {name}")
            return None

    previous = list(sys.meta_path)
    sys.meta_path.insert(0, Blocker())
    try:
        for module in list(sys.modules):
            if module == "chronicle" or module.startswith("chronicle."):
                del sys.modules[module]
        import chronicle.vfm.compile  # noqa: F401
        import chronicle.vfm.query  # noqa: F401
    finally:
        sys.meta_path[:] = previous


def test_caption_budget_stride_counts_only_calls() -> None:
    from chronicle.semantic.florence import StubCaptioner

    metrics = MetricsTracker(track_vram=False)
    captioner = StubCaptioner(default_caption="a cup on a table")
    assert instrument_captioner(captioner, metrics, stride=2)
    outputs = [captioner.caption(None) for _ in range(5)]
    assert outputs == ["a cup on a table", "", "a cup on a table", "", "a cup on a table"]
    assert metrics.vlm_calls == 3
    assert metrics.vlm_skips == 2

    never = MetricsTracker(track_vram=False)
    captioner = StubCaptioner(default_caption="unused")
    assert instrument_captioner(captioner, never, stride=0)
    assert captioner.caption(None) == ""
    assert never.vlm_calls == 0
    assert never.vlm_skips == 1

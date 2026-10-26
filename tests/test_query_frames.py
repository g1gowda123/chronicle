"""Model-free tests for the answer evidence-frame feature."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import DT, cup, make_frame

from chronicle import frames as frame_utils
from chronicle.integration import ChronicleRuntime
from dashboard.frames import AnswerFrameSelector
from dashboard.runtime import Runtime
from dashboard.store import Entity, MemoryStore

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHRONICLE_LLM_BACKEND", "template")
    monkeypatch.delenv("CHRONICLE_QUERY_URL", raising=False)

    def build(start: float) -> Runtime:
        return Runtime(data_dir=DATA, demo_enabled=True, start_t=start)

    return build


def test_entity_answer_uses_latest_interaction_strip(demo) -> None:
    runtime = demo(19.0)
    result = runtime.query.answer("Where are my keys?", runtime.playhead)

    frame = result["frame"]
    assert frame is not None
    assert frame["url"] == "/media/strips/keys_insert.png"
    assert frame["kind"] == "interaction_strip"
    assert frame["source"] == "graph"
    assert frame["entity_id"] == "keys_1"
    assert frame["action"] == "INSERT_INTO"
    assert frame["t"] == pytest.approx(17.15)


def test_answer_frame_never_uses_future_evidence(demo) -> None:
    runtime = demo(10.0)
    result = runtime.query.answer("Where are my keys?", runtime.playhead)

    frame = result["frame"]
    assert frame is not None
    assert frame["url"] == "/media/crops/keys.png"
    assert frame["kind"] == "entity_crop"
    assert frame["t"] <= runtime.playhead


def test_zone_answer_uses_room_keyframe(demo) -> None:
    runtime = demo(19.0)
    result = runtime.query.answer("Where is the kitchen?", runtime.playhead)

    frame = result["frame"]
    assert frame is not None
    assert frame["url"] == "/media/zones/kitchen.png"
    assert frame["kind"] == "zone_keyframe"
    assert frame["zone_id"] == "kitchen"


def test_timeline_answer_uses_interaction_strip(demo) -> None:
    runtime = demo(19.0)
    result = runtime.query.answer("What did I do in this episode?", runtime.playhead)

    frame = result["frame"]
    assert frame is not None
    assert frame["kind"] == "interaction_strip"
    assert frame["url"] == "/media/strips/keys_insert.png"


def test_lost_entity_uses_its_crop(demo) -> None:
    runtime = demo(19.0)
    result = runtime.query.answer("Where is my wallet?", runtime.playhead)

    frame = result["frame"]
    assert frame is not None
    assert frame["url"] == "/media/crops/wallet.png"
    assert frame["entity_id"] == "wallet_1"


def test_frame_contract_is_seekable(demo) -> None:
    runtime = demo(19.0)
    frame = runtime.query.answer("Where are my keys?", runtime.playhead)["frame"]

    assert frame is not None
    assert frame["seek"] == frame["t"]
    assert isinstance(frame["caption"], str)
    assert isinstance(frame["clip_sim"], float)
    assert len(frame["bbox"]) == 4


def test_selector_extracts_video_frame_when_graph_has_no_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"video placeholder")
    store = MemoryStore()
    store.session["video_url"] = "/media/clip.mp4"
    selector = AnswerFrameSelector(store, data_dir=tmp_path)

    monkeypatch.setattr(
        "dashboard.frames.extract_video_frame_data_url",
        lambda video, t: ("data:image/jpeg;base64,ZmFrZQ==", float(t)),
    )
    result = {
        "route": "entity",
        "matched_entity": "keys_1",
        "citations": [
            {"t": 4.2, "label": "PICK_UP", "entity_id": "keys_1", "kind": "action"}
        ],
        "entities": [],
        "zones": [],
    }

    frame = selector.for_answer(result, playhead=6.0)

    assert frame is not None
    assert frame["kind"] == "video_frame"
    assert frame["source"] == "video"
    assert frame["url"].startswith("data:image/jpeg;base64,")
    assert frame["t"] == pytest.approx(4.2)


def test_selector_ignores_missing_media_files(tmp_path: Path) -> None:
    store = MemoryStore()
    store.entities["keys_1"] = Entity(
        id="keys_1",
        label="Keys",
        crop_url="/media/crops/keys.png",
        last_seen=4.0,
    )
    selector = AnswerFrameSelector(store, data_dir=tmp_path)
    result = {
        "route": "entity",
        "matched_entity": "keys_1",
        "citations": [{"t": 4.0, "label": "VISIBLE_STATIC", "entity_id": "keys_1"}],
        "entities": [{"id": "keys_1", "label": "Keys", "crop_url": "/media/crops/keys.png"}],
        "zones": [],
    }

    assert selector.for_answer(result, playhead=6.0) is None


def test_extract_video_frame_returns_browser_ready_data_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video placeholder")

    def fake_frame(path: Path, t: float):
        return np.zeros((24, 32, 3), dtype=np.uint8), t

    monkeypatch.setattr(frame_utils, "_read_video_frame", fake_frame)

    extracted = frame_utils.extract_video_frame_data_url(video, 4.25)

    assert extracted is not None
    data_url, actual_t = extracted
    assert data_url.startswith("data:image/jpeg;base64,")
    assert actual_t == pytest.approx(4.25)


def test_integrated_runtime_attaches_frame_to_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = ChronicleRuntime(":memory:", semantic_enabled=False, zone_label="kitchen", reset=True)
    try:
        for index in range(30):
            runtime.memory.observe(make_frame(index * DT, objects=[cup()], zone_label="kitchen"))

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video placeholder")
        runtime._video = str(video)  # noqa: SLF001 - test seam for the frame source
        monkeypatch.setattr(
            "chronicle.integration.extract_video_frame_data_url",
            lambda path, t: ("data:image/jpeg;base64,ZmFrZQ==", float(t)),
        )

        answer = runtime.ask("where is my cup?")

        assert answer.frame is not None
        assert answer.frame["kind"] == "video_frame"
        assert answer.to_dict()["frame"] == answer.frame
    finally:
        runtime.close()


def _write_evidence_index(
    episode_dir: Path,
    rows: list[dict],
    *,
    missing: bool = False,
) -> None:
    evidence = episode_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if missing and row["file"].startswith("missing"):
            continue
        image = evidence / row["file"]
        assert cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))
    (evidence / "index.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_load_stored_evidence_reads_index_and_skips_missing(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    rows = [
        {
            "file": "shown.jpg",
            "t": 4.0,
            "kind": "interaction_strip",
            "label": "cup",
            "entity_id": "cup_1",
            "action": "PICK_UP",
        },
        {
            "file": "missing.jpg",
            "t": 5.0,
            "kind": "zone_keyframe",
            "label": "kitchen",
        },
    ]
    _write_evidence_index(episode, rows, missing=True)

    loaded = frame_utils.load_stored_evidence(episode)

    assert len(loaded) == 1
    assert loaded[0].path.name == "shown.jpg"
    assert loaded[0].entity_id == "cup_1"


def test_pick_stored_evidence_prefers_matching_label_and_encodes_data_url(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode"
    rows = [
        {
            "file": "cup.jpg",
            "t": 5.0,
            "kind": "interaction_strip",
            "label": "cup",
            "entity_id": "cup_1",
            "action": "PICK_UP",
        },
        {
            "file": "zone.jpg",
            "t": 0.0,
            "kind": "zone_keyframe",
            "label": "kitchen",
            "zone_id": "kitchen",
        },
    ]
    _write_evidence_index(episode, rows)
    loaded = frame_utils.load_stored_evidence(episode)

    picked = frame_utils.pick_stored_evidence(
        loaded,
        playhead=10.0,
        anchor_t=5.0,
        matched_label="cup",
    )

    assert picked is not None
    assert picked.kind == "interaction_strip"
    assert picked.label == "cup"
    assert picked.url.startswith("data:image/jpeg;base64,")


def test_bundled_evidence_indexes_are_usable() -> None:
    indexes = sorted(ROOT.glob("data/**/evidence/index.jsonl"))
    assert indexes, "no bundled evidence indexes found"
    for index in indexes:
        rows = frame_utils.load_stored_evidence(index.parent.parent)
        assert rows, index
        assert all(row.path.is_file() for row in rows)


def test_integrated_runtime_prefers_stored_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = tmp_path / "episode"
    _write_evidence_index(
        episode,
        [
            {
                "file": "cup.jpg",
                "t": 0.5,
                "kind": "interaction_strip",
                "label": "cup",
                "entity_id": "cup_1",
                "action": "PICK_UP",
            }
        ],
    )
    runtime = ChronicleRuntime(
        str(episode / "chronicle.db"),
        semantic_enabled=False,
        zone_label="kitchen",
        reset=True,
    )
    try:
        for index in range(30):
            runtime.memory.observe(
                make_frame(index * DT, objects=[cup()], zone_label="kitchen")
            )
        runtime._video = "video-placeholder.mp4"  # noqa: SLF001
        monkeypatch.setattr(
            "chronicle.integration.extract_video_frame_data_url",
            lambda path, t: (_ for _ in ()).throw(
                AssertionError("stored evidence should win")
            ),
        )

        answer = runtime.ask("where is my cup?")

        assert answer.frame is not None
        assert answer.frame["kind"] == "interaction_strip"
        assert answer.frame["label"] == "cup"
        assert answer.frame["url"].startswith("data:image/jpeg;base64,")
    finally:
        runtime.close()


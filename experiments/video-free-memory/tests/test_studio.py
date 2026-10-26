"""Tests for the upload caption vault and the Qwen context builder."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import nlq  # noqa: E402
import query  # noqa: E402
import video_vault  # noqa: E402


def _ledger() -> dict:
    return {
        "version": 1,
        "model": "test-vlm",
        "mode": "caption-vault",
        "window_s": 30.0,
        "video": "test.mp4",
        "duration_s": 45.0,
        "windows": [
            {
                "t_start": 0.0,
                "t_end": 30.0,
                "caption": "A person slices a tomato on a cutting board in the kitchen.",
                "objects": ["tomato", "knife", "cutting board"],
                "actions": ["slice"],
                "zone": "kitchen",
                "tags": ["tomato", "knife"],
                "frames": [
                    {"t": 2.0, "caption": "Tomato on the board.", "objects": ["tomato"], "actions": ["slice"], "zone": "kitchen"},
                    {"t": 15.0, "caption": "Knife cuts the tomato.", "objects": ["knife", "tomato"], "actions": ["slice"], "zone": "kitchen"},
                ],
            },
            {
                "t_start": 30.0,
                "t_end": 45.0,
                "caption": "The person places the tomato slices on a plate.",
                "objects": ["tomato", "plate"],
                "actions": ["place"],
                "zone": "kitchen",
                "tags": ["tomato", "plate"],
                "frames": [
                    {"t": 37.0, "caption": "Tomato slices on the plate.", "objects": ["tomato", "plate"], "actions": ["place"], "zone": "kitchen"},
                ],
            },
        ],
    }


def test_caption_vault_per_frame_events_and_queries(tmp_path: Path) -> None:
    video = tmp_path / "test.mp4"
    video.write_bytes(b"not a real video")
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(_ledger()), encoding="utf-8")
    vault = tmp_path / "vault.db"

    summary = video_vault.compile_caption_vault(video, ledger_path, vault)
    assert summary["mode"] == "caption-vault"
    assert summary["windows"] == 2
    assert summary["events"] >= 6

    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        times = [float(event["t"]) for event in store.events]
        assert 2.0 in times and 15.0 in times and 37.0 in times

        count = query.answer(store, "how many tomatoes?", use_cache=False)
        assert count.method == "vault-count-captions"
        assert "caption" in count.answer.lower()

        inventory = query.answer(store, "what did I see?", use_cache=False)
        assert inventory.method == "vault-inventory-captions"

        where = query.answer(store, "where is the knife?", use_cache=False)
        assert where.method == "vault-where"
        assert "kitchen" in where.answer.lower()
    finally:
        con.close()


def test_nlq_context_is_text_only(tmp_path: Path) -> None:
    video = tmp_path / "test.mp4"
    video.write_bytes(b"not a real video")
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(_ledger()), encoding="utf-8")
    vault = tmp_path / "vault.db"
    video_vault.compile_caption_vault(video, ledger_path, vault)

    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        deterministic = query.answer(store, "what did I do with the tomato?", use_cache=False)
        context = nlq._context(store, "what did I do with the tomato?", deterministic)
        assert "CAPTION MEMORY" in context
        assert "tomato" in context.lower()
        assert "test.mp4" not in context
        assert "VIDEO" not in context
    finally:
        con.close()

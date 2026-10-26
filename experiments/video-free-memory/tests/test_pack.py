"""Tests for the caption-first vault and LLM memory pack."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import memory_pack  # noqa: E402
import nlq  # noqa: E402
import query  # noqa: E402
import video_vault  # noqa: E402


def _ledger(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": "test",
                "window_s": 30.0,
                "duration_s": 45.0,
                "windows": [
                    {
                        "t_start": 0.0,
                        "t_end": 30.0,
                        "caption": "A person slices a tomato on a cutting board in the kitchen.",
                        "objects": ["tomato", "knife", "cutting board"],
                        "actions": ["slice tomato"],
                        "zone": "kitchen",
                        "tags": ["tomato", "slicing"],
                    },
                    {
                        "t_start": 30.0,
                        "t_end": 45.0,
                        "caption": "A person places the tomato on a plate.",
                        "objects": ["tomato", "plate"],
                        "actions": ["place tomato"],
                        "zone": "kitchen",
                        "tags": ["tomato", "plate"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_caption_vault_and_memory_pack(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    _ledger(ledger_path)
    vault = tmp_path / "vault.db"
    summary = video_vault.compile_caption_vault(
        Path("clip.mp4"), ledger_path, vault, title="clip"
    )
    assert summary["events"] >= 4
    assert summary["entities"] >= 3

    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        answer = query.answer(store, "what did I do with the tomato?", use_cache=False)
        assert "tomato" in answer.answer.lower()

        pack = memory_pack.build_pack(
            store, "where is the tomato?", verified_answer=answer, max_chars=5000
        )
        assert "Verified structured answer" in pack
        assert "Timestamped caption windows" in pack
        assert "cutting board" in pack

        payload = memory_pack.build_pack_data(store)
        assert payload["duration_s"] == 45.0
        assert payload["windows"]
        assert payload["events"]
    finally:
        con.close()


def _big_ledger(path: Path, windows: int = 36) -> None:
    payload = {
        "version": 1,
        "model": "test",
        "window_s": 30.0,
        "duration_s": windows * 30.0,
        "windows": [
            {
                "t_start": index * 30.0,
                "t_end": (index + 1) * 30.0,
                "caption": (
                    f"A person slices a tomato with a knife on a wooden cutting board "
                    f"beside a plate and a bowl during window {index}."
                ),
                "objects": ["tomato", "knife", "cutting board", "plate", "bowl"],
                "actions": ["slice tomato", "move knife"],
                "zone": "kitchen",
                "tags": ["tomato", "slicing", "kitchen"],
                "frames": [
                    {
                        "t": index * 30.0 + offset,
                        "caption": f"Frame {offset}: hands and a tomato near the knife.",
                        "objects": ["tomato", "knife"],
                    }
                    for offset in (1.0, 10.0, 20.0)
                ],
            }
            for index in range(windows)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_token_budget_fits_and_keeps_instructions(tmp_path: Path) -> None:
    ledger_path = tmp_path / "big_ledger.json"
    _big_ledger(ledger_path)
    vault = tmp_path / "big.db"
    video_vault.compile_caption_vault(Path("clip.mp4"), ledger_path, vault)

    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        verified = query.answer(store, "where is the tomato?", use_cache=False)
        sizes = {}
        for budget in memory_pack.TOKEN_BUDGETS:
            pack = memory_pack.build_pack(
                store,
                "where is the tomato?",
                verified_answer=verified,
                token_budget=budget,
            )
            assert len(pack) <= budget * memory_pack.CHARS_PER_TOKEN
            assert "## Verified structured answer" in pack
            assert "## Answering rules" in pack
            assert "## Timestamped caption windows" in pack
            assert f"Budget: ~{budget} tokens" in pack
            sizes[budget] = len(pack)
        assert sizes[2000] <= sizes[4000] <= sizes[8000]
        assert sizes[2000] < sizes[8000]
        assert len(
            memory_pack.build_pack(store, token_budget=2000)
        ) < len(memory_pack.build_pack(store, token_budget=8000))

        payload = memory_pack.build_pack_data(store, token_budget=2000)
        assert payload["stats"]["token_budget"] == 2000
        assert payload["stats"]["chars"] <= 2000 * memory_pack.CHARS_PER_TOKEN
    finally:
        con.close()


def test_nlq_uses_pack_and_parses_citations(tmp_path: Path, monkeypatch) -> None:
    ledger_path = tmp_path / "ledger.json"
    _ledger(ledger_path)
    vault = tmp_path / "vault.db"
    video_vault.compile_caption_vault(Path("clip.mp4"), ledger_path, vault)

    monkeypatch.setattr(
        nlq,
        "_ollama",
        lambda model, prompt, **kwargs: "The tomato was sliced at [t=0.0s] and placed at [t=30.0s].",
    )
    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        result = nlq.answer_nl(store, "what happened to the tomato?", model="test")
        assert result.method.startswith("qwen-pack:")
        assert "[t=0.0s]" in result.answer
        assert [round(float(cite["t"])) for cite in result.citations] == [0, 30]
    finally:
        con.close()

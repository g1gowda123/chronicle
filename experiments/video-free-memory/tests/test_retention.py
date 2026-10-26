"""Tests for the dense ledger, answer cache and retention budget."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import attach_ledger  # noqa: E402
import caption_ledger  # noqa: E402
import compile as vault_compile  # noqa: E402
import query as vault_query  # noqa: E402
import retention  # noqa: E402

COOK_DB = Path("/home/g1/projects/chronicle/data/real/cook/chronicle.db")


def test_parse_caption_accepts_model_key_variants() -> None:
    parsed = caption_ledger.parse_caption(
        json.dumps({"description": "A person slices a tomato.", "objects": "tomato"}),
        "test-model",
    )
    assert "tomato" in parsed["caption"]
    assert parsed["objects"] == ["tomato"]

    empty = caption_ledger.parse_caption("{}", "test-model")
    assert empty["caption"] == ""

    fenced = caption_ledger.parse_caption(
        '```json\n{"caption": "The cup is on the counter."}\n```', "test-model"
    )
    assert "cup" in fenced["caption"]


def test_window_times_are_sorted_and_inside_window() -> None:
    times = caption_ledger.window_times(10.0, 40.0, 3)
    assert times == sorted(times)
    assert all(10.0 < value < 40.0 for value in times)
    assert caption_ledger.window_times(0.0, 3.0, 3) == [1.5]


@pytest.mark.skipif(not COOK_DB.is_file(), reason="cook episode database not present")
def test_attach_ledger_keyword_and_window(tmp_path: Path) -> None:
    vault = tmp_path / "cook.db"
    vault_compile.compile_vault(COOK_DB, vault)
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "model": "test-vlm",
                "prompt_version": 1,
                "window_s": 30.0,
                "windows": [
                    {
                        "t_start": 0.0,
                        "t_end": 30.0,
                        "caption": "A person slices a tomato on a cutting board.",
                        "objects": ["tomato", "knife"],
                        "actions": ["slice"],
                        "zone": "kitchen",
                        "tags": ["tomato"],
                    },
                    {
                        "t_start": 30.0,
                        "t_end": 45.0,
                        "caption": "A person places avocado on a plate.",
                        "objects": ["avocado", "plate"],
                        "actions": ["place"],
                        "zone": "kitchen",
                        "tags": ["avocado"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    attach_ledger.attach(vault, ledger_path)

    con = sqlite3.connect(str(vault))
    try:
        store = vault_query.load_store(con)
        assert len(store.ledger_windows) == 2
        keyword = vault_query.answer(store, "what did I do with the tomato?", use_cache=False)
        assert keyword.method == "vault-keyword"
        assert "tomato" in keyword.answer.lower()

        window = vault_query.answer(store, "what happened around 20 seconds?", use_cache=False)
        assert "memory caption" in window.answer.lower()
        assert "tomato" in window.answer.lower()
    finally:
        con.close()


@pytest.mark.skipif(not COOK_DB.is_file(), reason="cook episode database not present")
def test_answer_cache_hits_paraphrase(tmp_path: Path) -> None:
    vault = tmp_path / "cook.db"
    vault_compile.compile_vault(COOK_DB, vault)
    con = sqlite3.connect(str(vault))
    try:
        store = vault_query.load_store(con)
        first = vault_query.answer(store, "where is the cup?", use_cache=True)
        assert not first.cached
        second = vault_query.answer(store, "which room is the mug in?", use_cache=True)
        assert second.cached
        assert second.method.startswith("cache:")
        assert "kitchen" in second.answer.lower()
    finally:
        con.close()


@pytest.mark.skipif(not COOK_DB.is_file(), reason="cook episode database not present")
def test_retention_select_and_apply(tmp_path: Path) -> None:
    vault = tmp_path / "cook.db"
    vault_compile.compile_vault(COOK_DB, vault)
    con = sqlite3.connect(str(vault))
    try:
        con.execute(
            """
            create table if not exists evidence (
                id TEXT PRIMARY KEY, t REAL NOT NULL, kind TEXT NOT NULL, label TEXT,
                file TEXT NOT NULL, rel_path TEXT NOT NULL, bytes INTEGER NOT NULL DEFAULT 0,
                width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0,
                meta TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        rows = []
        for index in range(60):
            t = index * 60.0
            row = {
                "id": f"ev_{index:03d}",
                "t": t,
                "kind": "interaction" if index % 2 else "anchor",
                "label": "cup",
                "rel_path": f"evidence/ev_{index:03d}.webp",
                "bytes": 1000,
            }
            rows.append(row)
            con.execute(
                "insert into evidence(id,t,kind,label,file,rel_path,bytes) values (?,?,?,?,?,?,?)",
                (
                    row["id"],
                    row["t"],
                    row["kind"],
                    row["label"],
                    row["id"],
                    row["rel_path"],
                    row["bytes"],
                ),
            )
        con.commit()
    finally:
        con.close()

    selected = retention._select(rows, 12, 3600.0)
    assert len(selected) == 12
    assert len(retention._select(rows, 4, 3600.0)) == 4

    summary = retention.apply(vault, 4, drop_cache=True)
    assert summary["evidence_before"] == 60
    assert summary["evidence_after"] == 4
    assert summary["vault_bytes_after"] <= summary["vault_bytes_before"] + 1024


def test_video_baseline_scoring_and_missing_key(monkeypatch, tmp_path: Path) -> None:
    import vlm_baseline

    assert vlm_baseline.clean_answer('{"answer": "yes"}') == "yes"
    assert vlm_baseline.score_baseline("unknown", {"kind": "unknown"})[0]
    assert vlm_baseline.score_baseline(
        "The plate is on the countertop at t=43.4s.",
        {"must_contain": ["plate", "kitchen"], "t_near": 43.4, "t_tol": 1.0},
    )[0]

    import online_baseline

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    code = online_baseline.main(
        [
            "--provider",
            "gemini",
            "--questions",
            str(ROOT / "benchmark" / "questions.json"),
            "--source-root",
            "/tmp",
            "--vault-dir",
            str(tmp_path),
            "--limit",
            "1",
        ]
    )
    assert code == 2

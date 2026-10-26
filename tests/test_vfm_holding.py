"""Holding/carrying questions answered directly from caption phrases."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from chronicle.vfm import query, video_vault


def _ledger(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": "test",
                "window_s": 30.0,
                "duration_s": 30.0,
                "windows": [
                    {
                        "t_start": 0.0,
                        "t_end": 30.0,
                        "caption": (
                            "A person walks on a sidewalk next to a road, holding a cup of coffee. "
                            "Another person carries a black backpack."
                        ),
                        "objects": ["person", "cup of coffee", "backpack"],
                        "actions": ["walking", "holding"],
                        "zone": "Outdoor Path",
                        "tags": ["walking"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_holding_answer_is_direct(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    _ledger(ledger)
    vault = tmp_path / "vault.db"
    video_vault.compile_caption_vault(tmp_path / "clip.mp4", ledger, vault)
    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        result = query.answer(store, "what is the other person holding?", use_cache=False)
        assert result.method == "vault-holding"
        assert "cup of coffee" in result.answer.lower()
        assert "backpack" not in result.answer.lower()
        assert result.citations
    finally:
        con.close()


def test_time_hint_accepts_time_10s() -> None:
    assert query._time_hint("what happened at time 10s?") == 10.0
    assert query._time_hint("what happened around 42s?") == 42.0

"""First/last-seen answers use the entity bounds, not only salience events."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from chronicle.vfm import query, video_vault


def test_last_seen_uses_entity_bounds(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "version": 1,
                "model": "test",
                "window_s": 30.0,
                "duration_s": 60.0,
                "windows": [
                    {
                        "t_start": 0.0,
                        "t_end": 30.0,
                        "caption": "A cup is on the table.",
                        "objects": ["cup"],
                        "actions": [],
                        "zone": "Kitchen",
                        "tags": [],
                    },
                    {
                        "t_start": 30.0,
                        "t_end": 60.0,
                        "caption": "The cup is still on the table near the end.",
                        "objects": ["cup"],
                        "actions": [],
                        "zone": "Kitchen",
                        "tags": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    vault = tmp_path / "vault.db"
    video_vault.compile_caption_vault(tmp_path / "clip.mp4", ledger, vault)
    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        result = query.answer(store, "when was cup last seen?", use_cache=False)
        assert result.method == "vault-when-seen"
        assert "last seen at t=" in result.answer
        assert "t=0.0s" not in result.answer.split("last seen at")[1][:20]
    finally:
        con.close()

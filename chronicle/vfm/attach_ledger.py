#!/usr/bin/env python3
"""Attach a dense caption ledger (JSON) to a VFM vault.

The ledger is zlib-compressed and stored in the single-row `caption_ledger`
table. Query time only reads this blob; the video and the VLM are not needed.

Usage:
    python prototype/attach_ledger.py --vault /tmp/vfm/vault.db \
      --ledger /tmp/vfm/cook_ledger.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .compact import compress_json

PACKAGE_DIR = Path(__file__).resolve().parent
SCHEMA_V3 = PACKAGE_DIR / "schema_v3.sql"


def load_ledger(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("ledger must be a JSON object")
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise SystemExit("ledger has no windows")
    for index, window in enumerate(windows):
        if not isinstance(window, dict) or "caption" not in window:
            raise SystemExit(f"window {index} is missing a caption")
    return payload


def attach(vault: Path, ledger_path: Path) -> dict[str, Any]:
    if not vault.is_file():
        raise SystemExit(f"vault not found: {vault}")
    ledger = load_ledger(ledger_path)
    blob = compress_json(ledger)
    con = sqlite3.connect(str(vault))
    try:
        con.executescript(SCHEMA_V3.read_text(encoding="utf-8"))
        con.execute(
            """
            insert or replace into caption_ledger
                (id, model, window_s, blob, n, created_at)
            values (1, ?, ?, ?, ?, ?)
            """,
            (
                str(ledger.get("model") or ""),
                float(ledger.get("window_s") or 30.0),
                blob,
                len(ledger.get("windows") or []),
                time.time(),
            ),
        )
        meta = {
            "ledger_model": str(ledger.get("model") or ""),
            "ledger_prompt_version": str(ledger.get("prompt_version") or ""),
            "ledger_windows": str(len(ledger.get("windows") or [])),
            "ledger_blob_bytes": str(len(blob)),
            "ledger_attached_at": str(time.time()),
        }
        con.executemany(
            "insert or replace into meta(key, value) values (?, ?)",
            list(meta.items()),
        )
        # A new caption ledger invalidates answers computed before it existed,
        # especially cached "no-match" replies to holding/what-happened questions.
        try:
            con.execute("delete from qa_cache")
        except sqlite3.Error:
            pass
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()
    return {
        "vault": str(vault),
        "ledger": str(ledger_path),
        "model": str(ledger.get("model") or ""),
        "windows": len(ledger.get("windows") or []),
        "window_s": float(ledger.get("window_s") or 0.0),
        "raw_bytes": ledger_path.stat().st_size,
        "stored_bytes": len(blob),
        "vault_bytes": vault.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = attach(
        args.vault.expanduser().resolve(),
        args.ledger.expanduser().resolve(),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

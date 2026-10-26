"""Tests for the compressed vault primitives and the compile/query path."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import compact  # noqa: E402
import compile as vault_compile  # noqa: E402
import query as vault_query  # noqa: E402

COOK_DB = Path("/home/g1/projects/chronicle/data/real/cook/chronicle.db")


def test_string_table_roundtrip() -> None:
    table = compact.StringTable()
    assert table.get(0) == ""
    first = table.add("cup")
    second = table.add("cup")
    assert first == second
    assert table.get(first) == "cup"
    rows = table.rows()
    other = compact.StringTable()
    other.load(rows)
    assert other.get(first) == "cup"


def test_event_pack_roundtrip_and_density() -> None:
    table = compact.StringTable()
    events = [
        {
            "t": 1.234,
            "kind": "state",
            "subject": "cup",
            "relation": "NEW->VISIBLE_STATIC",
            "object": "cup",
            "zone": "Kitchen",
            "confidence": 0.87,
            "flags": 1,
            "evidence_id": 3,
        }
    ]
    blob = compact.pack_events(events, table)
    assert len(blob) == compact.HEADER.size + compact.EVENT_SIZE
    assert compact.EVENT_SIZE < 20
    decoded = compact.unpack_events(blob, table)
    assert decoded[0]["t"] == pytest.approx(1.234)
    assert decoded[0]["kind"] == "state"
    assert decoded[0]["subject"] == "cup"
    assert decoded[0]["zone"] == "Kitchen"
    assert decoded[0]["confidence"] == pytest.approx(0.87)


def test_caption_blob_compression() -> None:
    payload = [{"t_start": 0, "t_end": 30, "text": "the cup moved " * 50}]
    blob = compact.compress_json(payload)
    assert len(blob) < len(str(payload))
    assert compact.decompress_json(blob) == payload


def test_int8_quantization_is_4x_smaller() -> None:
    vector = [(index % 17) / 17.0 for index in range(512)]
    quantized = compact.quantize_int8(vector)
    blob = quantized.to_blob()
    assert len(blob) == 4 + 512
    assert len(vector) * 4 == 2048
    restored = compact.dequantize_int8(blob)
    assert len(restored) == 512
    worst = max(abs(a - b) for a, b in zip(vector, restored, strict=False))
    assert worst <= quantized.scale


@pytest.mark.skipif(not COOK_DB.is_file(), reason="cook episode database not present")
def test_compile_and_query_video_free(tmp_path: Path) -> None:
    out = tmp_path / "cook.db"
    summary = vault_compile.compile_vault(COOK_DB, out)
    assert summary["events"] > 10
    assert summary["event_blob_bytes"] < 20 * summary["events"]
    assert summary["vault_bytes"] < 200_000

    con = sqlite3.connect(str(out))
    try:
        store = vault_query.load_store(con)
        assert len(store.entities) >= 5
        answer = vault_query.answer(store, "where is the cup?")
        assert answer.method == "vault-where"
        assert "cup" in answer.answer.lower()
        missing = vault_query.answer(store, "where are my keys?")
        assert missing.method == "no-match"
        stats = vault_query.stats(store, out)
        assert stats["events"] == summary["events"]
    finally:
        con.close()

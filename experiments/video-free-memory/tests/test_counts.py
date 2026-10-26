"""Compile-time object-count reconciliation (P1-COUNTS).

The vault records two numbers per label: the number of tracker ids ("tracks",
an upper bound) and the largest number of same-label tracks visible in a single
observation frame ("peak", a lower bound on physical objects). These tests keep
the compile/query contract honest and make sure the old vault format still
works.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import compile as vfm_compile  # noqa: E402
import query  # noqa: E402


def _make_source(path: Path) -> None:
    con = sqlite3.connect(str(path))
    try:
        con.executescript(
            """
            create table zones (id text primary key, label text);
            create table entities (
                id text primary key, label text, state text, zone_id text,
                first_seen real, last_seen real, attrs text default '{}'
            );
            create table observations (
                id integer primary key autoincrement, entity_id text, t real,
                bbox text, state text
            );
            """
        )
        con.execute("insert into zones values ('z1', 'Kitchen')")
        entities = [
            ("bowl_a", "bowl", 0.0, 9.0),
            ("bowl_b", "bowl", 0.0, 9.0),
            ("bowl_c", "bowl", 10.0, 14.0),
            ("bowl_d", "bowl", 20.0, 21.0),
            ("cup_a", "cup", 0.0, 21.0),
        ]
        con.executemany(
            "insert into entities values (?, ?, 'VISIBLE', 'z1', ?, ?, '{}')",
            entities,
        )
        observations = [
            # bowl_a and bowl_b co-visible for two frames -> peak 2.
            ("bowl_a", 0.0),
            ("bowl_b", 0.0),
            ("bowl_a", 0.1),
            ("bowl_b", 0.1),
            ("bowl_a", 5.0),
            ("bowl_c", 10.0),
            ("bowl_c", 10.1),
            ("bowl_d", 20.0),
            ("cup_a", 0.0),
            ("cup_a", 5.0),
        ]
        con.executemany(
            "insert into observations (entity_id, t, bbox, state) values (?, ?, '[]', 'VISIBLE')",
            observations,
        )
        con.commit()
    finally:
        con.close()


def test_compile_stores_track_and_peak_counts(tmp_path: Path) -> None:
    source = tmp_path / "chronicle.db"
    _make_source(source)
    vault = tmp_path / "vault.db"

    report = vfm_compile.compile_vault(source, vault)

    bowl = report["counts"]["bowl"]
    assert bowl["tracks"] == 4
    assert bowl["peak"] == 2
    # The compiler may also emit an experimental duplicate-merged estimate.
    assert 0 < bowl.get("estimate", 4) <= 4
    cup = report["counts"]["cup"]
    assert cup["tracks"] == 1
    assert cup["peak"] == 1

    con = sqlite3.connect(str(vault))
    try:
        raw = con.execute("select value from meta where key='counts'").fetchone()[0]
        assert '"peak":2' in raw
        store = query.load_store(con)
        assert store.counts["bowl"]["peak"] == 2
        assert store.counts["cup"]["tracks"] == 1
    finally:
        con.close()


def test_count_answer_reports_physical_range(tmp_path: Path) -> None:
    source = tmp_path / "chronicle.db"
    _make_source(source)
    vault = tmp_path / "vault.db"
    vfm_compile.compile_vault(source, vault)

    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        result = query.answer(store, "how many bowls?", use_cache=False)
        assert "4 bowl" in result.answer
        assert "between 2 and 4" in result.answer
        assert result.method == "vault-count-visibility"

        single = query.answer(store, "how many cups?", use_cache=False)
        assert "1 cup" in single.answer
        assert "1 cup track" in single.answer

        # Singular grammar on the lower-bound branch.
        cars = query.answer(store, "how many bowls?", use_cache=False)
        assert "At most 2 bowl tracks were visible" in cars.answer
    finally:
        con.close()


def test_legacy_vault_without_counts_keeps_old_wording(tmp_path: Path) -> None:
    source = tmp_path / "chronicle.db"
    _make_source(source)
    vault = tmp_path / "vault.db"
    vfm_compile.compile_vault(source, vault)

    con = sqlite3.connect(str(vault))
    try:
        con.execute("delete from meta where key='counts'")
        con.commit()
        store = query.load_store(con)
        assert store.counts == {}
        result = query.answer(store, "how many bowls?", use_cache=False)
        assert result.method == "vault-count"
        assert "not a verified object count" in result.answer
    finally:
        con.close()

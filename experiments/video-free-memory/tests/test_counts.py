"""Compile-time object-count reconciliation (P1-COUNTS).

The vault records two numbers per label: the number of tracker ids ("tracks",
an upper bound) and the largest number of same-label tracks visible in a single
observation frame ("peak", a lower bound on physical objects). These tests keep
the compile/query contract honest and make sure the old vault format still
works.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

import compile as vfm_compile  # noqa: E402
import query  # noqa: E402
import report as vfm_report  # noqa: E402


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
            ("bowl_d", 20.1),
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
    assert bowl["physical"] == 4
    assert bowl["duplicates"] == 0
    cup = report["counts"]["cup"]
    assert cup["tracks"] == 1
    assert cup["peak"] == 1
    assert cup["physical"] == 1

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


def test_report_embeds_counts_and_uses_bounds(tmp_path: Path) -> None:
    source = tmp_path / "chronicle.db"
    _make_source(source)
    vault = tmp_path / "vault.db"
    vfm_compile.compile_vault(source, vault)
    out = tmp_path / "memory.html"

    summary = vfm_report.build_report(vault, out, data_root=tmp_path)
    html = out.read_text(encoding="utf-8")

    assert summary["html"] == str(out)
    assert '"counts"' in html
    assert '"peak": 2' in html
    assert "vault-count-visibility" in html
    assert "vault-count-dedup" in html


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


def test_cross_label_duplicate_tracks_are_counted_once(tmp_path: Path) -> None:
    source = tmp_path / "chronicle.db"
    con = sqlite3.connect(str(source))
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
        con.executemany(
            "insert into entities values (?, ?, 'VISIBLE', 'z1', 0.0, 9.0, '{}')",
            [("plate_a", "plate"), ("bowl_a", "bowl")],
        )
        # Same pixels, two labels: three strong duplicate frames. The plate
        # track lives longer, so it is the canonical label.
        for index, t in enumerate((1.0, 2.0, 3.0)):
            box = json.dumps([10.0, 10.0, 60.0, 60.0])
            con.execute(
                "insert into observations (entity_id, t, bbox, state) values (?, ?, ?, 'VISIBLE')",
                ("plate_a", t, box),
            )
            con.execute(
                "insert into observations (entity_id, t, bbox, state) values (?, ?, ?, 'VISIBLE')",
                ("bowl_a", t, box),
            )
        con.execute(
            "insert into observations (entity_id, t, bbox, state) values ('plate_a', 4.0, ?, 'VISIBLE')",
            (json.dumps([10.0, 10.0, 60.0, 60.0]),),
        )
        con.commit()
    finally:
        con.close()

    vault = tmp_path / "vault.db"
    report = vfm_compile.compile_vault(source, vault)
    assert report["counts"]["bowl"]["tracks"] == 1
    assert report["counts"]["bowl"]["duplicates"] == 1
    assert report["counts"]["bowl"]["physical"] == 0
    assert report["counts"]["plate"]["physical"] == 1
    assert report["counts"]["bowl"]["merged_into"] == {"plate": 1}

    con = sqlite3.connect(str(vault))
    try:
        store = query.load_store(con)
        result = query.answer(store, "how many bowls?", use_cache=False)
        assert result.method == "vault-count-dedup"
        assert "1 bowl track" in result.answer
        assert "duplicate detections of 1 plate track" in result.answer
        assert "physical bowl count is 0" in result.answer
    finally:
        con.close()

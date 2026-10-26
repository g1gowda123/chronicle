"""Model-free tests for the dashboard snapshot import path."""

from __future__ import annotations

import json

import pytest

from chronicle.ui.snapshot_io import (
    SnapshotImportError,
    parse_snapshot,
    snapshot_summary,
)


def _payload() -> dict:
    return {
        "nodes": [
            {"id": "zone:kitchen", "label": "Kitchen", "group": "zone"},
            {"id": "entity:cup", "label": "cup", "group": "entity"},
        ],
        "edges": [],
        "entities": [{"id": "cup", "label": "cup", "state": "VISIBLE_STATIC"}],
        "actions": [{"id": "action:1", "action": "PICK_UP"}],
    }


def test_parse_snapshot_accepts_utf8_and_bom() -> None:
    text = json.dumps(_payload())
    assert parse_snapshot(text) == _payload()
    assert parse_snapshot(text.encode("utf-8")) == _payload()
    assert parse_snapshot(b"\xef\xbb\xbf" + text.encode("utf-8")) == _payload()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{not json", "invalid JSON"),
        ("[]", "root must be a JSON object"),
        ('{"nodes": []}', '"entities"'),
        ('{"nodes": {}, "entities": []}', '"nodes" must be a list'),
        ('{"nodes": [], "entities": {}}', '"entities" must be a list'),
        (b"\xff\xfe\x00", "not valid UTF-8"),
    ],
)
def test_parse_snapshot_rejects_non_snapshots(raw, message: str) -> None:
    with pytest.raises(SnapshotImportError, match=message):
        parse_snapshot(raw)


def test_snapshot_summary_counts_the_useful_parts() -> None:
    assert snapshot_summary(_payload()) == "Imported 1 entities · 1 actions · 1 zones"
    assert snapshot_summary({"nodes": [], "entities": []}) == "Imported 0 entities · 0 actions"

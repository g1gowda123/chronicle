"""Tests for the fixed-label absence comparison helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from spore_absence_fixed import COCO_ALIASES, comparison  # noqa: E402


def _world(term: str, present: bool, hits: dict[str, int]) -> dict:
    return {"term": term, "present": present, "hits": hits}


def _fixed(term: str, present: bool, hits: dict[str, int], covered: bool = True) -> dict:
    return {
        "term": term,
        "present": present,
        "covered": covered,
        "hits": hits,
        "max_confidence": 0.5,
    }


def test_aliases_cover_open_vocabulary_terms() -> None:
    assert COCO_ALIASES["television"] == ("tv",)
    assert COCO_ALIASES["sofa"] == ("couch",)
    assert COCO_ALIASES["table"] == ("dining table",)


def test_comparison_counts_present_and_absent_terms() -> None:
    world = {
        "term_rows": [
            _world("cup", True, {"0.22": 5, "0.3": 5, "0.5": 5}),
            _world("bowl", False, {"0.22": 3, "0.3": 2, "0.5": 0}),
        ]
    }
    fixed = {
        "model": "yolov8s.pt",
        "terms": [
            _fixed("cup", True, {"0.25": 5, "0.5": 5}),
            _fixed("bowl", False, {"0.25": 0, "0.5": 0}),
        ],
    }
    rows = comparison(world, fixed, audited=False)
    by_key = {(row["detector"], row["floor"]): row for row in rows}
    assert by_key[("YOLO-World-S", 0.22)]["present_found"] == 1
    assert by_key[("YOLO-World-S", 0.22)]["absent_clean"] == 0
    assert by_key[("YOLO-World-S", 0.5)]["absent_clean"] == 1
    assert by_key[("yolov8s.pt", 0.25)]["present_found"] == 1
    assert by_key[("yolov8s.pt", 0.25)]["absent_clean"] == 1


def test_audited_flips_television_to_present() -> None:
    world = {
        "term_rows": [
            _world("cup", True, {"0.22": 5, "0.3": 5, "0.5": 5}),
            _world("television", False, {"0.22": 2, "0.3": 2, "0.5": 0}),
        ]
    }
    fixed = {
        "model": "yolov8s.pt",
        "terms": [
            _fixed("cup", True, {"0.25": 5, "0.5": 5}),
            _fixed("television", False, {"0.25": 3, "0.5": 1}),
        ],
    }
    original = {
        (row["detector"], row["floor"]): row
        for row in comparison(world, fixed, audited=False)
    }
    audited = {
        (row["detector"], row["floor"]): row
        for row in comparison(world, fixed, audited=True)
    }
    assert original[("YOLO-World-S", 0.22)]["present_total"] == 1
    assert original[("YOLO-World-S", 0.22)]["absent_total"] == 1
    assert audited[("YOLO-World-S", 0.22)]["present_total"] == 2
    assert audited[("YOLO-World-S", 0.22)]["absent_total"] == 0


def test_uncovered_terms_are_excluded() -> None:
    world = {
        "term_rows": [
            _world("cup", True, {"0.22": 5, "0.3": 5, "0.5": 5}),
            _world("plate", True, {"0.22": 5, "0.3": 5, "0.5": 5}),
        ]
    }
    fixed = {
        "model": "yolov8s.pt",
        "terms": [
            _fixed("cup", True, {"0.25": 5, "0.5": 5}),
            _fixed("plate", True, {"0.25": 0, "0.5": 0}, covered=False),
        ],
    }
    rows = comparison(world, fixed, audited=False)
    for row in rows:
        assert row["present_total"] == 1

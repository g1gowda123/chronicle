"""Tests for the independent Aria scene-annotation evaluation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aria_scene_eval import (  # noqa: E402
    SceneBox,
    boxes_near,
    load_scene_boxes,
    select_questions,
)

from chronicle.types import BBox  # noqa: E402


def test_select_questions_dedupes_boxes_by_object_uid() -> None:
    boxes = [
        SceneBox("table", 10.0, BBox(20, 20, 120, 100), "0"),
        SceneBox("table", 10.1, BBox(21, 20, 121, 100), "0"),
        SceneBox("table", 10.2, BBox(22, 20, 122, 100), "0"),
        SceneBox("table", 10.1, BBox(200, 20, 300, 100), "1"),
    ]
    questions = select_questions(
        boxes,
        duration=40.0,
        max_categories=1,
        max_per_category=1,
        min_area=1.0,
    )
    assert len(questions) == 1
    question = questions[0]
    assert question["query"] == "Where is a table?"
    assert question["simultaneous_instances"] == 2
    assert len(question["target_boxes"]) == 2


def test_boxes_near_accepts_any_instance_at_nearest_timestamp() -> None:
    boxes = [
        SceneBox("table", 10.0, BBox(0, 0, 20, 20), "0"),
        SceneBox("table", 12.0, BBox(30, 30, 50, 50), "1"),
        SceneBox("table", 12.1, BBox(31, 30, 51, 50), "2"),
        SceneBox("sofa", 12.0, BBox(60, 60, 90, 90), "3"),
    ]
    targets = boxes_near(boxes, "table", 12.2, tolerance=0.5)
    assert len(targets) == 2
    assert {tuple(box.xyxy) for box in targets} == {(30.0, 30.0, 50.0, 50.0), (31.0, 30.0, 51.0, 50.0)}


def test_boxes_near_rejects_annotations_without_a_nearby_sample() -> None:
    boxes = [SceneBox("table", 10.0, BBox(0, 0, 20, 20), "0")]
    assert boxes_near(boxes, "table", 15.0, tolerance=0.5) == []
    assert boxes_near(boxes, "sofa", 10.0) == []


def test_load_scene_boxes_scales_full_resolution_to_query_space(tmp_path: Path) -> None:
    sequence = "eat_0"
    scene = tmp_path / sequence / "scene" / "scene"
    scene.mkdir(parents=True)
    (scene / "instances.json").write_text(json.dumps({"0": {"category": "table"}}))
    with (scene / "2d_bounding_box.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stream_id",
                "object_uid",
                "timestamp[ns]",
                "x_min[pixel]",
                "x_max[pixel]",
                "y_min[pixel]",
                "y_max[pixel]",
                "visibility_ratio[%]",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stream_id": "214-1",
                "object_uid": "0",
                "timestamp[ns]": "0",
                "x_min[pixel]": "0",
                "x_max[pixel]": "2560",
                "y_min[pixel]": "0",
                "y_max[pixel]": "1920",
                "visibility_ratio[%]": "-1",
            }
        )
        writer.writerow(
            {
                "stream_id": "214-1",
                "object_uid": "0",
                "timestamp[ns]": "10000000000",
                "x_min[pixel]": "0",
                "x_max[pixel]": "2560",
                "y_min[pixel]": "0",
                "y_max[pixel]": "1920",
                "visibility_ratio[%]": "-1",
            }
        )
    item = {
        "name": sequence,
        "sequence": sequence,
        "duration_s": 45.0,
        "main_stream": "214-1",
    }
    boxes, meta = load_scene_boxes(tmp_path, item)
    # The first timestamp is filtered out because it is before 2 s.
    assert len(boxes) == 1
    assert boxes[0].bbox.xyxy == (0.0, 0.0, 640.0, 480.0)
    assert meta["origin_s"] == 0.0
    assert meta["boxes"] == 1

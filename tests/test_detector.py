from __future__ import annotations

import cv2
import numpy as np

from chronicle.perception.config import DetectorConfig, PerceptionConfig
from chronicle.perception.detector import (
    ColorDetector,
    YoloWorldDetector,
    _tile_windows,
    deduplicate_detections,
    filter_min_area,
)
from chronicle.perception.palette import PALETTE
from chronicle.types import BBox, Detection


def detection(
    bbox: tuple[float, float, float, float],
    *,
    label: str = "keys",
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        label=label,
        bbox=BBox(*bbox),
        confidence=confidence,
        source="test",
    )


def test_dedup_collapses_nested_same_label_boxes() -> None:
    outer = detection((100, 100, 200, 200), confidence=0.9)
    inner = detection((140, 110, 160, 190), confidence=0.7)
    # IoU is only 0.16 for this tall nested box, so this specifically exercises
    # the intersection-over-smaller gate that open-vocabulary detectors need.
    assert outer.bbox.iou(inner.bbox) < 0.55
    assert outer.bbox.ios(inner.bbox) > 0.8
    kept = deduplicate_detections([inner, outer], iou_threshold=0.55, ios_threshold=0.8)
    assert kept == [outer]


def test_dedup_never_merges_different_labels() -> None:
    keys = detection((100, 100, 200, 200), label="keys")
    phone = detection((105, 105, 195, 195), label="phone")
    kept = deduplicate_detections([keys, phone])
    assert {item.label for item in kept} == {"keys", "phone"}


def test_dedup_caps_instances_per_label() -> None:
    boxes = [
        detection((10.0, 10.0, 50.0, 50.0)),
        detection((100.0, 10.0, 140.0, 50.0)),
        detection((200.0, 10.0, 240.0, 50.0)),
    ]
    assert len(deduplicate_detections(boxes, max_per_label=2)) == 2


def test_filter_min_area_uses_per_label_floors() -> None:
    small_keys = detection((0, 0, 5, 5))  # 25 px^2
    large_phone = detection((0, 0, 100, 100), label="phone")  # 10000 px^2
    kept = filter_min_area([small_keys, large_phone], {"keys": 90.0, "phone": 500.0})
    assert kept == [large_phone]


def test_color_detector_finds_palette_object() -> None:
    frame = np.full((240, 320, 3), 20, dtype=np.uint8)  # dark neutral background
    cv2.rectangle(frame, (80, 90), (140, 130), PALETTE["keys"], -1)
    detector = ColorDetector(
        DetectorConfig(confidence=0.3), vocabulary=["keys", "phone", "table"]
    )
    detections = detector.detect(frame, frame_index=3)
    assert len(detections) == 1
    found = detections[0]
    assert found.label == "keys"
    assert found.source == "color-palette"
    assert found.bbox.ios(BBox(80, 90, 140, 130)) > 0.8
    assert found.metadata["frame"] == 3


def test_color_detector_ignores_objects_outside_vocabulary() -> None:
    frame = np.full((120, 160, 3), 20, dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (80, 80), PALETTE["backpack"], -1)
    detector = ColorDetector(DetectorConfig(confidence=0.3), vocabulary=["keys"])
    assert detector.detect(frame) == []


def test_yolo_prompt_aliases_expand_and_canonicalise() -> None:
    root = PerceptionConfig(vocabulary=("cup", "plate"))
    root.detector.prompt_aliases = {"cup": ("mug", "coffee cup")}
    detector = YoloWorldDetector(root.detector, root)
    assert detector._prompts == ("cup", "mug", "coffee cup", "plate")
    assert detector._aliases["mug"] == "cup"
    assert detector._aliases["coffee cup"] == "cup"

    detector.set_vocabulary(("cup",))
    assert detector._prompts == ("cup", "mug", "coffee cup")
    assert detector._labels == {"cup"}


def test_yolo_model_ensemble_is_configured_without_loading_weights() -> None:
    root = PerceptionConfig(vocabulary=("cup", "cutting board"))
    root.detector.model = "yolov8s-worldv2.pt"
    root.detector.models = ("yolov8s-worldv2.pt", "yolov8l-worldv2.pt")
    detector = YoloWorldDetector(root.detector, root)
    assert detector._model_names == ("yolov8s-worldv2.pt", "yolov8l-worldv2.pt")
    assert detector.name == "yolo-world-ensemble"

    single = YoloWorldDetector(DetectorConfig(model="yolov8s-worldv2.pt"), root)
    assert single._model_names == ("yolov8s-worldv2.pt",)
    assert single.name == "yolo-world"


def test_tile_windows_cover_the_frame() -> None:
    windows = _tile_windows(640, 480, 2, 0.15)
    assert len(windows) == 4
    assert windows[0][0] == 0 and windows[0][1] == 0
    assert windows[-1][2] == 640 and windows[-1][3] == 480
    assert all(x2 > x1 and y2 > y1 for x1, y1, x2, y2 in windows)

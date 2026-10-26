"""Model-free tests for cheap visual object attributes (colour, shape, ...)."""

from __future__ import annotations

import csv
import io

import numpy as np

from chronicle.attributes import (
    attribute_query_field,
    attribute_summary,
    describe_attribute,
    merge_attributes,
    public_attributes,
)
from chronicle.graph.adapter import frame_context_from_result
from chronicle.integration import ChronicleRuntime
from chronicle.perception.attributes import AttributeExtractor, extract_object_attributes
from chronicle.perception.config import PerceptionConfig
from chronicle.perception.tracker import ByteTrackLite
from chronicle.semantic.graphrag import GraphContext, template_answer
from chronicle.types import (
    BBox,
    CameraMotion,
    CameraState,
    Detection,
    FrameContext,
    FrameResult,
    MotionEstimate,
    ObjectObservation,
    TrackedObject,
)
from chronicle.ui.graph_view import graph_options

DT = 1.0 / 30.0


def _frame() -> np.ndarray:
    frame = np.full((240, 320, 3), 28, np.uint8)
    frame[80:160, 100:220] = (0, 0, 230)  # OpenCV BGR red
    return frame


def _red_object(**kwargs) -> ObjectObservation:
    return ObjectObservation(
        entity_id="cup",
        label="cup",
        bbox=(100.0, 80.0, 220.0, 160.0),
        attributes={
            "colour": "red",
            "colour_hex": "#e60000",
            "colour_confidence": 0.94,
            "shape": "wide",
            "size": "large",
            "texture": "plain",
            "brightness": "bright",
        },
        **kwargs,
    )


def test_extracts_primary_colour_and_simple_properties() -> None:
    attrs = extract_object_attributes(_frame(), BBox(100, 80, 220, 160))

    assert attrs["colour"] == "red"
    assert attrs["colour_hex"].startswith("#")
    assert attrs["shape"] == "wide"
    assert attrs["size"] == "large"
    assert attrs["texture"] == "plain"
    assert attrs["position"] == "centre"
    assert 0.0 < attrs["colour_confidence"] <= 1.0


def test_achromatic_colours_are_not_confused_with_hues() -> None:
    frame = np.full((120, 120, 3), 20, np.uint8)
    box = BBox(20, 20, 100, 100)
    cases = {
        "white": (245, 245, 245),
        "black": (5, 5, 5),
        "grey": (120, 120, 120),
    }
    for expected, bgr in cases.items():
        frame[20:100, 20:100] = bgr
        assert extract_object_attributes(frame, box)["colour"] == expected


def test_public_attributes_ignore_internal_track_metadata() -> None:
    metadata = {
        "detector": "yolo-world",
        "entity_id": "track_4",
        "reid_score": 0.91,
        "attributes": {
            "colour": "blue",
            "colour_hex": "#1e40af",
            "shape": "tall",
            "size": "small",
            "future_field": "ignored",
        },
    }

    attrs = public_attributes(metadata)

    assert attrs["colour"] == "blue"
    assert attrs["shape"] == "tall"
    assert "detector" not in attrs
    assert "future_field" not in attrs
    assert attribute_summary(attrs) == "blue · tall · small"
    assert "blue" in describe_attribute(attrs, "colour")


def test_attribute_query_parser_disambiguates_property_questions() -> None:
    assert attribute_query_field("what colour is the mug?") == "colour"
    assert attribute_query_field("how big is the cup?") == "size"
    assert attribute_query_field("describe the backpack") == "all"
    assert attribute_query_field("where is the cup?") is None


def test_colour_merge_keeps_stable_identity_across_noisy_frames() -> None:
    merged = merge_attributes(
        {"colour": "red", "colour_confidence": 0.8, "shape": "wide"},
        {"colour": "brown", "colour_confidence": 0.2, "shape": "square"},
    )
    assert merged["colour"] == "red"
    assert merged["shape"] == "square"

    switched = merge_attributes(
        {"colour": "red", "colour_confidence": 0.3},
        {"colour": "blue", "colour_confidence": 0.85},
    )
    assert switched["colour"] == "blue"


def test_tracker_and_adapter_carry_attributes_into_the_graph_contract() -> None:
    config = PerceptionConfig()
    config.tracker.min_hits = 1
    tracker = ByteTrackLite(config.tracker)
    detection = Detection(
        label="cup",
        bbox=BBox(100, 80, 220, 160),
        confidence=0.9,
        metadata={"attributes": {"colour": "red", "shape": "wide", "size": "large"}},
    )
    tracker.update(None, [detection], 0.0, 0)
    result = tracker.update(None, [detection], DT, 1)
    assert result.tracks[0].metadata["attributes"]["colour"] == "red"

    frame = FrameResult(
        frame_index=1,
        timestamp=DT,
        wall_time=DT,
        motion=MotionEstimate(state=CameraState.STATIONARY),
        detections=[detection],
        hands=[],
        tracks=[
            TrackedObject(
                track_id=1,
                label="cup",
                bbox=BBox(100, 80, 220, 160),
                confidence=0.9,
                first_seen=0.0,
                last_seen=DT,
                hits=2,
                age=2,
                metadata={"attributes": {"colour": "red", "shape": "wide"}},
            )
        ],
        interactions=[],
        interaction_events=[],
        clips=[],
        telemetry=[],
    )
    context = frame_context_from_result(frame)
    assert context.objects[0].attributes["colour"] == "red"
    assert context.objects[0].attributes["shape"] == "wide"


def test_attribute_extractor_throttles_without_losing_first_frame() -> None:
    extractor = AttributeExtractor(
        {"enabled": True, "interval": 5, "min_crop_px": 8, "colour_clusters": 3}
    )
    first = Detection(label="cup", bbox=BBox(100, 80, 220, 160))
    skipped = Detection(label="cup", bbox=BBox(100, 80, 220, 160))
    assert extractor.annotate(_frame(), [first], 0) == 1
    assert first.metadata["attributes"]["colour"] == "red"
    assert extractor.annotate(_frame(), [skipped], 3) == 0
    assert skipped.metadata == {}


def test_pipeline_wires_attribute_extraction_into_tracks() -> None:
    class OneBoxDetector:
        name = "one-box"

        def detect(self, frame, frame_index=0, timestamp=0.0):
            return [
                Detection(
                    label="mug",
                    bbox=BBox(100, 80, 220, 160),
                    confidence=0.9,
                    source=self.name,
                )
            ]

        def close(self) -> None:
            pass

    class NoHands:
        name = "no-hands"

        def detect(self, frame, frame_index=0, timestamp=0.0):
            return []

        def close(self) -> None:
            pass

    from chronicle.perception.embedder import ColorEmbedder
    from chronicle.perception.pipeline import PerceptionPipeline

    config = PerceptionConfig()
    config.motion.enabled = False
    config.interaction.enabled = False
    config.clip.enabled = False
    config.tracker.min_hits = 1
    pipeline = PerceptionPipeline(
        config,
        detector=OneBoxDetector(),
        hands=NoHands(),
        embedder=ColorEmbedder(config.embedder),
    )
    try:
        result = pipeline.process_frame(_frame(), 0.0, 0)
    finally:
        pipeline.close()

    assert result.tracks
    assert result.tracks[0].metadata["attributes"]["colour"] == "red"
    assert result.detections[0].metadata["attributes"]["shape"] == "wide"


def test_runtime_snapshot_query_and_csv_expose_appearance() -> None:
    runtime = ChronicleRuntime(
        ":memory:",
        semantic_enabled=False,
        zone_label="kitchen",
        reset=True,
    )
    try:
        for index in range(30):
            runtime.memory.observe(
                _make_frame(index * DT, objects=[_red_object()], zone_label="kitchen")
            )

        snapshot = runtime.snapshot()
        row = snapshot["entities"][0]
        assert row["colour"] == "red"
        assert row["attributes"]["shape"] == "wide"

        graph = {node["id"]: node for node in snapshot["nodes"]}
        assert graph["entity:cup"]["colour_hex"] == "#e60000"

        answer = runtime.ask("what colour is the cup?")
        assert "red" in answer.answer.lower()
        assert answer.grounded
        assert answer.citations

        inventory = runtime.ask("what red objects did you see?")
        assert "red" in inventory.answer.lower()
        assert inventory.method == "colour-filter"

        reader = csv.DictReader(io.StringIO(runtime.export_entities_csv()))
        csv_row = next(reader)
        assert csv_row["colour"] == "red"
        assert csv_row["colour_hex"] == "#e60000"
        assert "wide" in csv_row["attributes"]
    finally:
        runtime.close()


def test_template_answer_uses_appearance_and_cites_last_seen() -> None:
    context = GraphContext.from_mapping(
        {
            "entity": {"id": "cup", "label": "cup", "zone": "kitchen"},
            "last_seen": 7.5,
            "attributes": {
                "colour": "red",
                "colour_hex": "#e60000",
                "shape": "wide",
                "size": "large",
            },
            "facts": [
                {
                    "subject": "cup",
                    "relation": "RESTING_ON",
                    "object": "table",
                    "timestamp": 7.5,
                    "zone": "kitchen",
                }
            ],
        }
    )

    answer = template_answer("what colour is the cup?", context)

    assert answer.grounded
    assert "red" in answer.answer.lower()
    assert "[t=7.5s]" in answer.answer
    assert answer.citations[0].timestamp == 7.5


def test_graph_view_uses_detected_colour_as_an_identity_ring() -> None:
    options = graph_options(
        {
            "nodes": [
                {
                    "id": "entity:cup",
                    "label": "cup",
                    "group": "entity",
                    "state": "VISIBLE_STATIC",
                    "colour": "red",
                    "colour_hex": "#e60000",
                    "attributes": {"colour": "red", "shape": "wide"},
                }
            ],
            "edges": [],
        }
    )
    node = options["series"][0]["data"][0]
    assert node["itemStyle"]["borderColor"] == "#e60000"
    assert "red" in node["tooltip"]["formatter"]


def _make_frame(t: float, *, objects: list[ObjectObservation], zone_label: str | None):
    return FrameContext(
        t=t,
        objects=tuple(objects),
        hands=(),
        supports=(),
        camera=CameraMotion(0.0, 0.0, 0.0),
        frame_size=(320, 240),
        zone_id="kitchen",
        attributes={"zone_label": zone_label} if zone_label else {},
    )

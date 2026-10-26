"""Static-furniture routing: one persistent anchor per real-world object."""

from __future__ import annotations

from pathlib import Path

from chronicle.config import FurnitureConfig
from chronicle.graph.furniture import FurnitureRegistry, bbox_iou
from chronicle.graph.memory import ChronicleMemory
from chronicle.types import CameraMotion, FrameContext, ObjectObservation


def sofa(
    entity_id: str = "sofa-track",
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 300.0, 250.0),
    *,
    track_id: int = 1,
) -> ObjectObservation:
    return ObjectObservation(
        entity_id=entity_id,
        label="sofa",
        bbox=bbox,
        track_id=track_id,
    )


def frame(t: float, objects: tuple[ObjectObservation, ...], camera: CameraMotion | None = None) -> FrameContext:
    return FrameContext(
        t=t,
        objects=objects,
        camera=camera or CameraMotion(),
        zone_id="living-room",
        frame_size=(800, 600),
    )


def test_two_covisible_sofas_stay_separate() -> None:
    registry = FurnitureRegistry(FurnitureConfig(min_hits=1))
    left = registry.observe("sofa", (0.0, 100.0, 220.0, 260.0), "living-room", 0.0, track_id=1)
    right = registry.observe("sofa", (560.0, 100.0, 800.0, 260.0), "living-room", 0.0, track_id=2)

    assert left is not None and right is not None
    assert left.anchor_id != right.anchor_id
    assert len(registry) == 2
    assert bbox_iou(left.bbox, right.bbox) == 0.0


def test_camera_compensation_rebinds_a_lost_furniture_track() -> None:
    registry = FurnitureRegistry(FurnitureConfig(min_hits=1))
    first = registry.observe("sofa", (100.0, 100.0, 300.0, 250.0), "living-room", 0.0, track_id=1)
    assert first is not None

    # The tracker expires the old id; the camera pans 40 px and the sofa is
    # re-detected with a new track id.  Motion-compensated IoU must recover it.
    again = registry.observe(
        "sofa",
        (140.0, 100.0, 340.0, 250.0),
        "living-room",
        2.0,
        track_id=9,
        camera_dx=40.0,
        camera_dy=0.0,
    )
    assert again is not None
    assert again.anchor_id == first.anchor_id
    assert not again.is_new
    assert len(registry) == 1


def test_memory_routes_sofa_to_background_not_foreground() -> None:
    memory = ChronicleMemory(":memory:")
    try:
        for index in range(3):
            memory.observe(frame(index / 30.0, (sofa(),)))

        foreground_labels = {row["label"] for row in memory.foreground.entities()}
        assert "sofa" not in foreground_labels
        furniture = memory.background.furniture()
        assert len(furniture) == 1
        assert furniture[0]["label"] == "sofa"
        assert furniture[0]["bbox"] is not None
        assert furniture[0]["attrs"]["confirmed"] is True
    finally:
        memory.close()


def test_furniture_anchor_survives_reopen_without_duplication(tmp_path: Path) -> None:
    db = tmp_path / "chronicle.db"
    memory = ChronicleMemory(db)
    try:
        for index in range(3):
            memory.observe(frame(index / 30.0, (sofa(),)))
        assert len(memory.background.furniture()) == 1
    finally:
        memory.close()

    reopened = ChronicleMemory(db)
    try:
        assert len(reopened.background.furniture()) == 1
        assert len(reopened.furniture) == 1
        reopened.observe(frame(2.0, (sofa(track_id=7),)))
        furniture = reopened.background.furniture()
        assert len(furniture) == 1
        assert furniture[0]["attrs"]["track_ids"] == [1, 7]
    finally:
        reopened.close()

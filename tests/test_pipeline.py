from __future__ import annotations

import json

import numpy as np

from chronicle.perception.config import PerceptionConfig
from chronicle.perception.embedder import ColorEmbedder
from chronicle.perception.pipeline import PerceptionPipeline
from chronicle.perception.stream import PerceptionStream
from chronicle.types import BBox, CameraState, Detection, HandObservation, MotionEstimate


class StaticDetector:
    name = "test-detector"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame, frame_index=0, timestamp=0.0):
        self.calls += 1
        return [
            Detection(
                label="keys",
                bbox=BBox(100, 100, 140, 140),
                confidence=0.92,
                source=self.name,
            )
        ]

    def close(self) -> None:
        pass


class ScriptedHands:
    """A hand sits on the keys for the first half of the clip, then leaves."""

    name = "test-hands"

    def __init__(self, *, leave_at: int = 15) -> None:
        self.leave_at = leave_at
        self.calls = 0

    @staticmethod
    def _hand() -> HandObservation:
        points = np.stack(
            [np.linspace(108, 132, 21), np.linspace(108, 132, 21)], axis=1
        ).astype(np.float32)
        return HandObservation(landmarks=points, confidence=0.99, handedness="Right")

    def detect(self, frame, frame_index=0, timestamp=0.0):
        self.calls += 1
        return [] if frame_index >= self.leave_at else [self._hand()]

    def close(self) -> None:
        pass


class ShiftedMotion:
    """Pretend the camera panned 10 px, to check relative-velocity removal."""

    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, frame, exclude=(), *, timestamp=0.0) -> MotionEstimate:
        self.calls += 1
        return MotionEstimate(dx=10.0, dy=0.0, valid=True, state=CameraState.STATIONARY)

    def reset(self) -> None:
        self.calls = 0


def make_config() -> PerceptionConfig:
    config = PerceptionConfig()
    config.tracker.min_hits = 1
    config.interaction.dwell_seconds = 0.2
    config.interaction.release_grace = 0.1
    config.clip.length = 12
    config.clip.pre_roll = 5
    config.clip.post_roll = 2
    config.clip.max_width = 320
    config.telemetry_interval = 5
    return config


def frames(count: int, width: int = 320, height: int = 240) -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return [base.copy() for _ in range(count)]


def test_pipeline_emits_jsonl_clips_and_telemetry(tmp_path) -> None:
    config = make_config()
    detector = StaticDetector()
    hands = ScriptedHands()
    jsonl_path = tmp_path / "frames.jsonl"
    clips_dir = tmp_path / "clips"
    telemetry_categories: set[str] = set()

    pipeline = PerceptionPipeline(
        config,
        detector=detector,
        hands=hands,
        embedder=ColorEmbedder(config.embedder),
        output_jsonl=jsonl_path,
        clips_dir=clips_dir,
        telemetry_sink=lambda event: telemetry_categories.add(event.category.value),
    )
    results = list(pipeline.iter_results(frames(30), fps_override=30.0))
    summary = pipeline.summary()
    pipeline.close()

    assert len(results) == 30
    assert detector.calls == 30
    assert summary.frames == 30
    assert summary.tracks == 1
    assert summary.interactions == 1
    assert summary.clips == 1
    assert summary.camera_states["STATIONARY"] >= 20
    assert {"SYSTEM", "GEOM", "TRACK", "PERCEPTION", "CLIP"} <= telemetry_categories

    # Every interaction clip is a structured, 30-frame-window artifact.
    assert results[-1].clips == []  # clips are emitted on the frame they finish
    emitted = [clip for result in results for clip in result.clips]
    assert len(emitted) == 1
    clip = emitted[0]
    assert clip.label == "keys"
    assert clip.track_id == 1
    assert len(clip.frames) == 12
    assert 0 <= clip.contact_index < len(clip.frames)
    # The interaction starts at t=0 in this fixture, so there is no pre-roll.
    assert clip.contact_index == 0
    assert clip.composite is not None
    assert (clips_dir / f"{clip.clip_id}.jpg").exists()
    metadata = json.loads((clips_dir / f"{clip.clip_id}.json").read_text())
    assert metadata["label"] == "keys"
    assert metadata["frames"]

    # One JSON object per frame, plus the run-summary trailer written at EOF.
    lines = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(lines) == 31
    assert [line["frame_index"] for line in lines[:-1]] == list(range(30))
    assert lines[-1]["type"] == "run_summary"
    assert lines[-1]["summary"]["frames"] == 30
    frame_events = [
        event
        for line in lines[:-1]
        for event in line["interaction_events"]
    ]
    assert [event["kind"] for event in frame_events] == ["start", "end"]
    assert frame_events[0]["label"] == "keys"


def test_pipeline_on_clip_callback_is_isolated(caplog) -> None:
    config = make_config()
    seen = []

    def callback(clip) -> None:
        seen.append(clip)
        raise RuntimeError("semantic worker down")

    pipeline = PerceptionPipeline(
        config,
        detector=StaticDetector(),
        hands=ScriptedHands(),
        embedder=ColorEmbedder(config.embedder),
        on_clip=callback,
    )
    results = list(pipeline.iter_results(frames(10), fps_override=30.0))
    pipeline.close()

    assert len(results) == 10
    assert len(seen) == 1
    assert seen[0].label == "keys"
    # The clip is still pending at EOF, so this also covers the flush path.
    assert pipeline.flushed_clips == seen
    assert "clip callback failed" in caplog.text


def test_pipeline_subtracts_camera_motion_from_track_velocity() -> None:
    config = make_config()
    config.tracker.min_hits = 1
    pipeline = PerceptionPipeline(
        config,
        detector=StaticDetector(),
        hands=ScriptedHands(leave_at=0),
        embedder=ColorEmbedder(config.embedder),
        motion=ShiftedMotion(),
    )
    result = None
    for index in range(3):
        result = pipeline.process_frame(frames(1)[0], index / 30.0, index)
    pipeline.close()

    assert result is not None
    track = result.tracks[0]
    assert abs(track.relative_velocity[0] + 300.0) < 1e-6
    assert abs(track.velocity[0]) < 1e-6


def test_stream_delivers_every_frame_and_stops_cleanly() -> None:
    config = make_config()
    pipeline = PerceptionPipeline(
        config,
        detector=StaticDetector(),
        hands=ScriptedHands(leave_at=0),
        embedder=ColorEmbedder(config.embedder),
        output_jsonl=None,
        clips_dir=None,
    )
    stream = PerceptionStream(pipeline, frames(8), queue_size=0, fps_override=30.0)
    with stream:
        received = list(stream)
    assert [result.frame_index for result in received] == list(range(8))
    assert not stream.running

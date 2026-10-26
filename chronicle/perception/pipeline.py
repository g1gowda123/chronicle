"""Role A orchestration: video -> gated perception events + interaction clips."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from chronicle.perception.clip_buffer import FrameRingBuffer, InteractionClipRecorder
from chronicle.perception.config import PerceptionConfig
from chronicle.perception.detector import build_detector
from chronicle.perception.embedder import build_embedder
from chronicle.perception.hands import build_hands
from chronicle.perception.interactions import InteractionTracker
from chronicle.perception.motion import AffineMotionEstimator
from chronicle.perception.sources import VideoSource
from chronicle.perception.support import SupportTracker
from chronicle.perception.tracker import ByteTrackLite
from chronicle.types import (
    CameraState,
    EventCategory,
    FrameResult,
    InteractionClip,
    InteractionEvent,
    MotionEstimate,
    RunSummary,
    SupportSurface,
    TelemetryEvent,
    TrackedObject,
)

LOGGER = logging.getLogger(__name__)

# Detector labels that can contain a manipulated object.  This is only an
# evidence hint for Role C; Role B owns the final containment semantics.
_CONTAINER_HINTS = frozenset(
    {
        "backpack",
        "bag",
        "basket",
        "bin",
        "bowl",
        "box",
        "bucket",
        "case",
        "container",
        "drawer",
        "fridge",
        "jar",
        "pot",
        "shelf",
        "trash can",
    }
)


class PerceptionPipeline:
    """Compose motion gating, detection, hands, tracking and clip assembly.

    The pipeline is single-threaded. :meth:`run` and :meth:`iter_results` are
    the two supported entry points; use :class:`PerceptionStream` (or a worker
    thread in the consumer) if you need non-blocking UI ingestion.
    """

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        *,
        detector: object | None = None,
        hands: object | None = None,
        embedder: object | None = None,
        tracker: ByteTrackLite | None = None,
        motion: AffineMotionEstimator | None = None,
        interactions: InteractionTracker | None = None,
        telemetry_sink: Callable[[TelemetryEvent], None] | None = None,
        on_clip: Callable[[InteractionClip], None] | None = None,
        output_jsonl: str | Path | None = None,
        clips_dir: str | Path | None = None,
    ) -> None:
        self.config = config or PerceptionConfig()
        self._detector_override = detector
        self._hands_override = hands
        self._embedder_override = embedder
        self._tracker_override = tracker
        self._motion_override = motion
        self._interactions_override = interactions
        self.telemetry_sink = telemetry_sink
        self.on_clip = on_clip
        self.output_jsonl = Path(output_jsonl).expanduser() if output_jsonl else None
        self.clips_dir = Path(clips_dir).expanduser() if clips_dir else None

        self.detector = None
        self.hands = None
        self.embedder = None
        self.tracker = None
        self.motion = None
        self.interactions = None
        self.support: SupportTracker | None = None

        self._buffer = FrameRingBuffer(
            maxlen=self.config.clip.ring_size, max_width=self.config.clip.max_width
        )
        self._recorder = InteractionClipRecorder(self.config.clip)
        self._prepared = False
        self._closed = False
        self._source_name = ""
        self._fps = 30.0
        self._last_hands: list = []
        self._last_supports: list[SupportSurface] = []
        self._last_tracks: list[TrackedObject] = []
        self._last_dynamic_boxes: list = []
        self._previous_timestamp: float | None = None
        self._last_timestamp = 0.0
        self._frame_count = 0
        self._run_started = time.monotonic()
        self._detections_seen = 0
        self._track_ids: set[int] = set()
        self._interaction_starts = 0
        self._clip_count = 0
        self._camera_counts: Counter[str] = Counter()
        self._pending_telemetry: list[TelemetryEvent] = []
        self._jsonl_handle = None
        self.flushed_events: list[InteractionEvent] = []
        self.flushed_clips: list[InteractionClip] = []

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------
    @property
    def prepared(self) -> bool:
        return self._prepared

    @property
    def fps(self) -> float:
        return self._fps

    def set_vocabulary(self, vocabulary: Iterable[str]) -> None:
        """Update the open-vocabulary targets, before or after ``prepare``."""
        labels = tuple(
            dict.fromkeys(str(label).strip().lower() for label in vocabulary if str(label).strip())
        )
        if not labels:
            return
        self.config.vocabulary = labels
        if self.detector is not None:
            self.detector.set_vocabulary(labels)

    def prepare(self, *, source_name: str = "", fps: float = 30.0) -> None:
        if self._prepared:
            return
        if self._closed:
            raise RuntimeError("pipeline is closed")
        self._source_name = source_name or "frames"
        self._fps = fps if fps > 0 else 30.0
        self._run_started = time.monotonic()

        self.detector = self._detector_override or build_detector(
            self.config, source_name=self._source_name
        )
        self.hands = self._hands_override or build_hands(
            self.config, source_name=self._source_name
        )
        self.embedder = self._embedder_override or build_embedder(
            self.config, source_name=self._source_name
        )
        self.tracker = self._tracker_override or ByteTrackLite(
            self.config.tracker, embedder=self.embedder
        )
        self.motion = self._motion_override or AffineMotionEstimator(self.config.motion)
        self.interactions = self._interactions_override or InteractionTracker(
            self.config.interaction
        )
        self.support = SupportTracker(self.config.support)
        self._prepared = True

        system_event = TelemetryEvent(
            category=EventCategory.SYSTEM,
            message=(
                f"perception ready | detector={self._detector_name} "
                f"hands={self._hands_name} embedder={self._embedder_name} "
                f"tracker=bytetrack-lite fps={self._fps:.1f}"
            ),
            data={
                "detector": self._detector_name,
                "hands": self._hands_name,
                "embedder": self._embedder_name,
                "fps": self._fps,
            },
        )
        self._pending_telemetry.append(system_event)
        self._publish(system_event)
        LOGGER.info(system_event.message)

    @property
    def _detector_name(self) -> str:
        return getattr(self.detector, "name", type(self.detector).__name__)

    @property
    def _hands_name(self) -> str:
        return getattr(self.hands, "name", type(self.hands).__name__)

    @property
    def _embedder_name(self) -> str:
        backend = getattr(self.embedder, "backend", None)
        return backend or getattr(self.embedder, "name", type(self.embedder).__name__)

    def reset(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        if self.motion is not None:
            self.motion.reset()
        if self.interactions is not None:
            self.interactions.reset()
        if self.support is not None:
            self.support.reset()
        self._buffer.clear()
        self._recorder = InteractionClipRecorder(self.config.clip)
        self._last_hands = []
        self._last_dynamic_boxes = []
        self._previous_timestamp = None
        self._last_timestamp = 0.0
        self._frame_count = 0
        self._detections_seen = 0
        self._track_ids.clear()
        self._interaction_starts = 0
        self._clip_count = 0
        self._camera_counts.clear()
        self._run_started = time.monotonic()
        self.flushed_events = []
        self.flushed_clips = []

    def close(self) -> None:
        if self._closed:
            return
        for component in (self.detector, self.hands, self.embedder):
            try:
                component.close()  # type: ignore[union-attr]
            except Exception:
                pass
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None
        self._buffer.clear()
        self._closed = True

    def __enter__(self) -> PerceptionPipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, timestamp: float, frame_index: int = 0
    ) -> FrameResult:
        self.prepare(source_name=self._source_name, fps=self._fps)
        assert self.detector is not None
        assert self.hands is not None
        assert self.tracker is not None
        assert self.motion is not None
        assert self.interactions is not None

        detections = self.detector.detect(frame, frame_index, timestamp)
        hand_interval = max(1, int(self.config.hands.interval))
        if frame_index % hand_interval == 0 or not self._last_hands:
            current_hands = self.hands.detect(frame, frame_index, timestamp)
        else:
            current_hands = self._last_hands

        exclude = (
            [detection.bbox for detection in detections]
            + [hand.bbox for hand in current_hands]
            + list(self._last_dynamic_boxes)
        )
        if self.config.motion.enabled:
            motion = self.motion.estimate(frame, exclude, timestamp=timestamp)
        else:
            motion = MotionEstimate(valid=False, state=CameraState.UNKNOWN)

        tracker_result = self.tracker.update(frame, detections, timestamp, frame_index)
        relative_to = (motion.dx, motion.dy) if motion.valid else (0.0, 0.0)
        dt = (
            max(1e-3, timestamp - self._previous_timestamp)
            if self._previous_timestamp is not None
            else 1.0 / max(1e-3, self._fps)
        )
        camera_velocity = (relative_to[0] / dt, relative_to[1] / dt)
        for track in tracker_result.tracks:
            track.relative_velocity = (
                track.velocity[0] - camera_velocity[0],
                track.velocity[1] - camera_velocity[1],
            )

        supports: list[SupportSurface] = []
        if self.support is not None:
            supports = self.support.update(
                tracker_result.tracks,
                timestamp,
                camera_dx=relative_to[0],
                camera_dy=relative_to[1],
            )

        interaction_observations, interaction_events = self.interactions.update(
            tracker_result.tracks, current_hands, timestamp, frame_index
        )

        telemetry: list[TelemetryEvent] = []
        if self._pending_telemetry:
            telemetry.extend(self._pending_telemetry)
            self._pending_telemetry.clear()

        # Geometry gate telemetry (transition always, otherwise throttled).
        if (
            motion.transition is not None
            or (motion.valid and frame_index % max(1, self.config.telemetry_interval) == 0)
        ):
            telemetry.append(
                self._make_event(
                    EventCategory.GEOM,
                    (
                        f"{motion.state.value} dx={motion.dx:+.2f} dy={motion.dy:+.2f} "
                        f"theta={motion.rotation_degrees:+.2f}deg "
                        f"inliers={motion.inlier_ratio:.2f}"
                        + (f" [{motion.transition}]" if motion.transition else "")
                    ),
                    timestamp=timestamp,
                    frame_index=frame_index,
                    data=motion.as_dict(),
                )
            )

        for track in tracker_result.new_tracks:
            telemetry.append(
                self._make_event(
                    EventCategory.TRACK,
                    f"new track #{track.track_id} {track.label} conf={track.confidence:.2f}",
                    timestamp=timestamp,
                    frame_index=frame_index,
                    data={"track_id": track.track_id, "label": track.label},
                )
            )
        for track in tracker_result.lost:
            telemetry.append(
                self._make_event(
                    EventCategory.TRACK,
                    f"lost track #{track.track_id} {track.label} "
                    f"after {max(0.0, timestamp - track.first_seen):.2f}s",
                    timestamp=timestamp,
                    frame_index=frame_index,
                    data={"track_id": track.track_id, "label": track.label},
                )
            )
        for track in tracker_result.rebound:
            score = track.metadata.get("reid_score", 0.0)
            source = track.metadata.get("reid_source", "appearance")
            telemetry.append(
                self._make_event(
                    EventCategory.REID,
                    f"rebound #{track.track_id} {track.label} SC={score:.3f} ({source})",
                    timestamp=timestamp,
                    frame_index=frame_index,
                    data={
                        "track_id": track.track_id,
                        "label": track.label,
                        "score": score,
                        "source": source,
                    },
                )
            )

        snapshot = None
        clips: list[InteractionClip] = []
        if self.config.clip.enabled:
            snapshot = self._buffer.append(
                frame, timestamp, frame_index, tracker_result.tracks, current_hands
            )
            clips = self._recorder.update(self._buffer, snapshot, interaction_events)
        for clip in clips:
            self._handle_clip(clip, timestamp, frame_index, telemetry)

        if frame_index % max(1, self.config.telemetry_interval) == 0:
            telemetry.append(
                self._make_event(
                    EventCategory.PERCEPTION,
                    (
                        f"frame={frame_index} t={timestamp:.2f}s det={len(detections)} "
                        f"hands={len(current_hands)} tracks={len(tracker_result.tracks)} "
                        f"interactions={len(interaction_observations)}"
                    ),
                    timestamp=timestamp,
                    frame_index=frame_index,
                    data={
                        "detections": len(detections),
                        "hands": len(current_hands),
                        "tracks": len(tracker_result.tracks),
                    },
                )
            )

        self._last_hands = list(current_hands)
        self._last_dynamic_boxes = [
            detection.bbox for detection in detections
        ] + [hand.bbox for hand in current_hands]
        self._previous_timestamp = timestamp
        self._last_timestamp = timestamp
        self._frame_count += 1
        self._detections_seen += len(detections)
        self._track_ids.update(track.track_id for track in tracker_result.tracks)
        self._interaction_starts += sum(
            1 for event in interaction_events if event.kind == "start"
        )
        self._camera_counts[motion.state.value] += 1

        result = FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            wall_time=time.time(),
            motion=motion,
            detections=detections,
            hands=list(current_hands),
            tracks=tracker_result.tracks,
            interactions=interaction_observations,
            interaction_events=interaction_events,
            clips=clips,
            telemetry=telemetry,
            image=frame.copy(),
            supports=supports,
        )
        for event in telemetry:
            self._publish(event)
        if self._jsonl_handle is not None:
            self._jsonl_handle.write(json.dumps(result.as_dict()) + "\n")
        return result

    # ------------------------------------------------------------------
    # Iteration / run
    # ------------------------------------------------------------------
    def iter_results(
        self,
        source: str | Path | int | Iterable[np.ndarray],
        *,
        stop_event: threading.Event | None = None,
        fps_override: float | None = None,
    ) -> Iterator[FrameResult]:
        video = VideoSource(source, fps_override=fps_override)
        self.prepare(source_name=video.name, fps=video.fps)
        if self.output_jsonl is not None:
            self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_handle = self.output_jsonl.open("w", encoding="utf-8")
        if self.clips_dir is not None:
            self.clips_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        last_index = 0
        try:
            for frame, timestamp, index in video:
                if stop_event is not None and stop_event.is_set():
                    break
                if self.config.frame_stride > 1 and index % self.config.frame_stride != 0:
                    continue
                if self.config.max_frames is not None and processed >= self.config.max_frames:
                    break
                last_index = index
                yield self.process_frame(frame, timestamp, index)
                processed += 1
        finally:
            self._finish(timestamp=self._last_timestamp, frame_index=last_index)
            video.close()

    def _finish(self, *, timestamp: float, frame_index: int) -> None:
        if self.interactions is not None:
            self.flushed_events = self.interactions.flush(timestamp, frame_index)
            for event in self.flushed_events:
                self._publish(
                    self._make_event(
                        EventCategory.TRACK,
                        f"interaction {event.kind} {event.label} #{event.track_id}",
                        timestamp=event.timestamp,
                        frame_index=event.frame_index,
                        data=event.as_dict(),
                    )
                )
        self.flushed_clips = self._recorder.flush(timestamp)
        telemetry: list[TelemetryEvent] = []
        for clip in self.flushed_clips:
            self._handle_clip(clip, timestamp, frame_index, telemetry)
        for event in telemetry:
            self._publish(event)
        if self._jsonl_handle is not None:
            trailer = {
                "type": "run_summary",
                "summary": self.summary().as_dict(),
                "flushed_events": [event.as_dict() for event in self.flushed_events],
                "flushed_clips": [clip.as_dict() for clip in self.flushed_clips],
            }
            self._jsonl_handle.write(json.dumps(trailer) + "\n")
            self._jsonl_handle.flush()
            self._jsonl_handle.close()
            self._jsonl_handle = None

    def run(
        self,
        source: str | Path | int | Iterable[np.ndarray],
        on_result: Callable[[FrameResult], None] | None = None,
        *,
        stop_event: threading.Event | None = None,
        fps_override: float | None = None,
    ) -> RunSummary:
        for result in self.iter_results(
            source, stop_event=stop_event, fps_override=fps_override
        ):
            if on_result is not None:
                on_result(result)
        return self.summary()

    def summary(self) -> RunSummary:
        return RunSummary(
            video=self._source_name,
            frames=self._frame_count,
            duration=self._last_timestamp,
            elapsed=time.monotonic() - self._run_started,
            detections=self._detections_seen,
            tracks=len(self._track_ids),
            interactions=self._interaction_starts,
            clips=self._clip_count,
            camera_states=dict(self._camera_counts),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_event(
        self,
        category: EventCategory,
        message: str,
        *,
        timestamp: float,
        frame_index: int,
        data: dict | None = None,
    ) -> TelemetryEvent:
        return TelemetryEvent(
            category=category,
            message=message,
            presentation_time=timestamp,
            frame_index=frame_index,
            data=data or {},
        )

    def _publish(self, event: TelemetryEvent) -> None:
        if self.telemetry_sink is not None:
            try:
                self.telemetry_sink(event)
            except Exception:
                LOGGER.exception("telemetry sink failed")

    def _handle_clip(
        self,
        clip: InteractionClip,
        timestamp: float,
        frame_index: int,
        telemetry: list[TelemetryEvent],
    ) -> None:
        self._clip_count += 1
        saved: dict[str, str] = {}
        if self.clips_dir is not None:
            try:
                saved = clip.save(
                    self.clips_dir, save_frames=self.config.clip.save_frames
                )
                clip.metadata["saved"] = saved
            except Exception:
                LOGGER.exception("could not save clip %s", clip.clip_id)
        message = (
            f"{clip.label} #{clip.track_id} hand={clip.hand} "
            f"{len(clip.frames)} frames trigger={clip.trigger}"
        )
        if saved.get("composite"):
            message += f" -> {saved['composite']}"
        telemetry.append(
            self._make_event(
                EventCategory.CLIP,
                message,
                timestamp=timestamp,
                frame_index=frame_index,
                data=clip.as_dict(),
            )
        )
        if self.on_clip is not None:
            try:
                self.on_clip(clip)
            except Exception:
                LOGGER.exception("clip callback failed for %s", clip.clip_id)

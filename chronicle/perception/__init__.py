"""Chronicle perception package (Role A).

Quick start::

    from chronicle.perception import PerceptionConfig, PerceptionPipeline

    pipeline = PerceptionPipeline(PerceptionConfig())
    for result in pipeline.iter_results("clip.mp4"):
        print(result.timestamp, result.camera_state, len(result.tracks))
"""

from chronicle.perception.clip_buffer import FrameRingBuffer, InteractionClipRecorder
from chronicle.perception.config import (
    ClipConfig,
    DetectorConfig,
    EmbedderConfig,
    HandConfig,
    InteractionConfig,
    MotionConfig,
    PerceptionConfig,
    TrackerConfig,
)
from chronicle.perception.detector import YoloWorldDetector, build_detector
from chronicle.perception.embedder import EmbeddingService, OpenClipEmbedder, build_embedder
from chronicle.perception.hands import HybridHands, MediaPipeHands, build_hands
from chronicle.perception.interactions import InteractionTracker
from chronicle.perception.motion import AffineMotionEstimator
from chronicle.perception.pipeline import PerceptionPipeline
from chronicle.perception.sources import VideoSource
from chronicle.perception.stream import PerceptionStream
from chronicle.perception.telemetry import TelemetryBus, TelemetryPrinter
from chronicle.perception.tracker import ByteTrackLite, TrackerResult
from chronicle.types import (
    BBox,
    CameraState,
    ClipFrame,
    Detection,
    EventCategory,
    FrameResult,
    HandObservation,
    InteractionClip,
    InteractionEvent,
    InteractionObservation,
    MotionEstimate,
    RunSummary,
    TelemetryEvent,
    TrackedObject,
)

__all__ = [
    "AffineMotionEstimator",
    "BBox",
    "ByteTrackLite",
    "CameraState",
    "ClipConfig",
    "ClipFrame",
    "Detection",
    "DetectorConfig",
    "EmbedderConfig",
    "EmbeddingService",
    "EventCategory",
    "FrameResult",
    "FrameRingBuffer",
    "HandConfig",
    "HandObservation",
    "HybridHands",
    "InteractionClip",
    "InteractionClipRecorder",
    "InteractionConfig",
    "InteractionEvent",
    "InteractionObservation",
    "InteractionTracker",
    "MediaPipeHands",
    "MotionConfig",
    "MotionEstimate",
    "OpenClipEmbedder",
    "PerceptionConfig",
    "PerceptionPipeline",
    "PerceptionStream",
    "RunSummary",
    "TelemetryBus",
    "TelemetryEvent",
    "TelemetryPrinter",
    "TrackedObject",
    "TrackerConfig",
    "TrackerResult",
    "VideoSource",
    "YoloWorldDetector",
    "build_detector",
    "build_embedder",
    "build_hands",
]

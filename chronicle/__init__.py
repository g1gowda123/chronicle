"""Chronicle: egocentric episodic memory graph.

Public surface for Role B (state machine + dual-graph engineering).
"""

from chronicle.config import DEFAULT_THRESHOLDS, Thresholds
from chronicle.geometry import (
    bbox_to_polygon,
    cosine_similarity,
    ios,
    support_contact,
    to_polygon,
)
from chronicle.graph.memory import ChronicleMemory
from chronicle.graph.query import QueryResult
from chronicle.state_machine import LifecycleEngine
from chronicle.telemetry import EventBus, TelemetryEvent, TelemetryTag
from chronicle.types import (
    CameraMotion,
    CameraState,
    EntitySnapshot,
    FrameContext,
    HandObservation,
    ObjectObservation,
    ObjectState,
    Relation,
    StateTransition,
    SupportSurface,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "CameraMotion",
    "CameraState",
    "ChronicleMemory",
    "EntitySnapshot",
    "EventBus",
    "FrameContext",
    "HandObservation",
    "LifecycleEngine",
    "ObjectObservation",
    "ObjectState",
    "QueryResult",
    "Relation",
    "StateTransition",
    "SupportSurface",
    "TelemetryEvent",
    "TelemetryTag",
    "Thresholds",
    "bbox_to_polygon",
    "cosine_similarity",
    "ios",
    "support_contact",
    "to_polygon",
]

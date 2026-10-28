"""Chronicle: egocentric episodic memory with spatio-temporal gating.

Public surface for Role B (state machine + dual-graph engineering).  Role A's
perception records are available from :mod:`chronicle.types`; the live event-bus
:class:`TelemetryEvent` is exported here from :mod:`chronicle.telemetry`.
"""

__version__ = "0.1.0"

from typing import Any

from chronicle.config import DEFAULT_THRESHOLDS, Thresholds
from chronicle.geometry import (
    bbox_to_polygon,
    cosine_similarity,
    ios,
    support_contact,
    to_polygon,
)
from chronicle.graph.adapter import frame_context_from_result
from chronicle.graph.consolidation import GraphConsolidator, MergeCandidate, MergeReport
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


def __getattr__(name: str) -> Any:
    """Load the integrated runtime only when it is actually requested.

    ``ChronicleMemory`` and the graph primitives are useful on their own; the
    full runtime pulls in the perception and semantic layers.  Keeping this
    import lazy means ``chronicle demo`` and ``chronicle doctor`` stay fast and
    do not initialize optional model code.
    """

    if name in {"ChronicleRuntime", "RuntimeAnswer"}:
        from chronicle.integration import ChronicleRuntime, RuntimeAnswer

        value = {
            "ChronicleRuntime": ChronicleRuntime,
            "RuntimeAnswer": RuntimeAnswer,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_THRESHOLDS",
    "CameraMotion",
    "CameraState",
    "ChronicleMemory",
    "ChronicleRuntime",
    "EntitySnapshot",
    "EventBus",
    "FrameContext",
    "GraphConsolidator",
    "HandObservation",
    "LifecycleEngine",
    "MergeCandidate",
    "MergeReport",
    "ObjectObservation",
    "ObjectState",
    "QueryResult",
    "Relation",
    "RuntimeAnswer",
    "StateTransition",
    "SupportSurface",
    "TelemetryEvent",
    "TelemetryTag",
    "Thresholds",
    "__version__",
    "bbox_to_polygon",
    "cosine_similarity",
    "frame_context_from_result",
    "ios",
    "support_contact",
    "to_polygon",
]

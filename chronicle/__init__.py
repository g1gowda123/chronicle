"""Chronicle: egocentric episodic memory with spatio-temporal gating.

The public names are still available as ``from chronicle import ...``, but they
are imported lazily.  This keeps the research path (``chronicle.vfm`` and
``chronicle.research``) importable with the standard library alone: a machine
that only wants to open and query a compiled vault does not need OpenCV,
Shapely, NetworkX, NiceGUI or a perception model stack.

Use a concrete submodule for new code::

    from chronicle.vfm.query import answer, load_store
    from chronicle.research import run_ablation
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

#: name -> (module, attribute)
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_THRESHOLDS": ("chronicle.config", "DEFAULT_THRESHOLDS"),
    "Thresholds": ("chronicle.config", "Thresholds"),
    "bbox_to_polygon": ("chronicle.geometry", "bbox_to_polygon"),
    "cosine_similarity": ("chronicle.geometry", "cosine_similarity"),
    "ios": ("chronicle.geometry", "ios"),
    "support_contact": ("chronicle.geometry", "support_contact"),
    "to_polygon": ("chronicle.geometry", "to_polygon"),
    "frame_context_from_result": ("chronicle.graph.adapter", "frame_context_from_result"),
    "GraphConsolidator": ("chronicle.graph.consolidation", "GraphConsolidator"),
    "MergeCandidate": ("chronicle.graph.consolidation", "MergeCandidate"),
    "MergeReport": ("chronicle.graph.consolidation", "MergeReport"),
    "ChronicleMemory": ("chronicle.graph.memory", "ChronicleMemory"),
    "QueryResult": ("chronicle.graph.query", "QueryResult"),
    "ChronicleRuntime": ("chronicle.integration", "ChronicleRuntime"),
    "RuntimeAnswer": ("chronicle.integration", "RuntimeAnswer"),
    "LifecycleEngine": ("chronicle.state_machine", "LifecycleEngine"),
    "EventBus": ("chronicle.telemetry", "EventBus"),
    "TelemetryEvent": ("chronicle.telemetry", "TelemetryEvent"),
    "TelemetryTag": ("chronicle.telemetry", "TelemetryTag"),
    "CameraMotion": ("chronicle.types", "CameraMotion"),
    "CameraState": ("chronicle.types", "CameraState"),
    "EntitySnapshot": ("chronicle.types", "EntitySnapshot"),
    "FrameContext": ("chronicle.types", "FrameContext"),
    "HandObservation": ("chronicle.types", "HandObservation"),
    "ObjectObservation": ("chronicle.types", "ObjectObservation"),
    "ObjectState": ("chronicle.types", "ObjectState"),
    "Relation": ("chronicle.types", "Relation"),
    "StateTransition": ("chronicle.types", "StateTransition"),
    "SupportSurface": ("chronicle.types", "SupportSurface"),
}

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


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})

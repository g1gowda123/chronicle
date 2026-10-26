"""Role C — semantic engine and multimodal verification.

Public surface used by the rest of Chronicle:

* :class:`SemanticEngine` — one facade for zones, interactions, and queries.
* :class:`VisualPlaceRecognizer` — OpenCLIP VPR with a 0.85 cosine threshold.
* :class:`TemporalStripCompositor` — 30-frame buffer -> 1x3 Florence input.
* :class:`ActionVerifier` — caption + LLM/rule -> strict JSON action.
* :class:`GraphRAG` — 2-hop subgraph -> cited natural-language answer.
"""

from .actions import (
    DEFAULT_ACTION_THRESHOLDS,
    ActionCalibrator,
    ActionParseError,
    ActionPrimitive,
    ActionVerification,
    RuleBasedActionClassifier,
    coerce_action,
    extract_container,
    is_ambiguous_caption,
    parse_action_json,
    resolve_container,
)
from .adapters import PerceptionEmbedderAdapter, adapt_embedder
from .config import SemanticConfig
from .embedders import (
    DeterministicImageEmbedder,
    ImageEmbedder,
    OpenClipEmbedder,
    build_embedder,
)
from .engine import InteractionVerification, SemanticEngine
from .florence import (
    FlorenceCaptioner,
    LazyCaptioner,
    ModelUnavailableError,
    StubCaptioner,
    build_captioner,
)
from .graphrag import (
    Citation,
    Fact,
    GraphContext,
    GraphRAG,
    QueryAnswer,
    extract_citations,
    format_context,
    serialize_subgraph,
    template_answer,
)
from .llm import (
    ChatClient,
    LLMUnavailableError,
    OllamaChatClient,
    TransformersChatClient,
    build_chat_client,
)
from .queue import QueueStats, VerificationQueue, VerificationService
from .temporal_strip import (
    FrameSample,
    TemporalStrip,
    TemporalStripCompositor,
    build_temporal_strip,
)
from .verification import ActionVerifier
from .vpr import PlaceMatch, VisualPlaceRecognizer, Zone, build_vpr

# Imported lazily so `python -m chronicle.semantic.{calibration,evaluation}`
# does not trip runpy's "found in sys.modules" warning.  `from
# chronicle.semantic import evaluate_graphrag` and `import
# chronicle.semantic.evaluation` both still work.
_LAZY_EXPORTS: dict[str, str] = {
    "CALIBRATION_CAPTIONS": "calibration",
    "CalibrationReport": "calibration",
    "CalibrationResult": "calibration",
    "CalibrationSample": "calibration",
    "calibrate_from_file": "calibration",
    "calibrate_rule_classifier": "calibration",
    "calibrator_from_samples": "calibration",
    "evaluate_calibration": "calibration",
    "load_calibration_labels": "calibration",
    "suggest_thresholds": "calibration",
    "DEFAULT_GRAPHRAG_CASES": "evaluation",
    "CaseOutcome": "evaluation",
    "EvaluationReport": "evaluation",
    "GraphRAGCase": "evaluation",
    "evaluate_graphrag": "evaluation",
    "DEFAULT_TIMELINE_TOLERANCE": "timeline",
    "ActionEvent": "timeline",
    "TimelineMatch": "timeline",
    "TimelineReport": "timeline",
    "evaluate_action_timeline": "timeline",
    "load_action_labels": "timeline",
    "load_action_predictions": "timeline",
    "load_annotation_actions": "timeline",
    "predictions_from_snapshot": "timeline",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "CALIBRATION_CAPTIONS",
    "DEFAULT_ACTION_THRESHOLDS",
    "DEFAULT_GRAPHRAG_CASES",
    "DEFAULT_TIMELINE_TOLERANCE",
    "ActionCalibrator",
    "ActionEvent",
    "ActionParseError",
    "ActionPrimitive",
    "ActionVerification",
    "ActionVerifier",
    "CalibrationReport",
    "CalibrationResult",
    "CalibrationSample",
    "CaseOutcome",
    "ChatClient",
    "Citation",
    "DeterministicImageEmbedder",
    "EvaluationReport",
    "Fact",
    "FlorenceCaptioner",
    "FrameSample",
    "GraphContext",
    "GraphRAG",
    "GraphRAGCase",
    "ImageEmbedder",
    "InteractionVerification",
    "LLMUnavailableError",
    "LazyCaptioner",
    "ModelUnavailableError",
    "OllamaChatClient",
    "OpenClipEmbedder",
    "PerceptionEmbedderAdapter",
    "PlaceMatch",
    "QueryAnswer",
    "QueueStats",
    "RuleBasedActionClassifier",
    "SemanticConfig",
    "SemanticEngine",
    "StubCaptioner",
    "TemporalStrip",
    "TemporalStripCompositor",
    "TimelineMatch",
    "TimelineReport",
    "TransformersChatClient",
    "VerificationQueue",
    "VerificationService",
    "VisualPlaceRecognizer",
    "Zone",
    "adapt_embedder",
    "build_captioner",
    "build_chat_client",
    "build_embedder",
    "build_temporal_strip",
    "build_vpr",
    "calibrate_from_file",
    "calibrate_rule_classifier",
    "calibrator_from_samples",
    "coerce_action",
    "evaluate_action_timeline",
    "evaluate_calibration",
    "evaluate_graphrag",
    "extract_citations",
    "extract_container",
    "format_context",
    "is_ambiguous_caption",
    "load_action_labels",
    "load_action_predictions",
    "load_annotation_actions",
    "load_calibration_labels",
    "parse_action_json",
    "predictions_from_snapshot",
    "resolve_container",
    "serialize_subgraph",
    "suggest_thresholds",
    "template_answer",
]

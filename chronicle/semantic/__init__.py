"""Role C — semantic engine and multimodal verification.

Public surface used by the rest of Chronicle:

* :class:`SemanticEngine` — one facade for zones, interactions, and queries.
* :class:`VisualPlaceRecognizer` — OpenCLIP VPR with a 0.85 cosine threshold.
* :class:`TemporalStripCompositor` — 30-frame buffer -> 1x3 Florence input.
* :class:`ActionVerifier` — caption + LLM/rule -> strict JSON action.
* :class:`GraphRAG` — 2-hop subgraph -> cited natural-language answer.
"""

from .actions import (
    ActionParseError,
    ActionPrimitive,
    ActionVerification,
    RuleBasedActionClassifier,
    parse_action_json,
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
from .temporal_strip import (
    FrameSample,
    TemporalStrip,
    TemporalStripCompositor,
    build_temporal_strip,
)
from .verification import ActionVerifier
from .vpr import PlaceMatch, VisualPlaceRecognizer, Zone, build_vpr

__all__ = [
    "ActionParseError",
    "ActionPrimitive",
    "ActionVerification",
    "ActionVerifier",
    "ChatClient",
    "Citation",
    "DeterministicImageEmbedder",
    "Fact",
    "FlorenceCaptioner",
    "FrameSample",
    "GraphContext",
    "GraphRAG",
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
    "RuleBasedActionClassifier",
    "SemanticConfig",
    "SemanticEngine",
    "StubCaptioner",
    "TemporalStrip",
    "TemporalStripCompositor",
    "TransformersChatClient",
    "VisualPlaceRecognizer",
    "Zone",
    "adapt_embedder",
    "build_captioner",
    "build_chat_client",
    "build_embedder",
    "build_temporal_strip",
    "build_vpr",
    "extract_citations",
    "format_context",
    "parse_action_json",
    "serialize_subgraph",
    "template_answer",
]

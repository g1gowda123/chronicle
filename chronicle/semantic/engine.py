"""Facade that wires Role C together for the ingestion pipeline.

Typical integration (see docs/ROLE_C.md):

    engine = SemanticEngine(config, on_event=dashboard.push)
    match = engine.identify_zone(keyframe, timestamp=video_t)
    result = engine.verify_interaction(buffer, "keys", contact_index=14, timestamp=14.2)
    answer = engine.answer_query("Where are my keys?", serialized_subgraph)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .actions import ActionVerification
from .adapters import adapt_embedder
from .config import SemanticConfig
from .embedders import ImageEmbedder, build_embedder
from .florence import build_captioner
from .graphrag import GraphContext, GraphRAG, QueryAnswer
from .llm import ChatClient, build_chat_client
from .queue import VerificationService
from .temporal_strip import TemporalStrip, TemporalStripCompositor
from .verification import ActionVerifier
from .vpr import PlaceMatch, VisualPlaceRecognizer

logger = logging.getLogger("chronicle.semantic.engine")

EventSink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class InteractionVerification:
    strip: TemporalStrip
    action: ActionVerification
    clip_id: str | None = None
    track_id: int | None = None
    trigger: str | None = None
    hand: str | None = None
    max_ios: float | None = None

    def to_dict(self, *, include_preview: bool = False) -> dict[str, Any]:
        payload = {
            "entity": self.action.object_label,
            "action": self.action.to_dict(),
            "panel_indices": list(self.strip.panel_indices),
            "panel_timestamps": [
                self.strip.frames[index].timestamp
                for index in self.strip.panel_indices
            ],
            "strip_size": list(self.strip.image.size),
        }
        for key in ("clip_id", "track_id", "trigger", "hand", "max_ios"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if include_preview:
            payload["preview"] = self.strip.to_data_url()
        return payload


@dataclass
class SemanticEngine:
    config: SemanticConfig = field(default_factory=SemanticConfig)
    embedder: ImageEmbedder | None = None
    captioner: Any = None
    chat_client: ChatClient | None | str = "auto"
    on_event: EventSink | None = None
    embedder_backend: str = "auto"
    captioner_backend: str = "auto"
    channel_order: str = "bgr"
    vpr: VisualPlaceRecognizer = field(init=False)
    graphrag: GraphRAG = field(init=False)
    verifier: ActionVerifier = field(init=False)
    compositor: TemporalStripCompositor = field(init=False)
    _verification_service: VerificationService | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        self.config.validate()
        if self.embedder is None:
            self.embedder = build_embedder(
                self.config,
                backend=self.embedder_backend,
                channel_order=self.channel_order,
            )
        else:
            # Accept Role A's EmbeddingService/OpenClipEmbedder directly.
            self.embedder = adapt_embedder(
                self.embedder, channel_order=self.channel_order
            )
        if self.captioner is None:
            self.captioner = build_captioner(
                self.config,
                backend=self.captioner_backend,
                channel_order=self.channel_order,
            )
        if isinstance(self.chat_client, str):
            if self.chat_client != "auto":
                raise ValueError("chat_client string must be 'auto'")
            self.chat_client = build_chat_client(self.config)

        self.vpr = VisualPlaceRecognizer.from_config(
            self.embedder,
            config=self.config,
            captioner=self.captioner,
        )
        self.compositor = TemporalStripCompositor(
            self.config, channel_order=self.channel_order
        )
        self.verifier = ActionVerifier(
            captioner=self.captioner,
            chat_client=self.chat_client if self.config.action_llm_enabled else None,
            config=self.config,
            channel_order=self.channel_order,
        )
        self.graphrag = GraphRAG(chat_client=self.chat_client, config=self.config)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def emit(self, tag: str, event: str, message: str, **data: Any) -> None:
        payload = {
            "tag": tag,
            "event": event,
            "message": message,
            "data": data,
            "emitted_at": time.time(),
        }
        logger.info("[%s] %s", tag, message)
        if self.on_event is not None:
            try:
                self.on_event(payload)
            except Exception:
                logger.exception("event sink failed (telemetry must not block ingestion)")

    # ------------------------------------------------------------------
    # Stage 1: visual place recognition
    # ------------------------------------------------------------------
    def identify_zone(
        self,
        keyframe: Any,
        *,
        caption: str | None = None,
        timestamp: float | None = None,
    ) -> PlaceMatch:
        match = self.vpr.resolve(keyframe, caption=caption, timestamp=timestamp)
        payload = match.as_event()
        event_type = payload.pop("event")
        payload.pop("tag", None)
        self.emit(
            "VLM",
            event_type,
            f"{'created' if match.is_new else 'relocalized'} zone "
            f"{match.zone.label!r} (sim={match.score:.3f})",
            **payload,
        )
        return match

    def merge_duplicate_zones(self) -> list[tuple[str, str]]:
        merged = self.vpr.merge_duplicates()
        if merged:
            self.emit(
                "GRAPH",
                "ZONES_MERGED",
                f"merged {len(merged)} duplicate zone(s)",
                merges=[{"removed": removed, "kept": kept} for removed, kept in merged],
            )
        return merged

    # ------------------------------------------------------------------
    # Stage 2: 1x3 strip + action verification
    # ------------------------------------------------------------------
    def build_strip(
        self,
        frames: Any,
        *,
        contact_index: int | None = None,
    ) -> TemporalStrip:
        return self.compositor.build(frames, contact_index=contact_index)

    def verify_interaction(
        self,
        frames: Any,
        entity_label: str | None = None,
        *,
        contact_index: int | None = None,
        container: str | None = None,
        context_hint: str = "",
        timestamp: float | None = None,
    ) -> InteractionVerification:
        clip = None
        if (
            not isinstance(frames, (list, tuple))
            and hasattr(frames, "frames")
            and hasattr(frames, "contact_index")
        ):
            clip = frames
            frames = clip.frames
            if contact_index is None:
                contact_index = int(getattr(clip, "contact_index", 0))
            if entity_label is None:
                entity_label = getattr(clip, "label", None)
            if timestamp is None:
                contact_ts = getattr(clip, "contact_ts", None)
                if contact_ts is not None:
                    timestamp = float(contact_ts)
            metadata = getattr(clip, "metadata", {}) or {}
            if container is None:
                container = metadata.get("container")
            if not context_hint:
                context_hint = str(
                    metadata.get("context_hint")
                    or metadata.get("caption")
                    or ""
                )

        if not entity_label:
            raise ValueError("entity_label is required when frames are not an InteractionClip")

        strip = self.build_strip(frames, contact_index=contact_index)
        action = self.verifier.verify(
            strip,
            entity_label,
            container=container,
            context_hint=context_hint,
            timestamp=timestamp,
        )
        event_data = action.to_dict()
        if clip is not None:
            for key in ("clip_id", "track_id", "trigger", "hand", "max_ios"):
                value = getattr(clip, key, None)
                if value is not None:
                    event_data[key] = value
        self.emit(
            "VLM",
            "ACTION_VERIFIED",
            f"{action.action.value} {action.object_label} "
            f"(conf={action.confidence:.2f}, accepted={action.accepted})",
            **event_data,
        )
        return InteractionVerification(
            strip=strip,
            action=action,
            clip_id=getattr(clip, "clip_id", None),
            track_id=getattr(clip, "track_id", None),
            trigger=getattr(clip, "trigger", None),
            hand=getattr(clip, "hand", None),
            max_ios=getattr(clip, "max_ios", None),
        )

    # ------------------------------------------------------------------
    # Stage 2b: bounded verification queue
    # ------------------------------------------------------------------
    def verification_service(
        self,
        *,
        max_pending: int | None = None,
        policy: str | None = None,
        dedup_window: int | None = None,
    ) -> VerificationService:
        """Lazily build the Role C verification queue for this engine."""
        if self._verification_service is None:
            self._verification_service = VerificationService(
                engine_provider=lambda: self,
                max_pending=(
                    self.config.verification_queue_size
                    if max_pending is None
                    else max_pending
                ),
                policy=(
                    self.config.verification_queue_policy
                    if policy is None
                    else policy
                ),
                dedup_window=(
                    self.config.verification_dedup_window
                    if dedup_window is None
                    else dedup_window
                ),
            )
        return self._verification_service

    def submit_clip(self, clip: Any) -> bool:
        """Queue an ``InteractionClip``; returns False if duplicate/rejected."""
        return self.verification_service().submit(clip)

    def drain_verifications(self) -> int:
        """Verify every pending clip inline; returns the number processed."""
        return self.verification_service().drain()

    def queue_stats(self) -> dict[str, Any]:
        return self.verification_service().stats().to_dict()

    # ------------------------------------------------------------------
    # Stage 3: GraphRAG answer generation
    # ------------------------------------------------------------------
    def answer_query(
        self,
        query: str,
        context: GraphContext | Mapping[str, Any],
    ) -> QueryAnswer:
        answer = self.graphrag.answer(query, context)
        self.emit(
            "GRAPH",
            "QUERY_ANSWERED",
            f"{query!r} -> {answer.source} "
            f"({len(answer.citations)} citation(s), grounded={answer.grounded})",
            **answer.to_dict(),
        )
        return answer

    # Convenience wrappers used by teammates --------------------------------
    remember_zone = identify_zone

    def save_zones(self, path: str) -> None:
        self.vpr.save(path)

    def load_zones(self, path: str) -> None:
        loaded = VisualPlaceRecognizer.load(
            path,
            self.embedder,
            captioner=self.captioner,
            config=self.config,
        )
        self.vpr = loaded

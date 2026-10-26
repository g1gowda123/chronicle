"""Florence-2 + constrained classifier -> validated action primitive.

Flow:
    temporal strip -> dense caption -> local LLM (JSON mode) -> strict parse
                                  +-> deterministic rule classifier (offline)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ._utils import warn_once
from .actions import (
    ActionParseError,
    ActionPrimitive,
    ActionVerification,
    RuleBasedActionClassifier,
    action_schema_example,
    parse_action_json,
)
from .config import SemanticConfig
from .llm import ChatClient, LLMUnavailableError
from .temporal_strip import TemporalStrip

logger = logging.getLogger("chronicle.semantic.verification")

SYSTEM_PROMPT = """You verify egocentric hand-object interactions from three \
temporal frames (PRE, CONTACT, POST).

Return ONLY one JSON object. No markdown, no commentary.
Required keys:
  "action": one of PICK_UP, PUT_DOWN, INSERT_INTO, REMOVE_FROM, OPEN, CLOSE,
            PUSH, PULL, HAND_OFF, HOLD, OTHER
  "object": short lowercase object noun
  "container": string or null
  "confidence": number from 0.0 to 1.0
  "evidence": one sentence citing what changes between the frames

Use OTHER when the evidence is ambiguous. Never invent an object that is not
described."""


def build_action_prompt(
    *,
    entity_label: str,
    caption: str,
    container: str | None = None,
    context_hint: str = "",
    timestamp: float | None = None,
) -> str:
    lines = [
        f"Target entity: {entity_label}",
        f"Known container (may be empty): {container or 'unknown'}",
        f"Video timestamp: {'unknown' if timestamp is None else f'{timestamp:.2f}s'}",
        "Frames: PRE (left) -> CONTACT (middle) -> POST (right).",
        "",
        f"Florence-2 dense caption: {caption or '(no caption)'}",
    ]
    if context_hint:
        lines.append(f"Pipeline context: {context_hint}")
    lines.extend(
        [
            "",
            "Schema example:",
            json.dumps(action_schema_example()),
            "",
            "Classify the action. Return only JSON.",
        ]
    )
    return "\n".join(lines)


def _with_acceptance(
    result: ActionVerification,
    *,
    min_confidence: float,
) -> ActionVerification:
    accepted = (
        result.action is not ActionPrimitive.OTHER
        and result.confidence >= float(min_confidence)
    )
    return ActionVerification(
        action=result.action,
        object_label=result.object_label,
        container=result.container,
        confidence=result.confidence,
        evidence=result.evidence,
        source=result.source,
        caption=result.caption,
        raw_response=result.raw_response,
        timestamp=result.timestamp,
        accepted=accepted,
    )


@dataclass
class ActionVerifier:
    captioner: Any | None
    chat_client: ChatClient | None = None
    config: SemanticConfig = field(default_factory=SemanticConfig)
    channel_order: str = "bgr"
    rule_classifier: RuleBasedActionClassifier = field(
        default_factory=RuleBasedActionClassifier
    )
    last_parse_error: str | None = None

    def verify(
        self,
        strip_or_image: TemporalStrip | Any,
        entity_label: str,
        *,
        container: str | None = None,
        context_hint: str = "",
        timestamp: float | None = None,
    ) -> ActionVerification:
        """Caption the strip, then classify + validate the action."""
        image = strip_or_image.image if isinstance(strip_or_image, TemporalStrip) else strip_or_image
        if timestamp is None and isinstance(strip_or_image, TemporalStrip):
            contact = strip_or_image.frames[strip_or_image.contact_index]
            timestamp = contact.timestamp

        caption, _caption_source = self._caption(image, context_hint)
        return self.verify_caption(
            caption,
            entity_label,
            container=container,
            context_hint=context_hint,
            timestamp=timestamp,
        )

    def verify_caption(
        self,
        caption: str,
        entity_label: str,
        *,
        container: str | None = None,
        context_hint: str = "",
        timestamp: float | None = None,
    ) -> ActionVerification:
        """Classify an existing caption (handy for tests and batch re-runs)."""
        self.last_parse_error = None
        caption = (caption or "").strip()

        if self.chat_client is not None:
            prompt = build_action_prompt(
                entity_label=entity_label,
                caption=caption,
                container=container,
                context_hint=context_hint,
                timestamp=timestamp,
            )
            try:
                raw = self.chat_client.complete(SYSTEM_PROMPT, prompt, json_mode=True)
                try:
                    parsed = parse_action_json(
                        raw,
                        default_object=entity_label,
                        source="llm",
                        caption=caption,
                        timestamp=timestamp,
                    )
                except ActionParseError as first_error:
                    self.last_parse_error = str(first_error)
                    logger.warning("action JSON rejected (%s); retrying once", first_error)
                    repair_prompt = (
                        f"{prompt}\n\nYour previous response was rejected: {first_error}\n"
                        f"Previous response:\n{raw[:2000]}\n\n"
                        "Return corrected JSON only, matching the schema exactly."
                    )
                    raw = self.chat_client.complete(
                        SYSTEM_PROMPT, repair_prompt, json_mode=True
                    )
                    parsed = parse_action_json(
                        raw,
                        default_object=entity_label,
                        source="llm_repair",
                        caption=caption,
                        timestamp=timestamp,
                    )
                return _with_acceptance(
                    parsed, min_confidence=self.config.action_min_confidence
                )
            except (ActionParseError, LLMUnavailableError) as exc:
                self.last_parse_error = str(exc)
                warn_once(
                    "action-llm-fallback",
                    "LLM action verification failed (%s); using rule fallback.",
                    exc,
                )

        return self.rule_classifier.verification(
            caption,
            entity_label,
            container=container,
            context_hint=context_hint,
            timestamp=timestamp,
            min_confidence=self.config.action_min_confidence,
        )

    def _caption(self, image: Any, context_hint: str) -> tuple[str, str]:
        if self.captioner is not None:
            try:
                caption = (self.captioner.caption(image) or "").strip()
                if caption:
                    return caption, "florence"
            except Exception as exc:  # noqa: BLE001
                warn_once(
                    "captioner-fallback",
                    "Caption generation failed (%s); using pipeline context.",
                    exc,
                )
        if context_hint.strip():
            return context_hint.strip(), "context"
        return "", "none"

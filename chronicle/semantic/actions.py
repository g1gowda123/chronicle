"""Action primitives and strict JSON parsing for VLM verification.

Florence-2 produces a caption; a constrained classifier turns that caption
(plus the target entity) into a JSON object.  ``parse_action_json`` is the hard
boundary: malformed or out-of-schema output never leaks into the graph.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ActionPrimitive(StrEnum):
    PICK_UP = "PICK_UP"
    PUT_DOWN = "PUT_DOWN"
    INSERT_INTO = "INSERT_INTO"
    REMOVE_FROM = "REMOVE_FROM"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    PUSH = "PUSH"
    PULL = "PULL"
    HAND_OFF = "HAND_OFF"
    HOLD = "HOLD"
    OTHER = "OTHER"


class ActionParseError(ValueError):
    """Raised when a VLM/LLM response violates the action JSON schema."""


@dataclass(frozen=True)
class ActionVerification:
    action: ActionPrimitive
    object_label: str
    container: str | None
    confidence: float
    evidence: str
    source: str
    caption: str = ""
    raw_response: str | None = None
    timestamp: float | None = None
    accepted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "object": self.object_label,
            "container": self.container,
            "confidence": round(float(self.confidence), 4),
            "evidence": self.evidence,
            "source": self.source,
            "caption": self.caption,
            "timestamp": self.timestamp,
            "accepted": self.accepted,
        }


def _normalise_action_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


_ACTION_ALIASES: dict[str, ActionPrimitive] = {}


def _register(action: ActionPrimitive, *aliases: str) -> None:
    for alias in aliases:
        _ACTION_ALIASES[_normalise_action_key(alias)] = action


_register(
    ActionPrimitive.PICK_UP,
    "PICK_UP",
    "pick up",
    "pickup",
    "picked up",
    "picking up",
    "grab",
    "grabs",
    "grabbed",
    "grasp",
    "grasps",
    "lift",
    "lifts",
    "lifted",
)
_register(
    ActionPrimitive.PUT_DOWN,
    "PUT_DOWN",
    "put down",
    "putdown",
    "set down",
    "place",
    "places",
    "placed",
    "placing",
    "lay down",
    "drop",
    "drops",
)
_register(
    ActionPrimitive.INSERT_INTO,
    "INSERT_INTO",
    "insert",
    "inserts",
    "inserted",
    "put in",
    "put inside",
    "place in",
    "place inside",
    "put into",
    "place into",
    "deposit",
    "tuck",
)
_register(
    ActionPrimitive.REMOVE_FROM,
    "REMOVE_FROM",
    "remove",
    "removes",
    "removed",
    "take out",
    "takes out",
    "took out",
    "pull out",
    "pulls out",
    "extract",
)
_register(ActionPrimitive.OPEN, "OPEN", "opens", "opened", "opening", "unzip", "unlock")
_register(ActionPrimitive.CLOSE, "CLOSE", "closes", "closed", "closing", "shut", "zip", "lock")
_register(ActionPrimitive.PUSH, "PUSH", "pushes", "pushed", "pushing", "slide", "slides")
_register(ActionPrimitive.PULL, "PULL", "pulls", "pulled", "pulling", "drag", "drags")
_register(ActionPrimitive.HAND_OFF, "HAND_OFF", "hand off", "handoff", "give", "gives", "pass")
_register(ActionPrimitive.HOLD, "HOLD", "hold", "holds", "holding", "carry", "carries")


def _first(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _iter_json_objects(text: str) -> Iterable[str]:
    """Yield balanced {...} spans, ignoring braces inside JSON strings."""
    index = 0
    length = len(text)
    while index < length:
        start = text.find("{", index)
        if start < 0:
            return
        depth = 0
        in_string = False
        escaped = False
        completed = False
        for position in range(start, length):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : position + 1]
                    index = position + 1
                    completed = True
                    break
        if not completed:
            return


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract the first decodable JSON object from a model response.

    Handles bare JSON, ```json fences, and short prose wrappers.  Raises
    ``ActionParseError`` when no object can be decoded.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ActionParseError("empty model response")

    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced and "{" in fenced.group(1):
        text = fenced.group(1).strip()

    if "{" not in text:
        raise ActionParseError(f"no JSON object found in response: {raw[:200]!r}")

    candidates = list(_iter_json_objects(text))
    if not candidates:
        raise ActionParseError(
            f"invalid JSON: missing closing '}}' in response: {raw[:200]!r}"
        )

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(payload, dict):
            return payload

    if last_error is not None:
        raise ActionParseError(
            f"invalid JSON: {last_error.msg} at char {last_error.pos}"
        ) from last_error
    raise ActionParseError("action JSON must be an object")


def parse_action_json(
    raw: str,
    *,
    default_object: str | None = None,
    source: str = "llm",
    caption: str = "",
    timestamp: float | None = None,
) -> ActionVerification:
    """Parse and validate the action schema.

    Required keys: ``action``, ``confidence``, ``evidence``.
    ``object`` is required unless ``default_object`` is supplied.
    """
    payload = extract_json_object(raw)

    action_value = _first(payload, "action", "action_type", "primitive", "label")
    if action_value is None:
        raise ActionParseError("missing required key 'action'")
    action_key = _normalise_action_key(action_value)
    try:
        action = ActionPrimitive(action_key.upper())
    except ValueError:
        action = _ACTION_ALIASES.get(action_key)
        if action is None:
            raise ActionParseError(
                f"unknown action {action_value!r}; expected one of "
                f"{[item.value for item in ActionPrimitive]}"
            ) from None

    object_value = _first(payload, "object", "object_label", "target", "entity")
    object_label = str(object_value).strip() if object_value is not None else ""
    if not object_label and default_object:
        object_label = str(default_object).strip()
    if not object_label:
        raise ActionParseError("missing required key 'object'")

    confidence_value = _first(payload, "confidence", "score", "probability")
    if confidence_value is None:
        raise ActionParseError("missing required key 'confidence'")
    if isinstance(confidence_value, bool):
        raise ActionParseError("confidence must be a number in [0, 1]")
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError) as exc:
        raise ActionParseError("confidence must be a number in [0, 1]") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ActionParseError(f"confidence {confidence!r} is outside [0, 1]")

    evidence_value = _first(payload, "evidence", "reason", "justification")
    evidence = str(evidence_value).strip() if evidence_value is not None else ""
    if not evidence:
        raise ActionParseError("missing required key 'evidence'")

    container_value = _first(payload, "container", "destination", "receptacle")
    container = str(container_value).strip() if container_value is not None else ""
    if not container or container.lower() in {"null", "none", "n/a"}:
        container = None

    frame_timestamp = _first(payload, "timestamp", "t", "time")
    if timestamp is None and frame_timestamp is not None:
        try:
            timestamp = float(frame_timestamp)
        except (TypeError, ValueError):
            timestamp = None

    return ActionVerification(
        action=action,
        object_label=object_label,
        container=container,
        confidence=confidence,
        evidence=evidence,
        source=source,
        caption=caption,
        raw_response=raw,
        timestamp=timestamp,
    )


class RuleBasedActionClassifier:
    """Deterministic fallback used when no local LLM is reachable.

    It is intentionally conservative: unknown/ambiguous captions become
    ``OTHER`` with low confidence so the dashboard can flag them for review.
    """

    _PATTERNS: tuple[tuple[ActionPrimitive, re.Pattern[str], float], ...] = (
        (
            ActionPrimitive.REMOVE_FROM,
            re.compile(
                r"\b(?:take|takes|taking|pull|pulls|pulling|remove|removes|removing|"
                r"extract|extracts|lift|lifts)\b[^.]*\b(?:out of|out from|from inside|"
                r"from within)\b",
                re.IGNORECASE,
            ),
            0.78,
        ),
        (
            ActionPrimitive.INSERT_INTO,
            re.compile(
                r"\b(?:put|puts|putting|place|places|placing|insert|inserts|inserting|"
                r"drop|drops|dropping|slide|slides|sliding|tuck|tucks|deposit|deposits)\b"
                r"[^.]*\b(?:in|into|inside|within)\b",
                re.IGNORECASE,
            ),
            0.8,
        ),
        (
            ActionPrimitive.PUT_DOWN,
            re.compile(
                r"\b(?:put|puts|putting|place|places|placing|set|sets|setting|lay|lays|"
                r"drop|drops|dropping)\b[^.]*\b(?:down|on|onto|atop|against)\b",
                re.IGNORECASE,
            ),
            0.76,
        ),
        (
            ActionPrimitive.PICK_UP,
            re.compile(
                r"\b(?:pick|picks|picking|grab|grabs|grabbing|grasp|grasps|grasping|"
                r"lift|lifts|lifting|take|takes|taking|reach|reaches|reaching)\b",
                re.IGNORECASE,
            ),
            0.74,
        ),
        (
            ActionPrimitive.OPEN,
            re.compile(r"\b(?:open|opens|opening|unzip|unzips|unlock|unlocks)\b", re.IGNORECASE),
            0.72,
        ),
        (
            ActionPrimitive.CLOSE,
            re.compile(
                r"\b(?:close|closes|closing|shut|shuts|zip|zips|lock|locks)\b",
                re.IGNORECASE,
            ),
            0.72,
        ),
        (
            ActionPrimitive.PUSH,
            re.compile(r"\b(?:push|pushes|pushing)\b", re.IGNORECASE),
            0.68,
        ),
        (
            ActionPrimitive.PULL,
            re.compile(r"\b(?:pull|pulls|pulling|drag|drags|dragging)\b", re.IGNORECASE),
            0.68,
        ),
        (
            ActionPrimitive.HAND_OFF,
            re.compile(r"\b(?:hand off|hands off|handoff|give|gives|giving|pass|passes)\b", re.IGNORECASE),
            0.7,
        ),
        (
            ActionPrimitive.HOLD,
            re.compile(r"\b(?:hold|holds|holding|carry|carries|carrying)\b", re.IGNORECASE),
            0.62,
        ),
    )

    def classify(
        self,
        caption: str,
        object_label: str,
        *,
        container: str | None = None,
        context_hint: str = "",
    ) -> dict[str, Any]:
        text = " ".join(part for part in (caption, context_hint) if part).strip()
        if caption and context_hint and context_hint.lower() not in caption.lower():
            text = f"{caption} {context_hint}".strip()
        elif caption:
            text = caption.strip()
        else:
            text = context_hint.strip()
        if not text:
            return {
                "action": ActionPrimitive.OTHER.value,
                "object": object_label,
                "container": container,
                "confidence": 0.2,
                "evidence": "no visual caption was available",
            }

        for action, pattern, confidence in self._PATTERNS:
            match = pattern.search(text)
            if match:
                return {
                    "action": action.value,
                    "object": object_label,
                    "container": container,
                    "confidence": confidence,
                    "evidence": self._evidence(match, text),
                }

        return {
            "action": ActionPrimitive.OTHER.value,
            "object": object_label,
            "container": container,
            "confidence": 0.25,
            "evidence": f"no action keyword matched caption: {text[:160]}",
        }

    @staticmethod
    def _evidence(match: re.Match[str], text: str) -> str:
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 60)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        return snippet or match.group(0)

    def verification(
        self,
        caption: str,
        object_label: str,
        *,
        container: str | None = None,
        context_hint: str = "",
        timestamp: float | None = None,
        min_confidence: float = 0.45,
    ) -> ActionVerification:
        payload = self.classify(
            caption,
            object_label,
            container=container,
            context_hint=context_hint,
        )
        raw = json.dumps(payload)
        result = parse_action_json(
            raw,
            default_object=object_label,
            source="rule",
            caption=caption,
            timestamp=timestamp,
        )
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
            raw_response=raw,
            timestamp=result.timestamp,
            accepted=accepted,
        )


def action_schema_example() -> dict[str, Any]:
    return {
        "action": "INSERT_INTO",
        "object": "keys",
        "container": "drawer",
        "confidence": 0.91,
        "evidence": "the hand lowers the keys into the open drawer",
        "timestamp": 14.2,
    }

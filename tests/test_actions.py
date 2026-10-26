from __future__ import annotations

import json
import unittest

from chronicle.semantic.actions import (
    ActionParseError,
    ActionPrimitive,
    RuleBasedActionClassifier,
    parse_action_json,
)
from chronicle.semantic.config import SemanticConfig
from chronicle.semantic.florence import StubCaptioner
from chronicle.semantic.llm import LLMUnavailableError
from chronicle.semantic.verification import ActionVerifier

VALID = {
    "action": "PICK_UP",
    "object": "keys",
    "container": None,
    "confidence": 0.91,
    "evidence": "hand closes around keys and lifts",
    "timestamp": 14.2,
}


class ParseActionJsonTest(unittest.TestCase):
    def test_bare_json(self) -> None:
        result = parse_action_json(json.dumps(VALID))
        self.assertIs(result.action, ActionPrimitive.PICK_UP)
        self.assertEqual(result.object_label, "keys")
        self.assertAlmostEqual(result.confidence, 0.91)
        self.assertAlmostEqual(result.timestamp or 0.0, 14.2)

    def test_fenced_json_with_prose(self) -> None:
        raw = f"Here is my answer:\n```json\n{json.dumps(VALID)}\n```\nDone."
        result = parse_action_json(raw)
        self.assertIs(result.action, ActionPrimitive.PICK_UP)

    def test_alias_and_default_object(self) -> None:
        payload = dict(VALID)
        payload["action"] = "insert into"
        payload.pop("object")
        result = parse_action_json(json.dumps(payload), default_object="mug")
        self.assertIs(result.action, ActionPrimitive.INSERT_INTO)
        self.assertEqual(result.object_label, "mug")

    def test_null_container_becomes_none(self) -> None:
        payload = dict(VALID)
        payload["container"] = "null"
        self.assertIsNone(parse_action_json(json.dumps(payload)).container)

    def test_rejections(self) -> None:
        cases = [
            ("", "empty"),
            ("not json", "no json"),
            (json.dumps({**VALID, "action": "TELEPORT"}), "unknown action"),
            (json.dumps({k: v for k, v in VALID.items() if k != "action"}), "action"),
            (json.dumps({**VALID, "confidence": 1.4}), "outside"),
            (json.dumps({**VALID, "confidence": "high"}), "confidence"),
            (json.dumps({k: v for k, v in VALID.items() if k != "evidence"}), "evidence"),
            (json.dumps({k: v for k, v in VALID.items() if k != "object"}), "object"),
            (json.dumps({**VALID, "confidence": True}), "number"),
            ('{"action": "PICK_UP", "object": "keys",', "invalid json"),
        ]
        for raw, fragment in cases:
            with self.subTest(raw=raw[:40]):
                with self.assertRaises(ActionParseError) as context:
                    parse_action_json(raw)
                self.assertIn(fragment, str(context.exception).lower())


class RuleClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = RuleBasedActionClassifier()

    def test_caption_variants(self) -> None:
        cases = {
            "The hand picks up the keys from the table.": ActionPrimitive.PICK_UP,
            "She places the mug on the desk.": ActionPrimitive.PUT_DOWN,
            "He puts the keys inside the drawer.": ActionPrimitive.INSERT_INTO,
            "They take the charger out of the backpack.": ActionPrimitive.REMOVE_FROM,
            "The hand withdraws the Joy-Con from the console rail.": ActionPrimitive.REMOVE_FROM,
            "The hand slides the Joy-Con along the rail.": ActionPrimitive.PUSH,
            "The hand opens the cabinet door.": ActionPrimitive.OPEN,
            "The lid closes.": ActionPrimitive.CLOSE,
        }
        for caption, expected in cases.items():
            with self.subTest(caption=caption):
                result = self.classifier.verification(caption, "object")
                self.assertIs(result.action, expected, result.evidence)

    def test_ambiguous_caption_is_other_and_not_accepted(self) -> None:
        result = self.classifier.verification(
            "A first-person view of a room.", "keys", min_confidence=0.45
        )
        self.assertIs(result.action, ActionPrimitive.OTHER)
        self.assertFalse(result.accepted)

    def test_empty_caption(self) -> None:
        result = self.classifier.verification("", "keys")
        self.assertIs(result.action, ActionPrimitive.OTHER)

    def test_context_hint_is_used(self) -> None:
        result = self.classifier.verification(
            "", "keys", context_hint="hand places keys inside the drawer"
        )
        self.assertIs(result.action, ActionPrimitive.INSERT_INTO)


class FakeChat:
    def __init__(self, responses: list[str] | None = None, error: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[tuple[str, str, bool]] = []

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> str:
        self.calls.append((system_prompt, user_prompt, json_mode))
        if self.error is not None:
            raise self.error
        if not self.responses:
            return ""
        return self.responses.pop(0)


class ActionVerifierTest(unittest.TestCase):
    def test_rule_mode_uses_stub_caption(self) -> None:
        verifier = ActionVerifier(
            captioner=StubCaptioner("the hand picks up the keys"),
            chat_client=None,
            config=SemanticConfig(action_min_confidence=0.45),
        )
        result = verifier.verify("image", "keys")
        self.assertIs(result.action, ActionPrimitive.PICK_UP)
        self.assertEqual(result.source, "rule")
        self.assertTrue(result.accepted)

    def test_llm_valid_json(self) -> None:
        chat = FakeChat([json.dumps(VALID)])
        verifier = ActionVerifier(
            captioner=StubCaptioner("caption"),
            chat_client=chat,
            config=SemanticConfig(),
        )
        result = verifier.verify_caption("caption", "keys")
        self.assertIs(result.action, ActionPrimitive.PICK_UP)
        self.assertEqual(result.source, "llm")
        self.assertTrue(chat.calls[0][2])

    def test_llm_low_confidence_not_accepted(self) -> None:
        payload = {**VALID, "confidence": 0.2}
        verifier = ActionVerifier(
            captioner=StubCaptioner("caption"),
            chat_client=FakeChat([json.dumps(payload)]),
            config=SemanticConfig(action_min_confidence=0.45),
        )
        result = verifier.verify_caption("caption", "keys")
        self.assertFalse(result.accepted)

    def test_llm_repair_retry(self) -> None:
        chat = FakeChat(["not valid", json.dumps(VALID)])
        verifier = ActionVerifier(
            captioner=StubCaptioner("caption"), chat_client=chat
        )
        result = verifier.verify_caption("caption", "keys")
        self.assertEqual(result.source, "llm_repair")
        self.assertEqual(len(chat.calls), 2)
        self.assertIn("rejected", chat.calls[1][1])

    def test_llm_failure_falls_back_to_rules(self) -> None:
        verifier = ActionVerifier(
            captioner=StubCaptioner("the hand places the keys inside the drawer"),
            chat_client=FakeChat(["bad", "still bad"]),
        )
        result = verifier.verify("image", "keys")
        self.assertIs(result.action, ActionPrimitive.INSERT_INTO)
        self.assertEqual(result.source, "rule")
        self.assertIsNotNone(verifier.last_parse_error)

    def test_llm_unavailable_falls_back(self) -> None:
        verifier = ActionVerifier(
            captioner=StubCaptioner("hand picks up keys"),
            chat_client=FakeChat(error=LLMUnavailableError("down")),
        )
        result = verifier.verify("image", "keys")
        self.assertEqual(result.source, "rule")


if __name__ == "__main__":
    unittest.main()

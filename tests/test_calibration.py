"""Role C: container extraction, ambiguity rejection and action calibration."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from chronicle.semantic import (
    ActionCalibrator,
    ActionPrimitive,
    ActionVerifier,
    CalibrationSample,
    SemanticConfig,
    calibrate_from_file,
    calibrate_rule_classifier,
    calibration,
    evaluate_calibration,
    extract_container,
    is_ambiguous_caption,
    load_calibration_labels,
    parse_action_json,
    resolve_container,
    suggest_thresholds,
)
from chronicle.semantic.florence import StubCaptioner


class ContainerExtractionTest(unittest.TestCase):
    def test_insert_and_remove_phrases(self) -> None:
        self.assertEqual(
            extract_container("The hand puts the keys into the open drawer."),
            "open drawer",
        )
        self.assertEqual(
            extract_container(
                "The hand takes the charger out of the drawer.",
                action=ActionPrimitive.REMOVE_FROM,
            ),
            "drawer",
        )
        self.assertEqual(
            extract_container("A hand places the charger inside the bag."), "bag"
        )
        self.assertEqual(
            extract_container("She puts the mug on the kitchen counter."),
            "kitchen counter",
        )

    def test_non_containers_are_ignored(self) -> None:
        self.assertIsNone(extract_container("A first-person view of a room."))
        self.assertIsNone(extract_container("The hand moves into the frame."))
        self.assertIsNone(extract_container("puts the keys in the scene"))
        self.assertIsNone(extract_container(""))

    def test_action_preference(self) -> None:
        caption = "takes the keys out of the drawer and puts them in the pocket"
        self.assertEqual(
            extract_container(caption, action=ActionPrimitive.REMOVE_FROM), "drawer"
        )
        self.assertEqual(
            extract_container(caption, action=ActionPrimitive.INSERT_INTO), "pocket"
        )

    def test_resolve_priority(self) -> None:
        self.assertEqual(
            resolve_container(
                explicit="pocket",
                caption="puts the keys into the drawer",
            ),
            "pocket",
        )
        self.assertEqual(
            resolve_container(caption="puts the keys into the drawer"), "drawer"
        )
        self.assertEqual(
            resolve_container(
                context_hint="hand lowers the charger into the box"
            ),
            "box",
        )
        self.assertIsNone(resolve_container())


class AmbiguityTest(unittest.TestCase):
    def test_ambiguous_language(self) -> None:
        for caption in (
            "The hand appears to pick up the keys.",
            "It is unclear what the hand is holding.",
            "Maybe the mug is on the desk?",
            "Hard to tell from these frames.",
        ):
            self.assertTrue(is_ambiguous_caption(caption), caption)

    def test_assertive_language(self) -> None:
        for caption in (
            "The hand picks up the keys.",
            "The person places the mug into the drawer.",
        ):
            self.assertFalse(is_ambiguous_caption(caption), caption)

    def test_ambiguous_caption_is_penalised_and_rejected(self) -> None:
        calibrator = ActionCalibrator()
        raw = parse_action_json(
            '{"action": "PICK_UP", "object": "keys", "confidence": 0.95, '
            '"evidence": "hand reaches for the keys"}'
        )
        result = calibrator.apply(raw, ambiguous=True)
        self.assertFalse(result.accepted)
        self.assertLess(result.confidence, raw.confidence)

    def test_verifier_rejects_ambiguous_rule_caption(self) -> None:
        verifier = ActionVerifier(
            captioner=StubCaptioner("the hand appears to pick up the keys"),
            chat_client=None,
        )
        result = verifier.verify("image", "keys")
        self.assertIs(result.action, ActionPrimitive.PICK_UP)
        self.assertFalse(result.accepted)

    def test_reject_ambiguous_can_be_disabled(self) -> None:
        verifier = ActionVerifier(
            captioner=StubCaptioner("the hand appears to pick up the keys"),
            chat_client=None,
            config=SemanticConfig(action_reject_ambiguous=False),
        )
        self.assertTrue(verifier.verify("image", "keys").accepted)


class CalibratedAcceptanceTest(unittest.TestCase):
    def test_per_action_thresholds(self) -> None:
        calibrator = ActionCalibrator(floor=0.45)
        self.assertGreater(calibrator.threshold(ActionPrimitive.HOLD), 0.45)
        low = parse_action_json(
            '{"action": "HOLD", "object": "bottle", "confidence": 0.50, '
            '"evidence": "hand on bottle"}'
        )
        self.assertFalse(calibrator.apply(low).accepted)
        high = parse_action_json(
            '{"action": "HOLD", "object": "bottle", "confidence": 0.80, '
            '"evidence": "hand on bottle"}'
        )
        self.assertTrue(calibrator.apply(high).accepted)

    def test_other_is_never_accepted(self) -> None:
        calibrator = ActionCalibrator()
        result = parse_action_json(
            '{"action": "OTHER", "object": "keys", "confidence": 1.0, '
            '"evidence": "no action visible"}'
        )
        self.assertFalse(calibrator.apply(result).accepted)

    def test_floor_is_respected(self) -> None:
        calibrator = ActionCalibrator(floor=0.9)
        result = parse_action_json(
            '{"action": "OPEN", "object": "drawer", "confidence": 0.8, '
            '"evidence": "drawer opens"}'
        )
        self.assertFalse(calibrator.apply(result).accepted)


class ContainerWiringTest(unittest.TestCase):
    def test_insert_container_from_caption(self) -> None:
        verifier = ActionVerifier(
            captioner=StubCaptioner("the hand puts the keys into the drawer"),
            chat_client=None,
        )
        result = verifier.verify("image", "keys")
        self.assertIs(result.action, ActionPrimitive.INSERT_INTO)
        self.assertEqual(result.container, "drawer")
        self.assertTrue(result.accepted)

    def test_remove_container_from_caption(self) -> None:
        verifier = ActionVerifier(
            captioner=StubCaptioner("the hand takes the charger out of the drawer"),
            chat_client=None,
        )
        result = verifier.verify("image", "charger")
        self.assertIs(result.action, ActionPrimitive.REMOVE_FROM)
        self.assertEqual(result.container, "drawer")

    def test_explicit_container_wins(self) -> None:
        verifier = ActionVerifier(
            captioner=StubCaptioner("the hand puts the keys into the drawer"),
            chat_client=None,
        )
        result = verifier.verify("image", "keys", container="pocket")
        self.assertEqual(result.container, "pocket")

    def test_llm_payload_container_wins_over_caption(self) -> None:
        class FakeChat:
            def complete(self, system_prompt, user_prompt, *, json_mode=False):
                return (
                    '{"action": "INSERT_INTO", "object": "keys", '
                    '"container": "safe", "confidence": 0.9, '
                    '"evidence": "keys go into the safe"}'
                )

        verifier = ActionVerifier(
            captioner=StubCaptioner("the hand puts the keys into the drawer"),
            chat_client=FakeChat(),
        )
        result = verifier.verify_caption("the hand puts the keys into the drawer", "keys")
        self.assertEqual(result.container, "safe")


class CalibrationReportTest(unittest.TestCase):
    def test_rule_classifier_calibration(self) -> None:
        result = calibrate_rule_classifier()
        self.assertGreater(len(result.samples), 10)
        report = result.report.to_dict()
        self.assertGreaterEqual(report["precision"], 0.85)
        self.assertGreater(report["accepted"], 0)
        self.assertEqual(dict(result.thresholds)["OTHER"], 1.0)
        self.assertIn("per_action", report)

    def test_suggest_thresholds_prefers_precision(self) -> None:
        samples = (
            CalibrationSample(ActionPrimitive.OPEN, 0.9, True),
            CalibrationSample(ActionPrimitive.OPEN, 0.8, True),
            CalibrationSample(ActionPrimitive.OPEN, 0.6, False),
            CalibrationSample(ActionPrimitive.OPEN, 0.5, True),
        )
        thresholds = dict(suggest_thresholds(samples, target_precision=1.0))
        self.assertEqual(thresholds["OPEN"], 0.8)
        report = evaluate_calibration(samples, thresholds=thresholds)
        self.assertEqual(report.precision, 1.0)

    def test_calibrator_from_samples(self) -> None:
        samples = (
            CalibrationSample(ActionPrimitive.PICK_UP, 0.9, True),
            CalibrationSample(ActionPrimitive.PICK_UP, 0.7, True),
        )
        from chronicle.semantic import calibrator_from_samples

        calibrator = calibrator_from_samples(samples)
        parsed = parse_action_json(
            '{"action": "PICK_UP", "object": "keys", "confidence": 0.9, '
            '"evidence": "hand grabs keys"}'
        )
        self.assertTrue(calibrator.apply(parsed).accepted)


class ConfigTest(unittest.TestCase):
    def test_env_parses_action_thresholds(self) -> None:
        config = SemanticConfig.from_env(
            {
                "CHRONICLE_ACTION_THRESHOLDS": "HOLD=0.7, PICK_UP=0.8",
                "CHRONICLE_VERIFICATION_QUEUE_SIZE": "3",
                "CHRONICLE_VERIFICATION_QUEUE_POLICY": "reject",
            }
        )
        self.assertEqual(
            dict(config.action_thresholds), {"HOLD": 0.7, "PICK_UP": 0.8}
        )
        self.assertEqual(config.verification_queue_size, 3)
        self.assertEqual(config.verification_queue_policy, "reject")

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SemanticConfig(verification_queue_size=0).validate()
        with self.assertRaises(ValueError):
            SemanticConfig(verification_queue_policy="block").validate()
        with self.assertRaises(ValueError):
            SemanticConfig(action_thresholds=(("HOLD", 1.5),)).validate()


class CalibrationLabelFileTest(unittest.TestCase):
    """Labelled HoloLens captions should be consumable without editing Python."""

    LABELS = [
        {"caption": "The hand picks up the keys.", "action": "PICK_UP"},
        {
            "caption": "The hand puts the keys into the drawer.",
            "action": "INSERT_INTO",
        },
        {
            "caption": "The hand takes the charger out of the drawer.",
            "action": "REMOVE_FROM",
        },
        {"caption": "A first-person view of a room.", "action": "OTHER"},
    ]

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name: str, payload) -> Path:
        path = self.root / name
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_list_of_objects(self) -> None:
        labels = load_calibration_labels(self.write("labels.json", self.LABELS))
        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[1][1], ActionPrimitive.INSERT_INTO)
        self.assertEqual(labels[1][0], "The hand puts the keys into the drawer.")

    def test_loads_action_mapping_and_aliases(self) -> None:
        path = self.write(
            "mapping.json",
            {
                "pick up": ["The hand picks up the keys."],
                "REMOVE-FROM": [
                    {"text": "The hand takes the charger out of the drawer."}
                ],
            },
        )
        labels = dict(
            (action, caption) for caption, action in load_calibration_labels(path)
        )
        self.assertEqual(labels[ActionPrimitive.REMOVE_FROM],
                         "The hand takes the charger out of the drawer.")
        self.assertIn(ActionPrimitive.PICK_UP, labels)

    def test_loads_labels_wrapper_and_pairs(self) -> None:
        path = self.write(
            "wrapper.json",
            {"labels": [["The hand picks up the keys.", "pick_up"]]},
        )
        labels = load_calibration_labels(path)
        self.assertEqual(labels, (("The hand picks up the keys.", ActionPrimitive.PICK_UP),))

    def test_loads_jsonl(self) -> None:
        text = "\n".join(json.dumps(item) for item in self.LABELS) + "\n"
        labels = load_calibration_labels(self.write("labels.jsonl", text))
        self.assertEqual(len(labels), 4)

    def test_invalid_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_calibration_labels(self.write("bad_action.json", [
                {"caption": "x", "action": "FLY"}
            ]))
        with self.assertRaises(ValueError):
            load_calibration_labels(self.write("missing.json", [{"caption": "x"}]))
        with self.assertRaises(ValueError):
            load_calibration_labels(self.write("empty.json", []))
        bad_json = self.write("broken.json", "{not json")
        with self.assertRaises(ValueError):
            load_calibration_labels(bad_json)

    def test_calibrate_from_file_and_cli(self) -> None:
        path = self.write("labels.json", self.LABELS)
        result = calibrate_from_file(path, target_precision=1.0)
        self.assertEqual(len(result.samples), 4)
        self.assertEqual(result.report.precision, 1.0)
        self.assertGreater(result.report.accepted, 0)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = calibration.main(["--labels", str(path), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertIn("INSERT_INTO", payload["thresholds"])
        self.assertIn("CHRONICLE_ACTION_THRESHOLDS", payload["env"])
        self.assertIn("INSERT_INTO=", payload["env"]["CHRONICLE_ACTION_THRESHOLDS"])

    def test_cli_writes_threshold_file(self) -> None:
        labels = self.write("labels.json", self.LABELS)
        out = self.root / "thresholds.json"
        with redirect_stdout(io.StringIO()):
            code = calibration.main(
                ["--labels", str(labels), "--write", str(out)]
            )
        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["report"]["samples"], 4)
        self.assertIn("thresholds", payload)


if __name__ == "__main__":
    unittest.main()

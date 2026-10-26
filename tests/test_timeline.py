"""Role C: verified action timeline against hand-labelled ground truth."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from chronicle.semantic import (
    ActionEvent,
    ActionPrimitive,
    evaluate_action_timeline,
    load_action_labels,
    load_action_predictions,
    load_annotation_actions,
    predictions_from_snapshot,
    timeline,
)

ROOT = Path(__file__).resolve().parents[1]


def event(
    timestamp: float,
    action: str | ActionPrimitive,
    entity: str = "keys",
    container: str | None = None,
    *,
    accepted: bool = True,
) -> ActionEvent:
    return ActionEvent(
        timestamp=timestamp,
        action=action if isinstance(action, ActionPrimitive) else ActionPrimitive(action),
        entity=entity,
        container=container,
        accepted=accepted,
    )


class TimelineMatchingTest(unittest.TestCase):
    def test_exact_match_passes(self) -> None:
        report = evaluate_action_timeline(
            [event(1.9, "PICK_UP")], [event(1.9, "PICK_UP")]
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.precision, 1.0)
        self.assertEqual(report.recall, 1.0)
        self.assertEqual(report.f1, 1.0)
        self.assertEqual(report.matched, 1)
        self.assertEqual(report.mean_timestamp_error, 0.0)
        self.assertEqual(report.entity_accuracy, 1.0)

    def test_tolerance_is_inclusive(self) -> None:
        label = event(10.0, "OPEN", "drawer")
        inside = evaluate_action_timeline(
            [event(12.0, "OPEN", "drawer")], [label], tolerance=2.0
        )
        self.assertTrue(inside.passed)
        self.assertAlmostEqual(inside.mean_timestamp_error, 2.0)
        outside = evaluate_action_timeline(
            [event(12.01, "OPEN", "drawer")], [label], tolerance=2.0
        )
        self.assertFalse(outside.passed)
        self.assertEqual(outside.precision, 0.0)
        self.assertEqual(outside.recall, 0.0)

    def test_nearest_same_action_wins(self) -> None:
        labels = [event(1.0, "PICK_UP"), event(3.0, "PICK_UP")]
        predictions = [event(2.9, "PICK_UP"), event(1.1, "PICK_UP")]
        report = evaluate_action_timeline(predictions, labels, tolerance=0.5)
        self.assertTrue(report.passed)
        self.assertEqual(
            [match.prediction.timestamp for match in report.matches], [1.1, 2.9]
        )
        for match in report.matches:
            self.assertAlmostEqual(match.timestamp_error, 0.1)

    def test_wrong_action_is_extra_and_miss(self) -> None:
        report = evaluate_action_timeline(
            [event(1.9, "PUT_DOWN")], [event(1.9, "PICK_UP")]
        )
        self.assertEqual(report.matched, 0)
        self.assertEqual(len(report.unmatched_predictions), 1)
        self.assertEqual(len(report.unmatched_labels), 1)
        self.assertEqual(report.precision, 0.0)
        self.assertEqual(report.recall, 0.0)
        self.assertFalse(report.passed)

    def test_extra_prediction_lowers_precision(self) -> None:
        report = evaluate_action_timeline(
            [event(1.9, "PICK_UP"), event(5.0, "PUT_DOWN")],
            [event(1.9, "PICK_UP")],
        )
        self.assertEqual(report.recall, 1.0)
        self.assertEqual(report.precision, 0.5)
        self.assertFalse(report.passed)

    def test_entity_and_container_accuracy(self) -> None:
        label = event(4.0, "INSERT_INTO", "keys", "drawer")
        alias = event(4.1, "INSERT_INTO", "the keys", "open drawer")
        report = evaluate_action_timeline([alias], [label])
        self.assertTrue(report.passed)
        self.assertTrue(report.matches[0].entity_ok)
        self.assertTrue(report.matches[0].container_ok)

        missing = evaluate_action_timeline(
            [event(4.1, "INSERT_INTO", "keys")], [label]
        )
        self.assertIsNone(missing.matches[0].container_ok)
        self.assertEqual(missing.container_comparable, 0)

        wrong = evaluate_action_timeline(
            [event(4.1, "INSERT_INTO", "keys", "box")], [label]
        )
        self.assertFalse(wrong.matches[0].container_ok)
        self.assertFalse(wrong.passed)

        wrong_entity = evaluate_action_timeline(
            [event(4.1, "INSERT_INTO", "mug", "drawer")], [label]
        )
        self.assertFalse(wrong_entity.matches[0].entity_ok)
        self.assertFalse(wrong_entity.passed)

        near_miss = evaluate_action_timeline(
            [event(4.1, "INSERT_INTO", "cup", "drawer")],
            [event(4.0, "INSERT_INTO", "cupboard", "drawer")],
        )
        self.assertFalse(near_miss.matches[0].entity_ok)
        self.assertFalse(near_miss.passed)

    def test_compare_entities_can_be_disabled(self) -> None:
        report = evaluate_action_timeline(
            [event(4.4, "PUT_DOWN", "mug")],
            [event(4.2, "PUT_DOWN", "joy-con", "table")],
            compare_entities=False,
        )
        self.assertTrue(report.passed)
        self.assertIsNone(report.matches[0].entity_ok)
        self.assertIsNone(report.matches[0].container_ok)
        self.assertIsNone(report.entity_accuracy)
        self.assertIsNone(report.container_accuracy)

    def test_rejected_predictions_are_skipped_by_default(self) -> None:
        label = event(1.9, "PICK_UP")
        rejected = [event(1.9, "PICK_UP", accepted=False)]
        report = evaluate_action_timeline(rejected, [label])
        self.assertEqual(report.predictions, 0)
        self.assertEqual(report.rejected_predictions, 1)
        self.assertEqual(report.recall, 0.0)
        self.assertFalse(report.passed)

        included = evaluate_action_timeline(
            rejected, [label], accepted_only=False
        )
        self.assertTrue(included.passed)
        self.assertEqual(included.rejected_predictions, 0)


class TimelineLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name: str, payload) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_snapshot_rows_become_predictions(self) -> None:
        snapshot = {
            "actions": [
                {
                    "id": "edge:1",
                    "timestamp": 1.967,
                    "action": "PICK_UP",
                    "entity_id": "track_2",
                    "entity": "keys",
                    "confidence": 0.74,
                    "source": "rule",
                    "clip_ref": "clip_0001_t00001967",
                    "caption": "the hand grasps the keys",
                }
            ],
            "stats": {},
        }
        predictions = predictions_from_snapshot(snapshot)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].confidence, 0.74)
        self.assertEqual(predictions[0].caption, "the hand grasps the keys")
        report = evaluate_action_timeline(
            predictions, [{"timestamp": 1.9, "action": "pick up", "entity": "keys"}]
        )
        self.assertTrue(report.passed)

    def test_load_functions_accept_wrappers_and_snapshots(self) -> None:
        labels = load_action_labels(
            self.write(
                "labels.json",
                {"labels": [{"timestamp": 1.9, "action": "PICK_UP", "entity": "keys"}]},
            )
        )
        self.assertEqual(len(labels), 1)
        self.assertIs(labels[0].action, ActionPrimitive.PICK_UP)

        snapshot_path = self.write(
            "snapshot.json",
            {
                "actions": [
                    {"timestamp": 1.9, "action": "PICK_UP", "entity": "keys"}
                ]
            },
        )
        from_snapshot = load_action_predictions(snapshot_path)
        self.assertEqual(len(from_snapshot), 1)

        plain = load_action_predictions(
            self.write(
                "predictions.json",
                {"predictions": [{"timestamp": 1.9, "action": "PICK_UP"}]},
            )
        )
        self.assertEqual(len(plain), 1)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_action_timeline([], [], tolerance=-1)
        with self.assertRaises(ValueError):
            load_action_labels([{"action": "PICK_UP"}])
        with self.assertRaises(ValueError):
            load_action_labels([{"timestamp": 1.0, "action": "FLY"}])
        with self.assertRaises(ValueError):
            predictions_from_snapshot({"entities": []})
        with self.assertRaises(ValueError):
            load_action_labels(self.write("bad.json", {"nope": []}))


class HoloAssistLabelsTest(unittest.TestCase):
    LABELS = ROOT / "docs" / "action_timeline.holoassist.json"
    ANNOTATIONS = ROOT / "data" / "real" / "holoassist_switch_34s.json"

    def test_label_file_matches_published_annotations(self) -> None:
        labels = load_action_labels(self.LABELS)
        self.assertEqual(len(labels), 7)
        self.assertEqual(
            [label.action for label in labels],
            [
                ActionPrimitive.PICK_UP,
                ActionPrimitive.PUT_DOWN,
                ActionPrimitive.INSERT_INTO,
                ActionPrimitive.REMOVE_FROM,
                ActionPrimitive.PUT_DOWN,
                ActionPrimitive.REMOVE_FROM,
                ActionPrimitive.PUT_DOWN,
            ],
        )
        self.assertTrue(all(label.entity == "joy-con" for label in labels))

        annotations = load_annotation_actions(self.ANNOTATIONS)
        self.assertEqual(
            [(round(row.timestamp, 2), row.action) for row in annotations],
            [(round(row.timestamp, 2), row.action) for row in labels],
        )


class AnnotationLoaderTest(unittest.TestCase):
    def test_holoassist_annotations_pick_first_mappable_action(self) -> None:
        annotations = {
            "ground_truth": {
                "actions": [
                    {
                        "label": "joy-con",
                        "actions": ["PICK_UP", "TOUCH"],
                        "time": 4.2,
                        "source": "grab joy_con_controller",
                    },
                    {"label": "joy-con", "actions": ["MOVE"], "time": 28.11},
                    {"label": "joy-con", "actions": ["MOVE", "VISIBLE_MOVING"], "time": 28.9},
                    {
                        "label": "joy-con",
                        "actions": ["PLACE", "TOUCH"],
                        "time": 18.1,
                        "source": "place joy_con_controller",
                    },
                    {"label": "joy-con", "actions": ["REMOVE_FROM", "PICK_UP"], "time": 22.1},
                ]
            }
        }
        events = load_annotation_actions(annotations)
        self.assertEqual(
            [(event.timestamp, event.action.value) for event in events],
            [(4.2, "PICK_UP"), (18.1, "PUT_DOWN"), (22.1, "REMOVE_FROM")],
        )
        self.assertEqual({event.entity for event in events}, {"joy-con"})
        self.assertEqual(events[0].caption, "grab joy_con_controller")

    def test_bundled_holoassist_file_loads_mappable_rows(self) -> None:
        events = load_annotation_actions(
            ROOT / "data" / "real" / "holoassist_switch_34s.json"
        )
        self.assertEqual(len(events), 7)
        self.assertNotIn("MOVE", {event.action.value for event in events})
        self.assertEqual(events[0].action, ActionPrimitive.PICK_UP)
        self.assertEqual(events[0].entity, "joy-con")

    def test_bare_action_list_and_invalid_rows(self) -> None:
        events = load_annotation_actions(
            [{"time": 1.0, "action": "PICK_UP", "label": "keys"}]
        )
        self.assertEqual(len(events), 1)
        with self.assertRaises(ValueError):
            load_annotation_actions([{"actions": ["PICK_UP"]}])
        with self.assertRaises(ValueError):
            load_annotation_actions({"kind": "no actions here"})


class TimelineCliTest(unittest.TestCase):
    def test_example_files_pass(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = timeline.main(
                [
                    "--labels",
                    str(ROOT / "docs" / "action_timeline.example.json"),
                    "--predictions",
                    str(ROOT / "docs" / "action_predictions.example.json"),
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["matched"], 1)
        self.assertEqual(payload["matches"][0]["timestamp_error"], 0.067)

    def test_mismatch_exits_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.json"
            labels.write_text(
                json.dumps([{"timestamp": 1.9, "action": "PICK_UP", "entity": "keys"}]),
                encoding="utf-8",
            )
            predictions = root / "predictions.json"
            predictions.write_text(
                json.dumps([{"timestamp": 1.9, "action": "PUT_DOWN", "entity": "keys"}]),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = timeline.main(
                    [
                        "--labels",
                        str(labels),
                        "--predictions",
                        str(predictions),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("FAIL", buffer.getvalue())
            self.assertIn("MISS", buffer.getvalue())
            self.assertIn("EXTRA", buffer.getvalue())

    def test_annotations_mode_with_ignore_entity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "ground_truth": {
                            "actions": [
                                {
                                    "label": "joy-con",
                                    "actions": ["PICK_UP", "TOUCH"],
                                    "time": 4.2,
                                },
                                {
                                    "label": "joy-con",
                                    "actions": ["MOVE", "VISIBLE_MOVING"],
                                    "time": 15.16,
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            predictions = root / "predictions.json"
            predictions.write_text(
                json.dumps(
                    [
                        {
                            "timestamp": 4.4,
                            "action": "PICK_UP",
                            "entity": "mug",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = timeline.main(
                    [
                        "--annotations",
                        str(annotations),
                        "--predictions",
                        str(predictions),
                        "--ignore-entity",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("PASS", buffer.getvalue())
            self.assertIn("1/1 labels matched", buffer.getvalue())

            with redirect_stdout(io.StringIO()):
                strict = timeline.main(
                    ["--annotations", str(annotations), "--predictions", str(predictions)]
                )
            self.assertEqual(strict, 1)

    def test_snapshot_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.json"
            labels.write_text(
                json.dumps([{"timestamp": 1.9, "action": "PICK_UP", "entity": "keys"}]),
                encoding="utf-8",
            )
            snapshot = root / "snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "actions": [
                            {
                                "timestamp": 1.967,
                                "action": "PICK_UP",
                                "entity": "keys",
                                "accepted": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                code = timeline.main(
                    ["--labels", str(labels), "--snapshot", str(snapshot), "--json"]
                )
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()

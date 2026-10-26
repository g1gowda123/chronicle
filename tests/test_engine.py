from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from chronicle.semantic.config import SemanticConfig
from chronicle.semantic.embedders import DeterministicImageEmbedder
from chronicle.semantic.engine import SemanticEngine
from chronicle.semantic.florence import StubCaptioner
from chronicle.semantic.temporal_strip import FrameSample


def frame(index: int) -> FrameSample:
    image = Image.new("RGB", (64, 48), (100 + index, 120, 140))
    return FrameSample(
        image=image,
        timestamp=5.0 + index / 30.0,
        index=index,
        contact_score=1.0 if index == 12 else 0.1,
    )


class SemanticEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[dict] = []
        self.engine = SemanticEngine(
            config=SemanticConfig(
                llm_backend="template",
                strip_height=48,
                strip_panel_width=64,
                strip_gap=4,
            ),
            embedder=DeterministicImageEmbedder(),
            captioner=StubCaptioner("the hand places the keys inside the drawer"),
            chat_client=None,
            embedder_backend="deterministic",
            captioner_backend="stub",
            on_event=self.events.append,
        )

    def test_end_to_end_offline(self) -> None:
        zone = self.engine.identify_zone(
            Image.new("RGB", (96, 96), (180, 160, 120)),
            caption="kitchen counter",
            timestamp=1.0,
        )
        self.assertTrue(zone.is_new)

        result = self.engine.verify_interaction(
            [frame(index) for index in range(30)],
            "keys",
            container="drawer",
            timestamp=10.0,
        )
        self.assertEqual(result.strip.panel_indices, (7, 12, 17))
        self.assertEqual(result.action.action.value, "INSERT_INTO")
        self.assertTrue(result.action.accepted)
        self.assertEqual(result.strip.image.size, (64 * 3 + 4 * 2, 48))

        answer = self.engine.answer_query(
            "Where are my keys?",
            {
                "entity": {"id": "keys", "label": "keys", "state": "CONTAINED", "zone": "work desk"},
                "facts": [
                    {
                        "subject": "keys",
                        "relation": "INSERT_INTO",
                        "object": "drawer",
                        "timestamp": 14.2,
                        "zone": "work desk",
                    }
                ],
            },
        )
        self.assertIn("drawer", answer.answer)
        self.assertIn("[t=14.2s]", answer.answer)

        tags = {event["tag"] for event in self.events}
        self.assertTrue({"VLM", "GRAPH"}.issubset(tags))
        self.assertTrue(any(event["event"] == "ACTION_VERIFIED" for event in self.events))

    def test_preview_payload(self) -> None:
        result = self.engine.verify_interaction([frame(i) for i in range(20)], "keys")
        payload = result.to_dict(include_preview=True)
        self.assertIn("preview", payload)
        self.assertTrue(payload["preview"].startswith("data:image/jpeg;base64,"))

    def test_zone_persistence(self) -> None:
        self.engine.identify_zone(
            Image.new("RGB", (48, 48), (120, 180, 140)),
            caption="work desk",
            timestamp=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zones.json"
            self.engine.save_zones(str(path))
            self.engine.load_zones(str(path))
        self.assertEqual(len(self.engine.vpr), 1)

    def test_event_sink_failure_does_not_break_ingestion(self) -> None:
        def broken_sink(payload: dict) -> None:
            raise RuntimeError("dashboard disconnected")

        engine = SemanticEngine(
            config=SemanticConfig(llm_backend="template"),
            embedder=DeterministicImageEmbedder(),
            captioner=StubCaptioner(),
            chat_client=None,
            embedder_backend="deterministic",
            captioner_backend="stub",
            on_event=broken_sink,
        )
        match = engine.identify_zone(Image.new("RGB", (32, 32), (90, 90, 90)))
        self.assertTrue(match.is_new)

    def test_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            SemanticEngine(config=SemanticConfig(vpr_threshold=1.5))

    def test_from_env_overrides(self) -> None:
        config = SemanticConfig.from_env(
            {
                "CHRONICLE_VPR_THRESHOLD": "0.9",
                "CHRONICLE_STRIP_HEIGHT": "200",
                "CHRONICLE_STRIP_LABELS": "false",
                "CHRONICLE_LLM_MODEL": "llama3.2:3b",
            }
        )
        self.assertEqual(config.vpr_threshold, 0.9)
        self.assertEqual(config.strip_height, 200)
        self.assertFalse(config.strip_labels)
        self.assertEqual(config.llm_model, "llama3.2:3b")


if __name__ == "__main__":
    unittest.main()

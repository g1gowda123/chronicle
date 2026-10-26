"""Cross-role integration: Role C consumes Role A's native records."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from chronicle.semantic import SemanticConfig, SemanticEngine, TemporalStripCompositor
from chronicle.semantic.adapters import PerceptionEmbedderAdapter, adapt_embedder
from chronicle.semantic.embedders import DeterministicImageEmbedder
from chronicle.semantic.florence import StubCaptioner
from chronicle.types import ClipFrame, InteractionClip


def make_clip(frames: int = 30, contact_index: int = 14) -> InteractionClip:
    samples = []
    for index in range(frames):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 40 + index * 3  # BGR blue channel
        image[:, :, 1] = 90
        image[:, :, 2] = 140
        samples.append(
            ClipFrame(
                frame_index=index,
                timestamp=20.0 + index / 30.0,
                image=image,
            )
        )
    return InteractionClip(
        clip_id="clip-0007",
        track_id=7,
        label="keys",
        hand="Right",
        start_ts=20.0,
        end_ts=21.0,
        trigger="ios+dwell",
        frames=samples,
        contact_index=contact_index,
        max_ios=0.72,
    )


class FakePerceptionEmbedder:
    name = "fake-openclip"
    dimension = 8

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 3 and image.shape[2] == 3
        # A stable signature derived from mean BGR, like a tiny colour encoder.
        mean = image.reshape(-1, 3).mean(axis=0) / 255.0
        vector = np.zeros(8, dtype=np.float32)
        vector[:3] = mean
        vector[3] = float(np.std(image))
        vector[4] = 1.0
        return vector / max(float(np.linalg.norm(vector)), 1e-9)


class RoleAInteractionClipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SemanticEngine(
            config=SemanticConfig(llm_backend="template", strip_height=48, strip_panel_width=64),
            embedder=DeterministicImageEmbedder(),
            captioner=StubCaptioner("the hand places the keys inside the drawer"),
            chat_client=None,
            embedder_backend="deterministic",
            captioner_backend="stub",
        )

    def test_strip_accepts_interaction_clip(self) -> None:
        clip = make_clip()
        strip = TemporalStripCompositor(
            self.engine.config, channel_order="bgr"
        ).build(clip)
        self.assertEqual(strip.panel_indices, (9, 14, 19))
        self.assertEqual(strip.contact_index, 14)
        self.assertAlmostEqual(strip.frames[14].timestamp, 20.0 + 14 / 30.0, places=5)

    def test_verify_infers_clip_metadata(self) -> None:
        clip = make_clip()
        result = self.engine.verify_interaction(clip)
        self.assertEqual(result.action.object_label, "keys")
        self.assertEqual(result.action.action.value, "INSERT_INTO")
        self.assertEqual(result.clip_id, clip.clip_id)
        self.assertEqual(result.track_id, clip.track_id)
        self.assertEqual(result.trigger, clip.trigger)
        self.assertAlmostEqual(result.max_ios or 0.0, 0.72)
        payload = result.to_dict()
        self.assertEqual(payload["clip_id"], "clip-0007")
        self.assertEqual(payload["track_id"], 7)
        self.assertAlmostEqual(payload["action"]["timestamp"] or 0.0, clip.contact_ts or 0.0)


class PerceptionEmbedderAdapterTest(unittest.TestCase):
    def test_batch_protocol_and_text(self) -> None:
        adapter = adapt_embedder(FakePerceptionEmbedder())
        self.assertIsInstance(adapter, PerceptionEmbedderAdapter)
        vectors = adapter.embed_images(
            [Image.new("RGB", (32, 32), (200, 30, 30)), np.zeros((32, 32, 3), dtype=np.uint8)]
        )
        self.assertEqual(vectors.shape, (2, 8))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-6)
        texts = adapter.embed_text(["keys", "keys"])
        self.assertEqual(texts.shape, (2, 8))

    def test_engine_shares_role_a_embedder(self) -> None:
        engine = SemanticEngine(
            config=SemanticConfig(llm_backend="template"),
            embedder=FakePerceptionEmbedder(),
            captioner=StubCaptioner("kitchen"),
            chat_client=None,
            embedder_backend="deterministic",
            captioner_backend="stub",
        )
        self.assertEqual(engine.embedder.name, "perception:fake-openclip")
        first = engine.identify_zone(np.full((32, 32, 3), 90, dtype=np.uint8))
        again = engine.identify_zone(np.full((32, 32, 3), 90, dtype=np.uint8))
        self.assertFalse(again.is_new)
        self.assertEqual(first.zone.zone_id, again.zone.zone_id)


if __name__ == "__main__":
    unittest.main()

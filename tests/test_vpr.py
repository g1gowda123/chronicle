from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from chronicle.semantic.embedders import DeterministicImageEmbedder
from chronicle.semantic.florence import StubCaptioner
from chronicle.semantic.vpr import VisualPlaceRecognizer


def scene(color: tuple[int, int, int], seed: int = 0, noise: float = 0.0) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (96, 96), color)
    # A couple of fixed structures keep the signature discriminative.
    pixels = np.asarray(image).copy()
    pixels[20:40, 20:70] = (240, 240, 240)
    pixels[60:80, 30:60] = (20, 20, 20)
    if noise:
        pixels = pixels + rng.normal(0.0, noise, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


class VisualPlaceRecognizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = DeterministicImageEmbedder()
        self.vpr = VisualPlaceRecognizer(
            embedder=self.embedder,
            threshold=0.85,
            captioner=StubCaptioner("kitchen counter"),
        )

    def test_creates_then_relocalizes_zone(self) -> None:
        first = self.vpr.resolve(scene((200, 170, 120)), timestamp=1.0)
        self.assertTrue(first.is_new)
        self.assertEqual(first.zone.label, "kitchen counter")
        self.assertTrue(first.zone.zone_id.startswith("zone_kitchen_counter"))

        again = self.vpr.resolve(scene((200, 170, 120), noise=4.0), timestamp=2.0)
        self.assertFalse(again.is_new)
        self.assertEqual(again.zone.zone_id, first.zone.zone_id)
        self.assertGreaterEqual(again.score, 0.85)
        self.assertEqual(len(self.vpr), 1)
        self.assertEqual(again.zone.observations, 2)

    def test_different_room_creates_second_zone(self) -> None:
        self.vpr.resolve(scene((200, 170, 120)), timestamp=1.0)
        other = self.vpr.resolve(scene((80, 130, 210)), timestamp=2.0)
        self.assertTrue(other.is_new)
        self.assertEqual(len(self.vpr), 2)
        # Same caption must still produce a unique id.
        self.assertNotEqual(other.zone.zone_id, next(iter(self.vpr.zones)))

    def test_threshold_is_configurable(self) -> None:
        strict = VisualPlaceRecognizer(self.embedder, threshold=0.99)
        original = scene((200, 170, 120))
        strict.resolve(original, timestamp=1.0)
        # Change the top fifth of the frame: cosine ~0.95, which the default
        # 0.85 threshold would accept but the stricter one must reject.
        pixels = np.asarray(original).copy()
        pixels[:20, :, :] = (30, 30, 30)
        changed = Image.fromarray(pixels, mode="RGB")
        match = strict.resolve(changed, timestamp=2.0)
        self.assertTrue(match.is_new)

    def test_merge_duplicates(self) -> None:
        first = self.vpr.resolve(scene((200, 170, 120)), timestamp=1.0)
        # Force a duplicate zone, as can happen after a failed re-ID.
        duplicate = self.vpr.resolve(scene((200, 170, 120), noise=6.0), timestamp=2.0)
        self.assertFalse(duplicate.is_new)
        registry = VisualPlaceRecognizer(self.embedder, threshold=0.85)
        from chronicle.semantic.vpr import Zone

        prototype = self.embedder.embed_images([scene((200, 170, 120))])[0]
        registry.zones[first.zone.zone_id] = Zone(
            zone_id=first.zone.zone_id,
            label="kitchen",
            prototype=prototype,
            created_at=1.0,
            last_seen_at=1.0,
        )
        registry.zones["zone_duplicate"] = Zone(
            zone_id="zone_duplicate",
            label="Kitchen duplicate",
            prototype=prototype,
            created_at=2.0,
            last_seen_at=2.0,
        )
        merged = registry.merge_duplicates()
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(registry), 1)
        self.assertEqual(merged[0][1], first.zone.zone_id)

    def test_save_and_load(self) -> None:
        self.vpr.resolve(scene((200, 170, 120)), timestamp=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zones.json"
            self.vpr.save(path)
            loaded = VisualPlaceRecognizer.load(
                path, self.embedder, captioner=StubCaptioner("kitchen counter")
            )
        self.assertEqual(len(loaded), 1)
        zone = loaded.active_zones()[0]
        self.assertEqual(zone.label, "kitchen counter")

    def test_match_event_payload(self) -> None:
        match = self.vpr.resolve(scene((200, 170, 120)), timestamp=3.5)
        event = match.as_event()
        self.assertEqual(event["event"], "ZONE_CREATED")
        self.assertEqual(event["tag"], "VLM")
        self.assertIn("zone_id", event)
        self.assertAlmostEqual(event["timestamp"], 3.5)


if __name__ == "__main__":
    unittest.main()

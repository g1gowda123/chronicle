from __future__ import annotations

import unittest

from PIL import Image

from chronicle.semantic.config import SemanticConfig
from chronicle.semantic.temporal_strip import (
    FrameSample,
    TemporalStripCompositor,
    build_temporal_strip,
    coerce_frames,
)


def frames(count: int = 30) -> list[FrameSample]:
    samples = []
    for index in range(count):
        image = Image.new("RGB", (48, 36), (index * 6 % 255, 60, 120))
        samples.append(
            FrameSample(
                image=image,
                timestamp=10.0 + index / 30.0,
                index=index,
                contact_score=1.0 if index == 14 else 0.0,
            )
        )
    return samples


class TemporalStripTest(unittest.TestCase):
    def test_builds_three_panels_with_expected_indices(self) -> None:
        compositor = TemporalStripCompositor(
            SemanticConfig(strip_height=48, strip_panel_width=64, strip_gap=4, strip_labels=True)
        )
        strip = compositor.build(frames(), contact_index=14)
        self.assertEqual(strip.panel_indices, (9, 14, 19))
        self.assertEqual(strip.image.size, (64 * 3 + 4 * 2, 48))
        self.assertIn("CONTACT", strip.panel_labels[1])
        self.assertIn("t=10.47s", strip.panel_labels[1])
        self.assertEqual(strip.contact_index, 14)

    def test_auto_selects_max_contact_score(self) -> None:
        samples = frames()
        samples[21] = FrameSample(
            image=samples[21].image,
            timestamp=samples[21].timestamp,
            index=21,
            contact_score=5.0,
        )
        strip = build_temporal_strip(samples)
        self.assertEqual(strip.contact_index, 21)

    def test_dict_and_raw_images_are_accepted(self) -> None:
        raw = [Image.new("RGB", (20, 20), (i, i, i)) for i in range(5)]
        dicts = [
            {"frame": image, "t": float(index), "ios": 0.1}
            for index, image in enumerate(raw)
        ]
        self.assertEqual(len(coerce_frames(raw)), 5)
        self.assertEqual(len(coerce_frames(dicts)), 5)
        strip = build_temporal_strip(dicts, contact_index=2)
        self.assertEqual(strip.panel_indices, (1, 2, 3))

    def test_two_and_one_frame_buffers(self) -> None:
        two = build_temporal_strip(frames(2))
        self.assertEqual(two.panel_indices, (0, 0, 1))
        one = build_temporal_strip(frames(1))
        self.assertEqual(one.panel_indices, (0, 0, 0))

    def test_empty_buffer_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_temporal_strip([])

    def test_data_url_and_save(self) -> None:
        strip = build_temporal_strip(frames(10))
        url = strip.to_data_url()
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(strip.to_numpy().shape[2], 3)

    def test_labels_can_be_disabled(self) -> None:
        strip = build_temporal_strip(
            frames(10),
            config=SemanticConfig(strip_labels=False, strip_height=32, strip_panel_width=32),
        )
        self.assertEqual(strip.image.size, (32 * 3 + 6 * 2, 32))


if __name__ == "__main__":
    unittest.main()

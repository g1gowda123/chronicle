"""1x3 temporal strip compositor.

Role A's 30-frame ring buffer is collapsed into a single image containing the
pre-interaction, contact, and post-interaction moments.  Florence-2 consumes
this strip as one multi-moment input instead of paying for three separate VLM
calls.
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ._utils import to_pil_image
from .config import SemanticConfig

logger = logging.getLogger("chronicle.semantic.strip")


@dataclass(frozen=True)
class FrameSample:
    """One buffered frame plus the metadata needed for strip selection."""

    image: Any
    timestamp: float | None = None
    index: int | None = None
    contact_score: float | None = None

    def label_text(self, prefix: str) -> str:
        suffix = (
            f"t={self.timestamp:.2f}s"
            if self.timestamp is not None
            else f"frame {self.index}"
        )
        return f"{prefix}  {suffix}"


@dataclass(frozen=True)
class TemporalStrip:
    """Composited 1x3 strip and the frames it was built from."""

    image: Image.Image
    frames: tuple[FrameSample, ...]
    panel_indices: tuple[int, int, int]
    panel_labels: tuple[str, str, str]
    contact_index: int

    def to_numpy(self) -> Any:
        import numpy as np

        return np.asarray(self.image)

    def save(self, path: str | Path, quality: int = 92) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(target, quality=quality)
        return target

    def to_data_url(self, format: str = "JPEG", quality: int = 85) -> str:
        """Base64 data URL for NiceGUI/HTML previews (Role D)."""
        buffer = io.BytesIO()
        self.image.save(buffer, format=format, quality=quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        mime = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
        return f"data:{mime};base64,{encoded}"


def coerce_frames(frames: Sequence[Any]) -> list[FrameSample]:
    """Accept FrameSamples, dicts, or raw images from the perception layer."""
    samples: list[FrameSample] = []
    for position, item in enumerate(frames):
        if isinstance(item, FrameSample):
            samples.append(item)
            continue
        if hasattr(item, "image") and hasattr(item, "timestamp"):
            # Role A's ClipFrame (and any duck-typed equivalent).
            timestamp = getattr(item, "timestamp", None)
            try:
                timestamp = None if timestamp is None else float(timestamp)
            except (TypeError, ValueError):
                timestamp = None
            score = getattr(item, "contact_score", None)
            samples.append(
                FrameSample(
                    image=item.image,
                    timestamp=timestamp,
                    index=getattr(item, "frame_index", position),
                    contact_score=None if score is None else float(score),
                )
            )
            continue
        if isinstance(item, dict):
            image = item.get("image", item.get("frame", item.get("crop")))
            if image is None:
                raise ValueError(f"frame dict at {position} has no image/frame/crop key")
            score = item.get("contact_score", item.get("ios", item.get("hand_object_ios")))
            timestamp = item.get("timestamp", item.get("t"))
            if timestamp is not None:
                try:
                    timestamp = float(timestamp)
                except (TypeError, ValueError):
                    timestamp = None
            samples.append(
                FrameSample(
                    image=image,
                    timestamp=timestamp,
                    index=item.get("index", position),
                    contact_score=None if score is None else float(score),
                )
            )
            continue
        samples.append(FrameSample(image=item, index=position))
    return samples


def _select_indices(
    frames: Sequence[FrameSample],
    contact_index: int | None,
) -> tuple[int, int, int]:
    count = len(frames)
    if count == 0:
        raise ValueError("cannot compose a temporal strip from zero frames")
    if count == 1:
        return (0, 0, 0)
    if count == 2:
        return (0, 0, 1)

    if contact_index is None:
        scored = [
            (frame.contact_score, position)
            for position, frame in enumerate(frames)
            if frame.contact_score is not None
        ]
        if scored:
            contact = max(scored)[1]
        else:
            contact = count // 2
    else:
        contact = int(contact_index)

    contact = max(0, min(count - 1, contact))
    offset = max(1, count // 6)
    pre = max(0, contact - offset)
    post = min(count - 1, contact + offset)
    if pre == contact and contact > 0:
        pre = contact - 1
    if post == contact and contact < count - 1:
        post = contact + 1
    return (pre, contact, post)


class TemporalStripCompositor:
    """Build the 1x3 pre/contact/post composite consumed by Florence-2."""

    def __init__(
        self,
        config: SemanticConfig | None = None,
        *,
        height: int | None = None,
        panel_width: int | None = None,
        gap: int | None = None,
        labels: bool | None = None,
        channel_order: str = "bgr",
    ) -> None:
        self.config = config or SemanticConfig()
        self.height = int(height or self.config.strip_height)
        self.panel_width = int(panel_width or self.config.strip_panel_width)
        self.gap = int(self.config.strip_gap if gap is None else gap)
        self.labels = self.config.strip_labels if labels is None else bool(labels)
        self.channel_order = channel_order
        self._font = ImageFont.load_default()

    def build(
        self,
        frames: Sequence[Any],
        *,
        contact_index: int | None = None,
    ) -> TemporalStrip:
        # Role A hands the VLM worker a structured InteractionClip. Accept it
        # directly so the integration is a one-liner on their side.
        if (
            not isinstance(frames, (list, tuple))
            and hasattr(frames, "frames")
            and hasattr(frames, "contact_index")
        ):
            clip = frames
            if contact_index is None:
                contact_index = int(getattr(clip, "contact_index", 0))
            frames = clip.frames

        samples = coerce_frames(frames)
        pre, contact, post = _select_indices(samples, contact_index)
        indices = (pre, contact, post)

        prefixes = ("PRE", "CONTACT", "POST")
        labels = tuple(
            samples[index].label_text(prefix)
            for index, prefix in zip(indices, prefixes, strict=True)
        )

        panels = [
            self._panel(samples[index].image, label)
            for index, label in zip(indices, labels, strict=True)
        ]

        width = self.panel_width * 3 + self.gap * 2
        strip = Image.new("RGB", (width, self.height), (18, 18, 18))
        for position, panel in enumerate(panels):
            strip.paste(panel, (position * (self.panel_width + self.gap), 0))

        logger.debug(
            "composed temporal strip indices=%s size=%s", indices, strip.size
        )
        return TemporalStrip(
            image=strip,
            frames=tuple(samples),
            panel_indices=indices,
            panel_labels=labels,
            contact_index=contact,
        )

    def _panel(self, image: Any, label: str) -> Image.Image:
        panel = to_pil_image(image, channel_order=self.channel_order)
        panel = ImageOps.pad(
            panel,
            (self.panel_width, self.height),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )
        if not self.labels:
            return panel

        draw = ImageDraw.Draw(panel)
        bar_height = 20
        draw.rectangle((0, 0, self.panel_width, bar_height), fill=(0, 0, 0))
        draw.text((6, 4), label, fill=(255, 255, 255), font=self._font)
        return panel


def build_temporal_strip(
    frames: Sequence[Any],
    *,
    contact_index: int | None = None,
    config: SemanticConfig | None = None,
    channel_order: str = "bgr",
) -> TemporalStrip:
    """Functional shortcut used by tests and by Role A integration."""
    compositor = TemporalStripCompositor(config=config, channel_order=channel_order)
    return compositor.build(frames, contact_index=contact_index)

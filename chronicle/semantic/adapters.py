"""Adapters between Role A's perception records and Role C.

Role A already exposes an ``EmbeddingService`` / ``OpenClipEmbedder`` with a
single-image ``embed_image`` method.  Role C's VPR speaks a small batch
``ImageEmbedder`` protocol, so this adapter lets both packages share one loaded
OpenCLIP model instead of loading a second copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from ._utils import l2_normalize, to_pil_image
from .embedders import ImageEmbedder

ImageLike = Image.Image | np.ndarray | str


class PerceptionEmbedderAdapter:
    """Wrap Role A's single-image embedder in Role C's batch protocol."""

    def __init__(self, inner: Any, channel_order: str = "bgr") -> None:
        if not hasattr(inner, "embed_image"):
            raise TypeError("inner embedder must provide embed_image(image)")
        self.inner = inner
        self.channel_order = channel_order
        self._dim = int(getattr(inner, "dimension", getattr(inner, "dim", 512)))

    @property
    def name(self) -> str:
        inner_name = getattr(self.inner, "name", type(self.inner).__name__)
        return f"perception:{inner_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def _bgr(self, image: ImageLike) -> np.ndarray:
        pil = to_pil_image(image, channel_order=self.channel_order)
        return np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])

    def embed_images(self, images: Sequence[ImageLike]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = []
        for image in images:
            vector = self.inner.embed_image(self._bgr(image))
            if vector is None:
                vector = np.zeros(self.dim, dtype=np.float32)
            vectors.append(l2_normalize(np.asarray(vector, dtype=np.float32)))
        return np.stack(vectors)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = []
        for text in texts:
            method = getattr(self.inner, "embed_text", None)
            vector = None
            if callable(method):
                try:
                    vector = method(text)
                except Exception:  # noqa: BLE001 - optional backend
                    vector = None
            if vector is None:
                vector = np.zeros(self.dim, dtype=np.float32)
            vectors.append(l2_normalize(np.asarray(vector, dtype=np.float32)))
        return np.stack(vectors)


def adapt_embedder(candidate: Any, channel_order: str = "bgr") -> ImageEmbedder:
    """Return ``candidate`` if it already satisfies the batch protocol."""
    if candidate is None:
        raise TypeError("embedder is None")
    if hasattr(candidate, "embed_images"):
        return candidate
    if hasattr(candidate, "embed_image"):
        return PerceptionEmbedderAdapter(candidate, channel_order=channel_order)
    raise TypeError(
        "embedder must provide embed_images(...) or Role A-style embed_image(image)"
    )

"""Image/text embedders used by visual place recognition and Re-ID.

The production path is OpenCLIP ViT-B-32.  A deterministic colour/texture
embedder is provided as an offline fallback so the hackathon demo and the test
suite work on machines without torch weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from ._utils import l2_normalize, to_pil_image, warn_once
from .config import SemanticConfig

logger = logging.getLogger("chronicle.semantic.embedders")

ImageLike = Image.Image | np.ndarray | str | Path


@runtime_checkable
class ImageEmbedder(Protocol):
    """Minimal interface consumed by the VPR module."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_images(self, images: Sequence[ImageLike]) -> np.ndarray: ...

    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...


class DeterministicImageEmbedder:
    """Fast, dependency-light visual signature.

    The vector mixes a joint RGB histogram with a 4x4 spatial colour grid.
    It is not CLIP, but it is stable enough for place deduplication on synthetic
    frames and for exercising every code path without model downloads.
    """

    name = "deterministic-hsv-v3"
    dim = 18 + 8 + 8 + 4 * 4 * 3 + 6  # HSV histograms + grid + global stats

    def __init__(self, channel_order: str = "bgr") -> None:
        self.channel_order = channel_order

    @staticmethod
    def _one(image: ImageLike, channel_order: str) -> np.ndarray:
        pil = to_pil_image(image, channel_order=channel_order).resize(
            (32, 32), Image.Resampling.BILINEAR
        )
        rgb = np.asarray(pil, dtype=np.float32)
        hsv = np.asarray(pil.convert("HSV"), dtype=np.float32)

        parts: list[np.ndarray] = []
        for channel, bins in ((0, 18), (1, 8), (2, 8)):
            hist, _ = np.histogram(hsv[:, :, channel], bins=bins, range=(0.0, 255.0))
            hist = hist.astype(np.float32)
            parts.append(hist / max(float(hist.sum()), 1.0))

        grid = rgb.reshape(4, 8, 4, 8, 3).mean(axis=(1, 3)).reshape(-1) / 255.0
        parts.append(grid.astype(np.float32))

        parts.append(rgb.mean(axis=(0, 1)) / 255.0)
        parts.append(rgb.std(axis=(0, 1)) / 255.0)

        # Hue/saturation carry most of the place identity; down-weight the
        # spatial/statistical terms so muted rooms do not collapse together.
        weights = np.concatenate(
            [
                np.full(18, 2.0, dtype=np.float32),
                np.full(8, 1.5, dtype=np.float32),
                np.full(8, 1.5, dtype=np.float32),
                np.full(48, 0.5, dtype=np.float32),
                np.full(6, 0.5, dtype=np.float32),
            ]
        )
        return l2_normalize(np.concatenate(parts) * weights)

    def embed_images(self, images: Sequence[ImageLike]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._one(image, self.channel_order) for image in images])

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Signed feature hashing; good enough for label similarity in demos."""
        vectors = []
        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)
            tokens = [token for token in text.lower().replace("_", " ").split() if token]
            for token in tokens or [""]:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            vectors.append(vector)
        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack(vectors))


class OpenClipEmbedder:
    """OpenCLIP ViT-B-32 wrapper with lazy model loading.

    Importing torch/open_clip takes seconds and downloading weights can take
    minutes, so nothing happens until the first embedding request.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        channel_order: str = "bgr",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.requested_device = device
        self.channel_order = channel_order
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device: str | None = None
        self._dim = 512
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return f"openclip:{self.model_name}:{self.pretrained}"

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def is_importable() -> bool:
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("open_clip") is not None
        )

    def _resolve_device(self, torch: Any) -> str:
        requested = (self.requested_device or "auto").lower()
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import open_clip  # type: ignore
            import torch  # type: ignore

            self._torch = torch
            self._device = self._resolve_device(torch)
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained, device=self._device
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)

            projection = getattr(model, "visual", None)
            output_dim = getattr(projection, "output_dim", None)
            if isinstance(output_dim, int) and output_dim > 0:
                self._dim = output_dim

    def _to_tensor_batch(self, images: Sequence[ImageLike]) -> Any:
        tensor = self._torch
        pil_images = [to_pil_image(image, channel_order=self.channel_order) for image in images]
        batch = tensor.stack([self._preprocess(image) for image in pil_images])
        return batch.to(self._device)

    def embed_images(self, images: Sequence[ImageLike]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._load()
        with self._torch.inference_mode():
            batch = self._to_tensor_batch(images)
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return features.detach().float().cpu().numpy()

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._load()
        with self._torch.inference_mode():
            tokens = self._tokenizer(list(texts)).to(self._device)
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return features.detach().float().cpu().numpy()


def build_embedder(
    config: SemanticConfig | None = None,
    backend: str = "auto",
    channel_order: str = "bgr",
) -> ImageEmbedder:
    """Create the best available embedder.

    ``backend='auto'`` tries OpenCLIP and silently falls back to the
    deterministic embedder if torch/open_clip/weights are unavailable.
    ``backend='openclip'`` raises instead, which is useful in production
    validation.
    """
    config = config or SemanticConfig()
    config.validate()
    if backend not in {"auto", "openclip", "deterministic"}:
        raise ValueError(f"unknown embedder backend {backend!r}")

    if backend == "deterministic":
        return DeterministicImageEmbedder(channel_order=channel_order)

    if not OpenClipEmbedder.is_importable():
        if backend == "openclip":
            raise RuntimeError(
                "OpenCLIP backend requested but torch/open_clip are not installed. "
                "Install requirements-semantic-models.txt."
            )
        warn_once(
            "openclip-missing",
            "OpenCLIP not installed; using the deterministic embedder. "
            "Install the semantic-models extra for CLIP retrieval.",
        )
        return DeterministicImageEmbedder(channel_order=channel_order)

    embedder = OpenClipEmbedder(
        model_name=config.clip_model_name,
        pretrained=config.clip_pretrained,
        device=config.device,
        channel_order=channel_order,
    )
    if backend == "openclip":
        return embedder

    # Auto mode: validate that weights can actually be loaded.  If the machine
    # is offline, fall back instead of failing the whole ingestion run.
    try:
        embedder.embed_images([Image.new("RGB", (64, 64), (127, 127, 127))])
    except Exception as exc:  # pragma: no cover - depends on local weights  # noqa: BLE001
        warn_once(
            "openclip-load-failed",
            "OpenCLIP could not be loaded (%s); using deterministic embedder.",
            exc,
        )
        return DeterministicImageEmbedder(channel_order=channel_order)
    return embedder

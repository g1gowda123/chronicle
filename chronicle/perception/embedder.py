"""Appearance embeddings for Re-ID, with an offline colour fallback.

The production backend is OpenCLIP ViT-B-32 (512-D). The colour backend
computes a spatial HSV histogram projected into the same dimensionality, which
keeps the demo and the unit tests completely offline.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from chronicle.perception.config import EmbedderConfig, PerceptionConfig
from chronicle.perception.utils import crop_image, import_available, normalize, resolve_device
from chronicle.types import BBox

LOGGER = logging.getLogger(__name__)


class BaseEmbedder:
    name = "base"
    dimension = 512
    available = True

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def embed_crops(self, frame: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        vectors = []
        for bbox in boxes:
            crop = crop_image(frame, bbox)
            vectors.append(
                np.zeros(self.dimension, dtype=np.float32)
                if crop is None
                else self.embed_image(crop)
            )
        return np.stack(vectors) if vectors else np.zeros((0, self.dimension), np.float32)

    def embed_text(self, text: str) -> np.ndarray | None:
        return None

    def close(self) -> None:
        pass


class ColorEmbedder(BaseEmbedder):
    """Spatial HSV histogram projected to 512 dimensions.

    Deterministic and fast, but only an appearance proxy: useful for the
    synthetic demo and tests, not for production Re-ID.
    """

    name = "color-512"
    dimension = 512

    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self.dimension = int(config.dimension)
        self.grid = 4
        self._bins = (8, 4, 4)
        self._per_cell = int(np.prod(self._bins))
        rng = np.random.default_rng(7)
        self._projection = (
            rng.normal(0.0, 1.0, size=(self._per_cell * self.grid * self.grid, self.dimension))
            / np.sqrt(self.dimension)
        ).astype(np.float32)
        self._cache: dict[bytes, np.ndarray] = {}
        self._lock = threading.Lock()

    def _descriptor(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        cells: list[np.ndarray] = []
        for gy in range(self.grid):
            for gx in range(self.grid):
                y1, y2 = int(gy * height / self.grid), int((gy + 1) * height / self.grid)
                x1, x2 = int(gx * width / self.grid), int((gx + 1) * width / self.grid)
                cell = hsv[y1:y2, x1:x2]
                if cell.size == 0:
                    cells.append(np.zeros(self._per_cell, dtype=np.float32))
                    continue
                histogram = cv2.calcHist(
                    [cell], [0, 1, 2], None, list(self._bins), [0, 180, 0, 256, 0, 256]
                )
                flat = histogram.reshape(-1).astype(np.float32)
                norm = float(np.linalg.norm(flat))
                cells.append(flat / norm if norm > 0 else flat)
        descriptor = np.concatenate(cells)
        vector = normalize(descriptor @ self._projection)
        return vector if vector is not None else np.zeros(self.dimension, np.float32)

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros(self.dimension, dtype=np.float32)
        thumb = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        key = thumb.tobytes()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        vector = self._descriptor(thumb)
        with self._lock:
            if len(self._cache) >= max(16, int(self.config.cache_size)):
                self._cache.clear()
            self._cache[key] = vector
        return vector.copy()


class OpenClipEmbedder(BaseEmbedder):
    """OpenCLIP ViT-B-32 adapter (lazy-loaded on first embedding)."""

    name = "openclip"
    dimension = 512

    def __init__(self, config: EmbedderConfig, device: str = "auto") -> None:
        self.config = config
        self.requested_device = device
        self._model: object | None = None
        self._preprocess: object | None = None
        self._tokenizer: object | None = None
        self._torch: object | None = None
        self._device = "cpu"
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import open_clip
                import torch
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "OpenCLIP is not installed. Install perception dependencies with "
                    "`uv sync`, or run with --embedder color."
                ) from exc
            self._device = resolve_device(self.requested_device)
            LOGGER.info(
                "Loading OpenCLIP %s/%s on %s",
                self.config.model,
                self.config.pretrained,
                self._device,
            )
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.config.model,
                pretrained=self.config.pretrained,
                device=self._device,
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.config.model)
            self._torch = torch

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _embed_arrays(self, images: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        self.load()
        assert self._torch is not None and self._model is not None and self._preprocess is not None
        tensors = []
        for image in images:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensors.append(self._preprocess(Image.fromarray(rgb)))
        if not tensors:
            return np.zeros((0, self.dimension), dtype=np.float32)
        batch = self._torch.stack(tensors).to(self._device)
        with self._lock, self._torch.no_grad():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().astype(np.float32)

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros(self.dimension, dtype=np.float32)
        return self._embed_arrays([image])[0]

    def embed_crops(self, frame: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        images: list[np.ndarray] = []
        positions: list[int] = []
        output = np.zeros((len(boxes), self.dimension), dtype=np.float32)
        for index, bbox in enumerate(boxes):
            crop = crop_image(frame, bbox)
            if crop is not None and crop.size:
                images.append(crop)
                positions.append(index)
        if not images:
            return output
        vectors = self._embed_arrays(images)
        for position, vector in zip(positions, vectors):
            output[position] = vector
        return output

    def embed_text(self, text: str) -> np.ndarray | None:
        if not text:
            return None
        self.load()
        assert self._torch is not None and self._model is not None and self._tokenizer is not None
        tokens = self._tokenizer([text]).to(self._device)
        with self._lock, self._torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().astype(np.float32)[0]

    def close(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None


class EmbeddingService:
    """Backend selection plus crop/scene embedding helpers."""

    def __init__(self, config: PerceptionConfig, *, source_name: str = "") -> None:
        self.config = config
        backend = config.embedder.backend.lower()
        if backend == "auto":
            if config.is_demo_source(source_name):
                backend = "color"
            elif import_available("open_clip") and import_available("torch"):
                backend = "openclip"
            else:
                LOGGER.warning("OpenCLIP unavailable; using the colour embedding fallback.")
                backend = "color"
        if backend == "color":
            self._embedder: BaseEmbedder = ColorEmbedder(config.embedder)
        elif backend == "openclip":
            self._embedder = OpenClipEmbedder(config.embedder, config.device)
        else:
            raise ValueError(f"unknown embedder backend: {config.embedder.backend}")

    @property
    def backend(self) -> str:
        return self._embedder.name

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def available(self) -> bool:
        return True

    @property
    def embedder(self) -> BaseEmbedder:
        return self._embedder

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        return self._embedder.embed_image(image)

    def embed_crops(self, frame: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        return self._embedder.embed_crops(frame, boxes)

    def embed_text(self, text: str) -> np.ndarray | None:
        try:
            return self._embedder.embed_text(text)
        except Exception:
            return None

    def embed_scene(self, frame: np.ndarray) -> np.ndarray:
        return self._embedder.embed_image(frame)

    def close(self) -> None:
        self._embedder.close()


def build_embedder(config: PerceptionConfig, *, source_name: str = "") -> EmbeddingService:
    return EmbeddingService(config, source_name=source_name)

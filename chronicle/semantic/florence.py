"""Florence-2 captioning with lazy loading and offline stubs.

The model is only loaded on first use.  ``LazyCaptioner`` degrades to a stub if
transformers/torch or the weights are unavailable, so the ingestion pipeline
keeps running (and Role D keeps showing telemetry) instead of crashing.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from collections.abc import Callable
from typing import Any

from ._utils import to_pil_image, warn_once
from .config import SemanticConfig

logger = logging.getLogger("chronicle.semantic.florence")


class ModelUnavailableError(RuntimeError):
    """Raised when an optional foundation model cannot be loaded."""


class FlorenceCaptioner:
    """``microsoft/Florence-2-base`` wrapper for dense captions."""

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        *,
        device: str = "auto",
        max_new_tokens: int = 128,
        num_beams: int = 3,
        channel_order: str = "bgr",
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.channel_order = channel_order
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._lock = threading.RLock()

    @staticmethod
    def is_importable() -> bool:
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
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
            try:
                import torch  # type: ignore
                from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ModelUnavailableError(
                    "Florence-2 requires torch and transformers; install "
                    "requirements-semantic-models.txt"
                ) from exc

            self._torch = torch
            self._device = self._resolve_device(torch)
            self._dtype = torch.float16 if self._device == "cuda" else torch.float32

            logger.info("loading Florence-2 %s on %s", self.model_id, self._device)
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype=self._dtype,
                )
            except Exception as exc:  # pragma: no cover - depends on weights/network
                raise ModelUnavailableError(
                    f"could not load Florence-2 ({self.model_id}): {exc}"
                ) from exc

            self._model.to(self._device)
            self._model.eval()

    def caption(self, image: Any, task: str = "<MORE_DETAILED_CAPTION>") -> str:
        self._load()
        with self._lock, self._torch.inference_mode():
            pil = to_pil_image(image, channel_order=self.channel_order)
            inputs = self._processor(text=task, images=pil, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self._device)
            pixel_values = inputs["pixel_values"].to(
                device=self._device, dtype=self._dtype
            )
            generated = self._model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )
            text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]

        try:
            parsed = self._processor.post_process_generation(
                text, task=task, image_size=(pil.width, pil.height)
            )
            value = parsed.get(task, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception as exc:  # pragma: no cover - model-specific  # noqa: BLE001
            logger.debug("Florence post-processing failed: %s", exc)

        cleaned = str(text).replace(task, " ").strip()
        return cleaned


class StubCaptioner:
    """Offline captioner used by the demo/tests."""

    def __init__(
        self,
        default_caption: str = "",
        caption_fn: Callable[[Any], str] | None = None,
    ) -> None:
        self.default_caption = default_caption
        self.caption_fn = caption_fn

    def caption(self, image: Any, task: str = "<MORE_DETAILED_CAPTION>") -> str:
        if self.caption_fn is not None:
            return str(self.caption_fn(image))
        return self.default_caption


class LazyCaptioner:
    """Try Florence on first use, then stick to a stub after any failure."""

    def __init__(
        self,
        config: SemanticConfig | None = None,
        *,
        fallback: StubCaptioner | None = None,
        channel_order: str = "bgr",
    ) -> None:
        self.config = config or SemanticConfig()
        self.fallback = fallback or StubCaptioner()
        self.channel_order = channel_order
        self._impl: Any = None
        self._failed = False
        self._lock = threading.RLock()

    def _resolve(self) -> Any:
        if self._impl is not None:
            return self._impl
        with self._lock:
            if self._impl is not None:
                return self._impl
            if self._failed or not FlorenceCaptioner.is_importable():
                self._impl = self.fallback
            else:
                self._impl = FlorenceCaptioner(
                    self.config.florence_model_id,
                    device=self.config.device,
                    max_new_tokens=self.config.florence_max_new_tokens,
                    num_beams=self.config.florence_num_beams,
                    channel_order=self.channel_order,
                )
            return self._impl

    def caption(self, image: Any, task: str | None = None) -> str:
        target_task = task or self.config.action_task
        impl = self._resolve()
        if isinstance(impl, StubCaptioner):
            return impl.caption(image, target_task)
        try:
            return impl.caption(image, target_task)
        except Exception as exc:  # noqa: BLE001
            self._failed = True
            self._impl = self.fallback
            warn_once(
                "florence-failed",
                "Florence-2 unavailable (%s); falling back to stub captions.",
                exc,
            )
            return self.fallback.caption(image, target_task)


def build_captioner(
    config: SemanticConfig | None = None,
    backend: str = "auto",
    *,
    channel_order: str = "bgr",
) -> Any:
    """``auto``/``florence`` return a lazy model, ``stub`` is fully offline."""
    config = config or SemanticConfig()
    if backend == "stub":
        return StubCaptioner()
    if backend == "florence":
        return FlorenceCaptioner(
            config.florence_model_id,
            device=config.device,
            max_new_tokens=config.florence_max_new_tokens,
            num_beams=config.florence_num_beams,
            channel_order=channel_order,
        )
    if backend != "auto":
        raise ValueError(f"unknown captioner backend {backend!r}")
    return LazyCaptioner(config, channel_order=channel_order)

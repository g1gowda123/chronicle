"""Qwen2.5-VL captioner backed by a local Ollama server.

This is the drop-in alternative to :class:`FlorenceCaptioner`.  It implements
the same ``caption(image, task) -> str`` interface, so the perception pipeline,
event gating and graph code are unchanged.

The model never leaves the machine: Ollama serves ``qwen2.5vl:3b`` over
``http://127.0.0.1:11434``.  Prompts are short and deterministic because the
captioner is called inside the event gate, not once per frame.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any

from ._utils import to_pil_image

logger = logging.getLogger("chronicle.semantic.qwen_vl")

DEFAULT_PROMPT = (
    "Describe this egocentric video frame in one concise sentence. "
    "Mention the main objects and any action being performed."
)


class QwenVLUnavailableError(RuntimeError):
    """Raised when the local Ollama vision model cannot be reached."""


class QwenVLCaptioner:
    """``qwen2.5vl:3b`` (or any Ollama vision model) captioner."""

    def __init__(
        self,
        model: str = "qwen2.5vl:3b",
        *,
        host: str = "http://127.0.0.1:11434",
        timeout_s: float = 120.0,
        max_side: int = 448,
        num_predict: int = 64,
        temperature: float = 0.0,
        keep_alive: str = "10m",
        prompt: str = DEFAULT_PROMPT,
        channel_order: str = "bgr",
    ) -> None:
        self.model_id = model
        self.host = host.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_side = max(128, int(max_side))
        self.num_predict = max(16, int(num_predict))
        self.temperature = float(temperature)
        self.keep_alive = keep_alive
        self.prompt = prompt
        self.channel_order = channel_order
        self._lock = threading.RLock()
        self._warned = False

    @property
    def model(self) -> str:
        return self.model_id

    @staticmethod
    def _encode(image: Any, *, channel_order: str, max_side: int) -> str:
        from PIL import Image

        pil = to_pil_image(image, channel_order=channel_order).convert("RGB")
        longest = max(pil.width, pil.height)
        if longest > max_side:
            scale = max_side / float(longest)
            pil = pil.resize(
                (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def caption(self, image: Any, task: str | None = None) -> str:
        """Return one caption for ``image``.

        ``task`` is accepted for interface compatibility with Florence-2 and
        ignored: Qwen2.5-VL uses a natural-language prompt instead of task
        tokens such as ``<MORE_DETAILED_CAPTION>``.
        """

        encoded = self._encode(
            image, channel_order=self.channel_order, max_side=self.max_side
        )
        payload = {
            "model": self.model_id,
            "prompt": self.prompt,
            "images": [encoded],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._lock, urllib.request.urlopen(
                request, timeout=self.timeout_s
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise QwenVLUnavailableError(
                f"cannot reach Ollama at {self.host}; start `ollama serve` and "
                f"pull {self.model_id} ({exc})"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise QwenVLUnavailableError(
                f"Ollama returned an invalid response for {self.model_id}: {exc}"
            ) from exc

        text = str(data.get("response") or "").strip()
        if not text:
            raise QwenVLUnavailableError(
                f"{self.model_id} returned an empty caption at {self.host}"
            )
        return " ".join(text.split())

    def is_available(self, *, timeout_s: float = 1.5) -> bool:
        request = urllib.request.Request(f"{self.host}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return False
        names = [str(item.get("name") or item.get("model") or "") for item in payload.get("models", [])]
        return any(name == self.model_id or name.startswith(self.model_id) for name in names)

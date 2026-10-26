"""Local LLM clients for action JSON generation and GraphRAG answers.

Ollama is the default because it is a single HTTP call and works well in a
hackathon demo.  A transformers text-generation backend is available for fully
offline GPU boxes.  When neither is reachable, the semantic engine falls back to
deterministic templates so the pipeline never stalls.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

from ._utils import warn_once
from .config import SemanticConfig

logger = logging.getLogger("chronicle.semantic.llm")


class LLMUnavailableError(RuntimeError):
    """Raised when a local LLM cannot be reached or returns an error."""


@runtime_checkable
class ChatClient(Protocol):
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
    ) -> str: ...


class OllamaChatClient:
    """Tiny stdlib client for ``POST /api/chat`` (no requests dependency)."""

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def is_available(self, timeout_s: float = 1.0) -> bool:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return 200 <= int(response.status) < 300
        except Exception:  # noqa: BLE001
            return False

    def list_models(self, timeout_s: float = 3.0) -> list[str]:
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise LLMUnavailableError(f"Ollama is not reachable at {self.base_url}: {exc}") from exc
        return [str(item.get("name")) for item in payload.get("models", [])]

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 320},
        }
        if json_mode:
            payload["format"] = "json"

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc
        except Exception as exc:
            raise LLMUnavailableError(f"Ollama returned an invalid response: {exc}") from exc

        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise LLMUnavailableError(f"unexpected Ollama payload: {data!r}") from exc


class TransformersChatClient:
    """Lazy local ``transformers`` text-generation backend."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 320,
        timeout_s: float = 120.0,
    ) -> None:
        self.model_id = model
        self.requested_device = device
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_s = float(timeout_s)
        self._pipeline: Any = None
        self._tokenizer: Any = None
        self._lock = threading.RLock()

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            import torch  # type: ignore
            from transformers import pipeline  # type: ignore

            if (self.requested_device or "auto").lower() == "cpu":
                device_index = -1
            elif torch.cuda.is_available():
                device_index = 0
            else:
                device_index = -1
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                device=device_index,
            )
            self._tokenizer = getattr(self._pipeline, "tokenizer", None)

    def _format(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenizer = self._tokenizer
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                logger.debug("tokenizer chat template failed; using plain prompt", exc_info=True)
        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
    ) -> str:
        self._load()
        prompt = self._format(system_prompt, user_prompt)
        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
        except Exception as exc:
            raise LLMUnavailableError(f"transformers generation failed: {exc}") from exc
        if not outputs:
            raise LLMUnavailableError("transformers returned no output")
        text = outputs[0].get("generated_text", outputs[0])
        return str(text).strip()


def build_chat_client(config: SemanticConfig | None = None) -> ChatClient | None:
    """Build the configured client, or ``None`` for template-only mode."""
    config = config or SemanticConfig()
    backend = config.llm_backend.lower()

    if backend in {"template", "none"}:
        return None

    if backend == "ollama":
        return OllamaChatClient(
            model=config.llm_model,
            base_url=config.ollama_url,
            timeout_s=config.llm_timeout_s,
        )

    if backend == "transformers":
        return TransformersChatClient(
            model=config.llm_model,
            device=config.device,
            timeout_s=config.llm_timeout_s,
        )

    # auto: prefer a running Ollama daemon; never download multi-GB weights
    # implicitly during ingestion.
    client = OllamaChatClient(
        model=config.llm_model,
        base_url=config.ollama_url,
        timeout_s=config.llm_timeout_s,
    )
    if client.is_available():
        return client
    warn_once(
        "ollama-unavailable",
        "No Ollama daemon at %s; using deterministic template fallbacks.",
        config.ollama_url,
    )
    return None

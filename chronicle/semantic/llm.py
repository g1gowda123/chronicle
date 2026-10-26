"""Local LLM clients for action JSON generation and GraphRAG answers.

Ollama is the default because it is a single HTTP call and works well in a
hackathon demo.  A transformers text-generation backend is available for fully
offline GPU boxes.  When neither is reachable, the semantic engine falls back to
deterministic templates so the pipeline never stalls.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
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


def _has_cuda_runtime(binary: Path) -> bool:
    """True when this Ollama install ships a CUDA llama-server runtime."""
    libdir = binary.parent.parent / "lib" / "ollama"
    return any(libdir.glob("cuda_v*"))


def _ollama_binary(config: Any = None) -> str | None:
    explicit = getattr(config, "ollama_bin", None) or os.getenv("OLLAMA_BIN")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path)
    # Prefer local GPU-capable install beside local CUDA runtimes
    local = Path.home() / ".local" / "ollama" / "bin" / "ollama"
    if local.is_file() and _has_cuda_runtime(local):
        return str(local)
    found = shutil.which("ollama")
    if found:
        return found
    return str(local) if local.is_file() else None


def probe_ollama(url: str, model: str, timeout: float = 1.5) -> dict[str, Any]:
    """Return server/model availability without loading the model."""
    base = url.rstrip("/")
    status: dict[str, Any] = {
        "url": base,
        "model": model,
        "server": False,
        "model_present": False,
        "models": [],
        "error": None,
    }
    request = urllib.request.Request(f"{base}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                status["error"] = f"HTTP {response.status}"
                return status
            payload = json.loads(response.read().decode("utf-8"))
        names = [
            str(item.get("name") or item.get("model") or "")
            for item in payload.get("models", [])
        ]
        names = [name for name in names if name]
        status["server"] = True
        status["models"] = names
        wanted = model.strip()
        status["model_present"] = any(
            name == wanted
            or name.split(":")[0] == wanted.split(":")[0]
            or name.startswith(wanted)
            for name in names
        )
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def probe_ollama_runtime(url: str, model: str, timeout: float = 1.5) -> dict[str, Any]:
    """Probe the server, model, version, and currently loaded runner."""
    status = probe_ollama(url, model, timeout=timeout)
    status.update({"version": None, "loaded": [], "available": False})
    if not status.get("server"):
        return status
    status["available"] = bool(status.get("server") and status.get("model_present"))
    base = url.rstrip("/")
    try:
        req_ver = urllib.request.Request(f"{base}/api/version", method="GET")
        with urllib.request.urlopen(req_ver, timeout=timeout) as response:
            if response.status == 200:
                ver_data = json.loads(response.read().decode("utf-8"))
                status["version"] = ver_data.get("version")

        req_ps = urllib.request.Request(f"{base}/api/ps", method="GET")
        with urllib.request.urlopen(req_ps, timeout=timeout) as response:
            if response.status == 200:
                ps_data = json.loads(response.read().decode("utf-8"))
                loaded = []
                for item in ps_data.get("models", []):
                    size = int(item.get("size") or 0)
                    size_vram = int(item.get("size_vram") or 0)
                    gpu_percent = round(100 * size_vram / size) if size else 0
                    loaded.append(
                        {
                            "name": item.get("name") or item.get("model") or "",
                            "size": size,
                            "size_vram": size_vram,
                            "gpu_percent": min(100, max(0, gpu_percent)),
                            "context_length": item.get("context_length"),
                            "expires_at": item.get("expires_at"),
                        }
                    )
                status["loaded"] = loaded
    except Exception as exc:
        status["runtime_error"] = f"{type(exc).__name__}: {exc}"
    return status


def ensure_local_ollama(config: Any = None, *, wait_seconds: float = 12.0) -> dict[str, Any]:
    """Start user-local or bundled Ollama server when needed."""
    cfg = config or SemanticConfig()
    url = getattr(cfg, "ollama_url", "http://127.0.0.1:11434")
    model = getattr(cfg, "llm_model", "qwen2.5:7b-instruct")
    status = probe_ollama(url, model)
    if status["server"]:
        return status

    binary = _ollama_binary(cfg)
    if not binary:
        return status

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ollama.log"

    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
    try:
        with open(log_path, "ab") as log:
            subprocess.Popen(
                [binary, "serve"],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
    except Exception as exc:
        status["error"] = f"could not start Ollama: {type(exc).__name__}: {exc}"
        return status

    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() < deadline:
        time.sleep(0.35)
        status = probe_ollama(url, model)
        if status["server"]:
            return status
    return status


class OllamaChatClient:
    """Tiny stdlib client for ``POST /api/chat`` (no requests dependency)."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 120.0,
        *,
        keep_alive: str = "30m",
        num_ctx: int = 4096,
        num_predict: int = 220,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.keep_alive = keep_alive
        self.num_ctx = int(num_ctx)
        self.num_predict = int(num_predict)
        self.temperature = float(temperature)
        self._resolved_model: str | None = None
        self.calls: list[dict[str, int]] = []

    provider = "ollama"

    @property
    def last_usage(self) -> dict[str, int]:
        return self.calls[-1] if self.calls else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @property
    def active_model(self) -> str:
        """The model name actually used by the most recent request."""
        return self._resolved_model or self.model

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
        return [str(item.get("name") or item.get("model") or "") for item in payload.get("models", [])]

    def _resolve_model(self) -> str:
        """Resolve model name, picking the best matching installed Qwen if needed."""
        resolved = self._pick_model()
        self._resolved_model = resolved
        return resolved

    def _pick_model(self) -> str:
        try:
            available = self.list_models(timeout_s=1.0)
            if not available:
                return self.model
            if self.model in available:
                return self.model
            # Prefix / family match
            wanted_family = self.model.split(":")[0]
            for m in available:
                if m == self.model or m.split(":")[0] == wanted_family or m.startswith(self.model):
                    return m
            # If Qwen model requested, pick any available Qwen model
            if "qwen" in self.model.lower():
                qwens = [m for m in available if "qwen" in m.lower()]
                if qwens:
                    # Prefer smaller models if available
                    for preferred in ("0.5b", "1.5b", "3b", "7b"):
                        match = next((q for q in qwens if preferred in q), None)
                        if match:
                            return match
                    return qwens[0]
        except Exception:
            pass
        return self.model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
    ) -> str:
        model_name = self._resolve_model()
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
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

        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        self.calls.append(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        )
        try:
            text = str(data["message"]["content"]).strip()
            # Clean possible markdown wrap or ANSWER: header
            if "ANSWER:" in text:
                text = text.split("ANSWER:", 1)[1].strip()
            text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text).strip()
            return text
        except (KeyError, TypeError) as exc:
            raise LLMUnavailableError(f"unexpected Ollama payload: {data!r}") from exc


class GeminiChatClient:
    """Gemini via the OpenAI-compatible endpoint (stdlib only).

    The API key is read from ``GEMINI_API_KEY`` (never stored in the repo).
    Usage is accumulated in :attr:`calls` so benchmarks can report tokens.
    """

    provider = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        *,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_s: float = 120.0,
        num_predict: int = 220,
        temperature: float = 0.1,
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.num_predict = int(num_predict)
        self.temperature = float(temperature)
        self.max_retries = max(1, int(max_retries))
        self.calls: list[dict[str, int]] = []

    @property
    def active_model(self) -> str:
        return self.model

    @property
    def last_usage(self) -> dict[str, int]:
        return self.calls[-1] if self.calls else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def is_available(self, timeout_s: float = 3.0) -> bool:
        if not self.api_key:
            return False
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return 200 <= int(response.status) < 300
        except Exception:  # noqa: BLE001
            return False

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
    ) -> str:
        if not self.api_key:
            raise LLMUnavailableError("GEMINI_API_KEY is not set")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.num_predict,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(payload)
        try:
            text = str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(f"unexpected Gemini payload: {data!r}") from exc
        text = re.sub(r"^```(?:text|json)?\s*|\s*```$", "", text).strip()
        if "ANSWER:" in text:
            text = text.split("ANSWER:", 1)[1].strip()
        usage = data.get("usage") or {}
        self.calls.append(
            {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            }
        )
        return text

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:300]
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    retry_after = exc.headers.get("retry-after") if exc.headers else None
                    wait = float(retry_after) if retry_after else delay * attempt
                    time.sleep(min(wait, 30.0))
                    continue
                raise LLMUnavailableError(f"Gemini HTTP {exc.code}: {body}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(delay * attempt)
                    continue
                raise LLMUnavailableError(f"Gemini request failed: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                raise LLMUnavailableError(f"Gemini returned an invalid response: {exc}") from exc
        raise LLMUnavailableError("Gemini retries exhausted")


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
        ensure_local_ollama(config)
        return OllamaChatClient(
            model=config.llm_model,
            base_url=config.ollama_url,
            timeout_s=config.llm_timeout_s,
            keep_alive=config.llm_keep_alive,
            num_ctx=config.llm_num_ctx,
            num_predict=config.llm_num_predict,
            temperature=config.llm_temperature,
        )

    if backend == "transformers":
        return TransformersChatClient(
            model=config.hf_llm_model or config.llm_model,
            device=config.device,
            timeout_s=config.llm_timeout_s,
        )

    # auto: start Ollama if not running, then use it
    ensure_local_ollama(config, wait_seconds=3.0)
    client = OllamaChatClient(
        model=config.llm_model,
        base_url=config.ollama_url,
        timeout_s=config.llm_timeout_s,
        keep_alive=config.llm_keep_alive,
        num_ctx=config.llm_num_ctx,
        num_predict=config.llm_num_predict,
        temperature=config.llm_temperature,
    )
    if client.is_available():
        return client
    warn_once(
        "ollama-unavailable",
        "No Ollama daemon at %s; using deterministic template fallbacks.",
        config.ollama_url,
    )
    return None

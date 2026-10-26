"""Configuration for the Role C semantic engine.

Every field can be overridden with a ``CHRONICLE_<FIELD_NAME>`` environment
variable, which makes the demo easy to run on a laptop or on the GPU box
without editing code.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .actions import DEFAULT_ACTION_THRESHOLDS


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_pairs(value: str) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, raw = item.partition("=")
        if not raw:
            raise ValueError(f"expected NAME=VALUE in {item!r}")
        pairs.append((name.strip().upper(), float(raw)))
    return tuple(pairs)


@dataclass(frozen=True)
class SemanticConfig:
    # --- OpenCLIP visual place recognition (Role C) -----------------------
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"  # auto | cpu | cuda | mps
    vpr_threshold: float = 0.85
    vpr_min_margin: float = 0.0
    vpr_prototype_ema: float = 0.85
    vpr_caption_zones: bool = True
    vpr_max_zones: int = 512

    # --- 1x3 temporal strip ------------------------------------------------
    strip_height: int = 336
    strip_panel_width: int = 336
    strip_gap: int = 6
    strip_labels: bool = True

    # --- Florence-2 action verification -----------------------------------
    florence_model_id: str = "microsoft/Florence-2-base"
    florence_max_new_tokens: int = 128
    florence_num_beams: int = 3
    action_min_confidence: float = 0.45
    action_task: str = "<MORE_DETAILED_CAPTION>"
    action_reject_ambiguous: bool = True
    action_thresholds: tuple[tuple[str, float], ...] = DEFAULT_ACTION_THRESHOLDS

    # --- Bounded verification queue ---------------------------------------
    verification_queue_size: int = 8
    verification_queue_policy: str = "drop_oldest"  # drop_oldest | reject
    verification_dedup_window: int = 1024

    # --- Local LLM (Ollama by default, transformers optional) -------------
    llm_backend: str = "auto"  # auto | ollama | transformers | template | none
    llm_model: str = "qwen2.5:7b-instruct"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_bin: str | None = None
    hf_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_timeout_s: float = 120.0
    llm_keep_alive: str = "30m"
    llm_num_ctx: int = 4096
    llm_num_predict: int = 220
    llm_temperature: float = 0.1
    require_citations: bool = True
    max_facts: int = 40

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SemanticConfig:
        env = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            key = f"CHRONICLE_{field.name.upper()}"
            if key not in env:
                continue
            raw = env[key]
            default = field.default
            if isinstance(default, bool):
                values[field.name] = _as_bool(raw)
            elif isinstance(default, int):
                values[field.name] = int(raw)
            elif isinstance(default, float):
                values[field.name] = float(raw)
            elif isinstance(default, tuple):
                values[field.name] = _as_pairs(raw)
            else:
                values[field.name] = raw
        return cls(**values)

    def replace(self, **changes: Any) -> SemanticConfig:
        return dataclasses.replace(self, **changes)

    def validate(self) -> None:
        if not 0.0 <= self.vpr_threshold <= 1.0:
            raise ValueError("vpr_threshold must be in [0, 1]")
        if not 0.0 <= self.vpr_min_margin <= 1.0:
            raise ValueError("vpr_min_margin must be in [0, 1]")
        if not 0.0 <= self.vpr_prototype_ema <= 1.0:
            raise ValueError("vpr_prototype_ema must be in [0, 1]")
        if not 0.0 <= self.action_min_confidence <= 1.0:
            raise ValueError("action_min_confidence must be in [0, 1]")
        for name, value in self.action_thresholds:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"action threshold for {name!r} must be in [0, 1]")
        if self.verification_queue_size < 1:
            raise ValueError("verification_queue_size must be >= 1")
        if self.verification_queue_policy not in {"drop_oldest", "reject"}:
            raise ValueError(
                f"unknown verification_queue_policy {self.verification_queue_policy!r}"
            )
        if self.verification_dedup_window < 0:
            raise ValueError("verification_dedup_window must be >= 0")
        if self.strip_height <= 0 or self.strip_panel_width <= 0:
            raise ValueError("strip dimensions must be positive")
        if self.llm_backend not in {"auto", "ollama", "transformers", "template", "none"}:
            raise ValueError(f"unknown llm_backend {self.llm_backend!r}")

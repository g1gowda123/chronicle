#!/usr/bin/env python3
"""Headless evaluation entry point for Chronicle.

This module wraps the existing perception -> lifecycle -> semantic -> graph
pipeline in a UI-free class (`ChronicleEngine`) and instruments it with
`MetricsTracker`.  It deliberately does **not** reimplement the state machine,
gating heuristics, tracker or graph logic: those remain in ``chronicle.*`` and
are called through their public (or stable facade) APIs.

Usage
-----
    # Full temporal evaluation (Task 3): writes results.json + summary_report.md
    python eval_pipeline.py \
        --video data/real/cook/aria_cook_45s.mp4 \
        --qa data/qa/ego4d_example.json \
        --config config.yaml \
        --out runs/eval/cook \
        --evaluate

    # Fully offline smoke run: colour detector/hands/embedder + stub Florence
    python eval_pipeline.py --offline --max-frames 30

The engine loads Ego4D-shaped QA pairs.  ``run_evaluation`` pauses at each
requested timestamp, calls the existing GraphRAG query path on the current
graph, then resumes ingestion.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("chronicle.eval")

REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# QA loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class QAPair:
    """One Ego4D-shaped question.

    ``timestamp`` is the point in the video at which the question should be
    answered.  ``None`` means "answer from the final graph".
    """

    video_id: str
    query: str
    answer: str
    timestamp: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "query": self.query,
            "answer": self.answer,
            "timestamp": self.timestamp,
        }


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return default


def parse_timestamp(value: Any) -> float | None:
    """Accept seconds, ``MM:SS``, ``HH:MM:SS`` or ``{start, end}``."""

    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        for key in ("start", "timestamp", "t", "time", "start_time"):
            if key in value:
                return parse_timestamp(value[key])
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if ":" in text:
        pieces = [piece.strip() for piece in text.split(":")]
        if len(pieces) in {2, 3} and all(piece.replace(".", "", 1).isdigit() for piece in pieces):
            numbers = [float(piece) for piece in pieces]
            if len(numbers) == 2:
                return numbers[0] * 60.0 + numbers[1]
            return numbers[0] * 3600.0 + numbers[1] * 60.0 + numbers[2]
    return None


def _answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "answer", "caption", "value"):
            if key in value:
                return _answer_text(value[key])
    if isinstance(value, Sequence):
        return " ".join(_answer_text(item) for item in value if item is not None)
    return str(value)


def load_qa_document(
    path: str | Path, *, default_video_id: str | None = None
) -> tuple[list[QAPair], dict[str, Any]]:
    """Load QA pairs plus top-level metadata (video_id, zone_label, ...)."""

    qa_path = Path(path).expanduser()
    if not qa_path.is_file():
        raise FileNotFoundError(f"QA file not found: {qa_path}")
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        rows = payload.get("questions") or payload.get("data") or payload.get("qa")
        file_video_id = str(payload.get("video_id") or default_video_id or "")
        metadata = {key: value for key, value in payload.items() if key not in {"questions", "data", "qa"}}
    else:
        rows = payload
        file_video_id = str(default_video_id or "")
    if not isinstance(rows, list):
        raise ValueError(f"QA file {qa_path} does not contain a question list")

    pairs: list[QAPair] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"question {index} in {qa_path} is not an object")
        query = str(_first(row, "query", "question", "nlq_query", default="")).strip()
        if not query:
            raise ValueError(f"question {index} in {qa_path} has no query")
        answer = _answer_text(_first(row, "answer", "annotation", "answers", default=""))
        timestamp = parse_timestamp(
            _first(row, "timestamp", "t", "time", "start", "start_time")
        )
        video_id = str(_first(row, "video_id", "video", default=file_video_id))
        pairs.append(
            QAPair(
                video_id=video_id,
                query=query,
                answer=answer,
                timestamp=timestamp,
                raw=dict(row),
            )
        )
    pairs.sort(key=lambda pair: (pair.timestamp is None, pair.timestamp or 0.0))
    return pairs, metadata


def load_qa_pairs(path: str | Path, *, default_video_id: str | None = None) -> list[QAPair]:
    """Load a QA file in Ego4D / QAEgo4D-Closed shape."""

    return load_qa_document(path, default_video_id=default_video_id)[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class MetricsTracker:
    """Research metrics for one ingestion run.

    What is measured
    ----------------
    * ``frames_processed``: frames that reached the detector, after stride.
    * ``vlm_calls``: calls to the semantic captioner (Florence-2).  This counts
      an inference *trigger*, including the offline stub fallback.
    * ``llm_calls``: calls to the local text LLM used for GraphRAG phrasing.
      Template-only runs report zero and remain deterministic.
    * ``vlm_call_reduction_pct``: ``1 - vlm_calls / frames_processed``.
    * ``avg_latency_ms`` / percentiles: per-frame time inside
      ``PerceptionPipeline.process_frame`` (gating + detector + hands + tracker
      + attributes + clip assembly), excluding video decode and GraphRAG.
    * ``peak_vram_mb``: ``torch.cuda.max_memory_allocated()`` when CUDA is
      available; ``None`` on CPU / when torch is absent.
    * graph size: foreground MultiDiGraph and background DiGraph nodes/edges at
      the end of the run.
    """

    _MIB = 1024.0 * 1024.0

    def __init__(
        self,
        *,
        percentiles: Sequence[float] = (50, 95, 99),
        track_vram: bool = True,
        warmup_frames: int = 0,
    ):
        self.percentiles = tuple(float(value) for value in percentiles)
        self.track_vram = bool(track_vram)
        self.warmup_frames = max(0, int(warmup_frames))
        self.started_at: float | None = None
        self.finished_at: float | None = None

        self.frames_processed = 0
        self.frame_latencies_ms: list[float] = []
        self.observe_latencies_ms: list[float] = []
        self.vlm_calls = 0
        self.vlm_model_calls = 0
        self.vlm_stub_calls = 0
        self.vlm_skips = 0
        self.vlm_latencies_ms: list[float] = []
        self.vlm_backends: dict[str, int] = {}
        self.vlm_triggers: dict[str, int] = {}
        self.llm_calls = 0
        self.llm_errors = 0
        self.llm_latencies_ms: list[float] = []
        self.llm_models: dict[str, int] = {}
        self.llm_providers: dict[str, int] = {}
        self.llm_prompt_tokens = 0
        self.llm_completion_tokens = 0
        self.llm_total_tokens = 0
        self.detections = 0
        self.tracks = 0
        self.last_timestamp = 0.0
        self.peak_gpu_used_mb = 0.0
        self.error: str | None = None
        self._torch: Any = None
        self._cuda_available = False

    # ------------------------------------------------------------------ setup
    def start(self) -> None:
        self.started_at = time.perf_counter()
        if not self.track_vram:
            return
        try:
            import torch  # type: ignore

            self._torch = torch
            self._cuda_available = bool(torch.cuda.is_available())
            if self._cuda_available:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # pragma: no cover - torch is optional
            self._torch = None
            self._cuda_available = False

    # --------------------------------------------------------------- counters
    def record_frame(
        self,
        latency_ms: float,
        *,
        timestamp: float | None = None,
        detections: int = 0,
        tracks: int = 0,
    ) -> None:
        self.frames_processed += 1
        self.frame_latencies_ms.append(max(0.0, float(latency_ms)))
        self.detections += int(detections)
        self.tracks += int(tracks)
        if timestamp is not None:
            self.last_timestamp = float(timestamp)
        if self.frames_processed % 5 == 0:
            self._sample_gpu_memory()

    def _sample_gpu_memory(self) -> None:
        """Track device-wide used memory, including models served by Ollama.

        ``torch.cuda.max_memory_allocated`` only sees this process.  Sampling
        ``mem_get_info`` also captures the Qwen captioner when it runs as a
        separate Ollama process.
        """

        if not self._cuda_available or self._torch is None:
            return
        try:
            free, total = self._torch.cuda.mem_get_info()
            used = float(total - free) / self._MIB
            if used > self.peak_gpu_used_mb:
                self.peak_gpu_used_mb = used
        except Exception:  # pragma: no cover - defensive for exotic backends
            return

    def record_observe(self, latency_ms: float) -> None:
        self.observe_latencies_ms.append(max(0.0, float(latency_ms)))

    def record_vlm_call(
        self,
        *,
        latency_ms: float | None = None,
        backend: str = "unknown",
        trigger: str = "caption",
    ) -> None:
        self.vlm_calls += 1
        if "stub" in backend.lower() or "null" in backend.lower():
            self.vlm_stub_calls += 1
        else:
            self.vlm_model_calls += 1
        if latency_ms is not None:
            self.vlm_latencies_ms.append(max(0.0, float(latency_ms)))
        self.vlm_backends[backend] = self.vlm_backends.get(backend, 0) + 1
        self.vlm_triggers[trigger] = self.vlm_triggers.get(trigger, 0) + 1

    def record_vlm_skip(self, *, trigger: str = "caption") -> None:
        """Count an event trigger whose caption call was skipped by a budget."""

        self.vlm_skips += 1

    def record_llm_call(
        self,
        *,
        latency_ms: float | None = None,
        model: str = "unknown",
        provider: str = "unknown",
        usage: Mapping[str, Any] | None = None,
        error: bool = False,
    ) -> None:
        """Count one text-LLM request (GraphRAG phrasing or baseline answers)."""

        self.llm_calls += 1
        if error:
            self.llm_errors += 1
        if latency_ms is not None:
            self.llm_latencies_ms.append(max(0.0, float(latency_ms)))
        self.llm_models[str(model or "unknown")] = (
            self.llm_models.get(str(model or "unknown"), 0) + 1
        )
        self.llm_providers[str(provider or "unknown")] = (
            self.llm_providers.get(str(provider or "unknown"), 0) + 1
        )
        if usage:
            self.llm_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.llm_completion_tokens += int(usage.get("completion_tokens") or 0)
            self.llm_total_tokens += int(usage.get("total_tokens") or 0)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _percentile(values: Sequence[float], pct: float) -> float | None:
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        index = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
        return round(ordered[index], 4)

    def _peak_vram_mb(self) -> float | None:
        if not self.track_vram or not self._cuda_available or self._torch is None:
            return None
        try:
            self._torch.cuda.synchronize()
            return round(float(self._torch.cuda.max_memory_allocated()) / self._MIB, 2)
        except Exception:  # pragma: no cover - defensive for exotic backends
            return None

    @staticmethod
    def _graph_stats(memory: Any | None) -> dict[str, int]:
        if memory is None:
            return {}
        foreground = getattr(memory, "foreground", None)
        background = getattr(memory, "background", None)
        fg_graph = getattr(foreground, "graph", None)
        bg_graph = getattr(background, "graph", None)
        fg_nodes = int(fg_graph.number_of_nodes()) if fg_graph is not None else 0
        fg_edges = int(fg_graph.number_of_edges()) if fg_graph is not None else 0
        bg_nodes = int(bg_graph.number_of_nodes()) if bg_graph is not None else 0
        bg_edges = int(bg_graph.number_of_edges()) if bg_graph is not None else 0
        return {
            "foreground_nodes": fg_nodes,
            "foreground_edges": fg_edges,
            "background_nodes": bg_nodes,
            "background_edges": bg_edges,
            "total_nodes": fg_nodes + bg_nodes,
            "total_edges": fg_edges + bg_edges,
        }

    # --------------------------------------------------------------- snapshot
    def to_dict(self, *, memory: Any | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        finished = self.finished_at or time.perf_counter()
        wall = (finished - self.started_at) if self.started_at is not None else 0.0
        frames = self.frames_processed
        reduction = round(100.0 * (1.0 - self.vlm_calls / frames), 3) if frames else None
        effective_reduction = (
            round(100.0 * (1.0 - self.vlm_model_calls / frames), 3) if frames else None
        )
        latency = {
            f"p{int(pct)}_ms": self._percentile(self.frame_latencies_ms, pct)
            for pct in self.percentiles
        }
        warmup = min(self.warmup_frames, len(self.frame_latencies_ms))
        steady = self.frame_latencies_ms[warmup:]
        steady_latency = {
            f"steady_p{int(pct)}_ms": self._percentile(steady, pct)
            for pct in self.percentiles
        }
        payload: dict[str, Any] = {
            "frames_processed": frames,
            "duration_s": round(self.last_timestamp, 3),
            "wall_time_s": round(wall, 3),
            "frames_per_s": round(frames / wall, 3) if wall > 0 else None,
            "vlm_calls": self.vlm_calls,
            "vlm_model_calls": self.vlm_model_calls,
            "vlm_stub_calls": self.vlm_stub_calls,
            "vlm_skips": self.vlm_skips,
            "vlm_calls_per_100_frames": round(100.0 * self.vlm_calls / frames, 3) if frames else None,
            "vlm_call_reduction_pct": reduction,
            "vlm_model_call_reduction_pct": effective_reduction,
            "vlm_backends": dict(sorted(self.vlm_backends.items())),
            "vlm_triggers": dict(sorted(self.vlm_triggers.items())),
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
            "llm_models": dict(sorted(self.llm_models.items())),
            "llm_providers": dict(sorted(self.llm_providers.items())),
            "llm_tokens": {
                "prompt": self.llm_prompt_tokens,
                "completion": self.llm_completion_tokens,
                "total": self.llm_total_tokens,
            },
            "avg_llm_latency_ms": round(statistics.fmean(self.llm_latencies_ms), 4)
            if self.llm_latencies_ms
            else None,
            "avg_latency_ms": round(statistics.fmean(self.frame_latencies_ms), 4)
            if self.frame_latencies_ms
            else None,
            **latency,
            "warmup_frames": warmup,
            "avg_latency_steady_ms": round(statistics.fmean(steady), 4) if steady else None,
            **steady_latency,
            "avg_observe_ms": round(statistics.fmean(self.observe_latencies_ms), 4)
            if self.observe_latencies_ms
            else None,
            "avg_vlm_latency_ms": round(statistics.fmean(self.vlm_latencies_ms), 4)
            if self.vlm_latencies_ms
            else None,
            "peak_vram_mb": self._peak_vram_mb(),
            "peak_gpu_used_mb": round(self.peak_gpu_used_mb, 2)
            if self.peak_gpu_used_mb
            else None,
            "detections": self.detections,
            "tracks_touched": self.tracks,
            "graph": self._graph_stats(memory),
            "error": self.error,
        }
        if extra:
            payload.update(dict(extra))
        return payload


def instrument_captioner(
    captioner: Any,
    metrics: MetricsTracker,
    *,
    trigger: str = "semantic",
    stride: int = 1,
) -> bool:
    """Wrap ``captioner.caption`` so every Florence-2 trigger is counted.

    ``stride`` implements a VLM-budget ablation: only every ``stride``-th event
    trigger invokes the captioner (``0`` disables captions entirely).  Skipped
    triggers return an empty caption and are counted separately, so the
    accuracy/call curve uses measured model calls rather than inferred ones.

    Returns ``True`` when the captioner was instrumented.  The wrapper is
    idempotent, so `ChronicleEngine` can call it in tests more than once.
    """

    if captioner is None or getattr(captioner, "_chronicle_metrics_wrapped", False):
        return False
    original = getattr(captioner, "caption", None)
    if not callable(original):
        return False
    stride = int(stride)
    seen = [0]

    def _backend_name() -> str:
        impl = getattr(captioner, "_impl", None) or captioner
        return type(impl).__name__

    def counted(image: Any, task: Any = None) -> Any:
        index = seen[0]
        seen[0] += 1
        if stride <= 0 or (stride > 1 and index % stride != 0):
            metrics.record_vlm_skip(trigger=trigger)
            return ""
        started = time.perf_counter()
        try:
            if task is None:
                return original(image)
            return original(image, task)
        finally:
            metrics.record_vlm_call(
                latency_ms=(time.perf_counter() - started) * 1000.0,
                backend=_backend_name(),
                trigger=trigger,
            )

    captioner.caption = counted
    captioner._chronicle_metrics_wrapped = True
    return True


def instrument_chat_client(client: Any, metrics: MetricsTracker) -> bool:
    """Wrap ``chat_client.complete`` so local LLM calls are measured.

    The wrapper is intentionally provider agnostic: Ollama, transformers and
    Gemini clients all expose ``provider``/``model`` and a ``last_usage`` dict.
    """

    if client is None or getattr(client, "_chronicle_metrics_wrapped", False):
        return False
    original = getattr(client, "complete", None)
    if not callable(original):
        return False

    def model_name() -> str:
        return str(
            getattr(client, "active_model", None)
            or getattr(client, "model", None)
            or getattr(client, "model_id", None)
            or "unknown"
        )

    def provider_name() -> str:
        return str(getattr(client, "provider", "unknown"))

    def counted(system_prompt: str, user_prompt: str, *, json_mode: bool = False) -> Any:
        started = time.perf_counter()
        error = False
        try:
            result = original(system_prompt, user_prompt, json_mode=json_mode)
            return result
        except Exception:
            error = True
            raise
        finally:
            usage = getattr(client, "last_usage", None)
            metrics.record_llm_call(
                latency_ms=(time.perf_counter() - started) * 1000.0,
                model=model_name(),
                provider=provider_name(),
                usage=usage if isinstance(usage, Mapping) else None,
                error=error,
            )

    client.complete = counted
    client._chronicle_metrics_wrapped = True
    return True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "config.yaml support requires PyYAML; install it with "
            "`pip install pyyaml` or `uv sync --extra research`."
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"config file {path} must contain a mapping")
    return payload


def _resolve(path: str | Path | None, root: Path) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    candidate = Path(str(path)).expanduser()
    return candidate if candidate.is_absolute() else (root / candidate)


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(slots=True)
class EnginePaths:
    root: Path
    data_dir: Path
    model_dir: Path
    out_dir: Path
    db_path: Path
    video: Path
    qa: Path


class ChronicleEngine:
    """UI-free ingestion engine around Chronicle's existing core.

    The class owns the frame loop but delegates every decision to the existing
    modules:

    * ``PerceptionPipeline`` performs motion gating, detection, hands,
      tracking, attributes and clip assembly.
    * ``ChronicleMemory`` owns the nine-state lifecycle and both graphs.
    * ``ChronicleRuntime`` owns semantic verification and GraphRAG.
    * ``SemanticEngine`` triggers Florence-2; the captioner is instrumented
      only at the boundary so inference counts are exact.

    Parameters
    ----------
    video_path, qa_json_path:
        Explicit inputs.  When omitted, the paths in ``config.yaml`` are used.
    config_path:
        YAML file.  Relative values resolve from ``root``.
    overrides:
        Optional mapping merged over the YAML, e.g.
        ``{"perception": {"detector_backend": "color"}}``.
    """

    def __init__(
        self,
        video_path: str | Path | None = None,
        qa_json_path: str | Path | None = None,
        *,
        config_path: str | Path = _DEFAULT_CONFIG,
        root: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).resolve() if root is not None else REPO_ROOT
        self.raw_config = _load_yaml(_resolve(config_path, self.root) or Path(config_path))
        if overrides:
            self.raw_config = _deep_merge(self.raw_config, overrides)
        self._apply_profile()

        paths = _section(self.raw_config, "paths")
        video_cfg = _section(self.raw_config, "video")
        qa_cfg = _section(self.raw_config, "qa")
        self.data_dir = _resolve(paths.get("data_dir", "data"), self.root) or self.root / "data"
        self.model_dir = _resolve(paths.get("model_dir", "models"), self.root) or self.root / "models"
        self.out_dir = _resolve(paths.get("out_dir", "runs/eval"), self.root) or self.root / "runs/eval"
        self.db_path = _resolve(paths.get("db_path", "runs/eval/chronicle.db"), self.root)
        if self.db_path is None:
            self.db_path = self.out_dir / "chronicle.db"

        self.video_path = _resolve(video_path, self.root) if video_path else _resolve(video_cfg.get("path"), self.root)
        self.qa_path = _resolve(qa_json_path, self.root) if qa_json_path else _resolve(qa_cfg.get("path"), self.root)
        if self.video_path is None:
            raise ValueError("no video path supplied (argument or config.video.path)")
        if self.qa_path is None:
            raise ValueError("no QA path supplied (argument or config.qa.path)")

        self.qa_pairs, self.qa_metadata = load_qa_document(
            self.qa_path, default_video_id=self.video_path.stem
        )
        # QA exports can carry episode facts such as {"zone_label": "Kitchen"}.
        # Apply them without hardcoding per-video logic.
        runtime_section = _section(self.raw_config, "runtime")
        if not runtime_section.get("zone_label") and self.qa_metadata.get("zone_label"):
            runtime_section["zone_label"] = self.qa_metadata["zone_label"]
            self.raw_config["runtime"] = runtime_section
        if not runtime_section.get("zone_id") and self.qa_metadata.get("zone_id"):
            runtime_section["zone_id"] = self.qa_metadata["zone_id"]
            self.raw_config["runtime"] = runtime_section
        metrics_cfg = self.raw_config.get("metrics", {})
        if not isinstance(metrics_cfg, Mapping):
            metrics_cfg = {}
        self.metrics = MetricsTracker(
            percentiles=metrics_cfg.get("percentiles", [50, 95, 99]),
            track_vram=bool(metrics_cfg.get("track_vram", True)),
            warmup_frames=int(metrics_cfg.get("warmup_frames", 1)),
        )
        self.runtime: Any | None = None
        self.pipeline: Any | None = None
        self.semantic: Any | None = None
        self.perception_config: Any | None = None
        self.semantic_config: Any | None = None
        self.summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------ config
    def _apply_profile(self) -> None:
        runtime = _section(self.raw_config, "runtime")
        profile = str(runtime.get("profile", "baseline")).lower()
        if profile not in {"offline", "baseline"}:
            raise ValueError("runtime.profile must be 'baseline' or 'offline'")
        if profile != "offline":
            return
        # The offline profile changes only backend selection; the state machine,
        # gating thresholds and graph code are untouched.
        perception = _section(self.raw_config, "perception")
        perception.update(
            {
                "detector_backend": "color",
                "hands_backend": "color",
                "embedder_backend": "color",
            }
        )
        self.raw_config["perception"] = perception
        semantic = _section(self.raw_config, "semantic")
        semantic.update(
            {
                "require_florence": False,
                "captioner_backend": "stub",
                "embedder_backend": "deterministic",
                "llm_backend": "template",
            }
        )
        self.raw_config["semantic"] = semantic

    def _perception_config(self) -> Any:
        from chronicle.perception.config import PerceptionConfig

        section = _section(self.raw_config, "perception")
        cfg = PerceptionConfig()
        cfg.model_dir = self.model_dir
        cfg.device = str(section.get("device", cfg.device))
        cfg.frame_stride = max(1, int(section.get("frame_stride", cfg.frame_stride)))
        max_frames = section.get("max_frames")
        cfg.max_frames = None if max_frames in (None, "", 0) else int(max_frames)
        cfg.detector.backend = str(section.get("detector_backend", cfg.detector.backend))
        cfg.detector.model = str(section.get("detector_model", cfg.detector.model))
        raw_models = section.get("detector_models") or []
        if isinstance(raw_models, str):
            raw_models = [item.strip() for item in raw_models.split(",") if item.strip()]
        cfg.detector.models = tuple(
            dict.fromkeys(str(model).strip() for model in raw_models if str(model).strip())
        )
        cfg.detector.confidence = float(section.get("detector_confidence", cfg.detector.confidence))
        cfg.detector.imgsz = int(section.get("detector_imgsz", cfg.detector.imgsz))
        cfg.detector.tile_grid = max(1, int(section.get("tile_grid", getattr(cfg.detector, "tile_grid", 1))))
        cfg.detector.tile_overlap = float(section.get("tile_overlap", getattr(cfg.detector, "tile_overlap", 0.15)))
        tile_conf = section.get("tile_confidence")
        cfg.detector.tile_confidence = None if tile_conf in (None, "") else float(tile_conf)
        cfg.detector.tile_labels = tuple(
            str(v).strip().lower() for v in (section.get("tile_labels") or []) if str(v).strip()
        )
        raw_prompt_aliases = section.get("prompt_aliases") or {}
        if isinstance(raw_prompt_aliases, dict):
            cfg.detector.prompt_aliases = {
                str(label).strip().lower(): (
                    tuple(str(v).strip() for v in values if str(v).strip())
                    if isinstance(values, (list, tuple))
                    else (str(values).strip(),)
                )
                for label, values in raw_prompt_aliases.items()
                if str(label).strip()
            }
        cfg.hands.backend = str(section.get("hands_backend", cfg.hands.backend))
        cfg.hands.model = str(section.get("hands_model", cfg.hands.model))
        cfg.embedder.backend = str(section.get("embedder_backend", cfg.embedder.backend))
        cfg.embedder.model = str(section.get("embedder_model", cfg.embedder.model))
        cfg.embedder.pretrained = str(section.get("embedder_pretrained", cfg.embedder.pretrained))
        cfg.motion.enabled = bool(section.get("motion_enabled", cfg.motion.enabled))
        cfg.clip.enabled = bool(section.get("clip_enabled", cfg.clip.enabled))
        preset = str(section.get("vocabulary_preset", "default")).lower()
        if preset == "broad":
            from chronicle.perception.config import BROAD_VOCABULARY

            cfg.vocabulary = BROAD_VOCABULARY
        elif preset in {"none", "empty"}:
            cfg.vocabulary = ()
        vocab = section.get("vocabulary") or []
        if isinstance(vocab, str):
            vocab = [item.strip() for item in vocab.split(",") if item.strip()]
        cfg.vocabulary = tuple(dict.fromkeys([*cfg.vocabulary, *[str(item) for item in vocab]]))
        # Dataset metadata may declare a fixed scene vocabulary (the objects a
        # deployment cares about).  This is *not* derived from the questions.
        scene_vocab = self.qa_metadata.get("vocabulary") or []
        if isinstance(scene_vocab, str):
            scene_vocab = [item.strip() for item in scene_vocab.split(",") if item.strip()]
        if isinstance(scene_vocab, Sequence):
            cfg.vocabulary = tuple(
                dict.fromkeys([*cfg.vocabulary, *[str(item) for item in scene_vocab]])
            )
        # Prompt leakage ablation: optionally feed question nouns to the
        # open-vocabulary detector.  Off by default because a published
        # evaluation must not let test questions shape perception.
        if bool(section.get("derive_vocabulary_from_qa", False)):
            cfg.vocabulary = tuple(
                dict.fromkeys([*cfg.vocabulary, *_qa_vocabulary(self.qa_pairs)])
            )
        return cfg

    def _semantic_config(self) -> Any:
        from chronicle.semantic.config import SemanticConfig

        section = _section(self.raw_config, "semantic")
        florence_id = str(section.get("florence_model_id", "microsoft/Florence-2-base"))
        # A local directory may be written as ``models/Florence-2-base``.  A
        # bare HF id such as ``microsoft/Florence-2-base`` is left untouched so
        # the cached/downloaded model can still be resolved by transformers.
        florence_path = _resolve(florence_id, self.root)
        if (
            florence_path is not None
            and (
                Path(florence_id).is_absolute()
                or florence_id.startswith(("models/", "models\\", "./", "../"))
                or florence_path.exists()
            )
        ):
            florence_id = str(florence_path)
        cfg = SemanticConfig(
            clip_model_name=str(section.get("clip_model_name", "ViT-B-32")),
            clip_pretrained=str(section.get("clip_pretrained", "laion2b_s34b_b79k")),
            device=str(section.get("device", "auto")),
            florence_model_id=florence_id,
            florence_max_new_tokens=int(section.get("florence_max_new_tokens", 128)),
            florence_num_beams=int(section.get("florence_num_beams", 3)),
            qwen_vl_model=str(section.get("qwen_vl_model", "qwen2.5vl:3b")),
            qwen_vl_host=str(
                section.get("qwen_vl_host", "http://127.0.0.1:11434")
            ),
            qwen_vl_timeout_s=float(section.get("qwen_vl_timeout_s", 120.0)),
            qwen_vl_max_side=int(section.get("qwen_vl_max_side", 448)),
            action_task=str(section.get("action_task", "<MORE_DETAILED_CAPTION>")),
            action_min_confidence=float(section.get("action_min_confidence", 0.45)),
            llm_backend=str(section.get("llm_backend", "template")),
            llm_model=str(section.get("llm_model", "qwen2.5:7b-instruct")),
            action_llm_enabled=bool(section.get("action_llm_enabled", True)),
            ollama_url=str(section.get("ollama_url", "http://127.0.0.1:11434")),
            require_citations=bool(section.get("require_citations", True)),
            lenient_citations=bool(section.get("lenient_citations", True)),
            max_facts=int(section.get("max_facts", 40)),
        )
        return cfg

    @staticmethod
    def _preflight_captioner(captioner: Any, backend: str) -> None:
        """Fail before the rollout if the selected captioner is not usable."""

        selected = backend.lower()
        if selected in {"qwen", "qwen-vl", "qwen2.5vl", "qwen_vl"}:
            from chronicle.semantic.qwen_vl import QwenVLCaptioner

            if not isinstance(captioner, QwenVLCaptioner):
                raise RuntimeError(
                    "Qwen2.5-VL was selected but the captioner is not a "
                    "QwenVLCaptioner."
                )
            if not captioner.is_available():
                raise RuntimeError(
                    f"Qwen2.5-VL is not available at {captioner.host}. "
                    f"Start `ollama serve` and run `ollama pull {captioner.model_id}`."
                )
            LOGGER.info(
                "Qwen2.5-VL preflight ok: %s at %s",
                captioner.model_id,
                captioner.host,
            )
            return
        ChronicleEngine._preflight_florence(captioner)

    @staticmethod
    def _preflight_florence(captioner: Any) -> None:
        """Load local Florence-2 once, and fail clearly instead of using a stub."""

        from chronicle.semantic.florence import (
            FlorenceCaptioner,
            LazyCaptioner,
            StubCaptioner,
        )

        if isinstance(captioner, StubCaptioner):
            raise RuntimeError(
                "Florence-2 is required but the captioner is a StubCaptioner. "
                "Run `python scripts/fetch_models.py --check`."
            )
        target = captioner
        if isinstance(captioner, LazyCaptioner):
            target = captioner._resolve()  # noqa: SLF001 - explicit preflight
        if isinstance(target, StubCaptioner) or not isinstance(target, FlorenceCaptioner):
            raise RuntimeError(
                "Florence-2 is required but the local model could not be resolved. "
                "Install torch+transformers and run "
                "`python scripts/fetch_models.py --check`."
            )
        try:
            target._load()  # noqa: SLF001 - load before the run, not mid-rollout
        except Exception as exc:
            raise RuntimeError(
                "Florence-2 failed to load. Verify "
                "config.semantic.florence_model_id points at the local "
                f"models/Florence-2-base directory ({exc})."
            ) from exc
        LOGGER.info("Florence-2 preflight ok: %s", target.model_id)

    # ------------------------------------------------------------------ build
    def build(self) -> ChronicleEngine:
        """Build the runtime and pipeline.  Safe to call more than once."""

        if self.runtime is not None:
            return self
        if not self.video_path.is_file():
            raise FileNotFoundError(f"video not found: {self.video_path}")
        if not self.qa_path.is_file():
            raise FileNotFoundError(f"QA file not found: {self.qa_path}")

        from chronicle.integration import ChronicleRuntime
        from chronicle.perception.pipeline import PerceptionPipeline
        from chronicle.semantic.engine import SemanticEngine

        self.perception_config = self._perception_config()
        self.semantic_config = self._semantic_config()
        runtime_section = _section(self.raw_config, "runtime")
        semantic_section = _section(self.raw_config, "semantic")
        semantic_enabled = bool(semantic_section.get("enabled", True))

        semantic = None
        if semantic_enabled:
            embedder_backend = str(semantic_section.get("embedder_backend", "auto"))
            if self.perception_config.embedder.backend == "color" and embedder_backend == "auto":
                embedder_backend = "deterministic"
            semantic = SemanticEngine(
                config=self.semantic_config,
                captioner_backend=str(semantic_section.get("captioner_backend", "auto")),
                embedder_backend=embedder_backend,
                chat_client="auto",
            )
            if bool(semantic_section.get("require_florence", False)):
                self._preflight_captioner(
                    semantic.captioner,
                    str(semantic_section.get("captioner_backend", "auto")),
                )
            instrument_captioner(
                semantic.captioner,
                self.metrics,
                stride=int(
                    0
                    if semantic_section.get("caption_stride", 1) == 0
                    else semantic_section.get("caption_stride", 1) or 1
                ),
            )
            instrument_chat_client(semantic.chat_client, self.metrics)
        self.semantic = semantic

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = ChronicleRuntime(
            self.db_path,
            semantic=semantic,
            semantic_config=self.semantic_config,
            semantic_enabled=semantic_enabled,
            perception_config=self.perception_config,
            zone_label=runtime_section.get("zone_label"),
            zone_id=runtime_section.get("zone_id"),
            reset=bool(runtime_section.get("reset_db", True)),
        )
        self.pipeline = PerceptionPipeline(
            self.perception_config,
            on_clip=self.runtime.submit_clip,
            telemetry_sink=self.runtime.publish_perception_event,
        )
        self.runtime.attach_pipeline(self.pipeline)
        return self

    # ------------------------------------------------------------------ run
    def run_ingestion(
        self,
        *,
        stop_event: Any | None = None,
        on_frame: Any | None = None,
        on_progress: Any | None = None,
    ) -> dict[str, Any]:
        """Run the sequential frame loop and return the metrics payload.

        The loop is intentionally small: all perception and lifecycle decisions
        happen inside the existing components.  We only time ``process_frame``,
        call ``observe_result`` + ``drain_verifications``, and count VLM
        triggers at the captioner boundary.
        """

        self.build()
        assert self.runtime is not None and self.pipeline is not None
        runtime = self.runtime
        pipeline = self.pipeline
        self.metrics.start()
        actual_backends: dict[str, str] = {}

        if self.perception_config is not None and self.perception_config.device:
            LOGGER.info("perception device=%s", self.perception_config.device)

        zone_label = _section(self.raw_config, "runtime").get("zone_label")
        zone_id = _section(self.raw_config, "runtime").get("zone_id")
        if zone_label or zone_id:
            runtime.memory.enter_zone(zone_label or zone_id or "unmapped area", 0.0, zone_id=zone_id)

        original_process = pipeline.process_frame

        def timed_process(frame: Any, timestamp: float, frame_index: int = 0) -> Any:
            started = time.perf_counter()
            result = original_process(frame, timestamp, frame_index)
            self.metrics.record_frame(
                (time.perf_counter() - started) * 1000.0,
                timestamp=timestamp,
                detections=len(getattr(result, "detections", ()) or ()),
                tracks=len(getattr(result, "tracks", ()) or ()),
            )
            return result

        pipeline.process_frame = timed_process
        fps_override = _section(self.raw_config, "video").get("fps_override")
        try:
            for result in pipeline.iter_results(
                self.video_path,
                stop_event=stop_event,
                fps_override=float(fps_override) if fps_override not in (None, "") else None,
            ):
                started = time.perf_counter()
                runtime.observe_frame(result)
                runtime.drain_verifications()
                self.metrics.record_observe((time.perf_counter() - started) * 1000.0)
                if on_progress is not None and self.metrics.frames_processed % 10 == 0:
                    try:
                        on_progress(self.metrics)
                    except Exception:
                        LOGGER.debug("progress callback failed", exc_info=True)
                if on_frame is not None:
                    try:
                        on_frame(result)
                    except Exception:
                        LOGGER.exception("on_frame callback failed")
        except KeyboardInterrupt:
            LOGGER.warning("ingestion interrupted; finalising partial metrics")
        except Exception as exc:
            self.metrics.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("ingestion failed")
        finally:
            # Capture the actual loaded backends before detaching the pipeline.
            try:
                actual_backends = runtime.backends() if hasattr(runtime, "backends") else {}
            except Exception:
                actual_backends = {}
            # Flush any clip assembled by the pipeline's finalizer.
            try:
                runtime.drain_verifications()
            except Exception:
                LOGGER.exception("final verification drain failed")
            try:
                pipeline.close()
            except Exception:
                LOGGER.exception("pipeline close failed")
            runtime.detach_pipeline()
            self.metrics.finished_at = time.perf_counter()

        self.summary = self.metrics.to_dict(
            memory=runtime.memory,
            extra={
                "video": str(self.video_path),
                "qa_file": str(self.qa_path),
                "qa_pairs": len(self.qa_pairs),
                "compile_config": {
                    "perception": self.perception_config.as_dict() if self.perception_config else None,
                    "semantic": {
                        "captioner_backend": _section(self.raw_config, "semantic").get("captioner_backend"),
                        "embedder_backend": _section(self.raw_config, "semantic").get("embedder_backend"),
                        "llm_backend": _section(self.raw_config, "semantic").get("llm_backend"),
                    },
                },
                "backends": actual_backends,
            },
        )
        return self.summary

    def close(self) -> None:
        if self.runtime is not None:
            try:
                self.runtime.close()
            finally:
                self.runtime = None
        self.pipeline = None

    def __enter__(self) -> ChronicleEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


_QA_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "did", "do", "does", "what",
    "where", "when", "which", "who", "how", "many", "much", "my", "i", "me",
    "in", "on", "at", "to", "of", "for", "with", "and", "or", "this", "that",
    "it", "there", "memory", "last", "first", "before", "after", "go", "went",
}


def _qa_vocabulary(pairs: Iterable[QAPair]) -> tuple[str, ...]:
    """Extract candidate object nouns and noun phrases from QA text.

    YOLO-World is prompt driven, so a question such as "where is the cutting
    board?" must contribute the phrase ``cutting board`` rather than the two
    unrelated prompts ``cutting`` and ``board``.
    """

    words: list[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if term and term not in words and term not in _QA_STOPWORDS:
            words.append(term)

    for pair in pairs:
        tokens = [
            token
            for token in re.findall(r"[a-z][a-z0-9'-]{2,}", pair.query.lower())
            if token not in _QA_STOPWORDS
        ]
        for token in tokens:
            add(token)
        for left, right in zip(tokens, tokens[1:], strict=False):
            add(f"{left} {right}")
    return tuple(words[:60])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
class ProgressBar:
    """Small tqdm wrapper with a manual fallback (no hard dependency)."""

    def __init__(self, total: int | None = None, desc: str = "", unit: str = "frame"):
        self.desc = desc
        self.total = max(0, int(total)) if total else None
        self.unit = unit
        self.count = 0
        self._bar = None
        self._last_len = 0
        try:
            from tqdm import tqdm  # type: ignore

            self._bar = tqdm(total=self.total, desc=desc, unit=unit, dynamic_ncols=True)
        except Exception:
            self._bar = None

    def update(self, n: int = 1, postfix: str | None = None) -> None:
        self.count += max(0, int(n))
        if self._bar is not None:
            self._bar.update(n)
            if postfix:
                self._bar.set_postfix_str(postfix)
            return
        if self.total:
            pct = min(100.0, 100.0 * self.count / self.total)
            text = (
                f"\r{self.desc}: {self.count}/{self.total} "
                f"({pct:5.1f}%)" + (f" {postfix}" if postfix else "")
            )
        else:
            text = f"\r{self.desc}: {self.count}" + (f" {postfix}" if postfix else "")
        print(text.ljust(self._last_len), end="", flush=True)
        self._last_len = len(text)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        elif self.count:
            print()


def video_metadata(path: str | Path) -> dict[str, Any]:
    """Return fps/frame_count/duration without decoding every frame."""

    meta: dict[str, Any] = {"fps": None, "frames": None, "duration_s": None}
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()
    except Exception:
        return meta
    if fps > 0:
        meta["fps"] = round(fps, 3)
    if frames > 0:
        meta["frames"] = frames
    if fps > 0 and frames > 0:
        meta["duration_s"] = round(frames / fps, 3)
    return meta


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------
_ANSWER_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "of", "for", "with", "and", "or", "as", "by",
    "it", "its", "this", "that", "there", "here", "i", "me", "my", "you",
    "your", "he", "she", "they", "we", "us", "them", "from", "into", "onto",
    "then", "than", "about", "around", "approximately",
}
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_UNKNOWN_MARKERS = (
    "don't have", "do not have", "no record", "not in", "not recorded",
    "not visible", "unknown", "no memory", "cannot find", "can't find",
    "no such", "not mentioned", "never",
)


def _normalise(text: Any) -> str:
    value = str(text or "").lower()
    value = value.replace("’", "'").replace("“", '"').replace("”", '"')
    value = re.sub(r"[^a-z0-9.'%+-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _singular(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9][a-z0-9.'-]*", _normalise(text)):
        token = raw.strip(".'-")
        if token and token not in _ANSWER_STOPWORDS:
            tokens.add(_singular(token))
    return tokens


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for raw in _NUMBER_RE.findall(_normalise(text)):
        try:
            values.append(float(raw))
        except ValueError:
            pass
    return values


def _number_ok(expected: str, predicted: str, *, tolerance_s: float = 0.11) -> bool:
    targets = _numbers(expected)
    if not targets:
        return True
    found = _numbers(predicted)
    tolerance = max(0.0, float(tolerance_s))
    return all(any(abs(target - value) <= tolerance for value in found) for target in targets)


def _expected_answers(pair: QAPair) -> list[str]:
    raw = pair.raw or {}
    for key in ("answers", "acceptable_answers", "answer_aliases"):
        value = raw.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            answers = [_answer_text(item) for item in value]
            answers = [item for item in answers if item]
            if answers:
                return answers
    return [pair.answer] if pair.answer else []


def score_answer(predicted: str, expected: str, *, raw: Mapping[str, Any] | None = None) -> tuple[bool, float, str]:
    """Deterministic answer matcher for Ego4D-style free-text answers.

    It is deliberately conservative and fully offline:

    * explicit "unknown" answers require an unknown-style response;
    * exact/substring answers succeed;
    * otherwise content-token recall must be >= 0.6 and expected numbers must
      appear in the prediction;
    * ``must_contain`` in a QA row is treated as a hard requirement.
    """

    pred_norm = _normalise(predicted)
    exp_norm = _normalise(expected)
    raw = raw or {}
    if not exp_norm:
        return False, 0.0, "empty expected answer"
    if not pred_norm:
        return False, 0.0, "empty prediction"

    must = raw.get("must_contain")
    if isinstance(must, Sequence) and not isinstance(must, (str, bytes)):
        missing = [str(term) for term in must if _normalise(term) not in pred_norm]
        if missing:
            return False, 0.0, f"missing required term(s): {', '.join(missing)}"

    expects_unknown = any(marker in exp_norm for marker in _UNKNOWN_MARKERS)
    if str(raw.get("kind") or "").lower() == "unknown":
        expects_unknown = True
    if expects_unknown:
        ok = any(marker in pred_norm for marker in _UNKNOWN_MARKERS)
        return (ok, 1.0 if ok else 0.0, "unknown handled" if ok else "expected an explicit unknown")

    if any(marker in pred_norm for marker in _UNKNOWN_MARKERS):
        return False, 0.0, "prediction says unknown"

    # A bare count ("2") must not match a digit inside a timestamp ("21.8s").
    if re.fullmatch(r"-?\d+(?:\.\d+)?", exp_norm):
        tolerance = max(0.0, float(raw.get("tolerance_s", 0.11) or 0.11))
        ok = _number_ok(exp_norm, pred_norm, tolerance_s=tolerance)
        return (ok, 1.0 if ok else 0.0, "numeric match" if ok else "expected number missing")

    if exp_norm in pred_norm or pred_norm in exp_norm:
        return True, 1.0, "exact/substring match"

    exp_tokens = _content_tokens(exp_norm)
    pred_tokens = _content_tokens(pred_norm)
    if not exp_tokens:
        return False, 0.0, "expected answer has no content tokens"
    overlap = len(exp_tokens & pred_tokens) / len(exp_tokens)
    tolerance = float(raw.get("tolerance_s", 0.11) or 0.11)
    numbers_ok = _number_ok(exp_norm, pred_norm, tolerance_s=tolerance)
    if not numbers_ok:
        return False, round(overlap, 3), "expected number missing"
    if overlap >= 0.6:
        return True, round(overlap, 3), f"token recall {overlap:.2f}"
    # Short answers such as "4.5 seconds" have almost no lexical content after
    # stopword removal; accept when the number matches and the object token is
    # present in the prediction.
    if _numbers(exp_norm) and overlap >= 0.25:
        return True, round(overlap, 3), f"numeric match + token recall {overlap:.2f}"
    return False, round(overlap, 3), f"token recall {overlap:.2f}"


# ---------------------------------------------------------------------------
# Task 3: evaluation loop
# ---------------------------------------------------------------------------
def _answer_row(
    pair: QAPair,
    answer: Any,
    *,
    answered_at: float | None,
    query_latency_ms: float,
    unreachable: bool = False,
    final: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    predicted = str(getattr(answer, "answer", "") or "") if error is None else ""
    alternatives = _expected_answers(pair)
    best: tuple[bool, float, str] = (False, 0.0, "no expected answer")
    for expected in alternatives or [pair.answer]:
        candidate = score_answer(predicted, expected, raw=pair.raw)
        if candidate[0] or best[2] == "no expected answer" or candidate[1] > best[1]:
            best = candidate
    correct, score, reason = best
    payload = getattr(answer, "to_dict", None)
    answer_dict = payload() if callable(payload) else {}
    return {
        "video_id": pair.video_id,
        "query": pair.query,
        "expected": pair.answer,
        "predicted": predicted,
        "correct": bool(correct),
        "score": round(float(score), 3),
        "reason": reason,
        "category": str(pair.raw.get("category") or "uncategorized"),
        "qa_timestamp": pair.timestamp,
        "answered_at": answered_at,
        "unreachable_timestamp": bool(unreachable),
        "answered_after_video": bool(final),
        "query_latency_ms": round(float(query_latency_ms), 4),
        "confidence": answer_dict.get("confidence"),
        "source": answer_dict.get("source", getattr(answer, "source", None)),
        "method": answer_dict.get("method", getattr(answer, "method", None)),
        "citations": answer_dict.get("citations", []),
        "error": error,
    }


def _fmt(value: Any, *, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def _summary_markdown(results: Mapping[str, Any]) -> str:
    metrics = results.get("metrics") or {}
    graph = metrics.get("graph") or {}
    rows = results.get("rows") or []
    lines = [
        "# Chronicle evaluation summary",
        "",
        f"- **Method:** `{results.get('method', 'chronicle')}`",
        f"- **Video:** `{results.get('video', '')}`",
        f"- **QA file:** `{results.get('qa_file', '')}`",
        f"- **Questions:** {results.get('total', 0)}",
        f"- **Recall@1:** {results.get('recall_at_1', 0.0):.1%} "
        f"({results.get('correct', 0)}/{results.get('total', 0)})",
        f"- **Answered before video end:** {results.get('answered', 0)}",
        f"- **Unreachable timestamps:** {results.get('unreachable', 0)}",
        f"- **Average GraphRAG query latency:** {results.get('avg_query_latency_ms')} ms",
        "",
        "## Efficiency",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Frames processed | {metrics.get('frames_processed')} |",
        f"| VLM calls (Florence triggers) | {metrics.get('vlm_calls')} |",
        f"| VLM call reduction | {_fmt(metrics.get('vlm_call_reduction_pct'), suffix='%')} |",
        f"| Effective Florence calls | {metrics.get('vlm_model_calls')} |",
        f"| Text-LLM calls | {metrics.get('llm_calls')} |",
        f"| Text-LLM latency (mean) | {_fmt(metrics.get('avg_llm_latency_ms'), suffix=' ms')} |",
        f"| Text-LLM models | {_md_cell(metrics.get('llm_models'))} |",
        f"| Average frame latency | {metrics.get('avg_latency_ms')} ms |",
        f"| Steady-state average | {metrics.get('avg_latency_steady_ms')} ms |",
        f"| p50 / p95 latency | {metrics.get('p50_ms')} / {metrics.get('p95_ms')} ms |",
        f"| Peak VRAM | {metrics.get('peak_vram_mb')} MB |",
        f"| Foreground graph | {graph.get('foreground_nodes')} nodes / "
        f"{graph.get('foreground_edges')} edges |",
        f"| Background graph | {graph.get('background_nodes')} nodes / "
        f"{graph.get('background_edges')} edges |",
        "",
        "## Recall@1 by category",
        "",
        "| Category | Correct | Total | Recall@1 |",
        "|---|---:|---:|---:|",
    ]
    for category, bucket in (results.get("categories") or {}).items():
        lines.append(
            f"| {_md_cell(category)} | {bucket.get('correct')} | {bucket.get('total')} | "
            f"{bucket.get('recall_at_1', 0.0):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Per-question results",
            "",
            "| # | Question | Expected | Predicted | Correct | Answered at | Source | Latency (ms) |",
            "|---:|---|---|---|---:|---:|---|---:|",
        ]
    )
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {_md_cell(row.get('query'))} | {_md_cell(row.get('expected'))} | "
            f"{_md_cell(row.get('predicted'))[:240]} | {'yes' if row.get('correct') else 'no'} | "
            f"{_md_cell(row.get('answered_at'))} | {_md_cell(row.get('source'))} | "
            f"{row.get('query_latency_ms')} |"
        )
    failures = [row for row in rows if not row.get("correct")]
    lines.extend(["", "## Failures", ""])
    if failures:
        for row in failures:
            lines.append(f"- **{row.get('query')}** — {row.get('reason')} — `{row.get('predicted')[:240]}`")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Recall@1 uses deterministic token/number matching against the QA answer.",
            "- VLM calls count captioner triggers; `effective Florence calls` excludes stub fallbacks.",
            "- Text-LLM calls count GraphRAG phrasing requests; template answers make zero LLM calls.",
            "- Peak VRAM is `torch.cuda.max_memory_allocated()` after resetting peak stats at run start.",
            "- The first `warmup_frames` are excluded from steady-state latency.",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(
    video_path: str | Path | None = None,
    qa_json_path: str | Path | None = None,
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
    out_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run the full temporal evaluation loop.

    Questions are answered at their requested timestamp: ingestion pauses
    synchronously inside the frame callback, queries the current graph through
    ``ChronicleRuntime.ask``, then resumes.  Questions at or beyond the end of
    the video are answered after ingestion and flagged as unreachable.
    """

    engine = ChronicleEngine(
        video_path=video_path,
        qa_json_path=qa_json_path,
        config_path=config_path,
        overrides=overrides,
    )
    if out_dir is not None:
        engine.out_dir = Path(out_dir).expanduser().resolve()
        engine.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(engine.qa_pairs)
    timed = [pair for pair in pairs if pair.timestamp is not None]
    untimed = [pair for pair in pairs if pair.timestamp is None]
    pending = sorted(timed, key=lambda pair: float(pair.timestamp))  # type: ignore[arg-type]
    rows: list[dict[str, Any]] = []
    meta = video_metadata(engine.video_path)
    bar = ProgressBar(meta.get("frames"), desc="Chronicle", unit="frame") if show_progress else None

    def query_pair(pair: QAPair, *, answered_at: float | None, unreachable: bool = False, final: bool = False) -> None:
        started = time.perf_counter()
        answer = None
        error = None
        try:
            answer = engine.runtime.ask(pair.query)  # type: ignore[union-attr]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("query failed: %s", pair.query)
        latency = (time.perf_counter() - started) * 1000.0
        row = _answer_row(
            pair,
            answer,
            answered_at=answered_at,
            query_latency_ms=latency,
            unreachable=unreachable,
            final=final,
            error=error,
        )
        rows.append(row)
        if bar is not None:
            bar.update(0, postfix=f"QA {len(rows)}/{len(pairs)}")

    def on_frame(result: Any) -> None:
        nonlocal pending
        while pending and float(pending[0].timestamp) <= float(result.timestamp) + 1e-6:  # type: ignore[arg-type]
            pair = pending.pop(0)
            query_pair(pair, answered_at=float(result.timestamp))
        if bar is not None:
            bar.update(1, postfix=f"QA {len(rows)}/{len(pairs)}")

    try:
        metrics = engine.run_ingestion(on_frame=on_frame)
    except Exception as exc:
        LOGGER.exception("evaluation ingestion failed")
        metrics = {
            "error": f"{type(exc).__name__}: {exc}",
            "frames_processed": 0,
            "vlm_calls": 0,
            "vlm_call_reduction_pct": None,
            "avg_latency_ms": None,
            "peak_vram_mb": None,
            "graph": {},
        }
    finally:
        if bar is not None:
            bar.close()

    # Anything still pending was after the final frame (or the video ended
    # early).  Answer against the final graph and mark it clearly.
    last_t = float(metrics.get("duration_s") or 0.0)
    for pair in pending:
        query_pair(pair, answered_at=last_t, unreachable=True, final=True)
    for pair in untimed:
        query_pair(pair, answered_at=last_t, final=True)

    correct = sum(1 for row in rows if row["correct"])
    total = len(rows)
    answered = sum(1 for row in rows if not row["unreachable_timestamp"] and not row["answered_after_video"])
    unreachable = sum(1 for row in rows if row["unreachable_timestamp"])
    categories: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = categories.setdefault(str(row["category"]), {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(row["correct"])
    for bucket in categories.values():
        bucket["recall_at_1"] = round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0

    payload: dict[str, Any] = {
        "method": "chronicle",
        "video": str(engine.video_path),
        "qa_file": str(engine.qa_path),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "video_metadata": meta,
        "total": total,
        "correct": correct,
        "recall_at_1": round(correct / total, 4) if total else 0.0,
        "answered": answered,
        "unreachable": unreachable,
        "avg_query_latency_ms": round(
            statistics.fmean([row["query_latency_ms"] for row in rows]), 4
        )
        if rows
        else None,
        "categories": dict(sorted(categories.items())),
        "metrics": metrics,
        "rows": rows,
        "config": engine.raw_config,
    }
    engine.close()

    _write_json(engine.out_dir / "results.json", payload)
    (engine.out_dir / "summary_report.md").write_text(_summary_markdown(payload), encoding="utf-8")

    if show_progress:
        print(
            f"\nRecall@1 {payload['recall_at_1']:.1%} ({correct}/{total}) · "
            f"VLM calls {metrics.get('vlm_calls')} · "
            f"peak VRAM {metrics.get('peak_vram_mb')} MB"
        )
        print(f"wrote {engine.out_dir / 'results.json'}")
        print(f"wrote {engine.out_dir / 'summary_report.md'}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=None, help="video path (default: config.video.path)")
    parser.add_argument("--qa", type=Path, default=None, help="QA JSON path (default: config.qa.path)")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG, help="YAML config")
    parser.add_argument("--out", type=Path, default=None, help="output directory for metrics.json")
    parser.add_argument("--db", type=Path, default=None, help="SQLite path for this run")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames")
    parser.add_argument("--fps", type=float, default=None, help="override video FPS for sampling")
    parser.add_argument(
        "--vocabulary-preset",
        choices=("default", "broad", "none"),
        default=None,
        help="detector prompt set; scene metadata vocabulary is always added",
    )
    parser.add_argument("--offline", action="store_true", help="colour backends + stub Florence (no weights)")
    parser.add_argument(
        "--captioner",
        choices=("auto", "florence", "qwen", "qwen-vl", "stub"),
        default=None,
        help="semantic captioner backend; qwen uses Ollama qwen2.5vl:3b",
    )
    parser.add_argument("--no-semantic", action="store_true", help="disable VLM verification entirely")
    parser.add_argument(
        "--caption-stride",
        type=int,
        default=None,
        help="VLM budget ablation: caption every Nth event trigger (0 disables captions)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="run the temporal QA loop and write results.json + summary_report.md",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("template", "none", "ollama", "auto", "transformers"),
        default=None,
        help="GraphRAG phrasing backend; ollama uses local qwen2.5:7b-instruct",
    )
    parser.add_argument("--llm-model", default=None, help="local Ollama model name")
    parser.add_argument("--json", action="store_true", help="print the full JSON payload")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides: dict[str, Any] = {}
    if args.captioner is not None:
        overrides.setdefault("semantic", {})["captioner_backend"] = args.captioner
    if args.offline:
        overrides["runtime"] = {"profile": "offline"}
    if args.no_semantic:
        overrides.setdefault("semantic", {})["enabled"] = False
    if args.caption_stride is not None:
        overrides.setdefault("semantic", {})["caption_stride"] = int(args.caption_stride)
    if args.max_frames is not None:
        overrides.setdefault("perception", {})["max_frames"] = int(args.max_frames)
    if args.fps is not None:
        overrides.setdefault("video", {})["fps_override"] = float(args.fps)
    if args.vocabulary_preset is not None:
        overrides.setdefault("perception", {})["vocabulary_preset"] = args.vocabulary_preset
    if args.llm_backend is not None:
        overrides.setdefault("semantic", {})["llm_backend"] = args.llm_backend
    if args.llm_model is not None:
        overrides.setdefault("semantic", {})["llm_model"] = args.llm_model
    if args.out is not None:
        overrides.setdefault("paths", {})["out_dir"] = str(args.out)
        if args.db is None:
            overrides["paths"]["db_path"] = str(Path(args.out) / "chronicle.db")
    if args.db is not None:
        overrides.setdefault("paths", {})["db_path"] = str(args.db)

    if args.evaluate:
        payload = run_evaluation(
            args.video,
            args.qa,
            config_path=args.config,
            out_dir=args.out,
            overrides=overrides,
            show_progress=not args.json,
        )
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if not (payload.get("metrics") or {}).get("error") else 1

    engine = ChronicleEngine(
        video_path=args.video,
        qa_json_path=args.qa,
        config_path=args.config,
        overrides=overrides,
    )
    bar = ProgressBar(video_metadata(engine.video_path).get("frames"), desc="Chronicle", unit="frame")
    try:
        metrics = engine.run_ingestion(on_frame=lambda _result: bar.update(1))
    finally:
        bar.close()
        engine.close()

    out_dir = engine.out_dir
    _write_json(out_dir / "metrics.json", metrics)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        reduction = metrics.get("vlm_call_reduction_pct")
        print(
            f"frames={metrics['frames_processed']} "
            f"vlm_calls={metrics['vlm_calls']} "
            f"reduction={reduction if reduction is None else f'{reduction:.1f}%'} "
            f"avg_frame_ms={metrics['avg_latency_ms']} "
            f"peak_vram_mb={metrics['peak_vram_mb']} "
            f"fg={metrics['graph'].get('foreground_nodes')}n/"
            f"{metrics['graph'].get('foreground_edges')}e "
            f"bg={metrics['graph'].get('background_nodes')}n/"
            f"{metrics['graph'].get('background_edges')}e "
            f"-> {out_dir / 'metrics.json'}"
        )
    if metrics.get("error"):
        LOGGER.error("run finished with error: %s", metrics["error"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

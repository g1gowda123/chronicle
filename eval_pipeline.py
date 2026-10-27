#!/usr/bin/env python3
"""Headless evaluation entry point for Chronicle.

This module wraps the existing perception -> lifecycle -> semantic -> graph
pipeline in a UI-free class (`ChronicleEngine`) and instruments it with
`MetricsTracker`.  It deliberately does **not** reimplement the state machine,
gating heuristics, tracker or graph logic: those remain in ``chronicle.*`` and
are called through their public (or stable facade) APIs.

Usage
-----
    python eval_pipeline.py \
        --video data/real/cook/aria_cook_45s.mp4 \
        --qa data/qa/ego4d_example.json \
        --config config.yaml \
        --out runs/eval/cook

    # Fully offline smoke run: colour detector/hands/embedder + stub Florence
    python eval_pipeline.py --offline --max-frames 30

The engine accepts an Ego4D-shaped QA file.  Task 3 (the evaluation loop) will
consume ``ChronicleEngine.qa_pairs`` and query the graph at each timestamp; the
engine already loads and normalises those pairs so the query code has a stable
contract.
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


def load_qa_pairs(path: str | Path, *, default_video_id: str | None = None) -> list[QAPair]:
    """Load a QA file in Ego4D / QAEgo4D-Closed shape.

    Accepted top-level shapes::

        {"questions": [{"video_id", "query", "answer", "timestamp"}, ...]}
        [{"video_id", "query", "answer", "timestamp"}, ...]

    Aliases ``question``/``nlq_query`` and ``annotation``/``answers`` are
    accepted so exports from the Ego4D tooling and the internal benchmark can
    share one loader.
    """

    qa_path = Path(path).expanduser()
    if not qa_path.is_file():
        raise FileNotFoundError(f"QA file not found: {qa_path}")
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        rows = payload.get("questions") or payload.get("data") or payload.get("qa")
        file_video_id = str(payload.get("video_id") or default_video_id or "")
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
    return pairs


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
        self.vlm_latencies_ms: list[float] = []
        self.vlm_backends: dict[str, int] = {}
        self.vlm_triggers: dict[str, int] = {}
        self.detections = 0
        self.tracks = 0
        self.last_timestamp = 0.0
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
            "vlm_calls_per_100_frames": round(100.0 * self.vlm_calls / frames, 3) if frames else None,
            "vlm_call_reduction_pct": reduction,
            "vlm_model_call_reduction_pct": effective_reduction,
            "vlm_backends": dict(sorted(self.vlm_backends.items())),
            "vlm_triggers": dict(sorted(self.vlm_triggers.items())),
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
            "detections": self.detections,
            "tracks_touched": self.tracks,
            "graph": self._graph_stats(memory),
            "error": self.error,
        }
        if extra:
            payload.update(dict(extra))
        return payload


def instrument_captioner(captioner: Any, metrics: MetricsTracker, *, trigger: str = "semantic") -> bool:
    """Wrap ``captioner.caption`` so every Florence-2 trigger is counted.

    Returns ``True`` when the captioner was instrumented.  The wrapper is
    idempotent, so `ChronicleEngine` can call it in tests more than once.
    """

    if captioner is None or getattr(captioner, "_chronicle_metrics_wrapped", False):
        return False
    original = getattr(captioner, "caption", None)
    if not callable(original):
        return False

    def _backend_name() -> str:
        impl = getattr(captioner, "_impl", None) or captioner
        return type(impl).__name__

    def counted(image: Any, task: Any = None) -> Any:
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

        self.qa_pairs = load_qa_pairs(self.qa_path, default_video_id=self.video_path.stem)
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
        cfg.detector.confidence = float(section.get("detector_confidence", cfg.detector.confidence))
        cfg.detector.imgsz = int(section.get("detector_imgsz", cfg.detector.imgsz))
        cfg.hands.backend = str(section.get("hands_backend", cfg.hands.backend))
        cfg.hands.model = str(section.get("hands_model", cfg.hands.model))
        cfg.embedder.backend = str(section.get("embedder_backend", cfg.embedder.backend))
        cfg.embedder.model = str(section.get("embedder_model", cfg.embedder.model))
        cfg.embedder.pretrained = str(section.get("embedder_pretrained", cfg.embedder.pretrained))
        cfg.motion.enabled = bool(section.get("motion_enabled", cfg.motion.enabled))
        cfg.clip.enabled = bool(section.get("clip_enabled", cfg.clip.enabled))
        vocab = section.get("vocabulary") or []
        if isinstance(vocab, str):
            vocab = [item.strip() for item in vocab.split(",") if item.strip()]
        cfg.vocabulary = tuple(dict.fromkeys([*cfg.vocabulary, *[str(item) for item in vocab]]))
        # QA nouns are useful YOLO-World prompts; the detector ignores unknown
        # labels, so over-including them only costs a little compute.
        cfg.vocabulary = tuple(dict.fromkeys([*cfg.vocabulary, *_qa_vocabulary(self.qa_pairs)]))
        return cfg

    def _semantic_config(self) -> Any:
        from chronicle.semantic.config import SemanticConfig

        section = _section(self.raw_config, "semantic")
        cfg = SemanticConfig(
            clip_model_name=str(section.get("clip_model_name", "ViT-B-32")),
            clip_pretrained=str(section.get("clip_pretrained", "laion2b_s34b_b79k")),
            device=str(section.get("device", "auto")),
            florence_model_id=str(section.get("florence_model_id", "microsoft/Florence-2-base")),
            florence_max_new_tokens=int(section.get("florence_max_new_tokens", 128)),
            florence_num_beams=int(section.get("florence_num_beams", 3)),
            action_task=str(section.get("action_task", "<MORE_DETAILED_CAPTION>")),
            action_min_confidence=float(section.get("action_min_confidence", 0.45)),
            llm_backend=str(section.get("llm_backend", "template")),
            llm_model=str(section.get("llm_model", "qwen2.5:7b-instruct")),
            ollama_url=str(section.get("ollama_url", "http://127.0.0.1:11434")),
            require_citations=bool(section.get("require_citations", True)),
            lenient_citations=bool(section.get("lenient_citations", True)),
            max_facts=int(section.get("max_facts", 40)),
        )
        return cfg

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
            instrument_captioner(semantic.captioner, self.metrics)
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
                "backends": runtime.backends() if hasattr(runtime, "backends") else {},
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
    """Extract candidate object nouns from QA text for YOLO-World prompts."""

    words: list[str] = []
    for pair in pairs:
        for token in re.findall(r"[a-z][a-z0-9'-]{2,}", pair.query.lower()):
            if token in _QA_STOPWORDS or token in words:
                continue
            words.append(token)
    return tuple(words[:40])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    parser.add_argument("--offline", action="store_true", help="colour backends + stub Florence (no weights)")
    parser.add_argument("--no-semantic", action="store_true", help="disable VLM verification entirely")
    parser.add_argument("--json", action="store_true", help="print the full metrics payload as JSON")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides: dict[str, Any] = {}
    if args.offline:
        overrides["runtime"] = {"profile": "offline"}
    if args.no_semantic:
        overrides.setdefault("semantic", {})["enabled"] = False
    if args.max_frames is not None:
        overrides.setdefault("perception", {})["max_frames"] = int(args.max_frames)
    if args.fps is not None:
        overrides.setdefault("video", {})["fps_override"] = float(args.fps)
    if args.out is not None:
        overrides.setdefault("paths", {})["out_dir"] = str(args.out)
        if args.db is None:
            overrides["paths"]["db_path"] = str(Path(args.out) / "chronicle.db")
    if args.db is not None:
        overrides.setdefault("paths", {})["db_path"] = str(args.db)

    engine = ChronicleEngine(
        video_path=args.video,
        qa_json_path=args.qa,
        config_path=args.config,
        overrides=overrides,
    )
    try:
        metrics = engine.run_ingestion()
    finally:
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

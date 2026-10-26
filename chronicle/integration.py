"""End-to-end runtime: Role A -> Role B -> Role C -> dashboard.

The team repo originally exposed three separate seams: ``PerceptionPipeline``
for Role A, ``ChronicleMemory`` for Role B and ``SemanticEngine`` for Role C.
This module wires them into one object so the dashboard, CLI and demo scripts
all use exactly the same path:

    runtime = ChronicleRuntime("data/demo/chronicle.db")
    runtime.ingest_video("data/demo/chronicle_demo.mp4")
    answer = runtime.ask("where are my keys?")

Design rules
------------
* Role A/B/C APIs stay untouched; this is an additive facade.
* Clips arrive on the pipeline's ``on_clip`` callback before the corresponding
  frame is observed, so they are queued and drained after ``observe_result``.
* Query results are normalised into Role C's ``GraphContext`` contract instead
  of leaking Role B's edge shape into GraphRAG.
* Everything is thread-safe enough for the NiceGUI worker thread: snapshots are
  plain dictionaries and SQLite readers use WAL.
"""

from __future__ import annotations

import copy
import json
import logging
import queue
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronicle.config import (
    DEFAULT_THRESHOLDS,
    STATIC_FURNITURE_LABELS,
    FurnitureConfig,
    Thresholds,
)
from chronicle.graph.ids import raw_entity_id
from chronicle.graph.memory import ChronicleMemory
from chronicle.graph.query import QueryResult
from chronicle.perception.config import PerceptionConfig
from chronicle.perception.pipeline import PerceptionPipeline
from chronicle.semantic.config import SemanticConfig
from chronicle.semantic.engine import SemanticEngine
from chronicle.semantic.graphrag import Fact, GraphContext, QueryAnswer
from chronicle.telemetry import EventBus, TelemetryTag
from chronicle.types import (
    FrameResult,
    InteractionClip,
)
from chronicle.types import (
    TelemetryEvent as PerceptionTelemetryEvent,
)

logger = logging.getLogger("chronicle.integration")

_TAG_BY_CATEGORY = {
    "GEOM": TelemetryTag.GEOM,
    "TRACK": TelemetryTag.GRAPH,
    "REID": TelemetryTag.GRAPH,
    "CLIP": TelemetryTag.VLM,
    "PERCEPTION": TelemetryTag.GEOM,
    "SYSTEM": TelemetryTag.SYSTEM,
}


@dataclass(frozen=True)
class RuntimeAnswer:
    """Serializable answer returned by :meth:`ChronicleRuntime.ask`."""

    query: str
    answer: str
    citations: tuple[dict[str, Any], ...] = ()
    source: str = "graph"
    grounded: bool = False
    method: str = "template"
    elapsed: float = 0.0
    entity: str | None = None
    state: str | None = None
    zone: str | None = None
    last_seen: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": list(self.citations),
            "source": self.source,
            "grounded": self.grounded,
            "method": self.method,
            "elapsed": round(self.elapsed, 4),
            "entity": self.entity,
            "state": self.state,
            "zone": self.zone,
            "last_seen": self.last_seen,
        }


@dataclass
class RuntimeSummary:
    video: str
    frames: int
    duration: float
    elapsed: float
    entities: int
    transitions: int
    actions: int
    zones: int
    furniture: int
    clips: int
    detections: int
    db_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "frames": self.frames,
            "duration": round(self.duration, 3),
            "elapsed": round(self.elapsed, 3),
            "entities": self.entities,
            "transitions": self.transitions,
            "actions": self.actions,
            "zones": self.zones,
            "furniture": self.furniture,
            "clips": self.clips,
            "detections": self.detections,
            "db_path": self.db_path,
        }


class ChronicleRuntime:
    """One façade for ingestion, semantic verification and GraphRAG queries."""

    def __init__(
        self,
        db_path: str | Path = "data/chronicle.db",
        *,
        bus: EventBus | None = None,
        thresholds: Thresholds | None = None,
        furniture_config: FurnitureConfig | None = None,
        furniture_labels: tuple[str, ...] | frozenset[str] = STATIC_FURNITURE_LABELS,
        semantic: SemanticEngine | None = None,
        semantic_config: SemanticConfig | None = None,
        semantic_enabled: bool = True,
        perception_config: PerceptionConfig | None = None,
        zone_label: str | None = None,
        zone_id: str | None = None,
        load: bool = True,
        reset: bool = False,
    ) -> None:
        self.db_path = str(db_path)
        self.bus = bus or EventBus()
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.furniture_config = furniture_config
        self.furniture_labels = tuple(furniture_labels)
        self.zone_label = zone_label
        self.zone_id = zone_id
        self.semantic_enabled = semantic_enabled
        self._semantic_config = semantic_config
        self._semantic = semantic
        self.perception_config = copy.deepcopy(perception_config or PerceptionConfig())

        if reset:
            _remove_database(self.db_path)

        self.memory = ChronicleMemory(
            self.db_path,
            thresholds=self.thresholds,
            bus=self.bus,
            furniture_labels=self.furniture_labels,
            furniture_config=self.furniture_config,
            load=load,
        )

        self._clip_queue: queue.Queue[InteractionClip] = queue.Queue()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._pipeline: PerceptionPipeline | None = None
        self._running = False
        self._frames = 0
        self._clips = 0
        self._actions = 0
        self._detections = 0
        self._transitions = 0
        self._started_at = 0.0
        self._last_timestamp = 0.0
        self._video: str | None = None
        self._summary: RuntimeSummary | None = None
        self._error: str | None = None
        self._last_progress = 0.0

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> ChronicleRuntime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.stop()
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass
        try:
            self.memory.close()
        except Exception:
            pass

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def summary(self) -> RuntimeSummary | None:
        with self._lock:
            return self._summary

    @property
    def semantic(self) -> SemanticEngine | None:
        return self._ensure_semantic()

    # ------------------------------------------------------------------ ingestion
    def ingest_video(
        self,
        video: str | Path,
        *,
        max_frames: int | None = None,
        fps_override: float | None = None,
        stop_event: threading.Event | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        reset: bool = False,
    ) -> RuntimeSummary:
        """Ingest a recorded video through A -> B -> C.

        ``reset=True`` recreates the in-memory/SQLite memory first.  The
        dashboard passes ``reset=True`` when replaying a source so repeated
        runs do not accumulate duplicate episodes.
        """

        with self._lock:
            if self._running:
                raise RuntimeError("runtime is already ingesting")
            self._running = True
            self._error = None
            self._frames = 0
            self._clips = 0
            self._actions = 0
            self._detections = 0
            self._transitions = 0
            self._started_at = time.monotonic()
            self._last_timestamp = 0.0
            self._video = str(video)
            self._summary = None
            self._last_progress = 0.0
        self._stop_event = stop_event or threading.Event()

        if reset:
            self._reset_memory()

        config = copy.deepcopy(self.perception_config)
        if max_frames is not None:
            config.max_frames = int(max_frames)
        pipeline = PerceptionPipeline(
            config,
            on_clip=self._queue_clip,
            telemetry_sink=self._on_perception_event,
        )
        with self._lock:
            self._pipeline = pipeline

        if self.zone_label or self.zone_id:
            self.memory.enter_zone(
                self.zone_label or self.zone_id or "unmapped area",
                0.0,
                zone_id=self.zone_id,
            )

        try:
            for result in pipeline.iter_results(
                video,
                stop_event=self._stop_event,
                fps_override=fps_override,
            ):
                self._observe_result(result)
                self._drain_clips()
                with self._lock:
                    self._frames += 1
                    self._last_timestamp = result.timestamp
                if on_progress is not None and self._frames % 5 == 0:
                    try:
                        on_progress(self.progress())
                    except Exception:
                        logger.debug("progress callback failed", exc_info=True)
        except Exception as exc:  # surfaced in the dashboard
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            logger.exception("runtime ingestion failed")
            raise
        finally:
            self._drain_clips()
            try:
                pipeline.close()
            except Exception:
                pass
            with self._lock:
                self._running = False
                self._pipeline = None
            summary = self._build_summary()
            with self._lock:
                self._summary = summary
            self.bus.emit(
                TelemetryTag.SYSTEM,
                "episode complete",
                self._last_timestamp,
                frames=summary.frames,
                entities=summary.entities,
                actions=summary.actions,
                furniture=summary.furniture,
            )
        return summary

    def _observe_result(self, result: FrameResult) -> None:
        zone_id = self.memory.active_zone
        transitions = self.memory.observe_result(
            result,
            zone_id=zone_id,
            zone_label=self.zone_label,
        )
        with self._lock:
            self._transitions += len(transitions)
            self._detections += len(result.detections)

    def _queue_clip(self, clip: InteractionClip) -> None:
        self._clip_queue.put(clip)

    def _drain_clips(self) -> None:
        while True:
            try:
                clip = self._clip_queue.get_nowait()
            except queue.Empty:
                return
            self._verify_clip(clip)

    def _verify_clip(self, clip: InteractionClip) -> None:
        engine = self._ensure_semantic()
        if engine is None:
            return
        try:
            context_hint = str(
                clip.metadata.get("context_hint")
                or clip.metadata.get("caption")
                or f"the hand grasps the {clip.label}"
            )
            verification = engine.verify_interaction(clip, context_hint=context_hint)
            action = verification.action
            with self._lock:
                self._clips += 1
            if not action.accepted:
                self.bus.vlm(
                    "action rejected",
                    clip.contact_ts,
                    clip=clip.clip_id,
                    action=action.action.value,
                    confidence=round(action.confidence, 3),
                )
                return
            entity_id = self.memory.foreground.entity_for_track(
                clip.track_id,
                label=clip.label,
                near_t=clip.contact_ts,
            )
            if entity_id is None:
                self.bus.vlm(
                    "clip entity not in graph",
                    clip.contact_ts,
                    clip=clip.clip_id,
                    track=clip.track_id,
                    label=clip.label,
                )
                return
            evidence = action.to_dict()
            evidence.update(
                {
                    "clip_id": clip.clip_id,
                    "track_id": clip.track_id,
                    "trigger": clip.trigger,
                    "max_ios": clip.max_ios,
                    "caption": action.caption or context_hint,
                }
            )
            self.memory.record_action(
                entity_id,
                action.action.value,
                action.timestamp if action.timestamp is not None else clip.contact_ts,
                confidence=action.confidence,
                source=action.source,
                clip_ref=clip.clip_id,
                evidence=evidence,
            )
            with self._lock:
                self._actions += 1
        except Exception as exc:  # a semantic failure must never stop ingestion
            self.bus.vlm(
                "clip verification failed",
                clip.contact_ts,
                clip=clip.clip_id,
                error=repr(exc),
            )
            logger.exception("clip verification failed for %s", clip.clip_id)

    # ------------------------------------------------------------------ queries
    def ask(self, query: str) -> RuntimeAnswer:
        started = time.monotonic()
        result = self.memory.query(query)
        if result is None:
            return RuntimeAnswer(
                query=query,
                answer="I don't have that in my memory.",
                source="graph",
                grounded=False,
                elapsed=time.monotonic() - started,
            )
        context = self.graph_context(result)
        engine = self._ensure_semantic()
        if engine is None:
            answer = result.fallback_answer()
            return RuntimeAnswer(
                query=query,
                answer=answer,
                citations=(),
                source="graph",
                grounded=True,
                method="graph-template",
                elapsed=time.monotonic() - started,
                entity=result.label,
                state=result.state,
                zone=result.zone_label,
                last_seen=result.last_seen,
            )
        try:
            answer: QueryAnswer = engine.answer_query(query, context)
            return RuntimeAnswer(
                query=query,
                answer=answer.answer,
                citations=tuple(citation.to_dict() for citation in answer.citations),
                source=answer.source,
                grounded=answer.grounded,
                method=answer.source,
                elapsed=time.monotonic() - started,
                entity=result.label,
                state=result.state,
                zone=result.zone_label,
                last_seen=result.last_seen,
            )
        except Exception as exc:
            logger.exception("GraphRAG answer failed; using graph template")
            return RuntimeAnswer(
                query=query,
                answer=result.fallback_answer(),
                source="graph",
                grounded=True,
                method=f"template ({type(exc).__name__})",
                elapsed=time.monotonic() - started,
                entity=result.label,
                state=result.state,
                zone=result.zone_label,
                last_seen=result.last_seen,
            )

    def graph_context(self, result: QueryResult) -> GraphContext:
        """Normalize Role B's serialized 2-hop graph into Role C's contract."""

        labels = {
            str(node.get("node_id")): str(node.get("label") or node.get("node_id"))
            for node in result.nodes
        }
        labels[result.node_id] = result.label
        facts: list[Fact] = []
        seen: set[tuple[str, str, str, float | None]] = set()
        for edge in result.edges:
            relation = str(edge.get("relation") or "")
            src = str(edge.get("src") or "")
            dst = str(edge.get("dst") or "")
            src_label = labels.get(src, src)
            dst_label = labels.get(dst, dst)
            timestamp = _as_float(edge.get("t_start"))
            zone = None
            if relation == "ACTION":
                subject = "USER"
                relation_label = str(edge.get("action") or "ACTION")
                object_label = dst_label
                zone = result.zone_label
            elif relation == "STATE_TRANSITION":
                subject = result.label
                relation_label = "STATE"
                to_state = str(edge.get("to_state") or "")
                object_label = to_state or dst_label
                zone = result.zone_label
            elif relation == "IN_ZONE":
                subject = src_label
                relation_label = "IN_ZONE"
                object_label = dst_label
                zone = dst_label
            elif relation in {"RESTING_ON", "INSIDE"}:
                subject = src_label
                relation_label = relation
                object_label = dst_label
                zone = result.zone_label
            else:
                subject = src_label
                relation_label = relation or "RELATED"
                object_label = dst_label
            key = (subject, relation_label, object_label, timestamp)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                Fact(
                    subject=subject,
                    relation=relation_label,
                    object_label=object_label,
                    timestamp=timestamp,
                    zone=zone,
                    confidence=_as_float(edge.get("confidence")),
                    source=str(edge.get("source") or "graph"),
                )
            )
        anchor = result.terminal_anchor or {}
        if anchor and anchor.get("label"):
            timestamp = _as_float(anchor.get("t_start"))
            key = (result.label, str(anchor.get("relation") or "RESTING_ON"), str(anchor["label"]), timestamp)
            if key not in seen:
                facts.append(
                    Fact(
                        subject=result.label,
                        relation=str(anchor.get("relation") or "RESTING_ON"),
                        object_label=str(anchor["label"]),
                        timestamp=timestamp,
                        zone=result.zone_label,
                        confidence=_as_float(anchor.get("confidence")),
                        source="anchor",
                    )
                )
        facts.sort(key=lambda fact: fact.timestamp if fact.timestamp is not None else -1.0)
        return GraphContext(
            entity_id=result.entity_id,
            entity_label=result.label,
            state=result.state,
            zone=result.zone_label,
            last_seen=result.last_seen,
            facts=tuple(facts),
            attributes={
                "container": result.container_label,
                "terminal_anchor": anchor,
                "match_score": result.match_score,
            },
        )

    # ------------------------------------------------------------------ dashboard
    def progress(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(1e-6, time.monotonic() - self._started_at)
            return {
                "frames": self._frames,
                "fps": self._frames / elapsed if self._frames else 0.0,
                "elapsed": elapsed,
                "timestamp": self._last_timestamp,
                "running": self._running,
                "actions": self._actions,
                "clips": self._clips,
                "detections": self._detections,
                "transitions": self._transitions,
                "error": self._error,
            }

    def snapshot(self) -> dict[str, Any]:
        """Graph + table payload consumed by the NiceGUI dashboard."""

        graph = self.memory.visualization_snapshot()
        progress = self.progress()
        entities = self._entity_rows()
        actions = self.memory.foreground.action_rows(limit=200)
        transitions = self._transition_rows(limit=500)
        furniture = self.memory.background.furniture()
        stats = self.memory.stats()
        with self._lock:
            summary = self._summary.as_dict() if self._summary else None
            video = self._video
            running = self._running
            error = self._error
        return {
            **graph,
            "entities": entities,
            "actions": actions,
            "transitions": transitions,
            "furniture": furniture,
            "stats": {
                **stats,
                "frames": progress["frames"],
                "fps": round(progress["fps"], 2),
                "elapsed": round(progress["elapsed"], 2),
                "timestamp": round(progress["timestamp"], 2),
                "actions": len(actions),
                "transitions": len(transitions),
                "clips": progress["clips"],
                "detections": progress["detections"],
                "camera_state": self.memory.engine.camera_state.value,
                "current_zone": self.memory.background.zone_label(self.memory.active_zone),
                "stage": "running" if running else ("error" if error else "complete"),
            },
            "summary": summary,
            "running": running,
            "error": error,
            "video": video,
            "zone": self.memory.background.zone_label(self.memory.active_zone),
            "backends": self.backends(),
        }

    def _entity_rows(self) -> list[dict[str, Any]]:
        zone_names = {
            node_id: attrs.get("label", node_id)
            for node_id, attrs in self.memory.background.graph.nodes(data=True)
            if attrs.get("kind") == "zone"
        }
        anchor_labels = {
            node_id: attrs.get("label", node_id)
            for node_id, attrs in self.memory.background.graph.nodes(data=True)
            if attrs.get("kind") == "furniture"
        }
        rows: list[dict[str, Any]] = []
        for attrs in self.memory.foreground.entities():
            entity_id = raw_entity_id(str(attrs["node_id"]))
            anchor = self.memory.store.terminal_anchor(entity_id)
            anchor_id = anchor.anchor_id if anchor else None
            container_id = attrs.get("container_id")
            rows.append(
                {
                    "id": entity_id,
                    "label": attrs.get("label", entity_id),
                    "state": attrs.get("state", "NEW"),
                    "zone": zone_names.get(str(attrs.get("zone_id") or ""), "-"),
                    "anchor": anchor_labels.get(str(anchor_id), _short_anchor(anchor_id)),
                    "container": anchor_labels.get(
                        f"anchor:{container_id}", str(container_id or "-")
                    ),
                    "first_seen": float(attrs.get("first_seen") or 0.0),
                    "last_seen": float(attrs.get("last_seen") or 0.0),
                    "span": f"{float(attrs.get('first_seen') or 0.0):.1f} – "
                    f"{float(attrs.get('last_seen') or 0.0):.1f}s",
                    "track_id": attrs.get("track_id"),
                    "confidence": 1.0,
                }
            )
        rows.sort(key=lambda row: row["first_seen"])
        return rows

    def _transition_rows(self, limit: int = 500) -> list[dict[str, Any]]:
        labels = {
            raw_entity_id(str(attrs["node_id"])): attrs.get("label")
            for attrs in self.memory.foreground.entities()
        }
        rows = []
        for record in reversed(self.memory.store.recent_transitions(limit=limit)):
            rows.append(
                {
                    "id": record.id,
                    "entity_id": record.entity_id,
                    "label": labels.get(record.entity_id, record.entity_id),
                    "from_state": record.from_state,
                    "to_state": record.to_state,
                    "timestamp": record.t_event,
                    "t_event": record.t_event,
                    "clip_ref": record.clip_ref,
                }
            )
        return rows

    def backends(self) -> dict[str, str]:
        pipeline = self._pipeline
        if pipeline is None:
            return {
                "detector": self.perception_config.detector.backend,
                "hands": self.perception_config.hands.backend,
                "embedder": self.perception_config.embedder.backend,
                "llm": self._semantic_config.llm_backend if self._semantic_config else "template",
            }
        return {
            "detector": getattr(pipeline.detector, "name", self.perception_config.detector.backend),
            "hands": getattr(pipeline.hands, "name", self.perception_config.hands.backend),
            "embedder": getattr(pipeline.embedder, "name", self.perception_config.embedder.backend),
            "llm": self._semantic_config.llm_backend if self._semantic_config else "template",
        }

    def llm_status(self, *, refresh: bool = False) -> dict[str, Any]:
        """Best-effort Ollama status for the header chip; never blocks ingestion."""

        config = self._semantic_config or SemanticConfig()
        model = config.llm_model
        base_url = config.ollama_url.rstrip("/")
        status: dict[str, Any] = {
            "backend": config.llm_backend,
            "model": model,
            "server": False,
            "available": False,
            "loaded": [],
            "version": None,
        }
        if config.llm_backend in {"template", "none"}:
            status["backend"] = "template"
            return status
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=0.8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            names = [str(item.get("name", "")) for item in payload.get("models", [])]
            status["server"] = True
            status["available"] = True
            status["model_present"] = any(
                name.split(":")[0] == model.split(":")[0] for name in names
            )
            try:
                with urllib.request.urlopen(f"{base_url}/api/ps", timeout=0.8) as response:
                    running = json.loads(response.read().decode("utf-8"))
                status["loaded"] = running.get("models", [])
            except Exception:
                pass
        except Exception:
            status["server"] = False
        return status

    # ------------------------------------------------------------------ internal
    def _ensure_semantic(self) -> SemanticEngine | None:
        if not self.semantic_enabled:
            return None
        if self._semantic is not None:
            return self._semantic
        with self._lock:
            if self._semantic is not None:
                return self._semantic
            config = self._semantic_config or SemanticConfig.from_env()
            # The dashboard must be usable without model downloads.  Florence is
            # opt-in; the deterministic captioner/rule classifier is the
            # documented offline fallback.  OpenCLIP auto-degrades too.
            self._semantic = SemanticEngine(
                config=config,
                captioner_backend="auto",
                embedder_backend="auto",
                on_event=self._on_semantic_event,
            )
            return self._semantic

    def _on_semantic_event(self, payload: dict[str, Any]) -> None:
        tag = str(payload.get("tag") or "VLM")
        message = str(payload.get("message") or payload.get("event") or "semantic event")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        self.bus.emit(tag, message, None, **data)

    def _on_perception_event(self, event: PerceptionTelemetryEvent) -> None:
        tag = _TAG_BY_CATEGORY.get(str(event.category.value), TelemetryTag.GEOM)
        self.bus.emit(
            tag,
            event.message,
            event.presentation_time,
            frame_index=event.frame_index,
            **dict(event.data),
        )

    def _reset_memory(self) -> None:
        try:
            self.memory.close()
        except Exception:
            pass
        _remove_database(self.db_path)
        self.memory = ChronicleMemory(
            self.db_path,
            thresholds=self.thresholds,
            bus=self.bus,
            furniture_labels=self.furniture_labels,
            furniture_config=self.furniture_config,
            load=False,
        )
        if self.zone_label or self.zone_id:
            self.memory.enter_zone(
                self.zone_label or self.zone_id or "unmapped area",
                0.0,
                zone_id=self.zone_id,
            )

    def _build_summary(self) -> RuntimeSummary:
        stats = self.memory.stats()
        with self._lock:
            return RuntimeSummary(
                video=self._video or "",
                frames=self._frames,
                duration=self._last_timestamp,
                elapsed=max(0.0, time.monotonic() - self._started_at),
                entities=int(stats.get("entities", 0)),
                transitions=int(self._transitions),
                actions=int(self._actions),
                zones=int(stats.get("zones", 0)),
                furniture=len(self.memory.background.furniture()),
                clips=int(self._clips),
                detections=int(self._detections),
                db_path=self.db_path,
            )


def _remove_database(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                logger.warning("could not remove %s", candidate, exc_info=True)


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _short_anchor(anchor_id: str | None) -> str:
    if not anchor_id:
        return "-"
    return str(anchor_id).removeprefix("anchor:").removeprefix("entity:")


__all__ = ["ChronicleRuntime", "RuntimeAnswer", "RuntimeSummary"]

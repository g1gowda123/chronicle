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
import csv
import io
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chronicle.attributes import (
    COLOUR_WORDS,
    attribute_query_field,
    attribute_summary,
    public_attributes,
)
from chronicle.config import (
    DEFAULT_THRESHOLDS,
    STATIC_FURNITURE_LABELS,
    FurnitureConfig,
    Thresholds,
)
from chronicle.frames import (
    FrameCandidate,
    extract_video_frame_data_url,
    latest_citation_time,
    load_stored_evidence,
    pick_stored_evidence,
)
from chronicle.graph.ids import raw_entity_id
from chronicle.graph.memory import ChronicleMemory
from chronicle.graph.query import QueryResult, is_event_query
from chronicle.perception.config import PerceptionConfig
from chronicle.perception.pipeline import PerceptionPipeline
from chronicle.semantic.config import SemanticConfig
from chronicle.semantic.engine import SemanticEngine
from chronicle.semantic.graphrag import Fact, GraphContext, QueryAnswer
from chronicle.semantic.queue import VerificationQueue
from chronicle.telemetry import EventBus, TelemetryTag
from chronicle.text import content_tokens, expand_token, singular
from chronicle.types import (
    FrameResult,
    InteractionClip,
    ObjectState,
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

_ENTITY_CSV_COLUMNS = (
    "id",
    "label",
    "state",
    "zone",
    "anchor",
    "container",
    "first_seen",
    "last_seen",
    "span",
    "track_id",
    "confidence",
    "colour",
    "colour_hex",
    "attributes",
)

_ACTION_CSV_COLUMNS = (
    "id",
    "timestamp",
    "action",
    "entity_id",
    "entity",
    "confidence",
    "source",
    "clip_ref",
    "caption",
)


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
    frame: dict[str, Any] | None = None

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
            "frame": self.frame,
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
        read_only: bool = False,
    ) -> None:
        self.db_path = str(db_path)
        self.read_only = bool(read_only)
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

        if reset and read_only:
            raise ValueError("read_only and reset are mutually exclusive")
        if reset:
            _remove_database(self.db_path)

        self.memory = ChronicleMemory(
            self.db_path,
            thresholds=self.thresholds,
            bus=self.bus,
            furniture_labels=self.furniture_labels,
            furniture_config=self.furniture_config,
            load=load,
            read_only=read_only,
        )

        self._clip_queue = VerificationQueue(
            max_pending=(
                self._semantic_config.verification_queue_size
                if self._semantic_config is not None
                else 8
            ),
            policy=(
                self._semantic_config.verification_queue_policy
                if self._semantic_config is not None
                else "drop_oldest"
            ),
            dedup_window=(
                self._semantic_config.verification_dedup_window
                if self._semantic_config is not None
                else 1024
            ),
            on_event=self._on_verification_event,
        )
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
        on_frame: Callable[[FrameResult], None] | None = None,
        reset: bool = False,
    ) -> RuntimeSummary:
        """Ingest a recorded video through A -> B -> C.

        ``reset=True`` recreates the in-memory/SQLite memory first.  The
        dashboard passes ``reset=True`` when replaying a source so repeated
        runs do not accumulate duplicate episodes.  ``on_frame`` observes every
        post-memory ``FrameResult`` (used by the Role D event exporter).
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
                if on_frame is not None:
                    try:
                        on_frame(result)
                    except Exception:
                        logger.exception("on_frame callback failed")
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
        self._clip_queue.submit(clip)

    def _drain_clips(self) -> None:
        while True:
            item = self._clip_queue.get_nowait()
            if item is None:
                return
            clip, submitted_at = item
            try:
                ok = self._verify_clip(clip)
            except Exception:  # defensive: _verify_clip should catch its own errors
                logger.exception("clip verification crashed for %s", clip.clip_id)
                ok = False
            latency = max(0.0, time.monotonic() - submitted_at)
            if ok:
                self._clip_queue.mark_verified(latency)
            else:
                self._clip_queue.mark_failed(latency)

    def _verify_clip(self, clip: InteractionClip) -> bool:
        engine = self._ensure_semantic()
        if engine is None:
            return False
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
                return True
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
                return True
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
            return True
        except Exception as exc:  # a semantic failure must never stop ingestion
            self.bus.vlm(
                "clip verification failed",
                clip.contact_ts,
                clip=clip.clip_id,
                error=repr(exc),
            )
            logger.exception("clip verification failed for %s", clip.clip_id)
            return False

    # ------------------------------------------------------------------ queries
    def ask(self, query: str) -> RuntimeAnswer:
        """Answer a question and attach the evidence frame that supports it."""

        answer = self._ask_impl(query)
        if answer.frame is None:
            frame = self._answer_frame(answer)
            if frame is not None:
                answer = replace(answer, frame=frame)
        return answer

    def _colour_inventory_answer(
        self, query: str, started: float
    ) -> RuntimeAnswer | None:
        """Answer "what red things did you see?" from appearance attributes."""

        lower = query.lower()
        if re.search(r"\bcolou?rs?\b", lower):
            return None
        if not re.search(r"\b(what|which|list|show|find|any|all|do i have)\b", lower):
            return None
        colour = next(
            (
                candidate
                for candidate in COLOUR_WORDS
                if re.search(rf"\b{re.escape(candidate)}\b", lower)
            ),
            None,
        )
        if colour is None:
            return None
        if colour == "gray":
            colour = "grey"
        matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for entity in self.memory.foreground.entities():
            appearance = public_attributes(entity.get("attrs") or {})
            if appearance.get("colour") == colour:
                matches.append((entity, appearance))
        if not matches:
            return RuntimeAnswer(
                query=query,
                answer=f"I don't remember any {colour} objects in this episode.",
                source="graph",
                grounded=False,
                method="colour-filter",
                elapsed=time.monotonic() - started,
            )
        matches.sort(
            key=lambda item: float(item[0].get("last_seen") or 0.0), reverse=True
        )
        described = ", ".join(
            f"the {entity.get('label', 'object')}"
            + (f" ({attribute_summary(appearance)})" if attribute_summary(appearance) else "")
            for entity, appearance in matches[:6]
        )
        newest_t = float(matches[0][0].get("last_seen") or 0.0)
        suffix = f" Last seen at [t={newest_t:.1f}s]."
        citations = tuple(
            {
                "timestamp": float(entity.get("last_seen") or 0.0),
                "relation": "APPEARANCE",
                "zone": self.memory.background.zone_label(
                    str(entity.get("zone_id") or self.memory.active_zone or "")
                ),
            }
            for entity, _ in matches[:4]
        )
        first = matches[0][0]
        return RuntimeAnswer(
            query=query,
            answer=(
                f"I remember {len(matches)} {colour} "
                f"{'object' if len(matches) == 1 else 'objects'}: {described}."
                f"{suffix}"
            ),
            citations=citations,
            source="graph",
            grounded=True,
            method="colour-filter",
            elapsed=time.monotonic() - started,
            entity=str(first.get("label") or ""),
            state=str(first.get("state") or ""),
            zone=self.memory.background.zone_label(str(first.get("zone_id") or "")),
            last_seen=newest_t,
        )

    def _count_answer(self, query: str, started: float) -> RuntimeAnswer | None:
        """Answer explicit object-count questions from tracked entities."""

        match = re.search(
            r"\bhow many\s+(.+?)(?:\s+(?:are|is|were|was|do|does|did|can|have|has)\b|\?|$)",
            query,
            re.IGNORECASE,
        )
        if match is None:
            match = re.search(
                r"\b(?:count|number of)\s+(?:the\s+)?(.+?)(?:\?|$)",
                query,
                re.IGNORECASE,
            )
        if match is None:
            return None
        phrase = match.group(1).strip()
        tokens = content_tokens(phrase)
        if not tokens or tokens == {"time"}:
            return None

        generic = tokens <= {"object", "objects", "thing", "things", "item", "items"}
        query_variants = {
            variant for token in tokens for variant in expand_token(token)
        }
        matches = []
        for entity in self.memory.foreground.entities():
            label = str(entity.get("label") or "")
            label_tokens = content_tokens(label)
            label_variants = {
                variant
                for token in label_tokens
                for variant in expand_token(token)
            }
            if generic or (
                label_tokens
                and (
                    label_tokens == tokens
                    or label_tokens <= tokens
                    or tokens <= label_tokens
                    or query_variants & label_variants
                )
            ):
                matches.append(entity)
        if not matches:
            return RuntimeAnswer(
                query=query,
                answer=f"I don't have any {phrase} in my memory.",
                source="graph",
                grounded=False,
                method="graph-count",
                elapsed=time.monotonic() - started,
            )

        matches.sort(key=lambda entity: float(entity.get("last_seen", 0.0) or 0.0))
        latest = matches[-1]
        count = len(matches)
        noun = "object" if generic else singular(str(latest.get("label") or phrase))
        plural = "" if count == 1 else "s"
        zone = self.memory.background.zone_label(
            str(latest.get("zone_id") or self.memory.active_zone or "")
        )
        last_seen = float(latest.get("last_seen", 0.0) or 0.0)
        return RuntimeAnswer(
            query=query,
            answer=f"There {'is' if count == 1 else 'are'} {count} {noun}{plural} in this memory.",
            citations=(
                {
                    "timestamp": last_seen,
                    "relation": "COUNT",
                    "zone": zone,
                },
            ),
            source="graph",
            grounded=True,
            method="graph-count",
            elapsed=time.monotonic() - started,
            entity=noun,
            state=str(latest.get("state") or ""),
            zone=zone,
            last_seen=last_seen,
        )

    def _event_overview(self, query: str, started: float) -> RuntimeAnswer | None:
        """Summarize the most recent recorded interactions for a generic event query."""

        entities = sorted(
            self.memory.foreground.entities(),
            key=lambda entity: float(entity.get("last_seen", 0.0) or 0.0),
            reverse=True,
        )
        summaries: list[str] = []
        results: list[QueryResult] = []
        for entity in entities:
            result = self.memory.query_entity(
                str(entity.get("node_id", "")),
                question=query,
            )
            summary = result.event_summary() if result is not None else None
            if not summary or summary in summaries:
                continue
            if "picked up" not in summary and "interacted with" not in summary:
                continue
            summaries.append(summary)
            results.append(result)
            if len(summaries) == 3:
                break
        if not results:
            return None
        first = results[0]
        return RuntimeAnswer(
            query=query,
            answer=" ".join(summaries),
            citations=tuple(
                {
                    "timestamp": result.last_seen,
                    "relation": "EVENT",
                    "zone": result.zone_label,
                }
                for result in results
            ),
            source="graph",
            grounded=True,
            method="graph-template-event",
            elapsed=time.monotonic() - started,
            entity=first.label,
            state=first.state,
            zone=first.zone_label,
            last_seen=first.last_seen,
        )

    def _ask_impl(self, query: str) -> RuntimeAnswer:
        started = time.monotonic()
        colour_answer = self._colour_inventory_answer(query, started)
        if colour_answer is not None:
            return colour_answer
        count_answer = self._count_answer(query, started)
        if count_answer is not None:
            return count_answer
        result = self.memory.query(query)
        if result is None:
            if is_event_query(query):
                event_answer = self._event_overview(query, started)
                if event_answer is not None:
                    return event_answer
            # Check for inventory / overview queries like "what did I see?", "what was tracked?"
            q_lower = query.lower()
            all_entities = list(self.memory.foreground.entities())
            if all_entities and re.search(
                r"\b(what did i see|what did i observe|what was tracked|summary|overview|what objects|inventory)\b",
                q_lower,
            ):
                counts: dict[str, int] = {}
                for e in all_entities:
                    label = str(e.get("label", "object"))
                    counts[label] = counts.get(label, 0) + 1
                items_str = ", ".join(f"{count} {label}{'s' if count > 1 else ''}" for label, count in counts.items())
                zone = (
                    self.memory.background.zone_label(self.memory.active_zone)
                    or "the scene"
                )
                engine = self._ensure_semantic()
                if engine and engine.chat_client:
                    facts = [
                        Fact(
                            subject=str(e.get("label", "entity")),
                            relation="OBSERVED_IN",
                            object_label=zone,
                            timestamp=float(e.get("last_seen", 0.0) or 0.0),
                            zone=zone,
                        )
                        for e in all_entities
                    ]
                    inv_context = GraphContext(
                        entity_id="inventory",
                        entity_label="scene objects",
                        zone=zone,
                        facts=tuple(facts),
                    )
                    ans = engine.answer_query(query, inv_context)
                    return RuntimeAnswer(
                        query=query,
                        answer=ans.answer,
                        citations=tuple(c.to_dict() for c in ans.citations),
                        source=ans.source,
                        grounded=True,
                        method=ans.source,
                        elapsed=time.monotonic() - started,
                        zone=zone,
                    )
                return RuntimeAnswer(
                    query=query,
                    answer=f"The episodic memory tracked {len(all_entities)} entities across {zone}: {items_str}.",
                    source="graph",
                    grounded=True,
                    method="inventory",
                    elapsed=time.monotonic() - started,
                    zone=zone,
                )

            if all_entities and re.search(r"\b(who|who's|anyone)\b", q_lower):
                people = [e for e in all_entities if str(e.get("label", "")).lower() in {"person", "people"}]
                if people:
                    best_person = max(people, key=lambda e: float(e.get("last_seen", 0.0) or 0.0))
                    result = self.memory.query_entity(str(best_person.get("node_id", "")), question=query)

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
            citations: tuple[dict[str, Any], ...] = ()
            if attribute_query_field(query) and result.attributes:
                citations = (
                    {
                        "timestamp": result.last_seen,
                        "relation": "APPEARANCE",
                        "zone": result.zone_label,
                    },
                )
            return RuntimeAnswer(
                query=query,
                answer=answer,
                citations=citations,
                source="graph",
                grounded=True,
                method="graph-template",
                elapsed=time.monotonic() - started,
                entity=result.label,
                state=result.state,
                zone=result.zone_label,
                last_seen=result.last_seen,
            )
        if engine.chat_client is None and is_event_query(query):
            summary = result.event_summary()
            if summary:
                return RuntimeAnswer(
                    query=query,
                    answer=summary,
                    source="graph",
                    grounded=True,
                    method="graph-template-event",
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

    def _answer_frame(self, answer: RuntimeAnswer) -> dict[str, Any] | None:
        """Prefer stored evidence images, then extract the video frame."""

        if not answer.answer or not answer.grounded:
            return None
        with self._lock:
            video = self._video
            playhead = float(self._last_timestamp or 0.0)

        # The evidence folder sits beside the episode SQLite file.  A hit here
        # is cheaper than decoding the video and keeps working without it.
        stored = load_stored_evidence(Path(self.db_path).expanduser().parent)

        anchor = latest_citation_time(answer.citations, playhead if playhead > 0 else 1e9)
        if anchor is None:
            anchor = answer.last_seen
        if anchor is None and stored:
            # Inventory answers ("what did I see?") have no citation; show the
            # newest stored evidence instead of returning no frame at all.
            anchor = max(row.t for row in stored)
        if anchor is None:
            return None
        target = max(0.0, float(anchor))
        if playhead > 0:
            target = min(target, playhead)
        # A loaded DB has no live playhead; rank against all stored evidence and
        # let the citation anchor + label decide.  A live run still protects the
        # "never show future evidence" rule.
        selection_playhead = playhead if playhead > 0 else float("inf")

        citation_entities = {
            str(citation["entity_id"])
            for citation in answer.citations
            if isinstance(citation, dict) and citation.get("entity_id")
        }
        citation_labels = {
            str(citation[key])
            for citation in answer.citations
            if isinstance(citation, dict)
            for key in ("entity", "label", "action")
            if citation.get(key)
        }
        if answer.entity:
            citation_labels.add(str(answer.entity))
        route = "entity" if answer.entity else ("zone" if answer.zone else "timeline")

        if stored:
            picked = pick_stored_evidence(
                stored,
                playhead=selection_playhead,
                anchor_t=target,
                matched_label=answer.entity,
                citation_entities=citation_entities,
                citation_labels=citation_labels,
                route=route,
                matched_zone=answer.zone,
            )
            if picked is not None:
                return picked.to_dict()

        if not video:
            return None
        extracted = extract_video_frame_data_url(video, target)
        if extracted is None:
            return None
        data_url, actual_t = extracted
        return FrameCandidate(
            url=data_url,
            t=actual_t,
            kind="video_frame",
            zone_id=answer.zone,
            label=answer.entity or "",
        ).to_dict()

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
                **dict(result.attributes),
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
        verification_stats = self.verification_stats
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
            "verification_stats": verification_stats,
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
                "verification_stats": verification_stats,
                "stage": "running" if running else ("error" if error else "complete"),
                "verification_queue_depth": self._clip_queue.pending,
                "verification_dropped": self._clip_queue.stats().dropped,
                "verification_verified": self._clip_queue.stats().verified,
            },
            "verification": self.verification_stats,
            "summary": summary,
            "running": running,
            "error": error,
            "video": video,
            "zone": self.memory.background.zone_label(self.memory.active_zone),
            "backends": self.backends(),
        }

    # ------------------------------------------------------------------- exports
    def export_snapshot_json(self, *, indent: int | None = 2) -> str:
        """Return the current dashboard snapshot as JSON text.

        The payload is the same dictionary served by ``snapshot()`` so the
        dashboard, CLI and HTTP export cannot drift apart.  Unknown object types
        fall back to ``str`` to keep the export usable when a backend surfaces a
        non-serializable thumbnail or numpy scalar.
        """

        return json.dumps(
            self.snapshot(),
            indent=indent,
            ensure_ascii=False,
            default=str,
        )

    def export_entities_csv(self) -> str:
        """Return the entity table as CSV text with a stable column order."""

        return _rows_to_csv(self._entity_rows(), _ENTITY_CSV_COLUMNS)

    def export_actions_csv(self) -> str:
        """Return the recent action edges as CSV text with a stable column order."""

        return _rows_to_csv(
            self.memory.foreground.action_rows(limit=1000),
            _ACTION_CSV_COLUMNS,
        )

    def load_snapshot_json(self, data: str | dict[str, Any]) -> None:
        """Reload snapshot state directly into memory without re-ingesting."""
        import json

        payload = json.loads(data) if isinstance(data, str) else dict(data)
        for f in payload.get("furniture", []):
            anchor_id = str(f.get("anchor_id") or f.get("id", "")).removeprefix("anchor:")
            label = str(f.get("label", "furniture"))
            zone_id = f.get("zone_id") or "unmapped"
            t = float(f.get("created_at") or 0.0)
            self.memory.background.ensure_zone(zone_id, t, zone_id=zone_id)
            self.memory.store.upsert_zone(zone_id, zone_id, t)
            self.memory.background.ensure_furniture(anchor_id, label, zone_id, t, attrs=f)
            self.memory.store.upsert_furniture(anchor_id, label, zone_id, t, attrs=f)
        for e in payload.get("entities", []):
            entity_id = str(e.get("id") or e.get("entity_id", "")).removeprefix("entity:")
            label = str(e.get("label", "object"))
            state = str(e.get("state", "VISIBLE_STATIC"))
            zone = e.get("zone")
            if zone in {"-", ""}:
                zone = None
            first_seen = float(e.get("first_seen") or 0.0)
            last_seen = float(e.get("last_seen") or first_seen)
            appearance = public_attributes(e.get("attributes") or e.get("attrs") or {})
            if zone:
                self.memory.background.ensure_zone(zone, first_seen, zone_id=zone)
                self.memory.store.upsert_zone(zone, zone, first_seen)
            state_val = (
                ObjectState(state)
                if state in ObjectState._value2member_map_
                else ObjectState.VISIBLE_STATIC
            )
            self.memory.foreground.upsert_entity(
                entity_id,
                label,
                state=state_val,
                state_since=last_seen,
                first_seen=first_seen,
                last_seen=last_seen,
                zone_id=zone,
                attrs=appearance,
            )
            self.memory.store.upsert_entity(
                entity_id,
                label,
                state_val.value,
                last_seen,
                first_seen,
                last_seen,
                zone_id=zone,
                attrs=appearance,
            )

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
            appearance = public_attributes(attrs.get("attrs") or {})
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
                    "colour": appearance.get("colour") or "-",
                    "colour_hex": appearance.get("colour_hex") or "",
                    "attributes": appearance,
                    "appearance": attribute_summary(appearance) or "-",
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
                "attributes": "crop-attributes"
                if self.perception_config.attributes.enabled
                else "off",
                "llm": self._semantic_config.llm_backend if self._semantic_config else "template",
            }
        return {
            "detector": getattr(pipeline.detector, "name", self.perception_config.detector.backend),
            "hands": getattr(pipeline.hands, "name", self.perception_config.hands.backend),
            "embedder": getattr(pipeline.embedder, "name", self.perception_config.embedder.backend),
            "attributes": getattr(
                pipeline.attributes,
                "name",
                "crop-attributes" if self.perception_config.attributes.enabled else "off",
            ),
            "llm": self._semantic_config.llm_backend if self._semantic_config else "template",
        }

    def llm_status(self, *, refresh: bool = False) -> dict[str, Any]:
        """Best-effort Ollama status for the header chip; never blocks ingestion."""

        config = self._semantic_config or SemanticConfig()
        model = config.llm_model
        base_url = config.ollama_url.rstrip("/")
        if config.llm_backend in {"template", "none"}:
            return {
                "backend": "template",
                "model": model,
                "server": False,
                "available": False,
                "loaded": [],
                "version": None,
            }
        from chronicle.semantic.llm import probe_ollama_runtime
        return probe_ollama_runtime(base_url, model)

    # ------------------------------------------------------------------ internal
    def _ensure_semantic(self) -> SemanticEngine | None:
        if not self.semantic_enabled:
            return None
        if self._semantic is not None:
            if self._semantic.chat_client is None:
                from chronicle.semantic.llm import build_chat_client
                client = build_chat_client(self._semantic.config)
                if client is not None:
                    self._semantic.chat_client = client
                    self._semantic.graphrag.chat_client = client
                    self._semantic.verifier.chat_client = client
            return self._semantic
        with self._lock:
            if self._semantic is not None:
                return self._semantic
            config = self._semantic_config or SemanticConfig.from_env()
            if os.getenv("CHRONICLE_LENIENT_CITATIONS") is None:
                config = config.replace(lenient_citations=True)
            from chronicle.semantic.llm import ensure_local_ollama
            ensure_local_ollama(config, wait_seconds=2.0)
            # The dashboard must be usable without model downloads.  Florence is
            # opt-in; the deterministic captioner/rule classifier is the
            # documented offline fallback.  OpenCLIP auto-degrades too.
            embedder = (
                getattr(self._pipeline, "embedder", None)
                if self._pipeline is not None
                else None
            )
            embedder_backend = os.getenv("CHRONICLE_EMBEDDER_BACKEND")
            if not embedder_backend:
                if self.perception_config and self.perception_config.embedder.backend == "color":
                    embedder_backend = "deterministic"
                else:
                    embedder_backend = "auto"
            self._semantic = SemanticEngine(
                config=config,
                captioner_backend="auto",
                embedder_backend=embedder_backend,
                embedder=embedder,
                on_event=self._on_semantic_event,
            )
            return self._semantic

    def _on_verification_event(
        self, tag: str, event: str, message: str, data: dict[str, Any]
    ) -> None:
        """Forward Role C queue metrics to the same bus the dashboard reads."""
        self.bus.emit(tag, message, None, **data)

    @property
    def verification_stats(self) -> dict[str, Any]:
        return self._clip_queue.stats().to_dict()

    def _on_semantic_event(self, payload: dict[str, Any]) -> None:
        tag = str(payload.get("tag") or "VLM")
        message = str(payload.get("message") or payload.get("event") or "semantic event")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        self.bus.emit(tag, message, None, **data)

    def _on_perception_event(self, event: PerceptionTelemetryEvent) -> None:
        tag = _TAG_BY_CATEGORY.get(str(event.category.value), TelemetryTag.GEOM)
        # Clip/semantic payloads may already carry a ``frame_index`` key.  Merge
        # explicitly so the bus never receives the keyword twice.
        data = dict(event.data)
        data["frame_index"] = event.frame_index
        self.bus.emit(tag, event.message, event.presentation_time, **data)

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


def _csv_cell(value: Any) -> str:
    """Render one CSV cell, JSON-encoding nested evidence instead of dropping it."""

    if value is None:
        return ""
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False, default=str)
    return str(value)


def _rows_to_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_csv_cell(row.get(column)) for column in columns])
    return buffer.getvalue()


def _short_anchor(anchor_id: str | None) -> str:
    if not anchor_id:
        return "-"
    return str(anchor_id).removeprefix("anchor:").removeprefix("entity:")


__all__ = ["ChronicleRuntime", "RuntimeAnswer", "RuntimeSummary"]

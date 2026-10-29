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
from chronicle.text import content_tokens, expand_token, is_plural, singular
from chronicle.types import (
    FrameResult,
    InteractionClip,
    ObjectState,
    Relation,
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

_EVENT_TARGET_RE = re.compile(
    r"\b(?:what\s+(?:happened|occurred)\s+to"
    r"|what\s+did\s+i\s+do\s+(?:with|to))\s+"
    r"(?:(?:the|my|our|your|a|an)\s+)?([a-z][a-z0-9_-]*)",
    re.IGNORECASE,
)
_EVENT_PRONOUN_TARGETS = {
    "it",
    "them",
    "that",
    "this",
    "those",
    "these",
    "everything",
    "anything",
}


def _event_target(query: str) -> str | None:
    """Return the named object in an event question, if there is one."""

    match = _EVENT_TARGET_RE.search(query)
    if match is None:
        return None
    target = match.group(1).lower()
    return None if target in _EVENT_PRONOUN_TARGETS else target


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


_LAST_SEEN_RE = re.compile(
    r"\b(?:when|what time)\b.*\b(?:last\s+)?(?:see|saw|seen|observed|spotted)\b",
    re.IGNORECASE,
)


def _zone_phrase(zone: str) -> str:
    zone = zone.strip()
    if not zone:
        return "the scene"
    if zone.lower().startswith("the "):
        return zone
    return f"the {zone}"


def _plural_noun(label: str) -> str:
    """Render a detector label as a plural noun without doubling plurals."""

    word = label.strip()
    if not word or is_plural(word):
        return word
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return f"{word}es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    return f"{word}s"


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
        interaction_only = bool(
            re.search(
                r"\b(?:touch(?:ed)?|interact(?:ed)?|handle(?:d)?|"
                r"pick(?:ed)?\s+up|put\s+down|place(?:d)?|set\s+down|"
                r"move(?:d)?)\b",
                query,
                re.IGNORECASE,
            )
        )
        placement_only = bool(
            re.search(
                r"\b(?:put\s+down|place(?:d)?|set\s+down|move(?:d)?)\b",
                query,
                re.IGNORECASE,
            )
        )
        surface_match = re.search(
            r"\b(on|in|at|to)\s+(?:(?:the|my|our)\s+)?"
            r"([a-z][a-z0-9 _-]*?)\s*\??$",
            query,
            re.IGNORECASE,
        )
        surface_tokens: set[str] = set()
        surface_phrase = ""
        preposition = "in"
        if surface_match is not None:
            preposition = surface_match.group(1).lower()
            surface_phrase = surface_match.group(2).strip()
            surface_tokens = content_tokens(surface_phrase)
        query_variants = {
            variant for token in tokens for variant in expand_token(token)
        }
        matches = []
        for entity in self.memory.foreground.entities():
            result = None
            if interaction_only or surface_tokens:
                result = self.memory.query_entity(
                    str(entity.get("node_id", "")), question=query
                )
                if interaction_only and (
                    result is None or result.last_interaction is None
                ):
                    continue
                if placement_only:
                    anchor = (result.terminal_anchor if result is not None else None) or {}
                    anchor_t = anchor.get("t_start")
                    interaction_t = (
                        (result.last_interaction or {}).get("t")
                        if result is not None
                        else None
                    )
                    if (
                        anchor_t is None
                        or interaction_t is None
                        or float(interaction_t) >= float(anchor_t)
                    ):
                        continue
                if surface_tokens:
                    anchor = (result.terminal_anchor if result is not None else None) or {}
                    anchor_label = str(
                        anchor.get("label") or anchor.get("anchor_id") or ""
                    )
                    zone_label = str(result.zone_label or "") if result is not None else ""
                    candidate_tokens = content_tokens(zone_label) | content_tokens(
                        anchor_label
                    )
                    if not (
                        surface_tokens <= candidate_tokens
                        or candidate_tokens <= surface_tokens
                    ):
                        continue
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
            where = (
                f" {preposition} the {surface_phrase}"
                if surface_tokens
                else " in my memory"
            )
            if interaction_only:
                return RuntimeAnswer(
                    query=query,
                    answer=(
                        f"I don't have any recorded interactions with {phrase}"
                        f"{where}."
                    ),
                    source="graph",
                    grounded=False,
                    method="graph-count",
                    elapsed=time.monotonic() - started,
                )
            return RuntimeAnswer(
                query=query,
                answer=f"I don't have any {phrase}{where}.",
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
        place = (
            f" {preposition} the {surface_phrase}"
            if surface_tokens
            else " in this memory"
        )
        if interaction_only:
            if re.search(
                r"\b(?:put\s+down|set\s+down)\b", query, re.IGNORECASE
            ):
                count_verb = "put down"
            elif re.search(r"\bplace(?:d)?\b", query, re.IGNORECASE):
                count_verb = "placed"
            elif re.search(r"\bpick(?:ed)?\s+up\b", query, re.IGNORECASE):
                count_verb = "picked up"
            elif re.search(r"\bmove(?:d)?\b", query, re.IGNORECASE):
                count_verb = "moved"
            else:
                count_verb = "interacted with"
            answer = f"You {count_verb} {count} {noun}{plural}{place}."
        else:
            answer = (
                f"There {'is' if count == 1 else 'are'} {count} "
                f"{noun}{plural}{place}."
            )
        return RuntimeAnswer(
            query=query,
            answer=answer,
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

    def _surface_inventory_answer(
        self, query: str, started: float
    ) -> RuntimeAnswer | None:
        """Answer "what is on the kitchen counter?" from terminal anchors."""

        match = re.search(
            r"\b(?:what(?:'s| is)|what objects? (?:are|were)|"
            r"which objects? (?:are|were)|list|show|name)\b[^?]*?\bon\s+"
            r"(?:(?:the|my|our)\s+)?([a-z][a-z0-9 _-]*?)\s*\??$",
            query,
            re.IGNORECASE,
        )
        if match is None:
            return None
        surface = match.group(1).strip()
        surface_tokens = content_tokens(surface)
        if not surface_tokens:
            return None

        found: list[tuple[str, str, float, float, str, str]] = []
        for entity in self.memory.foreground.entities():
            result = self.memory.query_entity(
                str(entity.get("node_id", "")), question=query
            )
            anchor = (result.terminal_anchor if result is not None else None) or {}
            anchor_label = str(anchor.get("label") or anchor.get("anchor_id") or "")
            anchor_tokens = content_tokens(anchor_label)
            if not anchor_tokens or not (
                surface_tokens <= anchor_tokens or anchor_tokens <= surface_tokens
            ):
                continue
            last_seen = float(entity.get("last_seen", 0.0) or 0.0)
            placed_at = float(
                anchor.get("t_start")
                if anchor.get("t_start") is not None
                else last_seen
            )
            found.append(
                (
                    str(entity.get("label") or "object"),
                    anchor_label,
                    last_seen,
                    placed_at,
                    str(result.zone_label or "") if result is not None else "",
                    str(result.state or "") if result is not None else "",
                )
            )
        if not found:
            return RuntimeAnswer(
                query=query,
                answer=f"I don't see anything on the {surface} in this memory.",
                source="graph",
                grounded=False,
                method="graph-inventory",
                elapsed=time.monotonic() - started,
            )

        found.sort(key=lambda item: item[2])
        labels = [item[0] for item in found]
        if len(labels) == 1:
            listed = f"the {labels[0]}"
        else:
            listed = (
                ", ".join(f"the {label}" for label in labels[:-1])
                + f" and the {labels[-1]}"
            )
        newest = found[-1]
        zone = newest[4] or self.memory.background.zone_label(
            str(self.memory.active_zone or "")
        )
        return RuntimeAnswer(
            query=query,
            answer=(
                f"On the {newest[1]} I can see {listed}. "
                f"Last seen at t={newest[2]:.1f}s."
            ),
            citations=tuple(
                {
                    "timestamp": item[2],
                    "placed_at": item[3],
                    "relation": "RESTING_ON",
                    "zone": item[4] or zone,
                }
                for item in found
            ),
            source="graph",
            grounded=True,
            method="graph-inventory",
            elapsed=time.monotonic() - started,
            entity=newest[0],
            state=newest[5],
            zone=zone,
            last_seen=newest[2],
        )

    def _action_sequence_answer(
        self, query: str, started: float
    ) -> RuntimeAnswer | None:
        """Answer "what did I do before/after I <action> <object>?".

        The generic temporal path resolves the referenced object itself and
        reports that object's earlier state, which reads as a non-answer
        ("before the keys, the keys were ...").  This handler builds a small
        ordered action timeline from each entity's latest pickup and terminal
        placement, then filters it around the referenced action.
        """

        query_text = query.lower()
        # Particle-separated references first: "picked the cup up",
        # "picking the cup up", "set the cup down".  Otherwise the
        # contiguous pattern would swallow the trailing particle into the
        # object label.
        match = re.search(
            r"\b(?:what happened|what did i do)\s+(before|after)\s+"
            r"(?:i\s+)?"
            r"(picked|picking|pick|grabbed|grabbing|put|putting|"
            r"set|setting|placed|placing|place|moved|moving|move|"
            r"touched|touching|touch)\s+"
            r"(?:(?:the|my|our)\s+)?([a-z][a-z0-9 _-]*?)\s+"
            r"(up|down)\s*\??$",
            query_text,
        )
        if match is not None:
            direction, verb, reference_label, particle = match.groups()
            verb_phrase = f"{verb} {particle}"
        else:
            match = re.search(
                r"\b(?:what happened|what did i do)\s+(before|after)\s+"
                r"(?:i\s+)?"
                r"(picked\s*up|picking\s*up|pick\s*up|grabbed|grabbing|"
                r"put\s*down|putting\s*down|set\s*down|setting\s*down|put|"
                r"placed|placing|place|moved|moving|move|"
                r"touched|touching|touch)\s+(?:(?:the|my|our)\s+)?"
                r"([a-z][a-z0-9 _-]*?)\s*\??$",
                query_text,
            )
            if match is None:
                return None
            direction, verb_phrase, reference_label = match.groups()
        verb_phrase = re.sub(r"\s+", " ", verb_phrase)
        if verb_phrase.startswith(("pick", "grab")):
            reference_action = "pickup"
        elif verb_phrase.startswith(("put", "plac", "set", "mov")):
            reference_action = "place"
        else:
            reference_action = "interaction"
        if reference_action == "pickup":
            ref_verb = "picked up"
        elif reference_action == "place":
            if "mov" in verb_phrase:
                ref_verb = "moved"
            elif "plac" in verb_phrase:
                ref_verb = "placed"
            elif "set" in verb_phrase:
                ref_verb = "set down"
            else:
                ref_verb = "put down"
        else:
            ref_verb = "touched"
        reference_tokens = content_tokens(reference_label)

        events: list[tuple[float, str, str, dict[str, Any] | None]] = []
        reference_times: list[float] = []
        for entity in self.memory.foreground.entities():
            result = self.memory.query_entity(
                str(entity.get("node_id", "")), question=query
            )
            if result is None:
                continue
            label = result.label
            interaction = result.last_interaction or {}
            interaction_t = interaction.get("t")
            interaction_kind = str(interaction.get("kind") or "")
            interaction_verb = {
                "HELD": "picked up",
                "INTERACTED": "touched",
            }.get(interaction_kind)
            if interaction_t is not None and interaction_verb:
                events.append(
                    (float(interaction_t), label, interaction_verb, None)
                )
                if reference_tokens <= content_tokens(label):
                    if reference_action == "pickup" and interaction_verb == "picked up":
                        reference_times.append(float(interaction_t))
                    elif reference_action == "interaction":
                        reference_times.append(float(interaction_t))
            anchor = result.terminal_anchor or {}
            anchor_t = anchor.get("t_start")
            # A terminal anchor at t=0.5 is the object's initial resting
            # place, not an action the user performed.  Only count a
            # placement when a pickup precedes it.
            if (
                anchor_t is not None
                and interaction_t is not None
                and float(interaction_t) < float(anchor_t)
            ):
                events.append((float(anchor_t), label, "placed", anchor))
                if reference_tokens <= content_tokens(label):
                    if reference_action == "place":
                        reference_times.append(float(anchor_t))

        if not reference_times:
            return None
        boundary = max(reference_times)
        if direction == "before":
            selected = [event for event in events if event[0] < boundary]
        else:
            selected = [event for event in events if event[0] > boundary]
        selected.sort(key=lambda event: event[0])
        if not selected:
            return RuntimeAnswer(
                query=query,
                answer=(
                    f"I don't have any recorded actions {direction} you "
                    f"{ref_verb} the {reference_label} in this memory."
                ),
                source="graph",
                grounded=False,
                method="graph-action-sequence",
                elapsed=time.monotonic() - started,
            )

        phrases: list[str] = []
        for timestamp, label, kind, anchor in selected:
            if kind in {"picked up", "touched"}:
                phrases.append(f"{kind} the {label} at t={timestamp:.1f}s")
                continue
            anchor_label = str(
                (anchor or {}).get("label")
                or (anchor or {}).get("anchor_id")
                or "surface"
            )
            relation = str((anchor or {}).get("relation") or "")
            preposition = "inside" if relation.endswith("INSIDE") else "on"
            phrases.append(
                f"placed the {label} {preposition} the {anchor_label} "
                f"at t={timestamp:.1f}s"
            )
        if len(phrases) == 1:
            listed = phrases[0]
        elif len(phrases) == 2:
            listed = f"{phrases[0]} and {phrases[1]}"
        else:
            listed = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"
        return RuntimeAnswer(
            query=query,
            answer=(
                f"{direction.capitalize()} you {ref_verb} the "
                f"{reference_label}, you {listed}."
            ),
            citations=tuple(
                {"timestamp": round(timestamp, 1), "relation": "ACTION"}
                for timestamp, _, _, _ in selected
            ),
            source="graph",
            grounded=True,
            method="graph-action-sequence",
            elapsed=time.monotonic() - started,
            entity=reference_label,
            last_seen=selected[-1][0],
        )

    def _object_destination_answer(
        self, query: str, started: float
    ) -> RuntimeAnswer | None:
        """Answer "where/what did I put <object>?" for one tracked object."""

        if re.search(r"\b(?:before|after)\b", query, re.IGNORECASE):
            return None
        match = re.match(
            r"^\s*(?:what|where)\s+did\s+i\s+(?P<verb>put|place|set)\s+"
            r"(?:down\s+)?(?P<phrase>.+?)\s*(?:(?:on|in|at|to)\s*)?\?\s*$",
            query,
            re.IGNORECASE,
        )
        if match is None:
            return None
        phrase = match.group("phrase")
        if re.match(r"^(?:on|in|at|to)\b", phrase, re.IGNORECASE):
            # "what did I put on the counter?" asks for every placement, not
            # for one named object; let the action overview handle it.
            return None
        target_tokens = content_tokens(phrase)
        target_tokens.discard("down")
        if not target_tokens:
            return None
        entity = self._entity_matching_tokens(target_tokens)
        if entity is None:
            return RuntimeAnswer(
                query=query,
                answer="I don't have that in my memory.",
                source="graph",
                grounded=False,
                method="graph-object-destination",
                elapsed=time.monotonic() - started,
            )
        node_id = str(entity.get("node_id") or "")
        result = self.memory.query_entity(node_id, question=query)
        if result is None:
            return RuntimeAnswer(
                query=query,
                answer="I don't have that in my memory.",
                source="graph",
                grounded=False,
                method="graph-object-destination",
                elapsed=time.monotonic() - started,
            )
        anchor = result.terminal_anchor or {}
        anchor_t = anchor.get("t_start")
        anchor_label = str(anchor.get("label") or anchor.get("anchor_id") or "").strip()
        interaction = result.last_interaction or {}
        interaction_t = interaction.get("t")
        if (
            interaction_t is None
            or anchor_t is None
            or not anchor_label
            or float(anchor_t) <= float(interaction_t)
        ):
            return RuntimeAnswer(
                query=query,
                answer=(
                    f"I don't have a recorded placement for the "
                    f"{result.label} in this memory."
                ),
                source="graph",
                grounded=False,
                method="graph-object-destination",
                elapsed=time.monotonic() - started,
                entity=result.label,
            )
        relation = str(anchor.get("relation") or "")
        preposition = "inside" if relation.endswith("INSIDE") else "on"
        verb = {
            "put": "put",
            "place": "placed",
            "set": "set down",
        }[match.group("verb").lower()]
        timestamp = float(anchor_t)
        return RuntimeAnswer(
            query=query,
            answer=(
                f"You {verb} the {result.label} {preposition} the "
                f"{anchor_label} at t={timestamp:.1f}s."
            ),
            citations=(
                {
                    "timestamp": timestamp,
                    "relation": "PLACEMENT",
                    "zone": result.zone_label,
                },
            ),
            source="graph",
            grounded=True,
            method="graph-object-destination",
            elapsed=time.monotonic() - started,
            entity=result.label,
            state=result.state,
            zone=result.zone_label,
            last_seen=timestamp,
        )

    def _object_source_answer(self, query: str, started: float) -> RuntimeAnswer | None:
        """Answer "where/what did I pick <object> up from?"."""

        if re.search(r"\b(?:before|after)\b", query, re.IGNORECASE):
            return None
        match = re.match(
            r"^\s*(?:what|where)\s+did\s+i\s+(?P<verb>pick|grab)(?:\s+up)?\s+"
            r"(?P<phrase>.+?)\s*(?:up\s+)?(?:from\s*)?\?\s*$",
            query,
            re.IGNORECASE,
        )
        if match is None:
            return None
        target_tokens = content_tokens(match.group("phrase"))
        target_tokens.discard("up")
        target_tokens.discard("from")
        if not target_tokens:
            return None
        entity = self._entity_matching_tokens(target_tokens)
        if entity is None:
            return RuntimeAnswer(
                query=query,
                answer="I don't have that in my memory.",
                source="graph",
                grounded=False,
                method="graph-object-source",
                elapsed=time.monotonic() - started,
            )
        node_id = str(entity.get("node_id") or "")
        result = self.memory.query_entity(node_id, question=query)
        if result is None or result.last_interaction is None:
            return RuntimeAnswer(
                query=query,
                answer=(
                    f"I don't have a recorded pick-up for the "
                    f"{entity.get('label') or 'object'} in this memory."
                ),
                source="graph",
                grounded=False,
                method="graph-object-source",
                elapsed=time.monotonic() - started,
                entity=str(entity.get("label") or ""),
            )
        support = self._support_before_last_interaction(result)
        if not support:
            return RuntimeAnswer(
                query=query,
                answer=(
                    f"I don't have a recorded location for the {result.label} "
                    "before it was picked up."
                ),
                source="graph",
                grounded=False,
                method="graph-object-source",
                elapsed=time.monotonic() - started,
                entity=result.label,
            )
        timestamp = float(
            result.last_interaction.get("t", result.last_seen) or result.last_seen
        )
        verb = "grabbed" if match.group("verb").lower() == "grab" else "picked up"
        return RuntimeAnswer(
            query=query,
            answer=(
                f"You {verb} the {result.label} from the {support} "
                f"at t={timestamp:.1f}s."
            ),
            citations=(
                {
                    "timestamp": timestamp,
                    "relation": "INTERACTION",
                    "zone": result.zone_label,
                },
            ),
            source="graph",
            grounded=True,
            method="graph-object-source",
            elapsed=time.monotonic() - started,
            entity=result.label,
            state=result.state,
            zone=result.zone_label,
            last_seen=timestamp,
        )

    def _entity_matching_tokens(self, tokens: set[str]) -> dict[str, Any] | None:
        """Find the most specific foreground entity whose label contains tokens."""

        best: tuple[int, dict[str, Any]] | None = None
        for entity in self.memory.foreground.entities():
            label = str(entity.get("label") or "")
            label_tokens = content_tokens(label)
            if not label_tokens or not label_tokens <= tokens:
                continue
            score = len(label_tokens)
            if best is None or score > best[0]:
                best = (score, entity)
        return best[1] if best is not None else None

    def _action_overview(self, query: str, started: float) -> RuntimeAnswer | None:
        """Summarize which objects share a requested interaction verb."""

        query_lower = query.lower()
        prefix = (
            r"\b(?:what|which)(?:\s+(?:object|objects|thing|things|item|items))?"
            r"\s+did\s+i\s+"
        )
        action = ""
        if re.search(prefix + r"(?:pick\s*up|pickup|grab)\b", query_lower):
            action = "pickup"
        elif re.search(
            prefix + r"(?:put(?:\s*(?:down|on|in|at))?|place|set\s*down)\b",
            query_lower,
        ):
            action = "place"
        elif re.search(prefix + r"(?:touch|interact\s+with|handle)\b", query_lower):
            action = "interaction"
        elif re.search(prefix + r"(?:move|moved|carry|carried)\b", query_lower):
            action = "move"
        if not action:
            return None

        surface_match = re.search(
            r"\b(on|in|at|to|from)\s+(?:(?:the|my|our)\s+)?"
            r"([a-z][a-z0-9 _-]*?)(?=\s+(?:before|after)\b|\s*\??$)",
            query_lower,
        )
        preposition = ""
        surface_phrase = ""
        surface_tokens: set[str] = set()
        if surface_match is not None:
            preposition = surface_match.group(1)
            surface_phrase = surface_match.group(2).strip()
            surface_tokens = content_tokens(surface_phrase)

        entries: list[tuple[QueryResult, float]] = []
        for entity in self.memory.foreground.entities():
            result = self.memory.query_entity(
                str(entity.get("node_id", "")), question=query
            )
            if result is None or result.last_interaction is None:
                continue
            summary = result.event_summary()
            if action == "pickup" and (not summary or "picked up" not in summary):
                continue
            if action == "place" and (not summary or "placed" not in summary):
                continue
            if action == "move":
                anchor = result.terminal_anchor or {}
                anchor_t = anchor.get("t_start")
                interaction_t = (result.last_interaction or {}).get("t")
                if (
                    not summary
                    or "picked up" not in summary
                    or anchor_t is None
                    or interaction_t is None
                    or float(interaction_t) >= float(anchor_t)
                ):
                    continue
            if surface_tokens:
                if action in {"place", "move"}:
                    anchor = (result.terminal_anchor or {})
                    anchor_label = str(
                        anchor.get("label") or anchor.get("anchor_id") or ""
                    )
                    candidate_tokens = content_tokens(anchor_label) | content_tokens(
                        result.zone_label or ""
                    )
                    if not surface_tokens <= candidate_tokens:
                        continue
                elif not surface_tokens <= content_tokens(summary or ""):
                    continue
            interaction = result.last_interaction
            if action in {"place", "move"}:
                anchor = (result.terminal_anchor or {})
                timestamp = float(
                    anchor.get("t_start")
                    if anchor.get("t_start") is not None
                    else result.last_seen
                )
            else:
                timestamp = float(interaction.get("t", result.last_seen) or 0.0)
            entries.append((result, timestamp))
        if not entries:
            where = (
                f" {preposition} the {surface_phrase}" if surface_tokens else ""
            )
            return RuntimeAnswer(
                query=query,
                answer=f"I don't have any recorded {action} events{where} in this memory.",
                source="graph",
                grounded=False,
                method="graph-action",
                elapsed=time.monotonic() - started,
            )

        entries.sort(key=lambda item: item[1])
        reference = re.search(
            r"\b(before|after)\s+(?:the\s+|my\s+|our\s+)?"
            r"([a-z][a-z0-9 _-]*?)\s*\??$",
            query_lower,
        )
        if reference is not None:
            ref_tokens = content_tokens(reference.group(2))
            ref_times: list[float] = []
            for entity in self.memory.foreground.entities():
                label = str(entity.get("label") or "")
                if not ref_tokens <= content_tokens(label):
                    continue
                ref_result = self.memory.query_entity(
                    str(entity.get("node_id", "")), question=query
                )
                if ref_result is None or ref_result.last_interaction is None:
                    continue
                if action == "place":
                    anchor = ref_result.terminal_anchor or {}
                    ref_times.append(
                        float(
                            anchor.get("t_start")
                            if anchor.get("t_start") is not None
                            else ref_result.last_seen
                        )
                    )
                else:
                    ref_times.append(
                        float(
                            ref_result.last_interaction.get(
                                "t", ref_result.last_seen
                            )
                            or 0.0
                        )
                    )
            if ref_times:
                boundary = max(ref_times)
                if reference.group(1) == "before":
                    entries = [item for item in entries if item[1] < boundary]
                else:
                    entries = [item for item in entries if item[1] > boundary]
                if not entries:
                    return RuntimeAnswer(
                        query=query,
                        answer=(
                            f"I don't have any recorded {action} events "
                            f"{reference.group(1)} the {reference.group(2)} "
                            "in this memory."
                        ),
                        source="graph",
                        grounded=False,
                        method="graph-action",
                        elapsed=time.monotonic() - started,
                    )
        labels = [result.label for result, _ in entries]
        order_match = re.search(r"\b(first|last)\b", query_lower)
        if order_match is not None:
            chosen, timestamp = (
                entries[0] if order_match.group(1) == "first" else entries[-1]
            )
            verb = {
                "pickup": "picked up",
                "place": "put down",
                "interaction": "interacted with",
                "move": "moved",
            }[action]
            where = (
                f" {preposition} the {surface_phrase}" if surface_tokens else ""
            )
            return RuntimeAnswer(
                query=query,
                answer=(
                    f"The {order_match.group(1)} object you {verb} was the "
                    f"{chosen.label} at t={timestamp:.1f}s{where}."
                ),
                citations=(
                    {
                        "timestamp": timestamp,
                        "relation": "INTERACTION",
                        "zone": chosen.zone_label,
                    },
                ),
                source="graph",
                grounded=True,
                method="graph-action",
                elapsed=time.monotonic() - started,
                entity=chosen.label,
                state=chosen.state,
                zone=chosen.zone_label,
                last_seen=timestamp,
            )
        if len(labels) == 1:
            listed = f"the {labels[0]}"
        else:
            listed = (
                ", ".join(f"the {label}" for label in labels[:-1])
                + f" and the {labels[-1]}"
            )
        verb = {
            "pickup": "picked up",
            "place": "put down",
            "interaction": "interacted with",
            "move": "moved",
        }[action]
        first = entries[0][0]
        where = (
            f" {preposition} the {surface_phrase}" if surface_tokens else ""
        )
        return RuntimeAnswer(
            query=query,
            answer=f"You {verb} {listed}{where}.",
            citations=tuple(
                {
                    "timestamp": timestamp,
                    "relation": "INTERACTION",
                    "zone": result.zone_label,
                }
                for result, timestamp in entries
            ),
            source="graph",
            grounded=True,
            method="graph-action",
            elapsed=time.monotonic() - started,
            entity=first.label,
            state=first.state,
            zone=first.zone_label,
            last_seen=entries[-1][1],
        )

    def _support_before_last_interaction(self, result: QueryResult) -> str:
        """Return the support surface an entity occupied before its last interaction."""

        interaction = result.last_interaction or {}
        interaction_t = interaction.get("t")
        if interaction_t is None:
            return ""
        supports = [
            edge
            for edge in result.timeline()
            if edge.get("relation") in {Relation.RESTING_ON.value, Relation.INSIDE.value}
            and edge.get("t_start") is not None
            and float(edge["t_start"]) < float(interaction_t)
        ]
        if not supports:
            return ""
        edge = max(supports, key=lambda item: float(item["t_start"]))
        node_labels = {
            str(node.get("node_id")): str(node.get("label") or "")
            for node in result.nodes
        }
        return node_labels.get(str(edge.get("dst")), str(edge.get("dst") or ""))

    def _yes_no_answer(self, query: str, started: float) -> RuntimeAnswer | None:
        """Answer direct yes/no location and interaction questions."""

        location_match = re.search(
            r"^\s*(?:is|are|was|were)\s+(?:the|my|our|a|an)?\s*"
            r"(?P<target>.+?)\s+(?P<prep>on|in|at)\s+"
            r"(?:the\s+)?(?P<surface>.+?)\s*\?$",
            query,
            re.IGNORECASE,
        )
        if location_match is not None:
            target = location_match.group("target").strip()
            surface = location_match.group("surface").strip()
            before_pickup = False
            if " before " in surface.lower():
                surface_text, _, before_clause = surface.partition(" before ")
                if re.search(
                    r"\b(?:pick(?:ed)?|picked\s+up|move(?:d)?|place(?:d)?|"
                    r"set\s+down)\b",
                    before_clause,
                    re.IGNORECASE,
                ):
                    surface = surface_text.strip()
                    before_pickup = True
            surface_tokens = content_tokens(surface)
            if not surface_tokens:
                return None
            result = self.memory.query(target)
            if result is None:
                return RuntimeAnswer(
                    query=query,
                    answer="I don't have that in my memory.",
                    source="graph",
                    grounded=False,
                    method="graph-confirm",
                    elapsed=time.monotonic() - started,
                )
            anchor = (result.terminal_anchor or {})
            actual = str(anchor.get("label") or anchor.get("anchor_id") or "")
            actual_tokens = content_tokens(actual)
            matches = bool(actual_tokens) and (
                surface_tokens <= actual_tokens or actual_tokens <= surface_tokens
            )
            if before_pickup:
                actual = self._support_before_last_interaction(result)
                actual_tokens = content_tokens(actual)
                matches = bool(actual_tokens) and (
                    surface_tokens <= actual_tokens or actual_tokens <= surface_tokens
                )
            if not actual:
                actual = result.zone_label or "the scene"
                actual_tokens = content_tokens(actual)
                matches = bool(actual_tokens) and (
                    surface_tokens <= actual_tokens or actual_tokens <= surface_tokens
                )
            preposition = "on" if anchor else "in"
            plural = is_plural(result.label)
            if before_pickup:
                copula = "were" if plural else "was"
                pronoun = "they" if plural else "it"
                answer = (
                    f"{'Yes' if matches else 'No'}. The {result.label} {copula} "
                    f"{preposition} the {actual} before {pronoun} {copula} "
                    "picked up."
                )
            else:
                copula = "are" if plural else "is"
                answer = (
                    f"{'Yes' if matches else 'No'}. The {result.label} {copula} "
                    f"{preposition} the {actual}."
                )
            return RuntimeAnswer(
                query=query,
                answer=answer,
                citations=(
                    {
                        "timestamp": result.last_seen,
                        "relation": "LOCATION_CHECK",
                        "zone": result.zone_label,
                    },
                ),
                source="graph",
                grounded=True,
                method="graph-confirm",
                elapsed=time.monotonic() - started,
                entity=result.label,
                state=result.state,
                zone=result.zone_label,
                last_seen=result.last_seen,
            )

        action_match = re.search(
            r"^\s*did\s+i\s+(?P<verb>move|pick\s*up|grab|put\s*down|put|"
            r"set\s*down|place|touch|interact\s+with|handle)\s+"
            r"(?:the|my|our|a|an)?\s*(?P<target>.+?)\s*\?$",
            query,
            re.IGNORECASE,
        )
        if action_match is None:
            return None
        target = action_match.group("target").strip()
        verb_text = re.sub(r"\s+", " ", action_match.group("verb").lower())
        verb = {
            "pick up": "pickup",
            "grab": "pickup",
            "put down": "place",
            "put": "place",
            "set down": "place",
            "place": "place",
        }.get(verb_text, verb_text)
        source = None
        destination = None
        if verb == "pickup":
            source_match = re.search(
                r"\s+from\s+(?:the\s+)?(.+?)\s*$", target, re.IGNORECASE
            )
            if source_match is not None:
                source = source_match.group(1).strip()
                target = target[: source_match.start()].strip()
        elif verb in {"move", "place"}:
            both_match = re.search(
                r"\s+from\s+(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?(.+?)\s*$",
                target,
                re.IGNORECASE,
            )
            if both_match is not None:
                source = both_match.group(1).strip()
                destination = both_match.group(2).strip()
                target = target[: both_match.start()].strip()
            else:
                destination_match = re.search(
                    r"\s+(?:to|onto|on|in|at)\s+(?:the\s+)?(.+?)\s*$",
                    target,
                    re.IGNORECASE,
                )
                if destination_match is not None:
                    destination = destination_match.group(1).strip()
                    target = target[: destination_match.start()].strip()
        result = self.memory.query(target)
        if result is None:
            return RuntimeAnswer(
                query=query,
                answer="I don't have that in my memory.",
                source="graph",
                grounded=False,
                method="graph-confirm",
                elapsed=time.monotonic() - started,
            )
        if result.last_interaction is None:
            return RuntimeAnswer(
                query=query,
                answer=(
                    f"No. I don't have a recorded interaction with the "
                    f"{result.label}."
                ),
                source="graph",
                grounded=True,
                method="graph-confirm",
                elapsed=time.monotonic() - started,
                entity=result.label,
                state=result.state,
                zone=result.zone_label,
                last_seen=result.last_seen,
            )
        if source is not None:
            actual_source = self._support_before_last_interaction(result)
            source_tokens = content_tokens(source)
            source_matches = bool(actual_source) and (
                source_tokens <= content_tokens(actual_source)
                or content_tokens(actual_source) <= source_tokens
            )
            if not source_matches:
                plural = is_plural(result.label)
                was = "were" if plural else "was"
                pronoun = "they" if plural else "it"
                answer = (
                    f"No. The {result.label} {was} on the {actual_source} before "
                    f"{pronoun} {was} picked up."
                    if actual_source
                    else f"No. I don't have a recorded location for the "
                    f"{result.label} before {pronoun} {was} picked up."
                )
                return RuntimeAnswer(
                    query=query,
                    answer=answer,
                    citations=(
                        {
                            "timestamp": result.last_seen,
                            "relation": "INTERACTION_CHECK",
                            "zone": result.zone_label,
                        },
                    ),
                    source="graph",
                    grounded=True,
                    method="graph-confirm",
                    elapsed=time.monotonic() - started,
                    entity=result.label,
                    state=result.state,
                    zone=result.zone_label,
                    last_seen=result.last_seen,
                )
        if destination is not None:
            anchor = result.terminal_anchor or {}
            actual = str(anchor.get("label") or anchor.get("anchor_id") or "")
            actual_tokens = content_tokens(actual) | content_tokens(
                result.zone_label or ""
            )
            destination_tokens = content_tokens(destination)
            matches = bool(actual_tokens) and (
                destination_tokens <= actual_tokens
                or actual_tokens <= destination_tokens
            )
            if not matches:
                copula = "are" if is_plural(result.label) else "is"
                answer = (
                    f"No. The {result.label} {copula} on the {actual}."
                    if actual
                    else f"No. I don't have a recorded placement for the "
                    f"{result.label}."
                )
                return RuntimeAnswer(
                    query=query,
                    answer=answer,
                    citations=(
                        {
                            "timestamp": result.last_seen,
                            "relation": "INTERACTION_CHECK",
                            "zone": result.zone_label,
                        },
                    ),
                    source="graph",
                    grounded=True,
                    method="graph-confirm",
                    elapsed=time.monotonic() - started,
                    entity=result.label,
                    state=result.state,
                    zone=result.zone_label,
                    last_seen=result.last_seen,
                )
        summary = result.event_summary()
        return RuntimeAnswer(
            query=query,
            answer=f"Yes. {summary}" if summary else "Yes.",
            citations=(
                {
                    "timestamp": result.last_seen,
                    "relation": "INTERACTION_CHECK",
                    "zone": result.zone_label,
                },
            ),
            source="graph",
            grounded=True,
            method="graph-confirm",
            elapsed=time.monotonic() - started,
            entity=result.label,
            state=result.state,
            zone=result.zone_label,
            last_seen=result.last_seen,
        )

    def _event_overview(self, query: str, started: float) -> RuntimeAnswer | None:
        """Summarize the most recent recorded interactions for a generic event query."""

        order = None
        if re.search(r"\b(?:first|earliest)\b", query, re.IGNORECASE):
            order = "first"
        elif re.search(r"\b(?:last|latest|most recent(?:ly)?)\b", query, re.IGNORECASE):
            order = "last"

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
            if len(summaries) == 3 and order is None:
                break
        if not results:
            return None

        def event_time(result: QueryResult) -> float:
            interaction_times = [
                float(edge["t_start"])
                for edge in result.timeline()
                if edge.get("t_start") is not None
                and (
                    edge.get("relation") == Relation.ACTION.value
                    or (
                        edge.get("relation") == Relation.STATE_TRANSITION.value
                        and edge.get("to_state") in {"HELD", "INTERACTED"}
                    )
                )
            ]
            if interaction_times:
                return max(interaction_times)
            times = [
                float(edge["t_start"])
                for edge in result.timeline()
                if edge.get("t_start") is not None
            ]
            return max(times) if times else float(result.last_seen)

        if order is not None:
            keep = (
                min(range(len(results)), key=lambda index: event_time(results[index]))
                if order == "first"
                else max(range(len(results)), key=lambda index: event_time(results[index]))
            )
            summaries = [summaries[keep]]
            results = [results[keep]]
        else:
            # A generic "what happened?" reads chronologically.  Selection
            # above used last-seen order to pick the most recent entities;
            # sort the chosen summaries by their event time for the answer.
            ordered = sorted(
                zip(results, summaries, strict=False),
                key=lambda pair: event_time(pair[0]),
            )
            results = [pair[0] for pair in ordered]
            summaries = [pair[1] for pair in ordered]

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
        sequence_answer = self._action_sequence_answer(query, started)
        if sequence_answer is not None:
            return sequence_answer
        destination_answer = self._object_destination_answer(query, started)
        if destination_answer is not None:
            return destination_answer
        source_answer = self._object_source_answer(query, started)
        if source_answer is not None:
            return source_answer
        action_answer = self._action_overview(query, started)
        if action_answer is not None:
            return action_answer
        confirm_answer = self._yes_no_answer(query, started)
        if confirm_answer is not None:
            return confirm_answer
        surface_answer = self._surface_inventory_answer(query, started)
        if surface_answer is not None:
            return surface_answer
        result = self.memory.query(query)
        if result is not None and _LAST_SEEN_RE.search(query):
            first = bool(re.search(r"\bfirst\b", query, re.IGNORECASE))
            if first:
                attrs = self.memory.foreground.get_entity(result.entity_id) or {}
                first_seen = attrs.get("first_seen")
                timestamp = float(
                    result.last_seen if first_seen is None else first_seen
                )
                return RuntimeAnswer(
                    query=query,
                    answer=f"You first saw the {result.label} at t={timestamp:.1f}s.",
                    citations=(
                        {"timestamp": timestamp, "relation": "FIRST_SEEN"},
                    ),
                    source="graph",
                    grounded=True,
                    method="graph-first-seen",
                    elapsed=time.monotonic() - started,
                    entity=result.label,
                    state=result.state,
                    zone=result.zone_label,
                    last_seen=timestamp,
                )
            zone = result.zone_label or self.memory.background.zone_label(
                str(result.zone_id or "")
            ) or ""
            zone_phrase = _zone_phrase(zone)
            return RuntimeAnswer(
                query=query,
                answer=(
                    f"You last saw the {result.label} at t={result.last_seen:.1f}s "
                    f"in {zone_phrase}."
                ),
                citations=(
                    {
                        "timestamp": result.last_seen,
                        "relation": "LAST_SEEN",
                        "zone": zone,
                    },
                ),
                source="graph",
                grounded=True,
                method="graph-last-seen",
                elapsed=time.monotonic() - started,
                entity=result.label,
                state=result.state,
                zone=zone,
                last_seen=result.last_seen,
            )
        if result is None:
            if is_event_query(query):
                # "what happened to the mug?" names a specific object.  If
                # lookup failed, answer honestly instead of falling back to an
                # overview of unrelated tracked objects.
                if _event_target(query) is not None:
                    return RuntimeAnswer(
                        query=query,
                        answer="I don't have that in my memory.",
                        source="graph",
                        grounded=False,
                        elapsed=time.monotonic() - started,
                    )
                event_answer = self._event_overview(query, started)
                if event_answer is not None:
                    return event_answer
            # Check for inventory / overview queries like "what did I see?", "what was tracked?"
            q_lower = query.lower()
            all_entities = list(self.memory.foreground.entities())
            if all_entities and re.search(
                r"\b(what did i see|what did i observe|what was tracked|summary|overview|"
                r"what objects|what things|which objects|which things|inventory)\b",
                q_lower,
            ):
                counts: dict[str, int] = {}
                for e in all_entities:
                    label = str(e.get("label", "object"))
                    counts[label] = counts.get(label, 0) + 1
                items_str = ", ".join(
                    f"{count} {singular(label) if count == 1 else _plural_noun(label)}"
                    for label, count in counts.items()
                )
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
                entity_word = "entity" if len(all_entities) == 1 else "entities"
                return RuntimeAnswer(
                    query=query,
                    answer=(
                        f"The episodic memory tracked {len(all_entities)} "
                        f"{entity_word} across the scene: {items_str}."
                    ),
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
        # The deterministic backends answer entirely from the graph.  Do not
        # construct the semantic engine just to discover that it has no chat
        # client: model imports dominate CLI latency on the demo memory.
        template_only = bool(
            self._semantic_config
            and self._semantic_config.llm_backend in {"template", "none"}
        )
        engine = None if template_only else self._ensure_semantic()
        if (engine is None or engine.chat_client is None) and is_event_query(query):
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
                method="template" if template_only else "graph-template",
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

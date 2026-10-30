"""Label-free visual capsules for late semantic binding.

SPORE separates *perception* from *naming*.  Ingest stores compact, label-free
evidence about things that look interesting (hand contact, visual change,
salient patches); a query later decides which of those capsules deserve an
open-vocabulary detector call and what label they should receive.

The classes in this module are deliberately model-free: they store crops,
embeddings and scores, never a class name.  ``label`` is an *attachment* made by
the query-time binder, not a precondition for creating a capsule.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chronicle.types import BBox


def _normalise(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return np.zeros_like(array)
    return array / norm


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _pack_vector(vector: np.ndarray | None) -> str | None:
    """Pack a float32 vector as base64 float16 (half the JSON size)."""
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float16).reshape(-1)
    return base64.b64encode(array.tobytes()).decode("ascii")


def _unpack_vector(payload: str | None, dimension: int | None = None) -> np.ndarray | None:
    if not payload:
        return None
    raw = base64.b64decode(payload.encode("ascii"))
    array = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dimension is not None and array.size != dimension:
        return None
    return array


def _encode_jpeg(image: np.ndarray | None, *, quality: int = 82) -> str | None:
    if image is None or image.size == 0:
        return None
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _decode_jpeg(payload: str | None) -> np.ndarray | None:
    if not payload:
        return None
    raw = base64.b64decode(payload.encode("ascii"))
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return image


def _externalize_images(payload: dict[str, Any], evidence_dir: Path) -> None:
    """Move base64 JPEGs out of an index payload into evidence blobs."""
    counter = 0

    def _extract(container: dict[str, Any]) -> None:
        nonlocal counter
        image = container.get("image")
        if not isinstance(image, str) or not image:
            return
        name = f"frame_{counter:06d}.jpg"
        counter += 1
        (evidence_dir / name).write_bytes(base64.b64decode(image))
        container["image"] = f"file:{evidence_dir.name}/{name}"

    for capsule in payload.get("capsules", []):
        for view in capsule.get("views", []):
            _extract(view)
    for scene in payload.get("scenes", []):
        _extract(scene)
    for context in payload.get("contexts", []):
        _extract(context)


def _resolve_image_refs(node: Any, base_dir: Path) -> None:
    """Inline ``file:`` evidence references before deserialisation."""
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key == "image" and isinstance(value, str) and value.startswith("file:"):
                blob = (base_dir / value[len("file:") :]).read_bytes()
                node[key] = base64.b64encode(blob).decode("ascii")
            else:
                _resolve_image_refs(value, base_dir)
    elif isinstance(node, list):
        for item in node:
            _resolve_image_refs(item, base_dir)


@dataclass
class SporeView:
    """One label-free crop of a capsule at a specific time.

    ``bbox`` is in *frame* coordinates.  The crop is stored as JPEG so a
    capsule store stays small enough to move with a Chronicle vault; the
    embedding stays in memory (and is packed as float16 on disk).
    """

    timestamp: float
    frame_index: int
    bbox: BBox
    score: float
    embedding: np.ndarray | None = None
    image: np.ndarray | None = None
    source: str = "saliency"
    sharpness: float = 0.0
    hand_overlap: float = 0.0
    motion: float = 0.0
    novelty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_image: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": round(float(self.timestamp), 4),
            "frame_index": int(self.frame_index),
            "bbox": [round(float(v), 2) for v in self.bbox.xyxy],
            "score": round(float(self.score), 4),
            "source": self.source,
            "sharpness": round(float(self.sharpness), 4),
            "hand_overlap": round(float(self.hand_overlap), 4),
            "motion": round(float(self.motion), 4),
            "novelty": round(float(self.novelty), 4),
            "embedding": _pack_vector(self.embedding),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        if include_image:
            payload["image"] = _encode_jpeg(self.image)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SporeView:
        return cls(
            timestamp=float(payload.get("timestamp", 0.0)),
            frame_index=int(payload.get("frame_index", 0)),
            bbox=BBox.from_sequence(payload.get("bbox", (0, 0, 0, 0))),
            score=float(payload.get("score", 0.0)),
            embedding=_unpack_vector(payload.get("embedding")),
            image=_decode_jpeg(payload.get("image")),
            source=str(payload.get("source", "saliency")),
            sharpness=float(payload.get("sharpness", 0.0)),
            hand_overlap=float(payload.get("hand_overlap", 0.0)),
            motion=float(payload.get("motion", 0.0)),
            novelty=float(payload.get("novelty", 0.0)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class ContextFrame:
    """A JPEG keyframe retained for query-time foveal re-detection.

    Capsule crops are deliberately small and sometimes lose the context a
    detector needs (a knife held under a hand is a classic failure).  Keeping
    one compressed keyframe per retained moment gives late binding a wider
    field of view without paying any detector call during ingest.
    """

    timestamp: float
    frame_index: int
    image_jpeg: bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    def image(self) -> np.ndarray | None:
        if not self.image_jpeg:
            return None
        array = np.frombuffer(self.image_jpeg, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def as_dict(self, *, include_image: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": round(float(self.timestamp), 4),
            "frame_index": int(self.frame_index),
            "metadata": self.metadata,
        }
        if include_image:
            payload["image"] = base64.b64encode(self.image_jpeg).decode("ascii")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContextFrame:
        return cls(
            timestamp=float(payload.get("timestamp", 0.0)),
            frame_index=int(payload.get("frame_index", 0)),
            image_jpeg=base64.b64decode(payload.get("image", "")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class ObjectCapsule:
    """A persistent, pre-semantic object track.

    The prototype is the mean of the best-view embeddings.  A capsule never
    stores a vocabulary label at ingest; ``label``/``label_confidence`` are
    written by :class:`chronicle.perception.spore.SporeQueryEngine` when a
    query actually needs semantics.
    """

    capsule_id: str
    created_at: float
    last_seen: float
    best_views: list[SporeView] = field(default_factory=list)
    max_views: int = 5
    observations: int = 0
    contact_windows: list[tuple[float, float]] = field(default_factory=list)
    last_bbox: BBox | None = None
    label: str | None = None
    label_confidence: float = 0.0
    label_views: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    _prototype: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.observations = max(self.observations, len(self.best_views))

    @property
    def prototype(self) -> np.ndarray | None:
        if self._prototype is not None:
            return self._prototype
        vectors = [view.embedding for view in self.best_views if view.embedding is not None]
        if not vectors:
            return None
        stacked = np.stack(vectors).astype(np.float32)
        self._prototype = _normalise(stacked.mean(axis=0))
        return self._prototype

    def invalidate_prototype(self) -> None:
        self._prototype = None

    def similarity(self, embedding: np.ndarray | None) -> float:
        return _cosine(self.prototype, embedding)

    @property
    def best_view(self) -> SporeView | None:
        return max(self.best_views, key=lambda view: view.score) if self.best_views else None

    @property
    def mean_view_score(self) -> float:
        if not self.best_views:
            return 0.0
        return float(np.mean([view.score for view in self.best_views]))

    @property
    def contact_count(self) -> int:
        return len(self.contact_windows)

    @property
    def last_contact(self) -> float | None:
        return self.contact_windows[-1][1] if self.contact_windows else None

    def add_view(self, view: SporeView) -> bool:
        """Insert a view, keeping only the highest-scoring ``max_views``.

        Returns ``True`` when the view was retained.  The prototype is
        invalidated lazily, so callers can add a burst of views and then read
        the prototype once.
        """
        self.observations += 1
        self.last_seen = max(self.last_seen, view.timestamp)
        self.last_bbox = view.bbox
        if len(self.best_views) < self.max_views:
            self.best_views.append(view)
            self.best_views.sort(key=lambda item: item.score, reverse=True)
            self.invalidate_prototype()
            return True
        weakest = min(self.best_views, key=lambda item: item.score)
        if view.score <= weakest.score:
            return False
        self.best_views.remove(weakest)
        self.best_views.append(view)
        self.best_views.sort(key=lambda item: item.score, reverse=True)
        self.invalidate_prototype()
        return True

    def add_contact(self, timestamp: float, *, merge_gap: float = 0.5) -> None:
        if self.contact_windows and timestamp - self.contact_windows[-1][1] <= merge_gap:
            start, _ = self.contact_windows[-1]
            self.contact_windows[-1] = (start, timestamp)
        else:
            self.contact_windows.append((timestamp, timestamp))

    def as_dict(self, *, include_images: bool = True) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "created_at": round(float(self.created_at), 4),
            "last_seen": round(float(self.last_seen), 4),
            "observations": int(self.observations),
            "max_views": int(self.max_views),
            "contact_windows": [
                [round(float(start), 4), round(float(end), 4)]
                for start, end in self.contact_windows
            ],
            "last_bbox": list(self.last_bbox.xyxy) if self.last_bbox is not None else None,
            "label": self.label,
            "label_confidence": round(float(self.label_confidence), 4),
            "label_views": int(self.label_views),
            "metadata": self.metadata,
            "views": [view.as_dict(include_image=include_images) for view in self.best_views],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObjectCapsule:
        capsule = cls(
            capsule_id=str(payload["capsule_id"]),
            created_at=float(payload.get("created_at", 0.0)),
            last_seen=float(payload.get("last_seen", 0.0)),
            max_views=int(payload.get("max_views", 5)),
            observations=int(payload.get("observations", 0)),
            contact_windows=[
                (float(row[0]), float(row[1])) for row in payload.get("contact_windows", [])
            ],
            last_bbox=(
                BBox.from_sequence(payload["last_bbox"])
                if payload.get("last_bbox") is not None
                else None
            ),
            label=payload.get("label"),
            label_confidence=float(payload.get("label_confidence", 0.0)),
            label_views=int(payload.get("label_views", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )
        capsule.best_views = [
            SporeView.from_dict(row) for row in payload.get("views", [])
        ]
        capsule.best_views.sort(key=lambda view: view.score, reverse=True)
        capsule.observations = max(capsule.observations, len(capsule.best_views))
        return capsule


@dataclass
class ActionCapsule:
    """A hand-object contact interval with geometric deltas (pre-semantic)."""

    action_id: str
    start: float
    end: float
    hand_id: str
    capsule_ids: list[str] = field(default_factory=list)
    displacement: tuple[float, float] = (0.0, 0.0)
    hand_speed: float = 0.0
    label: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "start": round(float(self.start), 4),
            "end": round(float(self.end), 4),
            "hand_id": self.hand_id,
            "capsule_ids": list(self.capsule_ids),
            "displacement": [round(float(v), 3) for v in self.displacement],
            "hand_speed": round(float(self.hand_speed), 3),
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionCapsule:
        return cls(
            action_id=str(payload["action_id"]),
            start=float(payload.get("start", 0.0)),
            end=float(payload.get("end", 0.0)),
            hand_id=str(payload.get("hand_id", "hand:0")),
            capsule_ids=list(payload.get("capsule_ids") or []),
            displacement=tuple(float(v) for v in payload.get("displacement", (0.0, 0.0))),
            hand_speed=float(payload.get("hand_speed", 0.0)),
            label=payload.get("label"),
            confidence=float(payload.get("confidence", 0.0)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class SceneCapsule:
    """A keyframe chosen by visual change, optionally with a global embedding."""

    timestamp: float
    frame_index: int
    score: float
    embedding: np.ndarray | None = None
    image: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, include_image: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": round(float(self.timestamp), 4),
            "frame_index": int(self.frame_index),
            "score": round(float(self.score), 4),
            "embedding": _pack_vector(self.embedding),
            "metadata": self.metadata,
        }
        if include_image:
            payload["image"] = _encode_jpeg(self.image)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SceneCapsule:
        return cls(
            timestamp=float(payload.get("timestamp", 0.0)),
            frame_index=int(payload.get("frame_index", 0)),
            score=float(payload.get("score", 0.0)),
            embedding=_unpack_vector(payload.get("embedding")),
            image=_decode_jpeg(payload.get("image")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class TextCapsule:
    """OCR evidence placeholder: text is bound to a region and a time."""

    text: str
    timestamp: float
    bbox: BBox
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeechCapsule:
    """ASR/VAD evidence placeholder: transcript segment plus timing."""

    text: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CapsuleStore:
    """In-memory capsule collection with compact JSON persistence."""

    def __init__(self) -> None:
        self.capsules: dict[str, ObjectCapsule] = {}
        self.actions: dict[str, ActionCapsule] = {}
        self.scenes: list[SceneCapsule] = []
        self.texts: list[TextCapsule] = []
        self.speech: list[SpeechCapsule] = []
        self.contexts: list[ContextFrame] = []
        self._counter = 0

    def new_capsule_id(self) -> str:
        self._counter += 1
        return f"spore-{self._counter:05d}"

    def new_action_id(self) -> str:
        self._counter += 1
        return f"action-{self._counter:05d}"

    def add_capsule(
        self, timestamp: float, view: SporeView, *, max_views: int = 5
    ) -> ObjectCapsule:
        capsule = ObjectCapsule(
            capsule_id=self.new_capsule_id(),
            created_at=float(timestamp),
            last_seen=float(timestamp),
            max_views=max_views,
        )
        capsule.add_view(view)
        self.capsules[capsule.capsule_id] = capsule
        return capsule

    def add_action(self, action: ActionCapsule) -> None:
        self.actions[action.action_id] = action

    def consolidate_actions(self, max_gap: float = 0.4) -> int:
        """Merge adjacent contact episodes on the same object.

        Two hands, or a momentarily-lost contact, can split one physical
        interaction into several capsules.  Merging by primary object and a
        small temporal gap keeps the temporal index compact without joining
        interactions on different objects.
        """
        if max_gap < 0 or not self.actions:
            return 0
        merged: list[ActionCapsule] = []
        removed = 0
        for action in sorted(self.actions.values(), key=lambda item: item.start):
            primary = action.capsule_ids[0] if action.capsule_ids else ""
            target = None
            if primary:
                for candidate in merged:
                    candidate_primary = (
                        candidate.capsule_ids[0] if candidate.capsule_ids else ""
                    )
                    if candidate_primary == primary and action.start <= candidate.end + max_gap:
                        target = candidate
                        break
            if target is None:
                merged.append(action)
                continue
            target.end = max(target.end, action.end)
            target.capsule_ids = list(
                dict.fromkeys([*target.capsule_ids, *action.capsule_ids])
            )
            target.metadata["merged_actions"] = int(
                target.metadata.get("merged_actions", 1)
            ) + 1
            removed += 1
        self.actions = {action.action_id: action for action in merged}
        return removed

    def add_scene(self, scene: SceneCapsule) -> None:
        self.scenes.append(scene)

    def add_text(self, text: TextCapsule) -> None:
        self.texts.append(text)

    def add_speech(self, speech: SpeechCapsule) -> None:
        self.speech.append(speech)

    def add_context(self, context: ContextFrame, *, max_frames: int = 400) -> None:
        self.contexts.append(context)
        self.contexts.sort(key=lambda item: item.timestamp)
        if len(self.contexts) > max_frames:
            del self.contexts[: len(self.contexts) - max_frames]

    def context_at(self, timestamp: float, *, max_gap: float = 1.5) -> ContextFrame | None:
        best: ContextFrame | None = None
        best_gap = max_gap
        for context in self.contexts:
            gap = abs(context.timestamp - timestamp)
            if gap < best_gap:
                best = context
                best_gap = gap
        return best

    def match(
        self,
        embedding: np.ndarray | None,
        timestamp: float,
        bbox: BBox,
        *,
        threshold: float = 0.80,
        spatial_gate: float = 0.75,
        max_gap: float = 8.0,
        frame_diagonal: float | None = None,
    ) -> ObjectCapsule | None:
        """Return the capsule that best explains this observation.

        Appearance is the primary key.  A slightly weaker appearance match is
        accepted when the patch is spatially close to where the capsule was
        last seen and the gap is short; this keeps identity through occlusion
        without merging two visually similar objects that are far apart.
        """
        if not self.capsules or embedding is None:
            return None
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query_norm = float(np.linalg.norm(query))
        if query_norm <= 1e-12:
            return None
        capsules: list[ObjectCapsule] = []
        prototypes: list[np.ndarray] = []
        for capsule in self.capsules.values():
            prototype = capsule.prototype
            if prototype is None:
                continue
            capsules.append(capsule)
            prototypes.append(prototype)
        if not capsules:
            return None
        matrix = np.stack(prototypes)
        norms = np.linalg.norm(matrix, axis=1)
        denominator = np.maximum(norms * query_norm, 1e-12)
        similarities = (matrix @ query) / denominator
        floor = max(0.0, threshold - 0.03)
        best: tuple[float, ObjectCapsule] | None = None
        for index in np.flatnonzero(similarities >= floor):
            capsule = capsules[int(index)]
            similarity = float(similarities[int(index)])
            if similarity < threshold:
                if capsule.last_bbox is None:
                    continue
                gap = timestamp - capsule.last_seen
                if gap < 0.0 or gap > max_gap:
                    continue
                distance = _center_distance(capsule.last_bbox, bbox)
                limit = spatial_gate * (frame_diagonal or max(bbox.width, bbox.height) * 4.0)
                if distance > limit:
                    continue
                similarity += 0.08 * max(0.0, 1.0 - distance / max(1e-6, limit))
            if best is None or similarity > best[0]:
                best = (similarity, capsule)
        if best is None:
            return None
        return best[1]

    def candidates(self) -> list[ObjectCapsule]:
        return list(self.capsules.values())

    def evict_to(self, limit: int) -> int:
        """Bound memory by dropping the least interaction-relevant capsules.

        A long egocentric clip otherwise accumulates thousands of background
        patches and blows the store size.  Contact windows are the strongest
        evidence that a capsule mattered, followed by hand overlap, motion, and
        view quality.  Recency only breaks ties, so an old but touched object
        survives a recent wall patch.
        """
        if limit <= 0 or len(self.capsules) <= limit:
            return 0
        def priority(capsule: ObjectCapsule) -> tuple[float, ...]:
            hand = max((view.hand_overlap for view in capsule.best_views), default=0.0)
            motion = max((view.motion for view in capsule.best_views), default=0.0)
            quality = max((view.score for view in capsule.best_views), default=0.0)
            return (
                1.0 if capsule.contact_windows else 0.0,
                1.0 if hand >= 0.05 else 0.0,
                float(motion),
                float(quality),
                capsule.last_seen,
            )

        ordered = sorted(self.capsules.values(), key=priority, reverse=True)
        removed = 0
        for capsule in ordered[limit:]:
            del self.capsules[capsule.capsule_id]
            removed += 1
        return removed

    def stats(self) -> dict[str, Any]:
        views = sum(len(capsule.best_views) for capsule in self.capsules.values())
        contacts = sum(capsule.contact_count for capsule in self.capsules.values())
        return {
            "capsules": len(self.capsules),
            "views": views,
            "actions": len(self.actions),
            "scenes": len(self.scenes),
            "texts": len(self.texts),
            "speech": len(self.speech),
            "contexts": len(self.contexts),
            "contacts": contacts,
        }

    def to_dict(
        self,
        *,
        include_images: bool = True,
        include_view_images: bool = True,
    ) -> dict[str, Any]:
        view_images = include_images and include_view_images
        return {
            "version": 1,
            "capsules": [
                capsule.as_dict(include_images=view_images)
                for capsule in self.capsules.values()
            ],
            "actions": [action.as_dict() for action in self.actions.values()],
            "scenes": [
                scene.as_dict(include_image=include_images) for scene in self.scenes
            ],
            "texts": [
                {
                    "text": text.text,
                    "timestamp": text.timestamp,
                    "bbox": list(text.bbox.xyxy),
                    "confidence": text.confidence,
                    "metadata": text.metadata,
                }
                for text in self.texts
            ],
            "speech": [
                {
                    "text": speech.text,
                    "start": speech.start,
                    "end": speech.end,
                    "confidence": speech.confidence,
                    "speaker": speech.speaker,
                    "metadata": speech.metadata,
                }
                for speech in self.speech
            ],
            "contexts": [
                context.as_dict(include_image=include_images)
                for context in self.contexts
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CapsuleStore:
        store = cls()
        for row in payload.get("capsules", []):
            capsule = ObjectCapsule.from_dict(row)
            store.capsules[capsule.capsule_id] = capsule
        for row in payload.get("actions", []):
            action = ActionCapsule.from_dict(row)
            store.actions[action.action_id] = action
        store.scenes = [SceneCapsule.from_dict(row) for row in payload.get("scenes", [])]
        for row in payload.get("texts", []):
            store.texts.append(
                TextCapsule(
                    text=str(row.get("text", "")),
                    timestamp=float(row.get("timestamp", 0.0)),
                    bbox=BBox.from_sequence(row.get("bbox", (0, 0, 0, 0))),
                    confidence=float(row.get("confidence", 1.0)),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
        for row in payload.get("speech", []):
            store.speech.append(
                SpeechCapsule(
                    text=str(row.get("text", "")),
                    start=float(row.get("start", 0.0)),
                    end=float(row.get("end", 0.0)),
                    confidence=float(row.get("confidence", 1.0)),
                    speaker=row.get("speaker"),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
        store.contexts = [
            ContextFrame.from_dict(row) for row in payload.get("contexts", [])
        ]
        store._counter = len(store.capsules) + len(store.actions)
        return store

    def save(
        self,
        path: str | Path,
        *,
        include_images: bool = True,
        include_view_images: bool = True,
        split: bool = False,
    ) -> Path:
        """Write the store as one JSON file or as a compact index + blobs.

        ``split=True`` moves every JPEG into a sibling ``<stem>_evidence/``
        directory and leaves ``file:<relative-path>`` references in the
        index.  An LLM can read the index without loading megabytes of
        base64, while query-time code transparently resolves the blobs.

        ``include_view_images=False`` keeps capsule view boxes and
        embeddings but drops their pixels.  Views then act as routing
        indices; the retained context frames remain the localisation
        evidence.  This roughly halves episode storage without changing
        late-binding answers on the evaluated clips.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(
            include_images=include_images,
            include_view_images=include_view_images,
        )
        if split and include_images:
            evidence_dir = target.parent / f"{target.stem}_evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            _externalize_images(payload, evidence_dir)
        target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> CapsuleStore:
        target = Path(path)
        payload = json.loads(target.read_text(encoding="utf-8"))
        _resolve_image_refs(payload, target.parent)
        return cls.from_dict(payload)

    def memory_bytes(
        self,
        *,
        include_images: bool = True,
        include_view_images: bool = True,
    ) -> int:
        return len(
            json.dumps(
                self.to_dict(
                    include_images=include_images,
                    include_view_images=include_view_images,
                )
            )
        )


def _center_distance(first: BBox, second: BBox) -> float:
    ax, ay = first.centroid
    bx, by = second.centroid
    return math.hypot(ax - bx, ay - by)


def view_iou(view: SporeView, boxes: Iterable[BBox]) -> float:
    """Largest IoU between a stored view and a set of boxes."""
    best = 0.0
    for box in boxes:
        value = view.bbox.iou(box)
        if value > best:
            best = value
    return best

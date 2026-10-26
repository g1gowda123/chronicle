"""Answer-frame selection for the dashboard query console.

The query engine attaches a ``frame`` dictionary to every grounded answer.  It
is chosen from images the graph already stores (interaction strips, object
crops and zone keyframes).  If none exist, the backend extracts the exact video
frame at the answer's newest citation and returns it as a data URL.

The renderer is intentionally untouched: this module only produces data.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from chronicle.frames import (
    FrameCandidate,
    extract_video_frame_data_url,
    latest_citation_time,
    pick_frame_candidate,
)

from .store import Entity, MemoryStore

VideoResolver = Callable[[], str | Path | None]


class AnswerFrameSelector:
    """Pick or extract the still frame that best supports one answer."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        data_dir: str | Path | None = None,
        video_path: str | Path | VideoResolver | None = None,
    ) -> None:
        self.store = store
        self.data_dir = Path(data_dir).expanduser().resolve() if data_dir else None
        self.video_path = video_path
        self._cache: dict[tuple[str, int], tuple[str, float] | None] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def for_answer(self, result: Mapping[str, Any], playhead: float) -> dict[str, Any] | None:
        route = str(result.get("route") or "none")
        if route not in {"entity", "zone", "timeline"}:
            return None

        citations = [item for item in (result.get("citations") or []) if isinstance(item, Mapping)]
        anchor_t = latest_citation_time(citations, playhead)
        matched_entity = result.get("matched_entity")
        matched_entity = str(matched_entity) if matched_entity else None
        citation_entities = {
            str(citation.get("entity_id"))
            for citation in citations
            if citation.get("entity_id")
        }
        citation_labels = {
            str(citation.get("label"))
            for citation in citations
            if citation.get("label")
        }

        zones = [zone for zone in (result.get("zones") or []) if isinstance(zone, Mapping)]
        matched_zone = str(zones[0].get("id")) if route == "zone" and len(zones) == 1 else None

        candidates = self._candidates(
            result,
            route=route,
            playhead=playhead,
            matched_entity=matched_entity,
            citation_entities=citation_entities,
        )
        picked = pick_frame_candidate(
            candidates,
            playhead=playhead,
            anchor_t=anchor_t,
            matched_entity=matched_entity,
            citation_entities=citation_entities,
            citation_labels=citation_labels,
            route=route,
            matched_zone=matched_zone,
        )
        if picked is not None:
            return picked.to_dict()

        return self._video_fallback(
            result,
            anchor_t=anchor_t,
            playhead=playhead,
            matched_entity=matched_entity,
            matched_zone=matched_zone,
        )

    # ------------------------------------------------------------------ #
    # Stored graph evidence
    # ------------------------------------------------------------------ #

    def _candidates(
        self,
        result: Mapping[str, Any],
        *,
        route: str,
        playhead: float,
        matched_entity: str | None,
        citation_entities: set[str],
    ) -> list[FrameCandidate]:
        candidates: list[FrameCandidate] = []
        seen_urls: set[str] = set()

        def add(candidate: FrameCandidate) -> None:
            url = self._clean_url(candidate.url)
            if url is None or url in seen_urls:
                return
            seen_urls.add(url)
            candidates.append(
                FrameCandidate(
                    url=url,
                    t=candidate.t,
                    kind=candidate.kind,
                    entity_id=candidate.entity_id,
                    zone_id=candidate.zone_id,
                    label=candidate.label,
                    action=candidate.action,
                    caption=candidate.caption,
                    clip_sim=candidate.clip_sim,
                    bbox=candidate.bbox,
                )
            )

        result_entities = [
            entity for entity in (result.get("entities") or []) if isinstance(entity, Mapping)
        ]
        relevant_ids = {
            str(entity.get("id"))
            for entity in result_entities
            if entity.get("id")
        }
        relevant_ids |= citation_entities
        if matched_entity:
            relevant_ids.add(matched_entity)

        # Subgraph object crops are the cheapest fallback when an interaction
        # strip was not exported by the producer.
        for entity_data in result_entities:
            self._add_entity_crop(add, str(entity_data.get("id") or ""))
        for entity_id in relevant_ids:
            self._add_entity_crop(add, entity_id)

        # Interaction strips/crops are the strongest answer evidence.  Timeline
        # answers have no resolved entity, so scan the whole episode instead.
        interaction_ids = set(self.store.entities) if route == "timeline" else relevant_ids
        for entity_id in interaction_ids:
            entity = self.store.entities.get(entity_id)
            if entity is None:
                continue
            for record in entity.interactions:
                if record.t > playhead + 0.5:
                    continue
                self._add_interaction(add, entity, record)

        for zone in (result.get("zones") or []):
            if not isinstance(zone, Mapping):
                continue
            url = self._clean_url(zone.get("keyframe_url"))
            if url is None:
                continue
            add(
                FrameCandidate(
                    url=url,
                    t=float(zone.get("last_seen") or 0.0),
                    kind="zone_keyframe",
                    zone_id=str(zone.get("id") or "") or None,
                    label=str(zone.get("label") or ""),
                )
            )

        return candidates

    def _add_entity_crop(self, add: Callable[[FrameCandidate], None], entity_id: str) -> None:
        entity = self.store.entities.get(entity_id)
        if entity is None or not entity.crop_url:
            return
        add(
            FrameCandidate(
                url=entity.crop_url,
                t=entity.last_seen,
                kind="entity_crop",
                entity_id=entity.id,
                zone_id=entity.zone_id,
                label=entity.label,
                clip_sim=entity.clip_sim,
                bbox=entity.bbox,
            )
        )

    def _add_interaction(self, add: Callable[[FrameCandidate], None], entity: Entity, record: Any) -> None:
        common = {
            "entity_id": entity.id,
            "zone_id": entity.zone_id,
            "label": entity.label,
            "action": record.action,
            "caption": record.caption,
            "clip_sim": record.clip_sim,
            "bbox": entity.bbox,
        }
        if record.strip_url:
            add(
                FrameCandidate(
                    url=record.strip_url,
                    t=record.t,
                    kind="interaction_strip",
                    **common,
                )
            )
        if record.crop_url:
            add(
                FrameCandidate(
                    url=record.crop_url,
                    t=record.t,
                    kind="interaction_crop",
                    **common,
                )
            )

    # ------------------------------------------------------------------ #
    # Video extraction fallback
    # ------------------------------------------------------------------ #

    def _video_fallback(
        self,
        result: Mapping[str, Any],
        *,
        anchor_t: float | None,
        playhead: float,
        matched_entity: str | None,
        matched_zone: str | None,
    ) -> dict[str, Any] | None:
        if anchor_t is None:
            return None
        video = self._video_file()
        if video is None:
            return None

        key = (str(video), round(max(0.0, min(anchor_t, playhead)) * 1000.0))
        with self._cache_lock:
            cached = key in self._cache
            extracted = self._cache.get(key)
        if not cached:
            extracted = extract_video_frame_data_url(video, key[1] / 1000.0)
            with self._cache_lock:
                if len(self._cache) >= 32:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = extracted
        if extracted is None:
            return None

        data_url, actual_t = extracted
        entity = self.store.entities.get(matched_entity or "")
        bbox = None
        if entity is not None:
            bbox = entity.bbox_at(actual_t) or entity.bbox
        zone = self.store.zones.get(matched_zone or "")
        label = entity.label if entity is not None else (zone.label if zone is not None else "")
        return FrameCandidate(
            url=data_url,
            t=actual_t,
            kind="video_frame",
            entity_id=entity.id if entity is not None else matched_entity,
            zone_id=(entity.zone_id if entity is not None else None) or matched_zone,
            label=label,
            bbox=bbox,
        ).to_dict()

    def _video_file(self) -> Path | None:
        value: Any = self.video_path
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value:
            path = Path(value).expanduser()
            if path.is_file():
                return path.resolve()

        url = str(self.store.session.get("video_url") or "").strip()
        if not url or url.startswith(("http://", "https://", "data:")):
            return None
        if url.startswith("/media/"):
            if self.data_dir is None:
                return None
            path = self.data_dir / url[len("/media/") :]
        else:
            path = Path(url).expanduser()
            if not path.is_absolute() and self.data_dir is not None:
                path = self.data_dir / path
        if path.is_file():
            return path.resolve()
        return None

    def _clean_url(self, value: Any) -> str | None:
        url = str(value or "").strip()
        if not url:
            return None
        if url.startswith(("http://", "https://", "data:image/")):
            return url
        if url.startswith("/media/"):
            if self.data_dir is None:
                return url
            path = self.data_dir / url[len("/media/") :]
            return url if path.is_file() else None
        path = Path(url).expanduser()
        if path.is_absolute() and path.is_file() and self.data_dir is not None:
            try:
                relative = path.resolve().relative_to(self.data_dir)
            except ValueError:
                return None
            return "/media/" + relative.as_posix()
        return None

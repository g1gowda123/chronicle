"""OpenCLIP visual place recognition (VPR) and zone deduplication.

Role C owns zone relocalisation: when a camera pan settles, the keyframe is
embedded and compared against active background-graph zones.  A cosine
similarity at or above ``vpr_threshold`` (0.85 by default) reuses the existing
zone; otherwise a new zone is minted and optionally captioned by Florence-2.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._utils import cosine_similarity, l2_normalize, shorten_label, slugify
from .config import SemanticConfig
from .embedders import ImageEmbedder

logger = logging.getLogger("chronicle.semantic.vpr")


@dataclass
class Zone:
    """A topological anchor in the background graph."""

    zone_id: str
    label: str
    prototype: np.ndarray
    caption: str = ""
    created_at: float = 0.0
    last_seen_at: float = 0.0
    observations: int = 1

    def similarity(self, embedding: np.ndarray) -> float:
        return cosine_similarity(self.prototype, embedding)

    def update(self, embedding: np.ndarray, ema: float = 0.85) -> None:
        """Blend a new observation into the prototype (EMA on the unit sphere)."""
        blended = ema * l2_normalize(self.prototype) + (1.0 - ema) * l2_normalize(embedding)
        self.prototype = l2_normalize(blended)
        self.observations += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "label": self.label,
            "prototype": [float(value) for value in np.asarray(self.prototype).ravel()],
            "caption": self.caption,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "observations": self.observations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Zone:
        return cls(
            zone_id=str(data["zone_id"]),
            label=str(data.get("label") or data["zone_id"]),
            prototype=l2_normalize(np.asarray(data["prototype"], dtype=np.float32)),
            caption=str(data.get("caption") or ""),
            created_at=float(data.get("created_at") or 0.0),
            last_seen_at=float(data.get("last_seen_at") or 0.0),
            observations=int(data.get("observations") or 1),
        )


@dataclass(frozen=True)
class PlaceMatch:
    """Result of relocalising a keyframe against the zone registry."""

    zone: Zone
    score: float
    is_new: bool
    runner_up_score: float | None
    timestamp: float
    margin: float | None = None

    @property
    def event_type(self) -> str:
        return "ZONE_CREATED" if self.is_new else "ZONE_RELOCALIZED"

    def as_event(self) -> dict[str, Any]:
        """Payload ready for Role B's background graph / Role D telemetry."""
        return {
            "event": self.event_type,
            "tag": "VLM",
            "zone_id": self.zone.zone_id,
            "label": self.zone.label,
            "caption": self.zone.caption,
            "similarity": round(float(self.score), 6),
            "runner_up": None if self.runner_up_score is None else round(self.runner_up_score, 6),
            "is_new": self.is_new,
            "timestamp": float(self.timestamp),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.as_event(),
            "zone": self.zone.to_dict(),
        }


@dataclass
class VisualPlaceRecognizer:
    """Cosine-similarity zone registry with caption-assisted zone creation."""

    embedder: ImageEmbedder
    threshold: float = 0.85
    min_margin: float = 0.0
    prototype_ema: float = 0.85
    captioner: Any | None = None
    max_zones: int = 512
    zones: dict[str, Zone] = field(default_factory=dict)
    clock: Any = time.monotonic

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0.0 <= self.min_margin <= 1.0:
            raise ValueError("min_margin must be in [0, 1]")
        if not 0.0 <= self.prototype_ema <= 1.0:
            raise ValueError("prototype_ema must be in [0, 1]")

    @classmethod
    def from_config(
        cls,
        embedder: ImageEmbedder,
        config: SemanticConfig | None = None,
        captioner: Any | None = None,
    ) -> VisualPlaceRecognizer:
        config = config or SemanticConfig()
        return cls(
            embedder=embedder,
            threshold=config.vpr_threshold,
            min_margin=config.vpr_min_margin,
            prototype_ema=config.vpr_prototype_ema,
            captioner=captioner,
            max_zones=config.vpr_max_zones,
        )

    def __len__(self) -> int:
        return len(self.zones)

    def active_zones(self) -> list[Zone]:
        return sorted(self.zones.values(), key=lambda zone: zone.last_seen_at, reverse=True)

    def resolve(
        self,
        image: Any,
        *,
        caption: str | None = None,
        timestamp: float | None = None,
    ) -> PlaceMatch:
        """Return the matching zone or create a new one for this keyframe."""
        now = float(self.clock() if timestamp is None else timestamp)
        embedding = np.asarray(
            self.embedder.embed_images([image])[0], dtype=np.float32
        )
        embedding = l2_normalize(embedding)

        scored = [(zone.similarity(embedding), zone) for zone in self.zones.values()]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        best_score, best_zone = (scored[0] if scored else (-1.0, None))
        runner_up = float(scored[1][0]) if len(scored) > 1 else None
        margin = None if runner_up is None else float(best_score - runner_up)

        if (
            best_zone is not None
            and best_score >= self.threshold
            and (runner_up is None or margin is None or margin >= self.min_margin)
        ):
            best_zone.update(embedding, ema=self.prototype_ema)
            best_zone.last_seen_at = now
            return PlaceMatch(
                zone=best_zone,
                score=float(best_score),
                is_new=False,
                runner_up_score=runner_up,
                timestamp=now,
                margin=margin,
            )

        zone = self._create_zone(image, embedding, caption=caption, timestamp=now)
        return PlaceMatch(
            zone=zone,
            score=1.0,
            is_new=True,
            runner_up_score=runner_up,
            timestamp=now,
            margin=margin,
        )

    # Friendly alias for callers that prefer "recognise".
    recognize = resolve

    def _create_zone(
        self,
        image: Any,
        embedding: np.ndarray,
        *,
        caption: str | None,
        timestamp: float,
    ) -> Zone:
        if len(self.zones) >= self.max_zones:
            self.merge_duplicates()
        if len(self.zones) >= self.max_zones:
            raise RuntimeError(
                f"zone registry is full ({len(self.zones)} >= max_zones={self.max_zones})"
            )

        caption_text = (caption or "").strip()
        if not caption_text and self.captioner is not None:
            try:
                caption_text = (self.captioner.caption(image) or "").strip()
            except Exception as exc:  # model unavailable / transient failure  # noqa: BLE001
                logger.warning("zone captioning failed: %s", exc)

        short = shorten_label(caption_text)
        base_label = short or f"Zone {len(self.zones) + 1}"
        base_id = f"zone_{slugify(base_label)}"
        zone_id = base_id
        suffix = 2
        while zone_id in self.zones:
            zone_id = f"{base_id}_{suffix}"
            suffix += 1

        zone = Zone(
            zone_id=zone_id,
            label=base_label,
            prototype=l2_normalize(embedding),
            caption=caption_text,
            created_at=timestamp,
            last_seen_at=timestamp,
            observations=1,
        )
        self.zones[zone_id] = zone
        logger.info("new zone %s (%s)", zone.zone_id, zone.label)
        return zone

    def merge_duplicates(self, threshold: float | None = None) -> list[tuple[str, str]]:
        """Merge zones whose prototypes exceed the similarity threshold.

        Returns ``(removed_id, kept_id)`` pairs.  This is the cleanup pass used
        after a demo run or a long ingestion session to guarantee that no two
        room anchors describe the same place.
        """
        limit = self.threshold if threshold is None else float(threshold)
        zones = list(self.zones.values())
        parent = list(range(len(zones)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for i in range(len(zones)):
            for j in range(i + 1, len(zones)):
                if zones[i].similarity(zones[j].prototype) >= limit:
                    union(i, j)

        clusters: dict[int, list[Zone]] = {}
        for index, zone in enumerate(zones):
            clusters.setdefault(find(index), []).append(zone)

        merged: list[tuple[str, str]] = []
        for cluster in clusters.values():
            if len(cluster) < 2:
                continue
            cluster.sort(key=lambda zone: zone.created_at)
            keeper = cluster[0]
            total_observations = sum(zone.observations for zone in cluster)
            prototype = sum(
                zone.prototype * max(zone.observations, 1) for zone in cluster
            ) / max(total_observations, 1)
            keeper.prototype = l2_normalize(prototype)
            keeper.observations = total_observations
            keeper.last_seen_at = max(zone.last_seen_at for zone in cluster)

            # Prefer a real caption over the generic "Zone N" placeholder.
            if keeper.label.lower().startswith("zone "):
                for candidate in cluster[1:]:
                    if not candidate.label.lower().startswith("zone "):
                        keeper.label = candidate.label
                        keeper.caption = candidate.caption
                        break

            for zone in cluster[1:]:
                self.zones.pop(zone.zone_id, None)
                merged.append((zone.zone_id, keeper.zone_id))
                logger.info("merged duplicate zone %s -> %s", zone.zone_id, keeper.zone_id)

        return merged

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "threshold": self.threshold,
            "zones": [zone.to_dict() for zone in self.zones.values()],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: ImageEmbedder,
        *,
        captioner: Any | None = None,
        config: SemanticConfig | None = None,
    ) -> VisualPlaceRecognizer:
        config = config or SemanticConfig()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        recognizer = cls.from_config(embedder=embedder, config=config, captioner=captioner)
        for item in payload.get("zones", []):
            zone = Zone.from_dict(item)
            recognizer.zones[zone.zone_id] = zone
        return recognizer


def build_vpr(
    embedder: ImageEmbedder,
    config: SemanticConfig | None = None,
    captioner: Any | None = None,
) -> VisualPlaceRecognizer:
    return VisualPlaceRecognizer.from_config(embedder, config=config, captioner=captioner)

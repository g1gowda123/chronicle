"""Bridge Role A perception output into the Role B frame contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from chronicle.types import (
    CameraMotion,
    FrameContext,
    FrameResult,
    ObjectObservation,
    SupportSurface,
)


def _embedding_tuple(embedding: Any) -> tuple[float, ...] | None:
    if embedding is None:
        return None
    array = np.asarray(embedding, dtype=np.float32).ravel()
    return tuple(float(value) for value in array)


def frame_context_from_result(
    result: FrameResult,
    *,
    supports: Sequence[SupportSurface] = (),
    zone_id: str | None = None,
    zone_label: str | None = None,
) -> FrameContext:
    """Convert one Role A ``FrameResult`` into a Role B ``FrameContext``.

    Role A does not detect support surfaces, so the caller (typically Role C or
    the background-graph keyframe pass) must provide the current ``supports``
    whenever it wants static/support reasoning.  Missing supports are allowed:
    the engine records ``support_verified=False`` instead of stalling.

    Entity IDs are derived from ByteTrack IDs (or an explicit
    ``metadata["entity_id"]`` override), keeping identity stable while a track
    is rebound by Re-ID.
    """

    objects: list[ObjectObservation] = []
    for track in result.tracks:
        metadata = dict(track.metadata or {})
        entity_id = str(metadata.get("entity_id") or f"track_{track.track_id}")
        reid_score = metadata.get("reid_score")
        objects.append(
            ObjectObservation(
                entity_id=entity_id,
                label=track.label or "object",
                bbox=tuple(float(value) for value in track.bbox.xyxy),
                confidence=track.confidence,
                track_id=track.track_id,
                reid_similarity=float(reid_score) if reid_score is not None else None,
                embedding=_embedding_tuple(track.embedding),
                attributes=metadata,
            )
        )

    frame_size = None
    if result.image is not None:
        frame_size = (int(result.image.shape[1]), int(result.image.shape[0]))

    attributes: dict[str, Any] = {
        "source": "role-a",
        "camera_state": result.motion.state.value,
    }
    if zone_label is not None:
        attributes["zone_label"] = zone_label

    return FrameContext(
        t=result.timestamp,
        objects=objects,
        hands=tuple(result.hands),
        supports=tuple(supports),
        camera=CameraMotion(
            dx=result.motion.dx,
            dy=result.motion.dy,
            dtheta_deg=result.motion.rotation_degrees,
        ),
        frame_size=frame_size,
        zone_id=zone_id,
        frame_index=result.frame_index,
        attributes=attributes,
    )

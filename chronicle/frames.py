"""Evidence-frame selection for grounded query answers.

The query backends expose a JSON ``frame`` descriptor alongside every answer so
a client can show the still image that supports the answer.  This module is UI
agnostic and deliberately small:

* :class:`FrameCandidate` describes one already-existing evidence image
  (interaction strip, object crop or zone keyframe).
* :func:`pick_frame_candidate` ranks those candidates against the answer's
  citations and playhead.
* :func:`extract_video_frame_data_url` extracts the exact video frame at a
  timestamp and returns a browser-ready ``data:image/jpeg;base64,...`` URL when
  the graph has no stored evidence image.

Nothing here writes into the UI package or assumes NiceGUI is installed.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Relative importance of a stored evidence image.  A temporal strip shows the
#: interaction itself, so it outranks a single object crop or a room keyframe.
_KIND_WEIGHTS: dict[str, float] = {
    "interaction_strip": 0.60,
    "interaction_crop": 0.48,
    "entity_crop": 0.36,
    "zone_keyframe": 0.24,
    "video_frame": 0.10,
}

_SOURCE_BY_KIND: dict[str, str] = {
    "interaction_strip": "graph",
    "interaction_crop": "graph",
    "entity_crop": "graph",
    "zone_keyframe": "graph",
    "video_frame": "video",
}


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    """One image that can support an answer."""

    url: str
    t: float
    kind: str
    entity_id: str | None = None
    zone_id: str | None = None
    label: str = ""
    action: str | None = None
    caption: str = ""
    clip_sim: float | None = None
    bbox: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, UI-ready frame contract."""

        frame: dict[str, Any] = {
            "url": self.url,
            "t": round(float(self.t), 3),
            "seek": round(float(self.t), 3),
            "kind": self.kind,
            "source": _SOURCE_BY_KIND.get(self.kind, "graph"),
            "label": self.label,
            "entity_id": self.entity_id,
            "zone_id": self.zone_id,
            "action": self.action,
            "caption": self.caption,
            "clip_sim": self.clip_sim,
        }
        if self.bbox is not None:
            frame["bbox"] = [round(float(value), 4) for value in self.bbox]
        return frame


def latest_citation_time(
    citations: Iterable[Mapping[str, Any]] | None,
    playhead: float,
) -> float | None:
    """Pick the newest grounded timestamp that is not in the future.

    Dashboard citations use ``t`` while Role C citations use ``timestamp``;
    accept both.  Negative sentinel times (for example a terminal anchor from
    before the episode) clamp to the beginning of the clip.  When every
    citation is in the future, the playhead is the best frame we can show.
    """

    times: list[float] = []
    for citation in citations or ():
        if not isinstance(citation, Mapping):
            continue
        value: Any = None
        for key in ("t", "timestamp"):
            if key in citation:
                value = citation[key]
                break
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            times.append(number)

    if not times:
        return None

    visible = [value for value in times if value <= playhead + 0.5]
    if visible:
        return max(0.0, max(visible))
    earliest = min(times)
    return max(0.0, min(earliest, playhead))


def pick_frame_candidate(
    candidates: Iterable[FrameCandidate],
    *,
    playhead: float,
    anchor_t: float | None = None,
    matched_entity: str | None = None,
    matched_label: str | None = None,
    citation_entities: Iterable[str] = (),
    citation_labels: Iterable[str] = (),
    route: str = "entity",
    matched_zone: str | None = None,
) -> FrameCandidate | None:
    """Rank stored evidence images and return the one that best fits the answer."""

    entity_ids = {str(value) for value in citation_entities if value}
    labels = {str(value) for value in citation_labels if value}
    best: FrameCandidate | None = None
    best_score = float("-inf")

    for candidate in candidates:
        if not candidate.url or not math.isfinite(candidate.t):
            continue
        score = _KIND_WEIGHTS.get(candidate.kind, 0.2)

        if matched_entity and candidate.entity_id == matched_entity:
            score += 0.30
        elif matched_label and candidate.label.casefold() == matched_label.casefold():
            score += 0.28
        elif candidate.entity_id and candidate.entity_id in entity_ids:
            score += 0.14

        if anchor_t is not None:
            score += max(0.0, 0.20 - 0.02 * abs(candidate.t - anchor_t))

        # Never prefer evidence the playhead has not reached yet.
        if candidate.t > playhead + 0.5:
            score -= 0.75

        if candidate.action and candidate.action in labels:
            score += 0.06
        if candidate.clip_sim is not None:
            score += min(max(float(candidate.clip_sim), 0.0), 1.0) * 0.04

        if route == "zone" and matched_zone and candidate.zone_id == matched_zone:
            score += 0.95 if candidate.kind == "zone_keyframe" else 0.10
        if route == "timeline" and candidate.kind.startswith("interaction"):
            score += 0.05

        if score > best_score:
            best_score = score
            best = candidate

    return best


@dataclass(frozen=True, slots=True)
class StoredEvidence:
    """One image already stored beside an episode database."""

    path: Path
    t: float
    kind: str
    label: str = ""
    entity_id: str | None = None
    zone_id: str | None = None
    action: str | None = None
    caption: str = ""
    clip_sim: float | None = None
    bbox: tuple[float, float, float, float] | None = None

    def to_frame_candidate(self) -> FrameCandidate | None:
        """Read the image and return a self-contained data URL candidate."""

        try:
            payload = self.path.read_bytes()
        except OSError:
            return None
        if not payload:
            return None
        mime = mimetypes.guess_type(self.path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(payload).decode("ascii")
        return FrameCandidate(
            url=f"data:{mime};base64,{encoded}",
            t=self.t,
            kind=self.kind,
            entity_id=self.entity_id,
            zone_id=self.zone_id,
            label=self.label,
            action=self.action,
            caption=self.caption,
            clip_sim=self.clip_sim,
            bbox=self.bbox,
        )

    def _proxy(self) -> FrameCandidate:
        return FrameCandidate(
            url=str(self.path),
            t=self.t,
            kind=self.kind,
            entity_id=self.entity_id,
            zone_id=self.zone_id,
            label=self.label,
            action=self.action,
            caption=self.caption,
            clip_sim=self.clip_sim,
            bbox=self.bbox,
        )


def load_stored_evidence(episode_dir: str | Path) -> tuple[StoredEvidence, ...]:
    """Read ``<episode>/evidence/index.jsonl`` if it exists.

    Each line is a JSON object with at least ``t`` and ``file``.  Optional keys
    mirror :class:`FrameCandidate`: ``kind``, ``label``, ``entity_id``,
    ``zone_id``/``zone``, ``action``, ``caption``, ``clip_sim`` and ``bbox``.
    Missing files are skipped so a moved clip degrades to video extraction.
    """

    base = Path(episode_dir).expanduser()
    index = base / "evidence" / "index.jsonl"
    if not index.is_file():
        return ()
    rows: list[StoredEvidence] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        if not isinstance(raw, Mapping):
            continue
        timestamp = raw.get("t", raw.get("timestamp"))
        relative = raw.get("file") or raw.get("rel_path")
        if timestamp is None or not relative:
            continue
        path = index.parent / str(relative)
        if not path.is_file() and raw.get("rel_path"):
            path = base / str(raw["rel_path"])
        if not path.is_file():
            continue
        try:
            t = float(timestamp)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(t):
            continue
        bbox_raw = raw.get("bbox")
        bbox: tuple[float, float, float, float] | None = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            try:
                bbox = tuple(float(value) for value in bbox_raw)  # type: ignore[assignment]
            except (TypeError, ValueError):
                bbox = None
        clip_sim_raw = raw.get("clip_sim")
        try:
            clip_sim = float(clip_sim_raw) if clip_sim_raw is not None else None
        except (TypeError, ValueError):
            clip_sim = None
        rows.append(
            StoredEvidence(
                path=path,
                t=t,
                kind=str(raw.get("kind") or "interaction_strip"),
                label=str(raw.get("label") or raw.get("action") or ""),
                entity_id=str(raw["entity_id"]) if raw.get("entity_id") else None,
                zone_id=str(raw.get("zone_id") or raw.get("zone") or "") or None,
                action=str(raw["action"]) if raw.get("action") else None,
                caption=str(raw.get("caption") or ""),
                clip_sim=clip_sim,
                bbox=bbox,
            )
        )
    return tuple(rows)


def pick_stored_evidence(
    candidates: Iterable[StoredEvidence],
    *,
    playhead: float,
    anchor_t: float | None = None,
    matched_entity: str | None = None,
    matched_label: str | None = None,
    citation_entities: Iterable[str] = (),
    citation_labels: Iterable[str] = (),
    route: str = "entity",
    matched_zone: str | None = None,
) -> FrameCandidate | None:
    """Rank stored images and encode only the winner as a data URL."""

    rows = list(candidates)
    if not rows:
        return None
    by_path = {str(row.path): row for row in rows}
    picked = pick_frame_candidate(
        (row._proxy() for row in rows),
        playhead=playhead,
        anchor_t=anchor_t,
        matched_entity=matched_entity,
        matched_label=matched_label,
        citation_entities=citation_entities,
        citation_labels=citation_labels,
        route=route,
        matched_zone=matched_zone,
    )
    if picked is None:
        return None
    stored = by_path.get(str(picked.url))
    return stored.to_frame_candidate() if stored is not None else None


def _read_video_frame(video: Path, t: float) -> tuple[Any, float] | None:
    """Decode one frame using OpenCV.

    Kept separate so tests can replace decoding without needing a codec in CI.
    """

    try:
        import cv2
    except Exception:  # pragma: no cover - OpenCV is a core dependency
        return None

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if frames > 0 and fps > 0 else None
        target = max(0.0, float(t))
        if duration is not None:
            target = min(target, max(0.0, duration - 1.0 / fps))

        capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(target * fps))
            ok, frame = capture.read()
        if not ok or frame is None:
            return None

        actual = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if actual <= 0.0 or not math.isfinite(actual):
            actual = target
        return frame, actual
    finally:
        capture.release()


def extract_video_frame_data_url(
    video_path: str | Path | None,
    t: float,
    *,
    max_width: int = 960,
    quality: int = 82,
) -> tuple[str, float] | None:
    """Return ``(data_url, actual_time)`` for the video frame nearest ``t``.

    ``None`` is returned for missing/undecodable footage so callers can simply
    omit the frame instead of failing a query.
    """

    if video_path is None:
        return None
    path = Path(video_path).expanduser()
    if not path.is_file():
        return None

    decoded = _read_video_frame(path, t)
    if decoded is None:
        return None
    frame, actual = decoded

    try:
        import cv2
    except Exception:  # pragma: no cover - OpenCV is a core dependency
        return None

    try:
        height, width = frame.shape[:2]
        if max_width > 0 and width > max_width:
            ratio = max_width / float(width)
            frame = cv2.resize(
                frame,
                (max_width, max(1, int(round(height * ratio)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(quality, 100)))],
        )
        if not ok:
            return None
    except Exception:  # pragma: no cover - defensive around exotic frame arrays
        return None

    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}", actual

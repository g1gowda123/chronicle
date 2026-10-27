"""Event-based clipping and per-event graph creation.

The dashboard can already replay one timeline for a whole clip.  This module
turns the *events* in that timeline into small, self-contained clips:

    data/event_clips/<source>/<01_event-slug>/
        clip.mp4        a pre/post-roll cut of the source video
        events.jsonl    only the events inside the cut, rebased to t=0
        graph.json      a Vis.js graph snapshot at the trigger time
        event.json      the trigger, salience score and artifact paths
    data/event_clips/<source>/index.json

That is the bounded recorder described in ``docs/VFM_ARCHITECTURE.tex``:
transition / low confidence / label conflict -> salience score -> 3 s pre +
3 s post clip -> graph + review metadata.  No VLM call is required to make a
clip; the events that already triggered it are the evidence.

The module is deliberately dependency-light: the UI calls it through
``POST /api/event-clips`` and the tests call :func:`build_event_clips` directly.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import Event, parse_event
from .store import MemoryStore

DEFAULT_PRE_S = 3.0
DEFAULT_POST_S = 3.0
DEFAULT_COOLDOWN_S = 2.5
DEFAULT_LIMIT = 12
DEFAULT_MIN_SCORE = 0.28
MIN_CLIP_S = 2.0
MAX_CUT_S = 30.0

OUTPUT_DIR = "event_clips"
MANIFEST_NAME = "index.json"

#: Event kinds that survive into a replay timeline and can feed a graph.
GRAPH_KINDS = frozenset({"entity", "state", "edge", "zone", "interaction", "telemetry"})
#: Kinds that are useful for salience scoring.
SCORE_KINDS = frozenset({"entity", "state", "edge", "zone", "interaction"})

_STATE_SCORES = {
    "NEW": 0.34,
    "VISIBLE_STATIC": 0.12,
    "VISIBLE_MOVING": 0.25,
    "INTERACTED": 0.82,
    "HELD": 0.92,
    "CONTAINED": 0.80,
    "OCCLUDED": 0.16,
    "OUT_OF_VIEW": 0.24,
    "LOST": 0.58,
}
_EDGE_SCORES = {
    "PLACED_ON": 0.70,
    "INSIDE": 0.66,
    "CONTAINS": 0.52,
    "RESTING_ON": 0.50,
    "HELD_BY": 0.68,
    "ACTION": 0.56,
}
_ACTION_SCORES = {
    "PICK_UP": 0.90,
    "PUT_DOWN": 0.82,
    "PLACE": 0.82,
    "HOLD": 0.74,
    "INSERT_INTO": 0.90,
    "REMOVE_FROM": 0.88,
    "OPEN": 0.72,
    "CLOSE": 0.72,
    "TOUCHED": 0.62,
    "SLIDE": 0.60,
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _slug(value: str, limit: int = 42) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return (text[:limit].rstrip("-") or "event")


def _clip_label(event: Event) -> str:
    data = event.data or {}
    if event.kind == "state":
        subject = data.get("label") or data.get("entity_id") or "object"
        return f"{data.get('to') or 'STATE'} · {subject}"
    if event.kind == "interaction":
        subject = data.get("label") or data.get("entity_id") or "object"
        return f"{data.get('action') or 'INTERACTION'} · {subject}"
    if event.kind == "edge":
        relation = data.get("relation") or "relation"
        return f"{relation} · {data.get('source') or '?'} → {data.get('target') or '?'}"
    if event.kind == "zone":
        return f"ZONE · {data.get('label') or data.get('id') or 'zone'}"
    return event.kind.upper()


def score_event(event: Event) -> tuple[float, list[str]]:
    """Return ``(salience, reasons)`` for one normalised event.

    The score is intentionally readable rather than learned: the demo has to
    explain why a moment became a clip.  A score below
    :data:`DEFAULT_MIN_SCORE` is not clipped.
    """

    data = event.data or {}
    reasons: list[str] = []
    score = 0.0

    if event.kind == "interaction":
        action = str(data.get("action") or "").upper()
        score = _ACTION_SCORES.get(action, 0.58)
        reasons.append(f"action:{action or 'interaction'}")
        confidence = _number(data.get("confidence"), _number(data.get("clip_sim"), 0.6))
        if confidence < 0.55:
            score += 0.16
            reasons.append("low-confidence")
        if data.get("accepted") is False:
            score += 0.10
            reasons.append("needs-review")
        if data.get("rejection_reason") or data.get("reason") in {"AMBIGUOUS", "REJECTED"}:
            score += 0.08
            reasons.append("ambiguity")
    elif event.kind == "state":
        to_state = str(data.get("to") or "NEW").upper()
        score = _STATE_SCORES.get(to_state, 0.20)
        reasons.append(f"state:{to_state}")
        trigger = data.get("trigger") if isinstance(data.get("trigger"), dict) else {}
        if trigger.get("new_entity"):
            score += 0.08
            reasons.append("new-entity")
        if trigger.get("reacquired"):
            score += 0.10
            reasons.append("reacquired")
        if trigger.get("container_id") or data.get("container"):
            score += 0.08
            reasons.append("container")
        if trigger.get("hand_contact"):
            score += 0.04
            reasons.append("hand-contact")
    elif event.kind == "edge":
        relation = str(data.get("relation") or "").upper()
        score = _EDGE_SCORES.get(relation, 0.34)
        reasons.append(f"edge:{relation or 'relation'}")
        if data.get("merged_into") or data.get("labels"):
            score = max(score, 0.84)
            reasons.append("label-conflict")
    elif event.kind == "zone":
        score = 0.30 if data.get("active", True) else 0.20
        reasons.append("zone")
    elif event.kind == "entity":
        confidence = _number(data.get("confidence"), 1.0)
        if data.get("merged_into") or (
            isinstance(data.get("labels"), (list, tuple)) and len(data.get("labels") or []) > 1
        ):
            score = 0.82
            reasons.append("label-conflict")
        elif confidence and confidence < 0.45:
            score = 0.34
            reasons.append("low-confidence")
        else:
            return 0.0, []

    return min(1.0, round(score, 4)), reasons


@dataclass(slots=True)
class ClipPlan:
    """One candidate event window, before any file is written."""

    anchor: Event
    start: float
    end: float
    score: float
    reasons: list[str] = field(default_factory=list)
    members: list[Event] = field(default_factory=list)

    @property
    def label(self) -> str:
        return _clip_label(self.anchor)

    @property
    def slug(self) -> str:
        return _slug(self.label)


def _apply_duration_window(
    anchor_t: float,
    duration: float | None,
    *,
    pre: float,
    post: float,
) -> tuple[float, float]:
    start = max(0.0, anchor_t - max(0.0, pre))
    end = anchor_t + max(0.0, post)
    if duration is not None and duration > 0:
        end = min(max(duration, 0.0), end)
    if end - start < MIN_CLIP_S:
        end = min(start + MIN_CLIP_S, duration) if duration and duration > 0 else start + MIN_CLIP_S
        if end - start < MIN_CLIP_S:
            start = max(0.0, end - MIN_CLIP_S)
    if duration and duration > 0:
        end = min(end, duration)
        start = max(0.0, min(start, end))
    return round(start, 3), round(end, 3)


def select_events(
    events: Iterable[Event],
    *,
    duration: float | None = None,
    pre: float = DEFAULT_PRE_S,
    post: float = DEFAULT_POST_S,
    cooldown: float = DEFAULT_COOLDOWN_S,
    limit: int = DEFAULT_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[ClipPlan]:
    """Turn raw events into coalesced, scored clip windows.

    Nearby events (within *cooldown* seconds) collapse into one clip whose
    anchor is the highest-salience event.  This is what keeps a held object
    from producing a clip for every frame of contact.
    """

    candidates: list[tuple[Event, float, list[str]]] = []
    for event in events:
        if event.kind not in SCORE_KINDS:
            continue
        score, reasons = score_event(event)
        if score >= min_score:
            candidates.append((event, score, reasons))
    candidates.sort(key=lambda item: (item[0].t, item[0].seq))

    groups: list[list[tuple[Event, float, list[str]]]] = []
    for candidate in candidates:
        if groups and candidate[0].t - groups[-1][-1][0].t <= cooldown:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    plans: list[ClipPlan] = []
    for group in groups:
        anchor_event, anchor_score, anchor_reasons = max(
            group, key=lambda item: (item[1], item[0].t)
        )
        members = [item[0] for item in group]
        reasons = list(dict.fromkeys(reason for item in group for reason in item[2]))
        start, end = _apply_duration_window(
            anchor_event.t, duration, pre=pre, post=post
        )
        plans.append(
            ClipPlan(
                anchor=anchor_event,
                start=start,
                end=end,
                score=min(1.0, round(anchor_score + 0.03 * (len(group) - 1), 4)),
                reasons=reasons,
                members=members,
            )
        )

    plans.sort(key=lambda plan: (plan.score, plan.anchor.t), reverse=True)
    selected = plans[: max(1, int(limit))]
    selected.sort(key=lambda plan: (plan.start, plan.anchor.t))
    return selected


def _load_events(path: Path) -> list[Event]:
    events: list[Event] = []
    if not path.is_file():
        return events
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = parse_event(raw, seq=index)
        if event is not None:
            events.append(event)
    events.sort(key=lambda event: (event.t, event.seq))
    return events


def _session_payload(events: Iterable[Event]) -> dict[str, Any]:
    for event in events:
        if event.kind == "session":
            return dict(event.data or {})
    return {}


def _under(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _source_key(path: Path, kind: str) -> str:
    """Stable, collision-free folder name for one source timeline/video."""

    if kind == "events":
        if path.name == "events.jsonl":
            return f"{path.parent.parent.name}-{path.parent.name}"
        stem = path.name.removesuffix(".events.jsonl")
    else:
        stem = path.stem
    if stem.lower() in {"source", "clip", "video", "upload", "events"}:
        stem = f"{path.parent.name}-{stem}"
    return stem


def _resolve_source(clip: str, data_dir: Path) -> tuple[Path, Path | None, list[Event], dict[str, Any], Path]:
    """Resolve ``events:<rel>`` / ``video:<rel>`` to events + optional video.

    Returns ``(source_key, video, events, session, output_root)``.
    """

    root = Path(data_dir).expanduser().resolve()
    kind, _, rel = str(clip or "").partition(":")
    if kind not in {"events", "video"} or not rel:
        raise ValueError("clip must be events:<path> or video:<path>")

    candidate = (root / rel).resolve()
    if not _under(root, candidate):
        raise ValueError("clip path must stay inside the data directory")

    video: Path | None = None
    if kind == "events":
        events_path = candidate
        if not events_path.is_file():
            raise ValueError(f"events file not found: {rel}")
        events = _load_events(events_path)
        session = _session_payload(events)
        video_url = str(session.get("video_url") or "")
        if video_url.startswith("/media/"):
            possible = (root / video_url.removeprefix("/media/")).resolve()
            if possible.is_file() and _under(root, possible):
                video = possible
        source_key = _source_key(events_path, "events")
    else:
        video = candidate
        if not video.is_file():
            raise ValueError(f"video file not found: {rel}")
        events_path = video.with_name(video.stem + ".events.jsonl")
        db_path = video.parent / "chronicle.db"
        if events_path.is_file():
            events = _load_events(events_path)
        elif db_path.is_file():
            from .replay_db import events_from_db

            raw_events = events_from_db(
                db_path,
                title=video.stem,
                video_url="/media/" + video.relative_to(root).as_posix(),
                video_path=video,
            )
            events = _load_events_from_rows(raw_events)
        else:
            events = []
        session = _session_payload(events) or {
            "title": video.stem,
            "video_url": "/media/" + video.relative_to(root).as_posix(),
        }
        source_key = _source_key(video, "video")

    output_root = root / OUTPUT_DIR / _slug(source_key, 60)
    return source_key, video, events, session, output_root


def _load_events_from_rows(rows: Iterable[dict[str, Any]]) -> list[Event]:
    events: list[Event] = []
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            continue
        event = parse_event(raw, seq=index)
        if event is not None:
            events.append(event)
    events.sort(key=lambda event: (event.t, event.seq))
    return events


def _window_events(events: Iterable[Event], start: float, end: float) -> list[Event]:
    """Events inside a cut plus the latest entity/zone/edge context.

    Without the context prelude a clip that starts mid-episode would begin with
    anonymous ``track_2`` nodes because the entity introduction happened before
    the window.  The prelude is rebased to t=0 by :func:`_write_events`.
    """

    inside = [
        event
        for event in events
        if start - 1e-6 <= event.t <= end + 1e-6
        and event.kind in GRAPH_KINDS
        and event.kind != "session"
    ]
    latest: dict[tuple[str, str], Event] = {}
    for event in events:
        if event.t >= start - 1e-6:
            break
        if event.kind not in {"entity", "zone", "edge"}:
            continue
        key = str(event.data.get("id") or event.data.get("entity_id") or "")
        if key:
            latest[(event.kind, key)] = event
    context = list(latest.values())
    context.sort(key=lambda event: (event.t, event.seq))
    return context + inside


def _graph_at(events: Iterable[Event], t: float) -> dict[str, Any]:
    store = MemoryStore()
    for event in events:
        if event.kind == "session":
            continue
        if event.t <= t + 1e-6:
            store.apply(event)
    try:
        return store.graph_snapshot(t, mode="full")
    finally:
        store.clear()


def _write_events(
    path: Path,
    session_event: dict[str, Any],
    events: Iterable[Event],
    *,
    offset: float = 0.0,
    seq_start: int = 0,
) -> None:
    """Write one replay timeline, rebasing event times by *offset*."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(session_event, separators=(",", ":"), default=str) + "\n")
        for index, event in enumerate(events, start=seq_start + 1):
            payload = dict(event.raw or {})
            payload.setdefault("kind", event.kind)
            payload["t"] = round(max(0.0, float(event.t) - offset), 3)
            payload["seq"] = index
            handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")


def _cut_video(video: Path, start: float, duration: float, target: Path) -> dict[str, Any] | None:
    """Cut one pre/post-roll mp4. Prefer ffmpeg, fall back to OpenCV."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(video),
            "-t",
            f"{max(0.1, duration):.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "25",
            "-movflags",
            "+faststart",
            str(target),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=max(20.0, duration * 8.0),
                check=False,
            )
            if result.returncode == 0 and target.is_file() and target.stat().st_size > 0:
                return {"path": target, "bytes": target.stat().st_size, "method": "ffmpeg"}
        except (OSError, subprocess.SubprocessError):
            pass

    try:
        import cv2

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0.1:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start) * 1000.0)
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        frames = 0
        limit = max(1, int(math.ceil(duration * fps)))
        while frames < limit:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
        writer.release()
        capture.release()
        if frames and target.is_file() and target.stat().st_size > 0:
            return {"path": target, "bytes": target.stat().st_size, "method": "opencv"}
    except Exception:
        pass
    try:
        target.unlink()
    except OSError:
        pass
    return None


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def build_event_clips(
    clip: str,
    data_dir: str | Path,
    *,
    pre: float = DEFAULT_PRE_S,
    post: float = DEFAULT_POST_S,
    cooldown: float = DEFAULT_COOLDOWN_S,
    limit: int = DEFAULT_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
    cut: bool = True,
    verify: bool = False,
    verify_limit: int = 12,
    vlm_model: str | None = None,
    vlm_host: str | None = None,
    vlm_timeout: float = 180.0,
) -> dict[str, Any]:
    """Build event clips for *clip* and return a UI-ready index.

    The output lives under ``<data-dir>/event_clips/<source>/``.  Rebuilding a
    source replaces only that source's generated folder; source videos and
    original timelines are never touched.
    """

    root = Path(data_dir).expanduser().resolve()
    source_key, video, events, session, output_root = _resolve_source(clip, root)
    if not events:
        raise ValueError("this clip has no events to clip")

    duration = _number(session.get("duration"), 0.0)
    if video is not None:
        try:
            from .runtime import probe_duration

            actual = probe_duration(video)
        except Exception:
            actual = 0.0
        if actual > 0:
            # The source video is the authority.  A replay timeline can outlive
            # a trimmed demo asset; cutting past the last frame is useless.
            duration = actual
            events = [event for event in events if event.t <= duration + 1e-6]
    if duration <= 0:
        duration = max((event.t for event in events), default=0.0) + post

    plans = select_events(
        events,
        duration=duration,
        pre=pre,
        post=post,
        cooldown=cooldown,
        limit=limit,
        min_score=min_score,
    )
    if not plans:
        raise ValueError("no salient events were found in this clip")

    if output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    source_title = str(session.get("title") or source_key)
    if vlm_model is None or vlm_host is None:
        from chronicle.vfm import caption_ledger

        vlm_model = vlm_model or caption_ledger.DEFAULT_MODEL
        vlm_host = vlm_host or caption_ledger.DEFAULT_HOST
    manifest_clips: list[dict[str, Any]] = []
    for index, plan in enumerate(plans, start=1):
        anchor_t = round(float(plan.anchor.t), 3)
        window = max(0.1, round(plan.end - plan.start, 3))
        folder = output_root / f"{index:02d}_{plan.slug}_t{int(round(anchor_t * 1000)):08d}"
        folder.mkdir(parents=True, exist_ok=True)
        title = f"{index:02d} · {plan.label} @ {anchor_t:.1f}s"

        # The replay timeline contains only this window.  Overview/telemetry
        # events are dropped; the graph kinds stay so the UI can rebuild a
        # graph for the clip.
        window_events = _window_events(events, plan.start, plan.end)
        clip_video = folder / "clip.mp4"
        video_info = _cut_video(video, plan.start, window, clip_video) if (cut and video is not None) else None

        video_rel = _relative(clip_video, root) if video_info else None
        video_url = "/media/" + video_rel if video_rel else str(session.get("video_url") or "")
        replay_session = {
            "v": 1,
            "kind": "session",
            "t": 0.0,
            "data": {
                "title": f"✂ {title}",
                "video_url": video_url,
                "duration": round(window, 3),
                "fps": session.get("fps", 30.0),
                "width": session.get("width", 1280),
                "height": session.get("height", 720),
                "source": "event-clip",
                "source_title": source_title,
                "event": {
                    "kind": plan.anchor.kind,
                    "label": plan.label,
                    "anchor_t": anchor_t,
                    "score": plan.score,
                    "reasons": plan.reasons,
                },
            },
        }
        replay_path = folder / "events.jsonl"
        _write_events(replay_path, replay_session, window_events, offset=plan.start)

        vlm: dict[str, Any] | None = None
        if verify and video_info and len(manifest_clips) < max(0, int(verify_limit)):
            try:
                from chronicle.vfm.event_verify import verify_event

                anchor_data = dict(plan.anchor.data or {})
                hint = {
                    "label": plan.label,
                    "subject": (
                        anchor_data.get("entity_id")
                        or anchor_data.get("id")
                        or anchor_data.get("label")
                        or plan.label
                    ),
                    "entity_id": anchor_data.get("entity_id") or anchor_data.get("id"),
                    "object_label": anchor_data.get("label") or anchor_data.get("entity_id"),
                    "state": anchor_data.get("to") if plan.anchor.kind == "state" else "",
                    "action": anchor_data.get("action") if plan.anchor.kind == "interaction" else "",
                    "zone": anchor_data.get("zone_id") or anchor_data.get("zone") or "",
                    "evidence": anchor_data.get("caption") or "",
                }
                vlm = verify_event(
                    clip_video,
                    anchor_rel=anchor_t - plan.start,
                    duration=window,
                    hint=hint,
                    frames_dir=folder / "frames",
                    replay_path=replay_path,
                    out_dir=folder,
                    title=title,
                    model=str(vlm_model),
                    host=str(vlm_host),
                    timeout=float(vlm_timeout),
                )
            except Exception as error:  # verification must not lose the clip
                vlm = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "provider": "none",
                }

        graph = _graph_at(events, anchor_t)
        graph_payload = {
            "version": 1,
            "source": clip,
            "source_title": source_title,
            "clip": title,
            "anchor_t": anchor_t,
            "window": [plan.start, plan.end],
            "score": plan.score,
            "reasons": plan.reasons,
            "event": {
                "kind": plan.anchor.kind,
                "t": anchor_t,
                "data": dict(plan.anchor.data or {}),
            },
            "vlm": vlm,
            "graph": graph,
        }
        graph_path = folder / "graph.json"
        graph_path.write_text(
            json.dumps(graph_payload, indent=2, default=str) + "\n", encoding="utf-8"
        )

        entry = {
            "id": f"{index:02d}",
            "title": title,
            "label": plan.label,
            "kind": plan.anchor.kind,
            "anchor_t": anchor_t,
            "start": plan.start,
            "end": plan.end,
            "duration": round(window, 3),
            "score": plan.score,
            "reasons": plan.reasons,
            "play": "events:" + _relative(replay_path, root),
            "events": _relative(replay_path, root),
            "video": video_rel,
            "graph": _relative(graph_path, root),
            "graph_nodes": len(graph.get("nodes") or []),
            "graph_edges": len(graph.get("edges") or []),
            "events_in_window": len(window_events),
            "vlm": vlm,
            "action": (vlm or {}).get("action"),
            "caption": (vlm or {}).get("caption"),
            "snapshot": (vlm or {}).get("snapshot"),
            "vault": (vlm or {}).get("vault"),
        }
        (folder / "event.json").write_text(
            json.dumps(entry, indent=2, default=str) + "\n", encoding="utf-8"
        )
        manifest_clips.append(entry)

    index = {
        "version": 1,
        "source": clip,
        "source_title": source_title,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pre_s": round(max(0.0, float(pre)), 3),
        "post_s": round(max(0.0, float(post)), 3),
        "cooldown_s": round(max(0.0, float(cooldown)), 3),
        "limit": int(limit),
        "verify": bool(verify),
        "vlm_model": str(vlm_model or ""),
        "verified": sum(
            1
            for entry in manifest_clips
            if (entry.get("vlm") or {}).get("status") == "ok"
        ),
        "duration": round(duration, 3),
        "clips": manifest_clips,
    }
    (output_root / MANIFEST_NAME).write_text(
        json.dumps(index, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return index


def read_index(clip: str, data_dir: str | Path) -> dict[str, Any] | None:
    """Read a previously built event-clip index without touching the video."""

    root = Path(data_dir).expanduser().resolve()
    kind, _, rel = str(clip or "").partition(":")
    if kind not in {"events", "video"} or not rel:
        return None
    path = (root / rel).resolve()
    if not _under(root, path):
        return None
    if kind == "events":
        if path.name == "events.jsonl":
            source_key = f"{path.parent.parent.name}-{path.parent.name}"
        else:
            source_key = _source_key(path, "events")
    else:
        source_key = _source_key(path, "video")
    manifest = root / OUTPUT_DIR / _slug(source_key, 60) / MANIFEST_NAME
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def list_event_clips(data_dir: str | Path) -> list[dict[str, Any]]:
    """Return every generated event clip, newest source first (UI registry)."""

    root = Path(data_dir).expanduser().resolve()
    base = root / OUTPUT_DIR
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for manifest in sorted(base.glob(f"*/{MANIFEST_NAME}"), reverse=True):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        for entry in payload.get("clips") or []:
            if isinstance(entry, dict) and entry.get("play"):
                out.append(entry)
    return out


__all__ = [
    "ClipPlan",
    "DEFAULT_COOLDOWN_S",
    "DEFAULT_LIMIT",
    "DEFAULT_POST_S",
    "DEFAULT_PRE_S",
    "OUTPUT_DIR",
    "build_event_clips",
    "list_event_clips",
    "read_index",
    "score_event",
    "select_events",
]

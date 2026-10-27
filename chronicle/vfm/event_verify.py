"""VLM verification for event clips.

The event recorder (``dashboard.event_clips``) decides *where* something
interesting happened.  This module asks a vision model *what* happened at that
moment and writes the answer back into the event memory:

    3 frames (pre / contact / post)  ->  Qwen2.5-VL or Florence-2
        -> strict JSON (event, caption, objects, actions, confidence)
        -> interaction event in events.jsonl
        -> one snapshot.jpg for audit
        -> a small queryable vault.db for this event

The VLM is optional.  If neither Ollama nor Florence is available, the existing
graph hint and the deterministic rule classifier still produce a verification
record, so the clip remains playable and auditable.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from chronicle.semantic.actions import RuleBasedActionClassifier

from . import caption_ledger
from .video_vault import compile_caption_vault

FRAME_OFFSETS = (-1.0, 0.0, 1.0)
FRAME_LABELS = ("pre", "contact", "post")
DEFAULT_EVENT_MODEL = caption_ledger.DEFAULT_MODEL
DEFAULT_EVENT_HOST = caption_ledger.DEFAULT_HOST
DEFAULT_EVENT_TIMEOUT = 180.0
MIN_DURATION = 0.2

EVENT_PROMPT = """You are verifying one event in an egocentric video.
The graph detected this candidate:
  candidate event: {label}
  candidate object: {subject}
  candidate zone: {zone}
  anchor time: {anchor:.2f}s of this event clip

The three images are, in order: 1 s before contact, at contact, and 1 s after.
Decide what actually happened.  Use concrete nouns and verbs.  Do not invent
objects or actions that are not visible.  If the candidate is wrong, correct it.

Return strict JSON only, with exactly these keys:
{{
  "event": "short lowercase action name (e.g. hold, pick up, place, insert, remove)",
  "caption": "one factual sentence about the event",
  "objects": ["lowercase object names"],
  "actions": ["lowercase verb phrases"],
  "zone": "kitchen|living room|office|hallway|outdoors|unknown",
  "confidence": 0.0,
  "agrees": true,
  "correction": "empty string when the candidate is right, otherwise one short sentence"
}}
"""


def _strip_json(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def _as_list(value: Any, limit: int = 24) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip().lower()
        if text and text not in out:
            out.append(text[:80])
    return out[:limit]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _frame_times(anchor_rel: float, duration: float) -> list[float]:
    duration = max(MIN_DURATION, float(duration))
    anchor = _clamp(float(anchor_rel), 0.0, duration)
    if duration < 1.2:
        return [
            round(duration * 0.15, 3),
            round(duration * 0.50, 3),
            round(duration * 0.85, 3),
        ]
    times = [_clamp(anchor + offset, 0.02, duration - 0.02) for offset in FRAME_OFFSETS]
    # Keep three distinct frames even for a very short pre/post window.
    if len(set(times)) < 3:
        return [
            round(duration * 0.15, 3),
            round(duration * 0.50, 3),
            round(duration * 0.85, 3),
        ]
    return [round(value, 3) for value in times]


def _hint_value(hint: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = hint.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _graph_action(hint: dict[str, Any]) -> str:
    action = _hint_value(hint, "action").upper()
    if action:
        return action
    state = _hint_value(hint, "state", "to").upper()
    return {"HELD": "HOLD", "INTERACTED": "TOUCHED", "CONTAINED": "INSERT_INTO"}.get(state, state)


def _normalise_action(text: str, object_label: str) -> tuple[str, float]:
    result = RuleBasedActionClassifier().classify(text, object_label)
    return str(result.get("action") or "OTHER"), float(result.get("confidence") or 0.0)


def _agreement(graph_action: str, vlm_action: str) -> bool:
    if not graph_action or not vlm_action:
        return False
    if graph_action == vlm_action:
        return True
    families = {
        "HOLD": {"HOLD", "PICK_UP", "HELD"},
        "PICK_UP": {"PICK_UP", "HOLD", "HELD"},
        "PUT_DOWN": {"PUT_DOWN", "PLACE", "PLACED_ON"},
        "INSERT_INTO": {"INSERT_INTO", "CONTAINED", "INSIDE"},
        "REMOVE_FROM": {"REMOVE_FROM", "WITHDRAW"},
        "TOUCHED": {"TOUCHED", "INTERACTED"},
    }
    return vlm_action in families.get(graph_action, {graph_action})


def _write_frames(
    video: Path,
    times: list[float],
    folder: Path,
) -> tuple[list[Path], Path | None]:
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for label, t in zip(FRAME_LABELS, times, strict=False):
        target = folder / f"{label}.jpg"
        caption_ledger.extract_frame(video, t, target, width=640)
        paths.append(target)
    snapshot = folder / "snapshot.jpg"
    contact = paths[1] if len(paths) > 1 else (paths[0] if paths else None)
    if contact is not None and contact.is_file():
        shutil.copyfile(contact, snapshot)
        return paths, snapshot
    return paths, None


def _ollama_verify(
    model: str,
    prompt: str,
    blobs: list[bytes],
    *,
    host: str,
    timeout: float,
) -> tuple[dict[str, Any], str]:
    raw = caption_ledger.ollama_generate(
        model, prompt, blobs, host=host, timeout=timeout
    )
    try:
        payload = json.loads(_strip_json(raw))
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict) or not str(payload.get("event") or payload.get("caption") or "").strip():
        raise RuntimeError("VLM returned no usable event JSON")
    return payload, raw


def _florence_verify(blobs: list[bytes], prompt: str) -> tuple[dict[str, Any], str]:
    import cv2
    import numpy as np

    from chronicle.semantic.florence import FlorenceCaptioner, ModelUnavailableError

    try:
        captioner = FlorenceCaptioner()
    except Exception as error:  # optional dependency
        raise RuntimeError(str(error)) from error
    captions: list[str] = []
    for blob in blobs:
        frame = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        captions.append(captioner.caption(frame))
    if not captions:
        raise ModelUnavailableError("Florence produced no captions")
    text = " ".join(captions).strip()
    payload = {
        "event": "",
        "caption": text,
        "objects": [],
        "actions": [],
        "zone": "unknown",
        "confidence": 0.55,
        "agrees": None,
        "correction": "",
    }
    return payload, text


def _parse_verification(
    payload: dict[str, Any],
    raw: str,
    *,
    model: str,
    hint: dict[str, Any],
) -> dict[str, Any]:
    caption_data = caption_ledger.parse_caption(raw, model)
    caption = str(caption_data.get("caption") or payload.get("caption") or "").strip()
    objects = _as_list(payload.get("objects")) or _as_list(caption_data.get("objects"))
    actions = _as_list(payload.get("actions")) or _as_list(caption_data.get("actions"))
    zone = str(payload.get("zone") or caption_data.get("zone") or "unknown").strip().lower()
    confidence = _clamp(_number(payload.get("confidence"), 0.5), 0.0, 1.0)
    event = str(payload.get("event") or "").strip().lower()[:120]
    subject = _hint_value(hint, "subject", "entity_id", "label")
    graph_action = _graph_action(hint)
    vlm_action, rule_confidence = _normalise_action(
        " ".join(part for part in (event, caption, " ".join(actions)) if part),
        subject,
    )
    agrees = payload.get("agrees")
    computed_agreement = _agreement(graph_action, vlm_action)
    if agrees is None:
        agrees = computed_agreement
    else:
        # Trust the model's flag only when the action family agrees too; a
        # small VLM can say "agrees" while returning a contradictory action.
        agrees = _bool(agrees, computed_agreement) and computed_agreement
    strong_graph = graph_action in {
        "HELD",
        "HOLD",
        "INTERACTED",
        "TOUCHED",
        "CONTAINED",
        "INSERT_INTO",
        "PICK_UP",
        "PUT_DOWN",
        "REMOVE_FROM",
        "OPEN",
        "CLOSE",
        "PUSH",
        "PULL",
        "HAND_OFF",
        "LOST",
    }
    if vlm_action == "OTHER":
        resolved = graph_action or "OTHER"
    elif agrees:
        resolved = vlm_action
    elif strong_graph:
        # The graph saw contact/containment; do not let a single VLM pass
        # overwrite a verified physical state.  The disagreement is recorded.
        resolved = graph_action
    else:
        resolved = vlm_action
    if resolved in {"HELD", "INTERACTED", "CONTAINED", "PLACED_ON", "INSIDE"}:
        resolved = {
            "HELD": "HOLD",
            "INTERACTED": "TOUCHED",
            "CONTAINED": "INSERT_INTO",
            "PLACED_ON": "PUT_DOWN",
            "INSIDE": "INSERT_INTO",
        }.get(resolved, resolved)
    accepted = bool(caption) and (
        (agrees and confidence >= 0.55)
        or (not strong_graph and confidence >= 0.6)
    )
    return {
        "event": event,
        "caption": caption,
        "objects": objects,
        "actions": actions,
        "zone": zone,
        "confidence": round(confidence, 4),
        "action": resolved or "OTHER",
        "vlm_action": vlm_action,
        "rule_confidence": round(rule_confidence, 4),
        "graph_action": graph_action,
        "agrees": agrees,
        "disagreement": bool(graph_action and vlm_action and not agrees),
        "correction": str(payload.get("correction") or "").strip()[:300],
        "accepted": accepted,
        "model": model,
    }


def _write_ledger(path: Path, *, model: str, duration: float, result: dict[str, Any], times: list[float]) -> None:
    frames = [
        {
            "t": t,
            "caption": result["caption"],
            "objects": result["objects"],
            "actions": result["actions"],
            "zone": result["zone"],
        }
        for t in times
    ]
    ledger = {
        "version": 1,
        "model": model,
        "mode": "event-vlm",
        "window_s": round(duration, 3),
        "duration_s": round(duration, 3),
        "windows": [
            {
                "t_start": 0.0,
                "t_end": round(duration, 3),
                "caption": result["caption"],
                "objects": result["objects"],
                "actions": result["actions"],
                "zone": result["zone"],
                "tags": list(dict.fromkeys(result["objects"] + result["actions"])),
                "frames": frames,
                "event": result["event"],
                "action": result["action"],
                "confidence": result["confidence"],
            }
        ],
    }
    path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_interaction(
    replay_path: Path,
    *,
    t: float,
    result: dict[str, Any],
    frames: dict[str, str],
    snapshot: str | None,
    subject: str,
) -> None:
    rows: list[dict[str, Any]] = []
    session: dict[str, Any] | None = None
    for line in replay_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("kind") == "session":
            session = payload
            continue
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if data.get("source") == "vlm-event":
            continue
        rows.append(payload)
    interaction = {
        "v": 1,
        "kind": "interaction",
        "t": round(max(0.0, t), 3),
        "data": {
            "entity_id": subject,
            "label": subject,
            "action": result["action"],
            "caption": result["caption"],
            "confidence": result["confidence"],
            "accepted": result["accepted"],
            "source": "vlm-event",
            "model": result["model"],
            "event": result["event"],
            "objects": result["objects"],
            "actions": result["actions"],
            "frames": frames,
            "snapshot": snapshot,
            "agrees": result["agrees"],
            "correction": result["correction"],
        },
    }
    rows.append(interaction)
    rows.sort(key=lambda row: float(row.get("t") or 0.0))
    lines: list[str] = []
    if session is not None:
        lines.append(json.dumps(session, separators=(",", ":"), ensure_ascii=False))
    for index, row in enumerate(rows, start=1):
        row["seq"] = index
        lines.append(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
    replay_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_event(
    video: Path,
    *,
    anchor_rel: float,
    duration: float,
    hint: dict[str, Any],
    frames_dir: Path,
    replay_path: Path,
    out_dir: Path,
    title: str,
    model: str = DEFAULT_EVENT_MODEL,
    host: str = DEFAULT_EVENT_HOST,
    timeout: float = DEFAULT_EVENT_TIMEOUT,
) -> dict[str, Any]:
    """Run the 3-frame VLM pass and update the event memory in place."""

    video = Path(video)
    if not video.is_file():
        return {"status": "unavailable", "reason": "event clip video is missing"}
    times = _frame_times(anchor_rel, duration)
    try:
        frame_paths, snapshot_path = _write_frames(video, times, frames_dir)
    except Exception as error:
        return {"status": "error", "error": f"frame extraction failed: {error}"}

    subject = _hint_value(hint, "entity_id", "subject", "label", default="object")
    object_label = _hint_value(hint, "object_label", "subject", "label", default=subject)
    label = _hint_value(hint, "label", "state", "action", default="event")
    zone = _hint_value(hint, "zone", "zone_id", default="unknown")
    prompt = EVENT_PROMPT.format(
        label=label,
        subject=object_label,
        zone=zone,
        anchor=float(anchor_rel),
    )
    blobs = [path.read_bytes() for path in frame_paths]
    raw = ""
    provider = ""
    payload: dict[str, Any] = {}
    errors: list[str] = []
    try:
        payload, raw = _ollama_verify(
            model, prompt, blobs, host=host, timeout=timeout
        )
        provider = f"ollama:{model}"
    except Exception as error:
        errors.append(f"ollama: {error}")
    if not payload:
        try:
            payload, raw = _florence_verify(blobs, prompt)
            provider = "florence-2-base"
        except Exception as error:
            errors.append(f"florence: {error}")

    if payload:
        result = _parse_verification(payload, raw, model=provider, hint=hint)
        status = "ok"
    else:
        result = _parse_verification(
            {
                "event": label,
                "caption": _hint_value(hint, "evidence", "caption", default=""),
                "objects": [subject] if subject else [],
                "actions": [],
                "zone": zone,
                "confidence": 0.4,
                "agrees": True,
            },
            "",
            model="rule-fallback",
            hint=hint,
        )
        result["model"] = "rule-fallback"
        result["accepted"] = False
        status = "unavailable"

    frame_rel = {
        label_name: path.relative_to(out_dir).as_posix()
        for label_name, path in zip(FRAME_LABELS, frame_paths, strict=False)
    }
    snapshot_rel = (
        snapshot_path.relative_to(out_dir).as_posix() if snapshot_path is not None else None
    )
    result.update(
        {
            "status": status,
            "provider": provider or "rule-fallback",
            "frames": frame_rel,
            "snapshot": snapshot_rel,
            "frame_times": times,
            "errors": errors,
        }
    )
    _append_interaction(
        replay_path,
        t=anchor_rel,
        result=result,
        frames=frame_rel,
        snapshot=snapshot_rel,
        subject=subject,
    )

    ledger_path = out_dir / "ledger.json"
    vault_path = out_dir / "vault.db"
    try:
        _write_ledger(ledger_path, model=result["model"], duration=duration, result=result, times=times)
        summary = compile_caption_vault(video, ledger_path, vault_path, title=title)
        result["ledger"] = ledger_path.relative_to(out_dir).as_posix()
        result["vault"] = vault_path.relative_to(out_dir).as_posix()
        result["vault_events"] = int(summary.get("events") or 0)
        result["vault_windows"] = int(summary.get("windows") or 1)
    except Exception as error:
        result["vault_error"] = f"{type(error).__name__}: {error}"
    return result


__all__ = ["DEFAULT_EVENT_HOST", "DEFAULT_EVENT_MODEL", "FRAME_OFFSETS", "verify_event"]

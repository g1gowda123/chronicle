#!/usr/bin/env python3
"""Build a dense timestamped caption ledger with an offline VLM.

This is a compile-time step: it may use a heavy local vision model. Query time
never calls the model; it reads the compressed ledger from the vault.

The script uses ffmpeg for frame extraction and the Ollama HTTP API for the VLM,
so it does not need OpenCV or a Python ML stack.

Usage:
    python prototype/caption_ledger.py \
      --video data/real/aria_cook_45s.mp4 \
      --out /tmp/vfm/cook_ledger.json \
      --model qwen2.5vl:3b --window-s 30

Then attach it to a vault:
    python prototype/attach_ledger.py --vault /tmp/vfm/vault.db \
      --ledger /tmp/vfm/cook_ledger.json
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROMPT_VERSION = 2
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5vl:3b"
DEFAULT_SYNTH_MODEL = "qwen2.5:1.5b"
FRAME_WIDTH = 640

FRAME_PROMPT = """You are compiling a timestamped memory from an egocentric video.
This is the frame at t={t:.1f}s of the episode.
Describe what is visible and what the person is doing. Mention the main objects,
actions and the room/zone. If the view is mostly empty, describe the visible
surfaces and setting instead of returning nothing. Do not guess identities,
brands, or events that are not visible. Prefer concrete nouns and verbs.

Return strict JSON only, with exactly these keys:
{{
  "caption": "1-2 factual sentences",
  "objects": ["lowercase object names"],
  "actions": ["lowercase verb phrases"],
  "zone": "kitchen|living room|office|hallway|outdoors|unknown",
  "tags": ["lowercase search tags"]
}}
"""

SYNTH_PROMPT = """Combine these timestamped frame captions into one dense memory
caption for the window {start:.1f}s to {end:.1f}s. Keep the chronological order,
merge duplicates, and do not invent anything that is not in the frame captions.

Frame captions:
{lines}

Return strict JSON only, with exactly these keys:
{{
  "caption": "2-4 factual sentences",
  "objects": ["lowercase object names"],
  "actions": ["lowercase verb phrases"],
  "zone": "kitchen|living room|office|hallway|outdoors|unknown",
  "tags": ["lowercase search tags"]
}}
"""

# Kept for callers that still want one call with multiple images.
PROMPT = """You are compiling a timestamped memory from an egocentric video.
The images are frames sampled from the window {start:.1f}s to {end:.1f}s.
Frame timestamps in order: {times}.

Write a dense, factual caption of what the person is doing across this window.
Mention the main objects, actions, and the room/zone. Do not guess identities,
brands, or events that are not visible. Prefer concrete nouns and verbs.

Return strict JSON only, with exactly these keys:
{{
  "caption": "2-4 sentences in chronological order",
  "objects": ["lowercase object names"],
  "actions": ["lowercase verb phrases"],
  "zone": "kitchen|living room|office|hallway|outdoors|unknown",
  "tags": ["lowercase search tags"]
}}
"""


def _run(cmd: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def probe_duration(video: Path) -> float:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ]
    )
    return float(result.stdout.strip() or 0.0)


def extract_frame(video: Path, t: float, out: Path, *, width: int = FRAME_WIDTH) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, t):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "3",
            "-y",
            str(out),
        ]
    )
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no frame at t={t:.2f}s")
    return out


def _strip_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip().lower()
        if text and text not in out:
            out.append(text[:80])
    return out[:24]


def ollama_generate_body(
    model: str,
    prompt: str,
    images: list[bytes],
    *,
    host: str = DEFAULT_HOST,
    timeout: float = 300.0,
    json_mode: bool = True,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image).decode("ascii") for image in images],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "top_p": 0.9,
            "num_predict": 400,
        },
    }
    if json_mode:
        payload["format"] = "json"
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"cannot reach ollama at {host}; start it with `ollama serve` and pull {model}"
        ) from exc


def ollama_generate(
    model: str,
    prompt: str,
    images: list[bytes],
    *,
    host: str = DEFAULT_HOST,
    timeout: float = 300.0,
    json_mode: bool = True,
) -> str:
    body = ollama_generate_body(
        model,
        prompt,
        images,
        host=host,
        timeout=timeout,
        json_mode=json_mode,
    )
    return str(body.get("response") or "")


def ollama_text(
    model: str,
    prompt: str,
    *,
    host: str = DEFAULT_HOST,
    timeout: float = 180.0,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "top_p": 0.9,
            "num_predict": 400,
        },
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach ollama at {host}") from exc
    return str(body.get("response") or "")


def parse_caption(text: str, model: str) -> dict[str, Any]:
    parsed_json = False
    try:
        payload = json.loads(_strip_json(text))
        parsed_json = True
    except (TypeError, ValueError):
        payload = {"caption": str(text or "").strip()}
    if not isinstance(payload, dict):
        payload = {"caption": str(text or "").strip()}
    caption = ""
    for key in ("caption", "description", "text", "summary", "scene", "content", "answer"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            caption = value.strip()
            break
    if not caption:
        strings = [value.strip() for value in payload.values() if isinstance(value, str)]
        if strings:
            caption = max(strings, key=len)
    if not caption and not parsed_json:
        caption = str(text or "").strip()
    return {
        "caption": caption[:1200],
        "objects": _as_list(payload.get("objects")),
        "actions": _as_list(payload.get("actions")),
        "zone": str(payload.get("zone") or "unknown").strip().lower()[:40],
        "tags": _as_list(payload.get("tags")),
        "model": model,
    }


def _union(*values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        for item in _as_list(value):
            if item not in out:
                out.append(item)
    return out[:24]


def _reset_model(model: str) -> None:
    """Restart the Ollama runner after a degenerate/empty generation."""
    binary = shutil.which("ollama")
    if not binary:
        return
    try:
        subprocess.run(
            [binary, "stop", model],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        time.sleep(1.0)
    except (OSError, subprocess.SubprocessError):
        return


def caption_frame(
    model: str,
    image: bytes,
    t: float,
    *,
    host: str = DEFAULT_HOST,
    timeout: float = 300.0,
) -> dict[str, Any]:
    prompt = FRAME_PROMPT.format(t=t)
    raw = ""
    for attempt in range(3):
        try:
            raw = ollama_generate(model, prompt, [image], host=host, timeout=timeout)
        except RuntimeError:
            raw = ""
        if raw.strip():
            parsed = parse_caption(raw, model)
            if parsed["caption"]:
                parsed["t"] = round(t, 3)
                return parsed
        if attempt < 2:
            _reset_model(model)
    return {
        "t": round(t, 3),
        "caption": "No clear objects or actions are visible in the frame.",
        "objects": [],
        "actions": [],
        "zone": "unknown",
        "tags": [],
        "model": model,
        "empty": True,
    }


def synthesize_window(
    frames: list[dict[str, Any]],
    start: float,
    end: float,
    *,
    synth_model: str,
    host: str = DEFAULT_HOST,
    timeout: float = 180.0,
) -> dict[str, Any]:
    usable = [frame for frame in frames if str(frame.get("caption") or "").strip()]
    objects = _union(*(frame.get("objects") for frame in frames))
    actions = _union(*(frame.get("actions") for frame in frames))
    tags = _union(*(frame.get("tags") for frame in frames), objects, actions)
    zones = [
        str(frame.get("zone") or "").strip().lower()
        for frame in frames
        if str(frame.get("zone") or "").strip().lower() not in {"", "unknown"}
    ]
    zone = max(set(zones), key=zones.count) if zones else "unknown"
    if not usable:
        return {
            "caption": "",
            "objects": objects,
            "actions": actions,
            "zone": zone,
            "tags": tags,
            "synth_fallback": True,
        }

    lines = "\n".join(
        f"- t={float(frame['t']):.1f}s: {str(frame['caption']).strip()}" for frame in usable
    )
    prompt = SYNTH_PROMPT.format(start=start, end=end, lines=lines)
    try:
        raw = ollama_text(synth_model, prompt, host=host, timeout=timeout)
        parsed = parse_caption(raw, synth_model)
    except RuntimeError:
        parsed = {"caption": ""}
    caption = str(parsed.get("caption") or "").strip()
    fallback = False
    if not caption:
        caption = " ".join(
            f"At t={float(frame['t']):.1f}s, {str(frame['caption']).strip()}"
            for frame in usable
        )
        fallback = True
    return {
        "caption": caption[:1200],
        "objects": _union(parsed.get("objects"), objects),
        "actions": _union(parsed.get("actions"), actions),
        "zone": str(parsed.get("zone") or zone).strip().lower()[:40] or zone,
        "tags": _union(parsed.get("tags"), tags, parsed.get("objects"), parsed.get("actions")),
        "synth_fallback": fallback,
    }


def window_times(start: float, end: float, count: int) -> list[float]:
    if end <= start:
        return []
    if count <= 1 or end - start < 4:
        return [round((start + end) / 2.0, 3)]
    margin = min(2.0, (end - start) * 0.1)
    low, high = start + margin, end - margin
    step = (high - low) / (count - 1)
    return [round(low + step * index, 3) for index in range(count)]


def build_ledger(
    video: Path,
    out: Path,
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    window_s: float = 30.0,
    frames_per_window: int = 3,
    width: int = FRAME_WIDTH,
    max_windows: int = 0,
    keep_frames: Path | None = None,
    timeout: float = 300.0,
    mode: str = "per-frame",
    synth_model: str = DEFAULT_SYNTH_MODEL,
) -> dict[str, Any]:
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    duration = probe_duration(video)
    if duration <= 0:
        raise SystemExit(f"cannot read duration: {video}")

    windows: list[dict[str, Any]] = []
    frame_root = keep_frames
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if frame_root is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="vfm-frames-")
        frame_root = Path(temp_dir.name)
    frame_root.mkdir(parents=True, exist_ok=True)

    start = 0.0
    index = 0
    try:
        while start < duration:
            end = min(start + window_s, duration)
            if end - start < 2.0:
                break
            times = window_times(start, end, frames_per_window)
            blobs: list[bytes] = []
            for t in times:
                path = extract_frame(
                    video, t, frame_root / f"win_{index:03d}_{t:08.2f}.jpg", width=width
                )
                blobs.append(path.read_bytes())

            if mode == "combined":
                time_text = ", ".join(f"{t:.1f}s" for t in times)
                prompt = PROMPT.format(start=start, end=end, times=time_text)
                raw = ollama_generate(model, prompt, blobs, host=host, timeout=timeout)
                parsed = parse_caption(raw, model)
                parsed["frame_times"] = times
            else:
                frame_rows = [
                    caption_frame(model, image, t, host=host, timeout=timeout)
                    for t, image in zip(times, blobs, strict=False)
                ]
                synthesized = synthesize_window(
                    frame_rows,
                    start,
                    end,
                    synth_model=synth_model,
                    host=host,
                    timeout=min(timeout, 180.0),
                )
                parsed = {
                    **synthesized,
                    "frame_times": times,
                    "frames": [
                        {
                            "t": frame["t"],
                            "caption": frame["caption"],
                            "objects": frame["objects"],
                            "actions": frame["actions"],
                            "zone": frame["zone"],
                        }
                        for frame in frame_rows
                    ],
                    "frame_model": model,
                    "synth_model": synth_model,
                }
            parsed.update(
                {
                    "t_start": round(start, 3),
                    "t_end": round(end, 3),
                }
            )
            windows.append(parsed)
            index += 1
            if max_windows and len(windows) >= max_windows:
                break
            start = end
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    ledger = {
        "version": 1,
        "model": model,
        "synth_model": synth_model if mode != "combined" else "",
        "mode": mode,
        "prompt_version": PROMPT_VERSION,
        "window_s": float(window_s),
        "video": video.name,
        "duration_s": round(duration, 3),
        "generated_at": time.time(),
        "windows": windows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    empty = sum(1 for window in windows if not str(window.get("caption") or "").strip())
    return {
        "ledger": str(out),
        "model": model,
        "mode": mode,
        "windows": len(windows),
        "window_s": window_s,
        "frames_per_window": frames_per_window,
        "duration_s": round(duration, 3),
        "bytes": out.stat().st_size,
        "empty_windows": empty,
        "objects": sorted({obj for window in windows for obj in window["objects"]}),
        "actions": sorted({action for window in windows for action in window["actions"]}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--synth-model", default=DEFAULT_SYNTH_MODEL)
    parser.add_argument("--mode", choices=("per-frame", "combined"), default="per-frame")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--frames-per-window", type=int, default=3)
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--keep-frames", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    summary = build_ledger(
        args.video.expanduser().resolve(),
        args.out.expanduser().resolve(),
        model=args.model,
        synth_model=args.synth_model,
        mode=args.mode,
        host=args.host,
        window_s=max(5.0, args.window_s),
        frames_per_window=max(1, args.frames_per_window),
        width=max(160, args.width),
        max_windows=max(0, args.max_windows),
        keep_frames=args.keep_frames.expanduser().resolve() if args.keep_frames else None,
        timeout=max(30.0, args.timeout),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

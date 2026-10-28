#!/usr/bin/env python3
"""Video-to-LLM baseline: send sampled frames to a local VLM, per question.

This is the cost/accuracy comparison the vault has to beat. A real video LLM is
not installed here, so the baseline samples N frames uniformly, composes one
contact sheet with burned-in timestamps, and sends that sheet to Qwen2.5-VL for
every question. The sheet is cached per episode, so only the VLM call is charged
per question.

This script is evaluation-only: it needs Pillow and Ollama. The vault query path
stays stdlib-only.

Usage:
    /home/g1/projects/chronicle/.venv/bin/python prototype/vlm_baseline.py \
      --questions benchmark/questions.json \
      --source-root /home/g1/projects/chronicle/data/real \
      --vault-dir /tmp/vfm \
      --model qwen2.5vl:3b --frames 9
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import benchmark
import caption_ledger
import query
from PIL import Image, ImageDraw, ImageFont, ImageOps

_NUMBERED = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)\b", re.IGNORECASE)
_LOCATION_CUES = {
    "kitchen": {"kitchen", "countertop", "counter", "sink", "cutting board", "toaster", "cabinet"},
    "outdoor path": {"outdoor", "path", "sidewalk", "road", "outside", "walkway"},
    "work table": {"work table", "table", "desk"},
}
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _video(source_root: Path, episode: str) -> Path | None:
    folder = source_root / episode
    for pattern in ("*.mp4", "*.webm", "*.mkv"):
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    prefix = episode.split("_")[0][:4].lower()
    for pattern in ("*.mp4", "*.webm", "*.mkv"):
        for candidate in sorted(source_root.glob(pattern)):
            if prefix and prefix in candidate.name.lower():
                return candidate
    return None


def _frame_times(duration: float, count: int) -> list[float]:
    if duration <= 0 or count <= 0:
        return []
    return [round(duration * (index + 0.5) / count, 3) for index in range(count)]


def contact_sheet(
    video: Path,
    duration: float,
    frames: int,
    out: Path,
    *,
    columns: int = 3,
    tile_width: int = 320,
    tile_height: int = 180,
) -> tuple[Path, list[float]]:
    times = _frame_times(duration, frames)
    rows = (frames + columns - 1) // columns
    pad = 4
    label_h = 24
    sheet = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * pad, rows * (tile_height + label_h) + (rows + 1) * pad),
        (12, 18, 30),
    )
    draw = ImageDraw.Draw(sheet)
    font = _font(18)
    with tempfile.TemporaryDirectory(prefix="vfm-baseline-") as tmp:
        for index, t in enumerate(times):
            frame_path = Path(tmp) / f"frame_{index:03d}.jpg"
            caption_ledger.extract_frame(video, t, frame_path, width=tile_width * 2)
            image = Image.open(frame_path).convert("RGB")
            image = ImageOps.fit(image, (tile_width, tile_height), method=Image.Resampling.LANCZOS)
            col = index % columns
            row = index // columns
            x = pad + col * (tile_width + pad)
            y = pad + row * (tile_height + label_h + pad)
            sheet.paste(image, (x, y))
            draw.rectangle(
                [x, y + tile_height, x + tile_width, y + tile_height + label_h],
                fill=(12, 18, 30),
            )
            draw.text((x + 6, y + tile_height + 2), f"t={t:.1f}s", font=font, fill=(45, 212, 191))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=85)
    return out, times


def clean_answer(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return "unknown"
    if value.startswith("{"):
        try:
            payload = json.loads(caption_ledger._strip_json(value))
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            for key in ("answer", "response", "caption", "text", "description"):
                if isinstance(payload.get(key), str) and payload[key].strip():
                    return payload[key].strip()[:1000]
    return value[:1000]


def question_prompt(question: str) -> str:
    return (
        "The image is a contact sheet of frames sampled uniformly from an egocentric video. "
        "Each tile has a timestamp. Answer the question using only these frames. "
        "If the answer is not visible, answer exactly 'unknown'. "
        "Be concise: one or two sentences, and include the timestamp if the question asks when.\n\n"
        f"Question: {question}"
    )


def ask_vlm(
    model: str,
    sheet: bytes,
    question: str,
    *,
    host: str,
    timeout: float,
) -> tuple[str, float, dict[str, int]]:
    prompt = question_prompt(question)
    started = time.perf_counter()
    usage: dict[str, int] = {}
    for attempt in range(2):
        try:
            body = caption_ledger.ollama_generate_body(
                model,
                prompt,
                [sheet],
                host=host,
                timeout=timeout,
                json_mode=False,
            )
        except RuntimeError:
            body = {}
        usage = {
            key: int(body.get(key) or 0)
            for key in (
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            )
        }
        raw = str(body.get("response") or "")
        if raw.strip():
            return (
                clean_answer(raw),
                (time.perf_counter() - started) * 1000.0,
                usage,
            )
        if attempt == 0:
            caption_ledger._reset_model(model)
    return (
        "unknown",
        (time.perf_counter() - started) * 1000.0,
        usage,
    )


def score_baseline(answer: str, expected: dict[str, Any]) -> tuple[bool, str]:
    text = answer.lower()
    if expected.get("kind") == "unknown":
        ok = any(
            marker in text
            for marker in ("unknown", "don't know", "do not know", "not visible", "no information", "cannot")
        )
        return ok, "unknown handled" if ok else "expected unknown"

    if any(marker in text for marker in ("unknown", "not visible", "cannot see", "can't see")):
        return False, "VLM said unknown"

    for term in expected.get("must_contain", []):
        wanted = str(term).lower()
        if wanted in text:
            continue
        cues = _LOCATION_CUES.get(wanted)
        if cues and any(cue in text for cue in cues):
            continue
        return False, f"missing term: {term}"

    if expected.get("t_near") is not None:
        target = float(expected["t_near"])
        tolerance = float(expected.get("t_tol", 1.0))
        times = [float(value) for value in _NUMBERED.findall(text)]
        if not any(abs(value - target) <= tolerance for value in times):
            return False, f"no timestamp within {tolerance}s of {target}s"

    # No citations/evidence in the video baseline; content + timestamp is the score.
    return True, "ok"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * pct / 100.0) - 1))
    return ordered[index]


def run(
    questions_path: Path,
    source_root: Path,
    vault_dir: Path,
    *,
    model: str,
    frames: int,
    limit: int = 0,
    host: str = caption_ledger.DEFAULT_HOST,
    timeout: float = 300.0,
    keep_sheets: Path | None = None,
    input_price_per_mtok: float = 0.15,
    output_price_per_mtok: float = 0.60,
    gpu_watts: float = 80.0,
    electricity_price_per_kwh: float = 0.15,
    json_output: bool = False,
    ask_fn: Any | None = None,
) -> dict[str, Any]:
    questions = benchmark._load_questions(questions_path)
    if limit:
        questions = questions[:limit]
    episodes = sorted({str(item["episode"]) for item in questions})

    sheets: dict[str, dict[str, Any]] = {}
    sheet_dir = keep_sheets
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if sheet_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="vfm-sheets-")
        sheet_dir = Path(temp_dir.name)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    try:
        for episode in episodes:
            video = _video(source_root, episode)
            if video is None:
                raise SystemExit(f"no video for episode {episode}")
            duration = caption_ledger.probe_duration(video)
            started = time.perf_counter()
            path, times = contact_sheet(
                video, duration, frames, sheet_dir / f"{episode}_sheet.jpg"
            )
            sheets[episode] = {
                "path": path,
                "bytes": path.stat().st_size,
                "times": times,
                "duration": duration,
                "video_bytes": video.stat().st_size,
                "build_ms": (time.perf_counter() - started) * 1000.0,
                "video": str(video),
            }

        stores: dict[str, Any] = {}
        connections: dict[str, Any] = {}
        try:
            for episode in episodes:
                con = sqlite3.connect(str(vault_dir / f"{episode}.db"))
                connections[episode] = con
                stores[episode] = query.load_store(con)

            rows: list[dict[str, Any]] = []
            baseline_latencies: list[float] = []
            vault_latencies: list[float] = []
            categories: dict[str, dict[str, int]] = {}
            for item in questions:
                episode = str(item["episode"])
                expected = item.get("expected") or {}
                store = stores[episode]

                started = time.perf_counter()
                vault_result = query.answer(store, str(item["question"]), use_cache=False)
                vault_ms = (time.perf_counter() - started) * 1000.0
                vault_latencies.append(vault_ms)
                vault_ok, vault_reason = benchmark.score_answer(vault_result, expected)

                sheet = sheets[episode]
                if ask_fn is not None:
                    answer, vlm_ms, usage = ask_fn(
                        sheet["path"].read_bytes(), str(item["question"])
                    )
                else:
                    answer, vlm_ms, usage = ask_vlm(
                        model,
                        sheet["path"].read_bytes(),
                        str(item["question"]),
                        host=host,
                        timeout=timeout,
                    )
                baseline_latencies.append(vlm_ms)
                baseline_ok, baseline_reason = score_baseline(answer, expected)
                prompt_tokens = int(usage.get("prompt_eval_count", 0))
                output_tokens = int(usage.get("eval_count", 0))
                total_tokens = prompt_tokens + output_tokens

                category = str(item["category"])
                bucket = categories.setdefault(category, {"vault": 0, "vlm": 0, "total": 0})
                bucket["total"] += 1
                bucket["vault"] += int(vault_ok)
                bucket["vlm"] += int(baseline_ok)
                rows.append(
                    {
                        "id": item["id"],
                        "episode": episode,
                        "category": category,
                        "question": item["question"],
                        "vault_correct": vault_ok,
                        "vault_method": vault_result.method,
                        "vault_answer": vault_result.answer.splitlines()[0][:220],
                        "vlm_correct": baseline_ok,
                        "vlm_reason": baseline_reason,
                        "vlm_answer": answer[:300],
                        "vault_ms": round(vault_ms, 3),
                        "vlm_ms": round(vlm_ms, 1),
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "vlm_frames": frames,
                        "sheet_bytes": sheet["bytes"],
                    }
                )
        finally:
            for con in connections.values():
                con.close()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    vault_correct = sum(1 for row in rows if row["vault_correct"])
    vlm_correct = sum(1 for row in rows if row["vlm_correct"])
    prompt_tokens_total = sum(row["prompt_tokens"] for row in rows)
    output_tokens_total = sum(row["output_tokens"] for row in rows)
    total_tokens_total = prompt_tokens_total + output_tokens_total
    video_bytes_total = sum(int(sheet["video_bytes"]) for sheet in sheets.values())
    sheet_bytes_total = sum(int(sheet["bytes"]) for sheet in sheets.values())

    vault_bytes_total = 0
    evidence_bytes_total = 0
    for episode in episodes:
        path = vault_dir / f"{episode}.db"
        vault_bytes_total += benchmark._size(path)
        con = sqlite3.connect(str(path))
        try:
            info = query.stats(query.load_store(con), path)
            evidence_bytes_total += int(info.get("evidence_bytes", 0))
        finally:
            con.close()
    memory_bytes_total = vault_bytes_total + evidence_bytes_total

    prompt_cost = prompt_tokens_total / 1_000_000.0 * input_price_per_mtok
    output_cost = output_tokens_total / 1_000_000.0 * output_price_per_mtok
    api_cost = prompt_cost + output_cost
    energy_wh = sum(baseline_latencies) / 1000.0 * gpu_watts / 3600.0
    energy_cost = energy_wh / 1000.0 * electricity_price_per_kwh
    report = {
        "questions": len(rows),
        "model": model,
        "frames_per_question": frames,
        "sheets": {
            episode: {
                key: value
                for key, value in sheet.items()
                if key != "path"
            }
            for episode, sheet in sheets.items()
        },
        "vault": {
            "correct": vault_correct,
            "accuracy": round(vault_correct / len(rows), 3) if rows else 0.0,
            "p50_ms": round(statistics.median(vault_latencies), 3),
            "p95_ms": round(_percentile(vault_latencies, 95), 3),
            "video_required_at_query": False,
            "bytes_at_rest": memory_bytes_total,
            "vault_bytes": vault_bytes_total,
            "evidence_bytes": evidence_bytes_total,
        },
        "video_llm": {
            "correct": vlm_correct,
            "accuracy": round(vlm_correct / len(rows), 3) if rows else 0.0,
            "p50_ms": round(statistics.median(baseline_latencies), 1),
            "p95_ms": round(_percentile(baseline_latencies, 95), 1),
            "video_required_at_query": True,
            "vlm_calls": len(rows),
            "frames_per_question": frames,
            "prompt_tokens": prompt_tokens_total,
            "output_tokens": output_tokens_total,
            "total_tokens": total_tokens_total,
            "mean_tokens_per_question": round(total_tokens_total / len(rows), 1) if rows else 0.0,
            "api_cost_usd": round(api_cost, 6),
            "api_cost_per_question_usd": round(api_cost / len(rows), 6) if rows else 0.0,
            "input_price_per_mtok": input_price_per_mtok,
            "output_price_per_mtok": output_price_per_mtok,
            "gpu_energy_wh": round(energy_wh, 2),
            "gpu_energy_cost_usd": round(energy_cost, 6),
            "video_bytes_at_query": video_bytes_total,
            "contact_sheet_bytes": sheet_bytes_total,
        },
        "size": {
            "video_bytes": video_bytes_total,
            "memory_bytes": memory_bytes_total,
            "contact_sheet_bytes": sheet_bytes_total,
            "compression_ratio_video_to_memory": round(
                video_bytes_total / memory_bytes_total, 1
            )
            if memory_bytes_total
            else None,
        },
        "categories": {
            name: {
                "vault": data["vault"],
                "vlm": data["vlm"],
                "total": data["total"],
            }
            for name, data in sorted(categories.items())
        },
        "rows": rows,
    }
    if json_output:
        print(json.dumps(report, indent=2))
        return report

    print(f"questions       : {report['questions']}")
    print(f"video-to-LLM    : {vlm_correct}/{len(rows)} = {report['video_llm']['accuracy']:.1%}")
    print(f"vault           : {vault_correct}/{len(rows)} = {report['vault']['accuracy']:.1%}")
    print(
        f"latency         : video-LLM p50 {report['video_llm']['p50_ms']:.1f} ms · "
        f"p95 {report['video_llm']['p95_ms']:.1f} ms; "
        f"vault p50 {report['vault']['p50_ms']:.3f} ms · p95 {report['vault']['p95_ms']:.3f} ms"
    )
    print(
        f"tokens          : {report['video_llm']['prompt_tokens']} prompt + "
        f"{report['video_llm']['output_tokens']} output = "
        f"{report['video_llm']['total_tokens']} total "
        f"({report['video_llm']['mean_tokens_per_question']}/question)"
    )
    print(
        f"API-equivalent  : ${report['video_llm']['api_cost_usd']:.4f} for "
        f"{report['questions']} questions "
        f"(${report['video_llm']['api_cost_per_question_usd']:.4f}/question) at "
        f"${report['video_llm']['input_price_per_mtok']}/M input and "
        f"${report['video_llm']['output_price_per_mtok']}/M output"
    )
    print(
        f"local energy    : {report['video_llm']['gpu_energy_wh']:.2f} Wh "
        f"(~${report['video_llm']['gpu_energy_cost_usd']:.4f}) at "
        f"{gpu_watts:g} W / ${electricity_price_per_kwh:g} per kWh"
    )
    print(
        f"size            : video {report['size']['video_bytes']/1048576:.2f} MB at query; "
        f"memory {report['size']['memory_bytes']/1024:.1f} KB at rest; "
        f"contact sheets {report['size']['contact_sheet_bytes']/1024:.1f} KB; "
        f"video/memory {report['size']['compression_ratio_video_to_memory']}x"
    )
    for episode, sheet in report["sheets"].items():
        print(
            f"sheet {episode:9}: {frames} frames, {sheet['bytes']/1024:.1f} KB, "
            f"{sheet['video_bytes']/1048576:.2f} MB video, build {sheet['build_ms']:.0f} ms"
        )
    print()
    print("| Category | Vault | Video-LLM | Total |")
    print("|---|---:|---:|---:|")
    for name, data in report["categories"].items():
        print(f"| {name} | {data['vault']} | {data['vlm']} | {data['total']} |")
    print()
    print("| ID | Category | Vault | VLM | VLM answer |")
    print("|---|---|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['id']} | {row['category']} | "
            f"{'yes' if row['vault_correct'] else 'no'} | "
            f"{'yes' if row['vlm_correct'] else 'no'} | "
            f"{row['vlm_answer'].replace('|', '/')[:180]} |"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--vault-dir", required=True, type=Path)
    parser.add_argument("--model", default=caption_ledger.DEFAULT_MODEL)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--host", default=caption_ledger.DEFAULT_HOST)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--keep-sheets", type=Path, default=None)
    parser.add_argument("--input-price-per-mtok", type=float, default=0.15)
    parser.add_argument("--output-price-per-mtok", type=float, default=0.60)
    parser.add_argument("--gpu-watts", type=float, default=80.0)
    parser.add_argument("--electricity-price-per-kwh", type=float, default=0.15)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    run(
        args.questions.expanduser().resolve(),
        args.source_root.expanduser().resolve(),
        args.vault_dir.expanduser().resolve(),
        model=args.model,
        frames=max(1, args.frames),
        limit=max(0, args.limit),
        host=args.host,
        timeout=max(30.0, args.timeout),
        keep_sheets=args.keep_sheets.expanduser().resolve() if args.keep_sheets else None,
        input_price_per_mtok=max(0.0, args.input_price_per_mtok),
        output_price_per_mtok=max(0.0, args.output_price_per_mtok),
        gpu_watts=max(1.0, args.gpu_watts),
        electricity_price_per_kwh=max(0.0, args.electricity_price_per_kwh),
        json_output=args.json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

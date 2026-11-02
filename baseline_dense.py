#!/usr/bin/env python3
"""Dense 1 FPS Florence-2 baseline for Chronicle evaluation.

The baseline is deliberately simple and strong on paper:

1. Sample the video at ``sample_fps`` (default 1 FPS).
2. Caption *every* sampled frame with the local Florence-2 model.
3. Store the captions as a flat timestamped text memory.
4. Answer the same Ego4D QA pairs using only captions at or before each
   question timestamp.
5. Write ``baseline_results.json`` with Recall@1 and efficiency metrics.

It is the direct comparison for event-gated Chronicle: same video, same QA,
same local model, but no gating, no graph and no event selection.

Usage:
    python baseline_dense.py \
      --video data/real/cook/aria_cook_45s.mp4 \
      --qa data/qa/ego4d_example.json \
      --out runs/eval/baseline_cook

For a no-model harness test:
    python baseline_dense.py --captioner stub --max-captions 10 --offline-paths
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_pipeline import (
    ChronicleEngine,
    MetricsTracker,
    ProgressBar,
    _content_tokens,
    _normalise,
    _write_json,
    instrument_chat_client,
    score_answer,
)

LOGGER = logging.getLogger("chronicle.baseline")
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Flat text memory
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CaptionEntry:
    t: float
    caption: str


@dataclass
class FlatTextMemory:
    entries: list[CaptionEntry] = field(default_factory=list)

    def add(self, t: float, caption: str) -> None:
        text = str(caption or "").strip()
        if text:
            self.entries.append(CaptionEntry(float(t), text))

    def _visible(self, at: float | None) -> list[CaptionEntry]:
        if at is None:
            return list(self.entries)
        # A small tolerance accepts the frame nearest the requested timestamp.
        return [entry for entry in self.entries if entry.t <= float(at) + 0.5]

    def retrieve(self, query: str, at: float | None, *, k: int = 8) -> list[CaptionEntry]:
        tokens = _content_tokens(query)
        visible = self._visible(at)
        if not visible:
            return []
        if not tokens:
            return visible[-k:]
        scored: list[tuple[float, CaptionEntry]] = []
        for entry in visible:
            caption_tokens = _content_tokens(entry.caption)
            overlap = len(tokens & caption_tokens) / max(1, len(tokens))
            temporal = 0.0
            if at is not None:
                temporal = 0.15 / (1.0 + abs(entry.t - float(at)))
            scored.append((overlap + temporal, entry))
        scored.sort(key=lambda item: (item[0], item[1].t), reverse=True)
        hits = [entry for score, entry in scored if score > 0]
        return hits[:k]

    def answer(self, query: str, at: float | None) -> tuple[str, list[CaptionEntry]]:
        """Answer from retrieved captions with a deterministic template."""

        lower = _normalise(query)
        hits = self.retrieve(query, at)
        if not hits:
            return "I don't have that in my memory.", []
        tokens = _content_tokens(query)
        object_tokens = {token for token in tokens if len(token) > 2}
        matching = [entry for entry in hits if object_tokens & _content_tokens(entry.caption)]
        if not matching:
            return "I don't have that in my memory.", []

        if re.search(r"\bwhere\b|\bwhich room\b|\blocated\b", lower):
            location = next(
                (
                    entry
                    for entry in matching
                    if re.search(r"\b(in|on|at|inside|near|under|behind)\b", _normalise(entry.caption))
                ),
                matching[0],
            )
            return f"At t={location.t:.1f}s, {location.caption}", [location]

        if re.search(r"\bwhen\b|\bat what time\b|\bwhat time\b", lower):
            latest = "last" not in lower
            entry = matching[0] if latest else matching[-1]
            return f"At t={entry.t:.1f}s, {entry.caption}", [entry]

        if re.search(r"\bhow many\b|\bcount\b", lower):
            label = next(iter(sorted(object_tokens)), "object")
            windows = {round(entry.t) for entry in matching}
            return (
                f"The captions mention the {label} in {len(windows)} one-second "
                f"window{'s' if len(windows) != 1 else ''}. This is caption "
                "coverage, not a verified physical count.",
                matching[:4],
            )

        if re.search(r"^\s*(did|do|does|have|has|is|are|was|were)\b", lower):
            entry = matching[0]
            return f"Yes. At t={entry.t:.1f}s, {entry.caption}", [entry]

        if re.search(r"\bwhat did i (see|observe)\b|\binventory\b|\boverview\b|\bwhat objects\b", lower):
            return (
                f"The flat caption memory has {len(self.entries)} one-second captions. "
                f"Most relevant: {matching[0].caption}",
                matching[:4],
            )

        entry = matching[0] if "first" in lower else matching[-1] if "last" in lower else matching[0]
        return f"At t={entry.t:.1f}s, {entry.caption}", [entry]


def answer_with_llm(
    memory: FlatTextMemory,
    query: str,
    at: float | None,
    client: Any,
    *,
    k: int = 8,
) -> tuple[str, list[CaptionEntry]]:
    """Optional local LLM phrasing over retrieved captions (no video)."""

    hits = memory.retrieve(query, at, k=k)
    if not hits:
        return "I don't have that in my memory.", []
    context = "\n".join(f"[t={entry.t:.1f}s] {entry.caption}" for entry in hits)
    system = (
        "Answer the question using ONLY the timestamped captions. "
        "If the answer is absent, say: I don't have that in my memory. "
        "Cite the timestamp as t=<seconds>s. Be concise."
    )
    prompt = f"Captions:\n{context}\n\nQuestion: {query}\nAnswer:"
    try:
        return str(client.complete(system, prompt)).strip(), hits
    except Exception:
        LOGGER.exception("baseline LLM query failed; using deterministic answer")
        return memory.answer(query, at)


# ---------------------------------------------------------------------------
# Dense ingestion
# ---------------------------------------------------------------------------
def build_captioner(config: Mapping[str, Any], *, backend: str, device: str | None = None) -> Any:
    from chronicle.semantic.config import SemanticConfig
    from chronicle.semantic.florence import build_captioner as _build

    semantic = _section(config, "semantic")
    model_id = str(semantic.get("florence_model_id", "models/Florence-2-base"))
    model_path = Path(model_id).expanduser()
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path).resolve()
    if (
        model_path.exists()
        or Path(model_id).is_absolute()
        or model_id.startswith(("models/", "models\\", "./", "../"))
    ):
        model_id = str(model_path)
    cfg = SemanticConfig(
        florence_model_id=model_id,
        device=device or str(semantic.get("device", "auto")),
        florence_max_new_tokens=int(semantic.get("florence_max_new_tokens", 128)),
        florence_num_beams=int(semantic.get("florence_num_beams", 3)),
        qwen_vl_model=str(semantic.get("qwen_vl_model", "qwen2.5vl:3b")),
        qwen_vl_host=str(semantic.get("qwen_vl_host", "http://127.0.0.1:11434")),
        qwen_vl_timeout_s=float(semantic.get("qwen_vl_timeout_s", 120.0)),
        qwen_vl_max_side=int(semantic.get("qwen_vl_max_side", 448)),
        action_task=str(semantic.get("action_task", "<MORE_DETAILED_CAPTION>")),
    )
    selected = str(backend or "florence").lower()
    if selected in {"qwen-vl", "qwen_vl", "qwen2.5vl"}:
        selected = "qwen"
    if selected not in {"florence", "stub", "qwen"}:
        raise ValueError("baseline captioner must be 'florence', 'qwen' or 'stub'")
    captioner = _build(cfg, backend=selected)
    if selected == "qwen":
        from chronicle.semantic.qwen_vl import QwenVLCaptioner

        if not isinstance(captioner, QwenVLCaptioner):
            raise RuntimeError("Qwen2.5-VL captioner could not be constructed")
        if not captioner.is_available():
            raise RuntimeError(
                f"Qwen2.5-VL is not available at {captioner.host}. Start "
                f"`ollama serve` and run `ollama pull {captioner.model_id}`."
            )
        return captioner
    if selected == "florence":
        # The script must not silently run the stub and call it a baseline.
        from chronicle.semantic.florence import FlorenceCaptioner, LazyCaptioner

        target = captioner._resolve() if isinstance(captioner, LazyCaptioner) else captioner  # noqa: SLF001
        if not isinstance(target, FlorenceCaptioner):
            raise RuntimeError(
                "Florence-2 is not available. Run `python scripts/fetch_models.py --check` "
                "or use --captioner stub only for harness tests."
            )
        target._load()  # noqa: SLF001 - preload before progress starts
        return target
    return captioner


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def sample_video(
    video: Path,
    memory: FlatTextMemory,
    metrics: MetricsTracker,
    captioner: Any,
    *,
    sample_fps: float = 1.0,
    max_captions: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Decode sequentially and caption at ``sample_fps``."""

    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else None
    interval = 1.0 / max(1e-6, sample_fps)
    total = int(duration * sample_fps) if duration else None
    bar = ProgressBar(total, desc="Dense 1 FPS", unit="caption") if show_progress else None

    next_sample = 0.0
    index = 0
    captions = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = index / fps
        if timestamp + 1e-9 >= next_sample:
            started = time.perf_counter()
            try:
                caption = captioner.caption(frame)
            except Exception as exc:
                LOGGER.exception("caption failed at t=%.2fs", timestamp)
                caption = ""
                metrics.error = f"{type(exc).__name__}: {exc}"
            latency = (time.perf_counter() - started) * 1000.0
            memory.add(timestamp, caption)
            metrics.record_frame(latency, timestamp=timestamp)
            metrics.record_vlm_call(
                latency_ms=latency,
                backend=type(getattr(captioner, "_impl", None) or captioner).__name__,
                trigger="dense_caption",
            )
            captions += 1
            next_sample += interval
            if bar is not None:
                bar.update(1, postfix=f"t={timestamp:.1f}s")
            if max_captions is not None and captions >= max_captions:
                break
        index += 1
    capture.release()
    if bar is not None:
        bar.close()
    return {
        "fps": round(fps, 3),
        "frames": frame_count,
        "duration_s": round(duration, 3) if duration else None,
        "sample_fps": sample_fps,
        "captions": captions,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _row(
    pair: Any,
    answer_text: str,
    hits: Sequence[CaptionEntry],
    *,
    answered_at: float | None,
    latency_ms: float,
    unreachable: bool,
    error: str | None = None,
) -> dict[str, Any]:
    expected_alternatives = [pair.answer] if pair.answer else [""]
    raw = pair.raw or {}
    for key in ("answers", "acceptable_answers", "answer_aliases"):
        value = raw.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            candidate = [str(item) for item in value if item]
            if candidate:
                expected_alternatives = candidate
                break
    best = (False, 0.0, "no expected answer")
    for expected in expected_alternatives:
        scored = score_answer(answer_text, expected, raw=pair.raw)
        if scored[0] or best[2] == "no expected answer" or scored[1] > best[1]:
            best = scored
    return {
        "video_id": pair.video_id,
        "query": pair.query,
        "expected": pair.answer,
        "predicted": answer_text,
        "correct": bool(best[0]),
        "score": round(float(best[1]), 3),
        "reason": best[2],
        "category": str(raw.get("category") or "uncategorized"),
        "qa_timestamp": pair.timestamp,
        "answered_at": answered_at,
        "unreachable_timestamp": bool(unreachable),
        "query_latency_ms": round(latency_ms, 4),
        "method": "dense-flat-text",
        "citations": [{"t": entry.t, "caption": entry.caption} for entry in hits[:6]],
        "error": error,
    }


def run_baseline(
    video_path: str | Path,
    qa_json_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    out_dir: str | Path | None = None,
    sample_fps: float | None = None,
    captioner_backend: str | None = None,
    max_captions: int | None = None,
    device: str | None = None,
    llm_backend: str | None = None,
    llm_model: str | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    engine = ChronicleEngine(
        video_path=video_path,
        qa_json_path=qa_json_path,
        config_path=config_path,
    )
    config = engine.raw_config
    baseline_cfg = _section(config, "baseline")
    fps = float(sample_fps or baseline_cfg.get("sample_fps", 1.0))
    backend = str(captioner_backend or baseline_cfg.get("captioner_backend", "florence"))
    budget = max_captions if max_captions is not None else baseline_cfg.get("max_captions")
    budget = int(budget) if budget not in (None, "", 0) else None
    out = Path(out_dir).expanduser().resolve() if out_dir else engine.out_dir / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    qa_pairs = list(engine.qa_pairs)

    metrics_cfg = _section(config, "metrics")
    metrics = MetricsTracker(
        percentiles=metrics_cfg.get("percentiles", [50, 95, 99]),
        track_vram=bool(metrics_cfg.get("track_vram", True)),
        warmup_frames=int(metrics_cfg.get("warmup_frames", 1)),
    )
    metrics.start()
    captioner = build_captioner(config, backend=backend, device=device)
    memory = FlatTextMemory()
    ingest_meta = sample_video(
        engine.video_path,
        memory,
        metrics,
        captioner,
        sample_fps=fps,
        max_captions=budget,
        show_progress=show_progress,
    )

    # Optional local text LLM (never required, never internet).
    semantic_cfg = _section(config, "semantic")
    llm_backend_selected = str(
        llm_backend or baseline_cfg.get("llm_backend", "template")
    ).lower()
    llm_model_selected = str(
        llm_model or semantic_cfg.get("llm_model", "qwen2.5:7b-instruct")
    )
    client = None
    if llm_backend_selected not in {"template", "none", ""}:
        from chronicle.semantic.config import SemanticConfig
        from chronicle.semantic.llm import build_chat_client

        client = build_chat_client(
            SemanticConfig(
                llm_backend=llm_backend_selected,
                llm_model=llm_model_selected,
                ollama_url=str(semantic_cfg.get("ollama_url", "http://127.0.0.1:11434")),
            )
        )
        instrument_chat_client(client, metrics)

    rows: list[dict[str, Any]] = []
    for pair in qa_pairs:
        at = pair.timestamp
        unreachable = bool(at is not None and ingest_meta["duration_s"] and at > ingest_meta["duration_s"] + 0.5)
        if at is not None and not unreachable:
            at = min(float(at), float(ingest_meta["duration_s"] or at))
        started = time.perf_counter()
        error = None
        try:
            if client is not None:
                answer_text, hits = answer_with_llm(memory, pair.query, at, client)
            else:
                answer_text, hits = memory.answer(pair.query, at)
        except Exception as exc:
            LOGGER.exception("baseline query failed: %s", pair.query)
            answer_text, hits = "", []
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            _row(
                pair,
                answer_text,
                hits,
                answered_at=at,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                unreachable=unreachable,
                error=error,
            )
        )
        if show_progress:
            print(f"\rBaseline QA {len(rows)}/{len(qa_pairs)}", end="", flush=True)
    if show_progress and rows:
        print()

    metrics.finished_at = time.perf_counter()
    metrics_payload = metrics.to_dict(
        extra={
            "method": "dense-1fps",
            "video_metadata": ingest_meta,
            "captioner": backend,
            "llm_backend": llm_backend_selected,
            "llm_model": llm_model_selected if client is not None else None,
            "memory_entries": len(memory.entries),
        }
    )
    correct = sum(1 for row in rows if row["correct"])
    total = len(rows)
    categories: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = categories.setdefault(str(row["category"]), {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(row["correct"])
    for bucket in categories.values():
        bucket["recall_at_1"] = round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0
    payload = {
        "method": "dense-1fps",
        "video": str(engine.video_path),
        "qa_file": str(engine.qa_path),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "video_metadata": ingest_meta,
        "total": total,
        "correct": correct,
        "recall_at_1": round(correct / total, 4) if total else 0.0,
        "categories": dict(sorted(categories.items())),
        "metrics": metrics_payload,
        "rows": rows,
    }
    _write_json(out / "baseline_results.json", payload)
    (out / "baseline_summary.md").write_text(_baseline_summary(payload), encoding="utf-8")
    if show_progress:
        print(
            f"Recall@1 {payload['recall_at_1']:.1%} ({correct}/{total}) · "
            f"VLM calls {metrics_payload.get('vlm_calls')} · "
            f"peak VRAM {metrics_payload.get('peak_vram_mb')} MB"
        )
        print(f"wrote {out / 'baseline_results.json'}")
    return payload


def _baseline_summary(results: Mapping[str, Any]) -> str:
    metrics = results.get("metrics") or {}
    rows = results.get("rows") or []
    lines = [
        "# Dense 1 FPS baseline summary",
        "",
        f"- **Video:** `{results.get('video', '')}`",
        f"- **QA file:** `{results.get('qa_file', '')}`",
        f"- **Questions:** {results.get('total', 0)}",
        f"- **Recall@1:** {results.get('recall_at_1', 0.0):.1%} "
        f"({results.get('correct', 0)}/{results.get('total', 0)})",
        f"- **Captions:** {metrics.get('memory_entries')}",
        "",
        "## Efficiency",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sampled frames / captions | {metrics.get('frames_processed')} |",
        f"| VLM calls | {metrics.get('vlm_calls')} |",
        f"| Text-LLM calls | {metrics.get('llm_calls')} |",
        f"| Text-LLM latency (mean) | {metrics.get('avg_llm_latency_ms')} ms |",
        f"| Text-LLM models | {metrics.get('llm_models')} |",
        f"| Average caption latency | {metrics.get('avg_latency_ms')} ms |",
        f"| Steady-state latency | {metrics.get('avg_latency_steady_ms')} ms |",
        f"| p50 / p95 | {metrics.get('p50_ms')} / {metrics.get('p95_ms')} ms |",
        f"| Peak VRAM | {metrics.get('peak_vram_mb')} MB |",
        "",
        "## Per-question results",
        "",
        "| # | Question | Expected | Predicted | Correct |",
        "|---:|---|---|---|---:|",
    ]
    for index, row in enumerate(rows, 1):
        def safe(value: Any) -> str:
            return str(value or "").replace("|", "/").replace("\n", " ")

        lines.append(
            f"| {index} | {safe(row.get('query'))} | {safe(row.get('expected'))} | "
            f"{safe(row.get('predicted'))[:240]} | {'yes' if row.get('correct') else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Every 1 FPS sample is captioned; VLM calls equal sampled frames by construction.",
            "- Retrieval only sees captions at or before the question timestamp.",
            "- Baseline answers use deterministic templates unless `baseline.llm_backend` is set.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None, help="sampling rate; default from config")
    parser.add_argument("--captioner", choices=("florence", "qwen", "stub"), default=None)
    parser.add_argument("--max-captions", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--llm-backend",
        choices=("template", "none", "ollama", "auto", "transformers"),
        default=None,
        help="answer phrasing backend; ollama uses a local Qwen model",
    )
    parser.add_argument("--llm-model", default=None, help="Ollama model name")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    payload = run_baseline(
        args.video,
        args.qa,
        config_path=args.config,
        out_dir=args.out,
        sample_fps=args.fps,
        captioner_backend=args.captioner,
        max_captions=args.max_captions,
        device=args.device,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        show_progress=not args.no_progress and not args.json,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not (payload.get("metrics") or {}).get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())

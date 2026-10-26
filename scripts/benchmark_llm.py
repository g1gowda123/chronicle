"""Benchmark GraphRAG answers across backends: accuracy, latency, tokens.

Runs the deterministic evaluation set through the template path, local Ollama
and/or Gemini (OpenAI-compatible endpoint, key from ``GEMINI_API_KEY``) and
writes a JSON report plus a console table.

Examples::

    python scripts/benchmark_llm.py --backend template
    python scripts/benchmark_llm.py --backend ollama --model qwen2.5:7b-instruct
    GEMINI_API_KEY=... python scripts/benchmark_llm.py --backend gemini --model gemini-2.5-flash
    GEMINI_API_KEY=... python scripts/benchmark_llm.py --backend all
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from chronicle.semantic.config import SemanticConfig
from chronicle.semantic.engine import SemanticEngine
from chronicle.semantic.evaluation import DEFAULT_GRAPHRAG_CASES, evaluate_graphrag
from chronicle.semantic.llm import GeminiChatClient, OllamaChatClient

OUT_DIR = Path("runs/benchmark")


def _engine(backend: str, model: str, api_key: str | None) -> tuple[SemanticEngine, Any]:
    if backend == "template":
        return (
            SemanticEngine(
                config=SemanticConfig(llm_backend="template"),
                chat_client=None,
                embedder_backend="deterministic",
                captioner_backend="stub",
            ),
            None,
        )
    config = SemanticConfig(llm_backend="ollama", llm_model=model)
    if backend == "ollama":
        client: Any = OllamaChatClient(model=model)
    elif backend == "gemini":
        client = GeminiChatClient(model=model, api_key=api_key)
    else:  # pragma: no cover - argparse restricts values
        raise ValueError(backend)
    return (
        SemanticEngine(
            config=config,
            chat_client=client,
            embedder_backend="deterministic",
            captioner_backend="stub",
        ),
        client,
    )


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def run_backend(backend: str, model: str, *, api_key: str | None, sleep: float) -> dict[str, Any]:
    engine, client = _engine(backend, model, api_key)
    latencies: list[float] = []

    def answer(query: str, context: Any) -> Any:
        started = time.perf_counter()
        result = engine.graphrag.answer(query, context)
        latencies.append(time.perf_counter() - started)
        if sleep:
            time.sleep(sleep)
        return result

    started = time.perf_counter()
    report = evaluate_graphrag(answer, DEFAULT_GRAPHRAG_CASES)
    elapsed = time.perf_counter() - started

    calls = list(getattr(client, "calls", [])) if client is not None else []
    tokens = {
        "prompt": sum(call.get("prompt_tokens", 0) for call in calls),
        "completion": sum(call.get("completion_tokens", 0) for call in calls),
        "total": sum(call.get("total_tokens", 0) for call in calls),
    }
    llm_sources = [case for case in report.cases if not str(case.source).startswith("template")]
    payload = report.to_dict()
    payload.update(
        {
            "backend": backend,
            "model": model,
            "elapsed_s": round(elapsed, 2),
            "latency_mean_s": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "latency_median_s": round(statistics.median(latencies), 3) if latencies else 0.0,
            "latency_p95_s": round(_p95(latencies), 3),
            "llm_calls": len(calls),
            "llm_answer_rate": round(len(llm_sources) / max(1, len(report.cases)), 3),
            "tokens": tokens,
            "sleep_s": sleep,
        }
    )
    return payload


def _print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'backend':<10} {'model':<26} {'pass':>6} {'cite':>6} {'llm%':>6} "
        f"{'mean_s':>7} {'p95_s':>6} {'tokens':>7}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['backend']:<10} {str(result['model'])[:26]:<26} "
            f"{result['pass_rate'] * 100:>5.0f}% {result['citation_precision'] * 100:>5.0f}% "
            f"{result['llm_answer_rate'] * 100:>5.0f}% {result['latency_mean_s']:>7.2f} "
            f"{result['latency_p95_s']:>6.2f} {result['tokens']['total']:>7}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark GraphRAG answer paths")
    parser.add_argument("--backend", choices=("template", "ollama", "gemini", "all"), default="all")
    parser.add_argument("--model", default="", help="model name (defaults per backend)")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--ollama-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--sleep", type=float, default=0.0, help="pause between cases (rate limits)")
    parser.add_argument("--out", default=str(OUT_DIR / "graphrag_benchmark.json"))
    args = parser.parse_args(argv)

    backends = ["template", "ollama", "gemini"] if args.backend == "all" else [args.backend]
    results: list[dict[str, Any]] = []
    for backend in backends:
        model = args.model or (
            args.gemini_model if backend == "gemini"
            else args.ollama_model if backend == "ollama"
            else "deterministic"
        )
        if backend == "gemini":
            client = GeminiChatClient(model=model)
            if not client.api_key:
                print("skip gemini: GEMINI_API_KEY is not set")
                continue
        if backend == "ollama" and not OllamaChatClient(model=model).is_available():
            print("skip ollama: daemon not reachable")
            continue
        pause = args.sleep or (4.5 if backend == "gemini" else 0.0)
        print(f"running {backend} ({model}) with {len(DEFAULT_GRAPHRAG_CASES)} cases…")
        results.append(run_backend(backend, model, api_key=None, sleep=pause))

    if not results:
        print("nothing to benchmark")
        return 1

    _print_table(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": len(DEFAULT_GRAPHRAG_CASES), "results": results}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

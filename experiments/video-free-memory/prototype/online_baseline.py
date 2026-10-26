#!/usr/bin/env python3
"""Online VLM baseline: send the sampled contact sheet to a hosted vision model.

The local baseline uses Ollama. This script runs the same 30 questions against a
popular online API so accuracy, latency, token use, cost and size can be compared
on equal footing.

Supported providers:
  openai     OPENAI_API_KEY       default model gpt-4o-mini
  anthropic  ANTHROPIC_API_KEY    default model claude-3-5-sonnet-latest
  gemini     GEMINI_API_KEY       default model gemini-2.0-flash

No key is bundled. If the environment variable is missing, the script exits with
a clear message instead of making a network call.

Usage:
    /home/g1/projects/chronicle/.venv/bin/python prototype/online_baseline.py \
      --provider openai --questions benchmark/questions.json \
      --source-root /home/g1/projects/chronicle/data/real \
      --vault-dir /tmp/vfm --frames 9
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import vlm_baseline

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-3.8-flash",
}
KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
# Published list prices at the time of writing; override on the command line.
DEFAULT_PRICES = {
    "openai": (0.15, 0.60),
    "anthropic": (3.00, 15.00),
    "gemini": (0.10, 0.40),
}


def _post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"request timed out after {timeout:.0f}s") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network error: {exc}") from exc


def _openai(
    sheet: bytes,
    question: str,
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout: float,
) -> tuple[str, dict[str, int]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vlm_baseline.question_prompt(question)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(sheet).decode("ascii")
                        },
                    },
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": 0,
    }
    body = _post(
        f"{base_url.rstrip('/')}/chat/completions",
        {"Authorization": f"Bearer {api_key}"},
        payload,
        timeout,
    )
    text = ""
    try:
        text = str(body["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""
    usage = body.get("usage") or {}
    return text, {
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "eval_count": int(usage.get("completion_tokens") or 0),
    }


def _anthropic(
    sheet: bytes,
    question: str,
    *,
    model: str,
    api_key: str,
    timeout: float,
) -> tuple[str, dict[str, int]]:
    payload = {
        "model": model,
        "max_tokens": 200,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(sheet).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": vlm_baseline.question_prompt(question)},
                ],
            }
        ],
    }
    body = _post(
        "https://api.anthropic.com/v1/messages",
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        payload,
        timeout,
    )
    text = ""
    try:
        blocks = body.get("content") or []
        text = " ".join(str(block.get("text") or "") for block in blocks if isinstance(block, dict))
    except (AttributeError, TypeError):
        text = ""
    usage = body.get("usage") or {}
    return text, {
        "prompt_eval_count": int(usage.get("input_tokens") or 0),
        "eval_count": int(usage.get("output_tokens") or 0),
    }


def _gemini(
    sheet: bytes,
    question: str,
    *,
    model: str,
    api_key: str,
    timeout: float,
) -> tuple[str, dict[str, int]]:
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": vlm_baseline.question_prompt(question)},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(sheet).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 200},
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    body = _post(url, {"x-goog-api-key": api_key}, payload, timeout)
    text = ""
    try:
        text = str(body["candidates"][0]["content"]["parts"][0]["text"] or "")
    except (KeyError, IndexError, TypeError):
        text = ""
    usage = body.get("usageMetadata") or {}
    return text, {
        "prompt_eval_count": int(usage.get("promptTokenCount") or 0),
        "eval_count": int(usage.get("candidatesTokenCount") or 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--vault-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--keep-sheets", type=Path, default=None)
    parser.add_argument("--input-price-per-mtok", type=float, default=None)
    parser.add_argument("--output-price-per-mtok", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    key_env = args.api_key_env or KEY_ENVS[args.provider]
    api_key = os.getenv(key_env, "").strip()
    if args.provider == "gemini" and not api_key and not args.api_key_env:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print(
            f"missing API key: set {key_env} (or pass --api-key-env). "
            "No network request was made."
        )
        return 2

    model = args.model or DEFAULT_MODELS[args.provider]
    default_in, default_out = DEFAULT_PRICES[args.provider]
    input_price = args.input_price_per_mtok if args.input_price_per_mtok is not None else default_in
    output_price = args.output_price_per_mtok if args.output_price_per_mtok is not None else default_out

    errors: list[str] = []

    def ask(sheet: bytes, question: str) -> tuple[str, float, dict[str, int]]:
        started = time.perf_counter()
        for attempt in range(2):
            try:
                if args.provider == "openai":
                    text, usage = _openai(
                        sheet,
                        question,
                        model=model,
                        api_key=api_key,
                        base_url=args.base_url,
                        timeout=max(10.0, args.timeout),
                    )
                elif args.provider == "anthropic":
                    text, usage = _anthropic(
                        sheet,
                        question,
                        model=model,
                        api_key=api_key,
                        timeout=max(10.0, args.timeout),
                    )
                else:
                    text, usage = _gemini(
                        sheet,
                        question,
                        model=model,
                        api_key=api_key,
                        timeout=max(10.0, args.timeout),
                    )
                return (
                    vlm_baseline.clean_answer(text),
                    (time.perf_counter() - started) * 1000.0,
                    usage,
                )
            except RuntimeError as exc:
                errors.append(str(exc))
                if "429" in str(exc) and attempt == 0:
                    time.sleep(10)
                    continue
                break
        return (
            "unknown",
            (time.perf_counter() - started) * 1000.0,
            {"prompt_eval_count": 0, "eval_count": 0},
        )

    report = vlm_baseline.run(
        args.questions.expanduser().resolve(),
        args.source_root.expanduser().resolve(),
        args.vault_dir.expanduser().resolve(),
        model=f"{args.provider}:{model}",
        frames=max(1, args.frames),
        limit=max(0, args.limit),
        timeout=max(10.0, args.timeout),
        keep_sheets=args.keep_sheets.expanduser().resolve() if args.keep_sheets else None,
        input_price_per_mtok=max(0.0, input_price),
        output_price_per_mtok=max(0.0, output_price),
        gpu_watts=0.0,
        json_output=args.json,
        ask_fn=ask,
    )
    if not args.json:
        cost = report["video_llm"]["api_cost_usd"]
        print(
            f"provider        : {args.provider}:{model} · "
            f"${input_price}/M input, ${output_price}/M output · "
            f"API cost ${cost:.4f} for {report['questions']} questions"
        )
        if errors:
            print(f"api errors      : {len(errors)}; first: {errors[0][:240]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

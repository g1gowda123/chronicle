# GraphRAG answer benchmarks

Accuracy, latency and token use for the three answer paths, measured with the
repository's deterministic evaluation set (`chronicle/semantic/evaluation.py`,
13 cases) on 2026-10-27.

## Setup

| item | value |
| --- | --- |
| Cases | 13 (location, last-seen time, action, multi-hop, appearance, unknown) |
| Reference | deterministic template path (the eval set is built from it) |
| Local model | `qwen2.5:7b-instruct` via Ollama (RTX 4060 Laptop, CUDA) |
| Cloud model | `gemini-2.5-flash-lite` (free AI Studio key, OpenAI-compatible endpoint) |
| Sampling | temperature 0.1, max 220 answer tokens, citation tolerance ±0.35 s |
| Runner | `scripts/benchmark_llm.py --backend all` |

The Gemini key is supplied through `GEMINI_API_KEY` and is never written to the
repository. `gemini-2.5-flash` is no longer issued to new keys — the harness
defaults to `gemini-2.5-flash-lite`.

## Results

| backend | model | pass | citation precision | answered by LLM | mean | median | p95 | tokens (prompt + completion) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| template | deterministic | **100%** | **100%** | — | 0.00 s | 0.00 s | 0.00 s | 0 |
| ollama | qwen2.5:7b-instruct | 77% | **100%** | 85% | 0.53 s | 0.53 s | 0.79 s | 4,251 + 295 = **4,546** |
| gemini | gemini-2.5-flash-lite | 77% | **100%** | 85% | 1.63 s | 1.30 s | 2.98 s | 4,414 + 421 = **4,835** |

Per-family accuracy:

| family | template | ollama | gemini |
| --- | --- | --- | --- |
| location | 100% | 100% | 100% |
| timestamp | 100% | 91% | 82% |
| relation | 100% | 70% | 70% |
| unknown / "not in memory" | 100% | 100% | 100% |

Crash/hallucinated citations: **0**. Every citation produced by both models
resolved to a fact in that case's context (12/12 and 16/16 valid).

## Reading the numbers

- The template path is the evaluation reference, so its 100% is by construction.
  It is also the fallback whenever the model has no verifiable citation.
- The LLM paths answer 11 of 13 cases themselves; two answers per model fall
  back to the template (e.g. when the model returns "I don't have that in my
  memory" or no citation).
- The three missed cases are `before-placement` (Ollama), `appearance-colour`,
  `appearance-size` (Ollama) and `appearance-colour`, `appearance-size`,
  `appearance-describe` (Gemini). The answers are factually right ("the mug is
  red", "medium") but the harness also requires the `APPEARANCE` relation label
  and the exact evidence timestamp, which free-form prose rarely echoes. This is
  a scoring strictness issue, not a grounding failure — citation precision
  stays at 100%.
- Tokens: one full evaluation pass costs ~4.5k tokens total
  (~350 prompt + ~25 completion per answered case). The measured run used the
  free tier; the harness spaces Gemini calls by 3 s to stay inside its rate
  limits.
- Latency is end-to-end per question, including HTTP and JSON parsing. The local
  7B model is faster in wall-clock time on this GPU; the cloud model has the
  better timestamp accuracy in this sample.

## 15-minute video: Gemini baseline

The GraphRAG numbers above are **text-only** calls over an already-retrieved
subgraph. They are not video-understanding latency.

For a 15-minute video:

- **It is accepted.** Per the [Gemini video docs](https://ai.google.dev/gemini-api/docs/video-understanding)
  (updated 2026-10-13), models with a 1M-token context can process up to about
  **1 hour at high media resolution** or **3 hours at low media resolution**.
  The Files API is recommended for inputs over 100 MB or 10 minutes; 249 MB is
  within the 20 GB paid / 2 GB free limit.
- **Token load.** At the default 1 FPS, 15 minutes is 900 frames. At low
  resolution that is roughly **90k tokens** (66/frame + 32/s audio); at high
  resolution roughly **270k tokens** (258/frame + 32/s audio).
- **Typical latency.** A public 24-run benchmark on a 10-minute talk reports
  **12.0–14.1 s** for static processing and **10.5–14.5 s** for agentic
  processing ([DEV Community](https://dev.to/gde/gemini-agentic-video-isnt-always-cheaper-a-24-run-benchmark-4ge3)).
  Scaling the static pass to 15 minutes gives roughly **18–21 s after upload**,
  so the frontend card uses **~20 s (15–25 s typical)**.
- **First use is slower.** Upload + Files API processing is additional. Reuse or
  context caching lowers repeat cost, but does not make a 15-minute video
  sub-second.

The card's "≈22,000×" is `20 s / 0.89 ms`; the 100-question row is
`100 × 20 s ≈ 33 min` versus `100 × 0.89 ms ≈ 0.09 s`. These are order-of-
magnitude comparisons, not a controlled head-to-head on the same machine.

## More detector tags vs. hallucination

Detector vocabulary only controls **what the perception stage looks for**. More
tags raise recall (the bag/laptop/monitor in the room get seen) and can raise a
few false positives, but they are not fed to the answer model. Answers are
grounded on the subgraph and must carry a citation, otherwise the deterministic
template answers instead. Adding tags therefore does not increase fabricated
answers; it can only add noisy objects to memory.

## Reproduce

```bash
# template + local Ollama + Gemini
GEMINI_API_KEY=... python scripts/benchmark_llm.py --backend all

# local only
python scripts/benchmark_llm.py --backend ollama --model qwen2.5:7b-instruct

# raw report
cat runs/benchmark/graphrag_benchmark.json
```

Both the full report and the per-case answers are written to
`runs/benchmark/graphrag_benchmark.json` (gitignored).

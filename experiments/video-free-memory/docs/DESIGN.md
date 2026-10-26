# Design — Chronicle VFM (video-free memory)

## 1. The inversion

Current system:

```
video -> perception -> graph -> query -> answer
          ^ video kept for playback, evidence, re-decoding
```

VFM:

```
video -> COMPILER (heavy models, once) -> VAULT + MEMORY PACK -> query -> answer
                                           |
                                           +-> event WebP (audit only)
          ^ source video optional for playback; never in the answer context
```

The compiler is allowed to be expensive. The query path must be cheap, deterministic where
possible, and independent of video and of any video LLM.

**Important distinction:** "video-free" means the LLM/query context contains no video frames.
A human-facing app may keep the source for playback, scrubbing and audit; that does not change
the compiled memory the model reads. The Studio keeps playback by default and can delete the
source with `--delete-video`.

## 2. Memory tiers

| Tier | What | Size target | Query role | Retention policy |
|---|---|---|---|---|
| T0 Source | raw video | transient or retained for UX | never in query/LLM context | keep for playback or delete |
| T1 Evidence | 1 WebP per major event, 480 px, q75 | ~12–18 KB/event | audit, UI, visual fallback | **12/h long-term; 40/h max recall** |
| T2 Caption ledger | 30 s window captions + frame captions + tags | ~1–3 KB/min | primary text retrieval | keep all; ~1 KB/episode |
| T3 Temporal graph | entities, states, anchors, relations, events | ~40–60 KB/h | identity/temporal reasoning | keep all salient events |
| T4 Embeddings | text + image vectors (optional) | ~2 KB/vector | semantic recall | only if retrieval needs it |
| T5 Answer cache | question → answer + citations + evidence ids | ~1–2 KB/answer | repeated/similar questions | regenerable; drop before quoting at-rest size |

Target: **1 hour of egocentric video < 1 MB of memory at rest**, with a 12-snapshot/hour
policy reaching ~282 KB/hour. Compare with ~240–790 MB/hour of raw video.

## 3. What counts as a "major event"

Minimum set:

- state transition (`NEW`, `INTERACTED`, `HELD`, `CONTAINED`, `LOST`, …)
- interaction start/end with a hand
- zone / place change
- first seen / last seen / re-acquired
- verified action (Florence-2 accepted)
- periodic scene anchor (every 30–60 s) so quiet periods are not invisible
- optional: scene-cut / large camera pan end

Each event stores: `t`, kind, subject, relation, object, zone, caption, confidence, source,
and an evidence JPEG id. Events are the join key between text, graph and pixels.

## 4. Compiler pipeline

```
1. Perception (existing)      -> tracks, states, interactions, actions
2. Salience coalescer         -> events, using physical `reason` labels
3. Keyframe selector          -> events + periodic anchors
4. Captioner (heavy, offline) -> per-frame VLM captions + 30 s window synthesis
5. Graph writer               -> entities/relations/events/packed blob
6. Ledger writer              -> zlib caption ledger + optional embeddings
7. Evidence pass              -> 480 px WebP snapshots, then delete video
8. Retention pass             -> snapshot budget, cache drop, vacuum
```

The compiler must be **idempotent** and **query-agnostic**. It should over-produce high-recall
captions and events, not guess future questions (MERIT).

### Caption policy

- Event caption: one sentence with subject, action, object, zone, timestamp.
- Window caption: a 30 s aggregated caption built from three per-frame VLM captions; this is the
  unit CapMem found competitive with direct VideoQA.
- Tags: object labels, action verbs, zone, people, colours/attributes.
- Store per-frame captions inside the same ledger so a 60 s window can still be traced to
  individual timestamps.
- Default model: `qwen2.5vl:3b` for frames + `qwen2.5:1.5b` for synthesis, both local. The
  multi-image Ollama path is unreliable; caption one frame per call and synthesize with text.

### Keyframe policy

- Always store the frame at the event timestamp (not a later one).
- **480 px wide WebP q75** is the default (knee measurement: 10 snapshots = 141 KB vs 457 KB for
  640 px/q82 JPEG, PSNR 35.8 vs 38.2 dB, SSIM 0.954 vs 0.967).
- Deduplicate near-identical frames (perceptual hash or CLIP cosine) before writing.
- Cap snapshots per hour: 12 for long-term retention, 40 for maximum visual audit.
- Delete the source video after the evidence pass; the vault must never reference it.

## 5. Query flow (the ladder)

```
question
  -> canonicalise to (intent, entity, time, action)  [cache key]
  -> qa_cache lookup (paraphrases hit the same key)
  -> intent + entity + time parser (regex/rules; optional tiny model)
  -> candidate retrieval:
       in-memory subject index (where/when)
       time index + bisect (around/before/after)
       compressed 30 s dense-caption ledger (keyword/window)
       SQL/table lookup (entities, evidence)
  -> temporal composition (before/after/last/first/count)
  -> confidence score
       high  -> answer from ledger + evidence ids
       low   -> retrieve 1–3 event JPEGs -> small VLM (frames-on-demand)
       still low -> "not in memory"
  -> cache answer, keyed by the canonical tuple
```

The small VLM is a **fallback**, not the default. Every fallback gets logged so we can measure
how often pixels are actually needed. The measured 9-frame video-to-LLM baseline is 8/30 at
3.4 s/question; the vault is 30/30 at 0.13 ms/question with no model.

### Why no video LLM at query time

- Video tokens are large and scale with duration.
- Re-decoding video makes latency dependent on file size/codec.
- The same episode is queried many times; recomputing is waste.
- Storing only captions/graph/JPEGs is privacy-preserving and portable.

## 6. Vault schema

See `schema_v2.sql` for the packed layout, `schema_v3.sql` for the dense ledger and answer
cache, and `schema.sql` for the legacy row layout. Key tables:

- `episodes` — one row per compiled episode, source metadata, `video_retained=0`.
- `entities` — current and first/last seen state.
- `event_pack` — chronological packed ledger; the source of truth for "what happened".
- `captions_blob` — synthetic 30 s captions from structured events (v2 fallback).
- `caption_ledger` — zlib dense VLM ledger: window caption, objects, actions, zone, tags,
  per-frame captions.
- `evidence` — event WebP metadata (path, bytes, width, height); **no video**.
- `embeddings` — optional vectors, keyed by `(kind, ref_id)` (int8).
- `qa_cache` — materialised answers with canonical hash, citations, evidence ids, hits.

Indexes: `events(t)`-equivalent in-memory `times`/`by_subject`, `captions(t_start,t_end)`,
`qa_cache(question_hash)`, `evidence(t)`. The vault uses **1 KB SQLite pages**, set before WAL
mode; this cut the 45 s cook vault from 77 KB to 27 KB.

## 7. Storage engineering (v2, implemented)

The v2 prototype replaces one SQLite row per event with four compression layers:

| Layer | Technique | Result |
|---|---|---|
| Strings | dictionary-encoded to uint16 | labels/zones/relations stored once |
| Events | struct-of-arrays pack `<IBHHHHBBH>` | **17 B/event** |
| Captions | zlib JSON, 30 s windows | **429 B for the cook episode** |
| Vectors (future) | float32 scale + int8 values | **512-d: 2,048 B -> 516 B** |

The v3 numbers (Aria cook, 45 s, 11 entities): **51 events** after salience coalescing,
**879 B event blob (17.2 B/event)**, **27 KB vault**, **996 B dense 30 s ledger**, **10 WebP
snapshots at 480 px/q75 = 137.5 KB**, and **164.5 KB total memory**. Query cold p50 is
**0.045 ms**. The walking episode (60 s) is 30 KB vault + 138.5 KB evidence = 168.5 KB, **1.25%
of the 13.18 MB video**.

The decisive storage change was **1 KB SQLite pages** set before WAL mode: 77 KB → 27 KB for the
cook vault. Evidence is now the dominant cost, not the ledger.

### 7.1 Retention budget

Measured on a synthetic 60-minute episode (2,474 events, 40-snapshot cap):

| Snapshots/hour | Vault | Memory/hour | 30 days at 1 h/day |
|---:|---:|---:|---:|
| 40 | 138 KB | 698 KB | 20.5 MB |
| 12 | 134 KB | 282 KB | 8.3 MB |
| 4 | 132 KB | 178 KB | 5.2 MB |

The ledger grows with events (~42 KB/hour); evidence grows with the snapshot budget. The cache is
regenerable and should be dropped before quoting at-rest size.

### 7.2 Original vs VFM vs video LLM

Same 30 questions:

| System | Accuracy | p50 / p95 | Query model | Video at query |
|---|---:|---:|---|---|
| Original Chronicle | 14/30 (46.7%) | 847 / 4,610 ms | Qwen2.5 7B | yes |
| VFM vault | 30/30 | 0.045 / 0.113 ms | none | no |
| 9-frame Qwen2.5-VL | 7/30 (23.3%) | 4,456 / 7,202 ms | Qwen2.5-VL 3B | yes |

The original is more fluent but reads terminal anchors, so its locations/timestamps are stale
(plate "on countertop at 38.1s" after it was placed in the bowl at 43.4s; knife first seen at
19.0s instead of 7.1s). The vault keeps the full ledger. Counts are track counts, not verified
physical counts.

### 7.3 LLM-facing memory pack (usability result)

The LLM does not get raw video or a wall of captions. `memory_pack.py` builds a compact,
timestamped pack containing:

- the verified structured answer (when one exists), marked authoritative;
- ranked 30 s caption windows with objects/actions/zone and per-frame captions;
- salient events with `[t=..s]` citations;
- evidence references;
- counts and explicit rules (digits, citations, unknown).

Qwen2.5 7B answering from this pack scores **30/30** on the same questions at **1,074 ms p50**
(p95 2,660 ms), using ~1,055 pack tokens/question (~4,000 chars). It never sees video. The
deterministic vault remains the 0.045 ms path; Qwen is the natural-language layer. This is the
usability contract: compile once, retrieve a verified answer, let the LLM phrase it from the
memory pack.

Studio keeps the uploaded video for playback; the vault, pack, report and zip do not reference it.
`--delete-video` is only for a strictly video-free session.

### 7.4 Captioner and deployment choice

Florence-2 is the main repo's existing captioner and a plausible CPU/Render choice for frame
captions, object tags and OCR. It has not been benchmarked against Qwen2.5-VL on this ledger or
memory-pack QA. Qwen2.5-VL was chosen for richer 30 s action narrative and strict JSON; Florence
may be good enough for a cheaper deployment. A/B them on object/action coverage, pack tokens and
the 30-question pack score before switching.

The current Studio assumes local Ollama, local SQLite/evidence and long jobs in-process. Render's
small CPU services cannot host the 2.8 GB VLM + 4.7 GB text model or durable local files. A Render
deployment needs async jobs, object storage, a background worker and either Florence/CPU captioning
or an external GPU VLM. Static precompiled vaults/reports can be hosted directly with no GPU.

## 8. Query API (v0 prototype)

`query.py` supports:

- `where is <object>?` / `which room is <object> in?`
- `when did <object> <verb>?` (first/last touch/place/see)
- `what happened at/around <t>?` (+ dense VLM window caption)
- `what happened before/after <event/time>?`
- `how many <object>?` (track count, with caveat)
- `did I <verb> <object>?` (yes/no)
- `what did I see?` / inventory
- keyword search over the dense ledger, captions and events
- semantic answer cache: canonical `(intent, entity, time, action)` key; 4/4 paraphrases hit

No LLM. Output is JSON: `answer`, `citations`, `evidence`, `confidence`, `method`, `cached`.
This is deliberately unambitious: if SQL/regex cannot answer the demo questions, the vault is
missing structure, not intelligence.

## 9. Radical ideas (ranked)

| # | Idea | Impact | Feasibility | Notes |
|---|------|--------|-------------|-------|
| R1 | **Compile-once, query-forever** | High | High | The core bet; delete video after ingest |
| R2 | **Answer materialisation / semantic cache** | High | High | Precompute common questions; cache paraphrases; measure hit rate |
| R3 | **Tiny query compiler** | High | Medium | Fine-tune 0.5B to emit SQL/graph ops; zero video at query |
| R4 | **Frames-on-demand fallback** | Medium | High | 1–3 keyframes + small VLM only when confidence is low |
| R5 | **Visual KV / token cache** | High | Hard | Store compressed visual tokens instead of pixels; attention over tokens |
| R6 | **Memory-native weights** | Very high | Hard | LoRA/cartridge/MemoryLLM per episode; no retrieval; catastrophic forgetting risk |
| R7 | **Event-sourced world model** | High | Medium | Store state transitions; answer counterfactuals ("what if…") without video |
| R8 | **Memory compression contest** | Medium | Medium | Target <1 MB/hour with accuracy loss curves; publish numbers |
| R9 | **Audio + speech ledger** | Medium | Medium | EgoCITE-style transcripts; cheap and high-value |
| R10 | **Query-budget router** | Medium | High | Spend 0 ms / 100 ms / 2 s depending on confidence and question type |

## 10. Experiments

### E1 — Size accounting

For 45 s cook, 60 s walking, 34 s desk: video size, current DB, event count, WebP bytes,
ledger bytes, vault size. Result: 27 KB vault, 137.5 KB evidence, 164.5 KB total for cook; 48.8×
smaller than the three source videos combined.

### E2 — Query accuracy vs ladder rung

30 questions across location, time, count, before/after, negation, paraphrase. Result: original
graph+Qwen 7B 14/30; vault 30/30; 9-frame Qwen2.5-VL 7/30. Per-category table in README/DESIGN
§7.2.

### E3 — Caption interval ablation

5 / 15 / 30 / 60 s windows; accuracy vs ledger size. 30 s vs 60 s measured: 30 s costs ~350 B
more per episode but carries 19 vs 13 objects and 26 vs 16 actions on cook. Structured accuracy
is unchanged because events answer exact times. Default 30 s.

### E4 — Event policy ablation

Transitions only vs +interactions vs +periodic anchors vs +scene cuts. Salience coalescing using
the source `reason` field cut 122 → 51 events on cook without losing the benchmark answers.

### E5 — Cache economics

Ask each question with paraphrases. Result: 4/4 paraphrases hit, reused accuracy 100%; warm
p50 0.056 ms vs cold 0.045 ms (the cache saves parsing, not much wall time at this scale).

### E6 — Long-horizon scaling

Concatenate the 45–60 s clips into 10/30/60 min episodes. Result: vault 51/86/139 KB, p50
0.68/1.86/3.64 ms, p95 1.9/5.3/10.4 ms. Size is linear in events; latency grows with event count,
not frames, but stays far below 50 ms. The ratio to video grows with duration because video grows
in MB/min while memory grows only with events + a bounded snapshot budget.

### E7 — Video-free stress test

Copy the clip + DB to a temp directory, compile, extract evidence, delete the only video, then
query and render the report. Result: zero videos remained; all queries and the 421 KB HTML report
worked.

### E8 — Privacy / audit

Show that the vault contains no raw video; inspect the bytes. Export a vault and query it on a
machine with no video and no model.

### E9 — Video-to-LLM cost/accuracy baseline

9-frame contact sheet, timestamps burned in, one VLM call per question. Result: 7/30, 4.46 s p50,
764 tokens/question, $0.0036/30 at illustrative prices, 19.24 MB video at query. Vault: 30/30,
0.12 ms, 0 tokens, 404 KB at rest. A Gemini free-tier probe was 234 s for one question and was
not run at 30 questions.

### E10 — Retention sweep

Simulate 40/12/4 snapshots/hour on a 60-minute synthetic vault. Result: 698/282/178 KB per hour;
20.5/8.3/5.2 MB for 30 days at 1 h/day. Default long-term policy: 12 snapshots/hour.

## 11. Risks

- **Caption loss** — if the compiler misses an event, the vault cannot answer. Mitigation:
  over-caption, periodic anchors, and a frames-on-demand fallback.
- **False confidence** — a clean SQL answer can still be wrong. Mitigation: citations + evidence
  ids + explicit unknown.
- **Schema drift** — the vault is derived from Chronicle's schema. Version it and keep the
  compiler idempotent.
- **Demo scope creep** — this branch must not destabilise the main demo. Keep it isolated until a
  clean interface exists.
- **Benchmark cherry-picking** — 30 questions is not a paper. Report per-category accuracy and
  failures, not just the mean. The current 30/30 is an internal regression suite.
- **Track fragmentation** — count answers are track counts, not physical object counts. Cook has
  2 plate tracks but a VLM may see 4 physical plates. Add compile-time physical-count calibration
  before claiming count accuracy.
- **VLM caption hallucination** — the 30 s cook caption said "another person" when only one
  person was tracked. Captions need spot-checks or a verification model before they are trusted
  as ground truth.
- **Baseline fairness** — a 9-frame contact sheet is a proxy for a video LLM, not a real one.
  Online Gemini is implemented but was not benchmarked (free-tier latency).

## 12. What "done" looks like

- **Met:** a 45–60 s episode compiles to <1 MB of memory (cook: 164.5 KB) with no video.
- **Met:** 30/30 on the curated set, without any vision model at query time.
- **Met:** query p50 0.045 ms, p95 0.113 ms (45 s); 3.64 ms / 10.42 ms at 60 min.
- **Met:** every answered question carries a timestamp and a nearest evidence snapshot.
- **Met:** deleting the only video copy and restarting changed nothing.
- **Met:** single-file 421 KB HTML report with timeline, dense captions and a client-side query box.
- **Not yet:** an external benchmark (CapMem/EgoSchema/Ego4D NLQ) and a real online video-LLM
  comparison on a paid/faster key.

That is a credible new system boundary, not just a feature.

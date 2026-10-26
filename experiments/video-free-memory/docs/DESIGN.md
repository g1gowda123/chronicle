# Design — Chronicle VFM (video-free memory)

## 1. The inversion

Current system:

```
video -> perception -> graph -> query -> answer
          ^ video kept for playback, evidence, re-decoding
```

VFM:

```
video -> COMPILER (heavy models, once) -> VAULT -> query -> answer
                                           |
                                           +-> event JPEGs (audit only)
          ^ video deleted after compilation
```

The compiler is allowed to be expensive. The query path must be cheap, deterministic where
possible, and independent of video and of any video LLM.

## 2. Memory tiers

| Tier | What | Size target | Query role |
|---|---|---|---|
| T0 Source | raw video | transient | never queried; deleted after compile |
| T1 Evidence | 1 JPEG per major event, 640 px, q82 | 30–80 KB/event | audit, UI, visual fallback |
| T2 Caption ledger | 30–60 s window captions + event captions + tags | ~1–3 KB/min | primary text retrieval |
| T3 Temporal graph | entities, states, anchors, relations, events | <1 MB per episode | identity/temporal reasoning |
| T4 Embeddings | text + image vectors (optional) | ~2 KB/vector | semantic recall |
| T5 Answer cache | question → answer + citations + evidence ids | ~1–2 KB/answer | repeated/similar questions |

Target: **1 hour of egocentric video < 20 MB of vault**, preferably < 5 MB. Compare with
~1 GB/hour raw video (CapMem's "1 GB+/h" raw visual token figure is a useful upper bound).

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
2. Keyframe selector          -> events + periodic anchors
3. Captioner (heavy, offline) -> event captions + 30/60 s window captions
4. Graph writer               -> entities/relations/events
5. Embedding writer           -> text and image vectors
6. Answer materialiser        -> precompute common questions (optional)
7. Delete source video
```

The compiler must be **idempotent** and **query-agnostic**. It should over-produce high-recall
captions and events, not guess future questions (MERIT).

### Caption policy

- Event caption: one sentence with subject, action, object, zone, timestamp.
- Window caption: one paragraph per 30 s (or 60 s), aggregating events; this is the unit that
  CapMem found competitive with direct VideoQA.
- Tags: object labels, action verbs, zone, people, colours/attributes.
- Optional audio transcript with speaker turns (EgoCITE).

### Keyframe policy

- Always store the frame at the event timestamp (not a later one).
- 640 px wide, JPEG q82; optionally a 320 px thumbnail for grids.
- Deduplicate near-identical frames (cosine of CLIP embeddings or perceptual hash).
- Cap events per minute; if exceeded, keep the highest-severity events (interaction > transition
  > periodic).

## 5. Query flow (the ladder)

```
question
  -> intent + entity + time parser (regex/rules; optional tiny model)
  -> candidate retrieval:
       SQL filters (entity, time)
       graph 2-hop
       text/vector search over captions
  -> temporal composition (before/after/last/first/count)
  -> confidence score
       high  -> answer from ledger + evidence ids
       low   -> retrieve 1–3 event JPEGs -> small VLM (frames-on-demand)
       still low -> "not in memory"
  -> cache answer (T5)
```

The small VLM is a **fallback**, not the default. Every fallback gets logged so we can measure
how often pixels are actually needed.

### Why no video LLM at query time

- Video tokens are large and scale with duration.
- Re-decoding video makes latency dependent on file size/codec.
- The same episode is queried many times; recomputing is waste.
- Storing only captions/graph/JPEGs is privacy-preserving and portable.

## 6. Vault schema (v0)

See `schema.sql`. Key tables:

- `episodes` — one row per compiled episode, source metadata, `video_retained=0`.
- `entities` — current and first/last seen state.
- `events` — chronological ledger; the source of truth for "what happened".
- `captions` — window/event captions with tags and a time range.
- `evidence` — event JPEG metadata (path, bytes, width, height).
- `embeddings` — optional vectors, keyed by `(kind, ref_id)`.
- `qa_cache` — materialised answers with hits and confidence.

Indexes: `events(t)`, `events(subject, t)`, `events(kind, t)`, `captions(t_start, t_end)`,
`qa_cache(question_hash)`.

## 7. Query API (v0 prototype)

`query.py` supports:

- `where is <object>?`
- `when did <object> <verb>?`
- `what happened at/around <t>?`
- `what did I see?` / inventory
- keyword search over captions/events

No LLM. Output is JSON: `answer`, `citations`, `evidence`, `confidence`, `method`.
This is deliberately unambitious: if SQL/regex cannot answer the demo questions, the vault is
missing structure, not intelligence.

## 8. Radical ideas (ranked)

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

## 9. Experiments

### E1 — Size accounting

For 45 s cook, 60 s walking, 13 s desk: video size, current DB, event count, JPEG bytes,
caption bytes, vault size. Target: vault <1% of video.

### E2 — Query accuracy vs ladder rung

Build 30 questions across location, time, action, count, before/after, unknown.
Compare:
- current GraphRAG + Qwen 7B,
- vault SQL only,
- vault + small LM phrasing,
- vault + frames-on-demand fallback (count fallback rate).

### E3 — Caption interval ablation

5 / 15 / 30 / 60 s windows; accuracy vs ledger size. CapMem predicts 30–60 s is the sweet spot.

### E4 — Event policy ablation

Transitions only vs +interactions vs +periodic anchors vs +scene cuts. Recall on a question set.

### E5 — Cache economics

Ask each question 5 times with paraphrases. Measure semantic-cache hit rate, latency, and
accuracy of reused answers.

### E6 — Long-horizon scaling

Concatenate the 45–60 s clips into 10/30/60 min episodes. Measure vault growth and query latency.
Hypothesis: size grows with events, latency stays ~constant.

### E7 — Video-free stress test

Delete the source video, restart, run the full question set. Any failure is a hidden video
dependency. Target: 0 failures.

### E8 — Privacy / audit

Show that the vault contains no raw video; inspect the bytes. Export a vault and query it on a
machine with no video and no model.

## 10. Risks

- **Caption loss** — if the compiler misses an event, the vault cannot answer. Mitigation:
  over-caption, periodic anchors, and a frames-on-demand fallback.
- **False confidence** — a clean SQL answer can still be wrong. Mitigation: citations + evidence
  ids + explicit unknown.
- **Schema drift** — the vault is derived from Chronicle's schema. Version it and keep the
  compiler idempotent.
- **Demo scope creep** — this branch must not destabilise the main demo. Keep it isolated until a
  clean interface exists.
- **Benchmark cherry-picking** — 30 questions is not a paper. Report per-category accuracy and
  failures, not just the mean.

## 11. What "done" looks like

- A 45–60 s episode compiles to a vault <5 MB with no video.
- 90% of a 30-question set answers without any vision model at query time.
- Query p50 <50 ms, p95 <200 ms on CPU.
- Every answer has a timestamp + evidence JPEG id.
- Deleting the video and restarting changes nothing.
- A one-page report with size, accuracy, fallback rate, and latency.

That is a credible new system boundary, not just a feature.

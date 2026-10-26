# Research journal — video-free episodic memory

> Append-only. One entry per work session. Record what was read, what was decided, what was
> measured, and what failed. Future entries should not rewrite old ones.

---

## 2026-10-26 — Session 1: framing the problem

### Why this branch exists

The shipped Chronicle build answers queries from a graph, but still keeps the source video and
uses it for playback and evidence-frame extraction. That is the wrong dependency if we want
long-horizon memory: video is large, private, and expensive to decode or re-encode. The query
path should not need it.

The new goal is not "better VideoQA". It is **compiled memory**:

- ingest once with a strong model,
- write a dense, timestamped, queryable ledger,
- delete the video,
- answer from the ledger + event snapshots,
- call a vision model only when the ledger is genuinely insufficient.

### Working thesis

1. A 30–60 s caption window is a better unit of memory than a frame.
2. An event graph (entities, states, relations, time) is a better index than a vector store alone.
3. One JPEG per major event is enough for auditability and most demo queries.
4. Query-time LLM compute should be optional; a small model only phrases or parses.
5. Repeated/similar questions should hit a materialised answer cache.

### Key readings (details in `RESEARCH.md`)

- **CapMem** (EMNLP 2026): on videos >20 min, 30/60 s caption memory beats direct VideoQA for
  10/12 and 8/12 models; a retrieve-and-verify harness adds up to 5.3 points.
- **EgoCITE** (2026): dense visual captions + audio transcripts as memory entries; time-aware
  retrieval for long-horizon egocentric QA.
- **EgoGraph** (2026): training-free temporal knowledge graph with an egocentric schema; SOTA on
  EgoLifeQA and EgoR1-bench.
- **MERIT** (ECCV 2026): query-agnostic multi-key memory construction, high-recall retrieval,
  temporal expansion only at query time.
- **FocusGraph** (2026): graph-based captions select clips; raw frames are never processed at
  question time until a short keyframe set is chosen.
- **R4DSG / VL-KnG**: anchor-relative compact memory and persistent spatio-temporal graphs;
  query latency decoupled from video length.
- **LightMem / HippoRAG / EM-LLM**: hierarchical, graph-structured long-term memory; consolidate
  offline, retrieve cheaply online.
- **Titans / MemoryLLM**: the more radical end — memory in model parameters, not in a database.

### Decisions

- Build the experiment in a separate worktree (`/home/g1/projects/chronicle-vfm`) so the demo
  branch stays stable.
- The first prototype will **compile the existing SQLite graph** into a vault schema. It proves
  the query path can be video-free before we invest in captioning.
- Query engine v0 is SQL + regex intent detection, deliberately no LLM. If v0 already answers
  the core questions, the LLM becomes a phrasing layer, not a dependency.
- Event snapshots are JPEGs at 640 px, one per state transition / interaction / zone change /
  periodic keyframe. Target budget: <100 KB per event, <5 MB per 45-minute episode.

### Prototype scope (session 1–2)

- `schema.sql`: episodes, entities, events, captions, embeddings, qa_cache.
- `compile.py`: read `zones, furniture, entities, state_transitions, edges, anchors, observations`
  and derive a chronological event ledger + captions.
- `query.py`: intents `where`, `when`, `what happened`, `list/inventory`, keyword search.
- `benchmark.py`: vault size vs source DB/video, event coverage, query latency.

### Open questions

- How many events per minute are enough? Transitions only, or transitions + periodic anchors?
- Do we store captions per event, per 30 s window, or both?
- How do we measure "the ledger is insufficient" without a VLM at query time?
- Can a 0.5B query compiler map questions to SQL/graph ops reliably enough to drop Qwen-7B?
- What is the smallest visual evidence unit: 640 px JPEG, 320 px thumbnail, or a compressed
  visual token / embedding?
- Does answer materialisation (semantic cache) actually help, or do users ask too variedly?

### Next session

1. Finish compile/query prototype and run it on `data/real/cook/chronicle.db`.
2. Measure: vault size, event count, caption count, query latency.
3. Write the "memory sufficiency" test set (20–30 questions) and score v0.
4. Read CapMem/EgoCITE more closely for caption granularity and retrieval design.
5. Draft the radical-idea matrix in `DESIGN.md`.

---

## Journal template

```
## YYYY-MM-DD — Session N: title

### Goal
### Read / learned
### Built
### Measured
### Failed / surprised by
### Decision
### Next
```

---

## 2026-10-26 — Session 2: v0 vault prototype

### Built

- `schema.sql` — episodes, entities, events, captions, evidence, embeddings, qa_cache.
- `prototype/compile.py` — turns an existing Chronicle SQLite graph into a vault:
  state transitions + anchors + relation edges, deduplicated, plus 30 s window captions
  and per-event captions.
- `prototype/query.py` — no video, no LLM. Intents: where / when / around-t / inventory /
  keyword. Returns citations and a confidence.
- `prototype/benchmark.py` — size, event/caption counts, accuracy on a fixed question set,
  p50/p95 latency.

### Measured (Aria cook, 45 s, 11 entities)

| Metric | Value |
|---|---:|
| Source video | 2.99 MB |
| Source graph DB | 436 KB |
| Vault | **300 KB (9.8% of video)** |
| Events / captions | 259 / 261 |
| Accuracy (10-question v0 set) | 10/10 |
| p50 / p95 query latency | **0.05 / 0.09 ms** |

All questions answered with the video path never touched. The vault is a single SQLite file.

### What worked

- The graph already contains enough structure for where/when/inventory/around-t questions.
- Time-indexed events make temporal questions trivial; no embeddings needed for v0.
- The 2-hop graph and terminal anchors translate directly into event rows.
- Query latency is effectively constant; there is no model or video cost.

### What failed / surprised

- **259 events in 45 s is too many.** Most are OCCLUDED/OUT_OF_VIEW flicker. "Major events"
  needs salience: transitions into/out of interaction, containment, lost, zone change, plus a
  periodic anchor. Flicker should be coalesced.
- **Vault is 9.8% of video, not <1%.** SQLite page/WAL overhead plus duplicated captions dominate.
  Need: coalesced events, one caption per window, optional gzip/JSONL archive, and a smaller
  page size/checkpoint.
- **Entity matching was too eager.** A state bonus made any visible entity match an unknown
  object ("where are my keys?" -> plate). Fixed: require lexical overlap before the bonus.
- **Duplicate events** from anchors and relation edges. Fixed with a dedupe key.
- **The benchmark set is self-defined.** 10/10 is not evidence of general accuracy; it only
  proves the query path works on the questions we chose. Next set must include paraphrases and
  negatives from CapMem/EgoLifeQA style data.

### Decisions

- Keep the query engine LLM-free for v0. The next addition is a caption ledger from a strong
  offline VLM, not a bigger query model.
- Add event salience before adding more query features. A smaller, cleaner ledger is the
  priority.
- Measure "video-free" by deleting the source and re-running the question set. The prototype
  already never references the video, but the test must be explicit.

### Next

1. Event salience/coalescing: target 20–40 events per 45 s, not 259.
2. Compile the existing evidence snapshots (JPEG) into the `evidence` table; one per salient
   event.
3. Add 30 s and 60 s caption-window modes and compare size/accuracy.
4. Build a 30-question paraphrase/negative set; report per-category accuracy.
5. Add a semantic answer cache and measure hit rate on paraphrases.
6. Run the long-horizon test by concatenating cook/walking/desk episodes.

---

## 2026-10-27 — Session 3: compressed vault v2 + evidence snapshots

### Built

- `prototype/compact.py`:
  - dictionary-encoded strings (uint16)
  - struct-of-arrays event pack (`<IBHHHHBBH`, 17 B/event)
  - zlib-compressed caption windows
  - symmetric int8 vector quantization (float32 -> 1 scale + int8, 4x smaller)
- `schema_v2.sql`: strings, event_pack, captions_blob, entities, evidence, embeddings, qa_cache.
- v2 compiler: stable-state coalescing (occlusion flicker is dropped), one anchor per
  (entity, relation, target), periodic 30 s scene anchors.
- v2 query engine loads the packed blob into memory and answers with the same intents.
- `prototype/extract_evidence.py`: seeks salient events in the video once, writes 640 px JPEGs,
  records them in `evidence`, optionally deletes the video.
- `prototype/report.py`: one-file offline HTML memory browser (timeline + entity filter + embedded
  evidence gallery). No video, no external JS, no network.
- Tests for dictionary/pack/compression/quantization and the compile→query path.

### Measured (Aria cook, 45 s, 11 entities)

| Metric | v0 | v2 |
|---|---:|---:|
| Vault | 300 KB | **76 KB** |
| Events | 259 | **122** |
| Event storage | SQLite rows | **2,086 B blob (17.1 B/event)** |
| Caption storage | text rows | **429 B zlib** |
| Vault / video | 9.8% | **2.48%** |
| Query p50 | 0.05 ms | **0.04 ms** |
| Accuracy (10-question set) | 10/10 | 10/10 |

Evidence pass: **15 snapshots, 674 KB** (avg 45 KB), video deleted. Combined memory ~750 KB
for a 2.99 MB clip. The snapshot cap is 40, so a one-hour episode does not grow linearly:
~40 × 45 KB ≈ 1.8 MB + ~0.3 MB vault, i.e. roughly **2 MB/hour vs ~240 MB/hour of video (<1%)**.

Generated report: 122 events + 15 embedded snapshots -> **915 KB single HTML file**.

### What worked

- Packed arrays are the big win: 17 B/event vs SQLite row overhead. The event blob is now
  smaller than the string dictionary for short episodes.
- Dropping OCCLUDED/OUT_OF_VIEW flicker cut events from 259 to 122 without losing the state
  history that queries need (entity table still carries the current state).
- The query engine can load the whole event pack in memory; latency is effectively zero.
- The HTML report makes the memory tangible without a video player.

### What is still weak

- 122 events for 45 s is still high. The snapshot selector already filters, but the *ledger*
  could coalesce VISIBLE_STATIC/VISIBLE_MOVING runs too. Target 40–60 events/45 s.
- Snapshots dominate bytes (674 KB vs 76 KB vault). Options: 480 px / q75, WebP, or one
  snapshot per 2 events. Measure quality vs size.
- Vault 76 KB includes ~70 KB of SQLite fixed overhead; per-hour scaling is what matters.
- The 10-question set is self-defined. Need a paraphrase/negative set and per-category reporting.
- Evidence extraction needs OpenCV (main venv), while query/compile/report are stdlib-only.

### Decisions

- Keep the packed v2 layout. It is simple, deterministic and already 4x smaller.
- The video is deleted after evidence extraction in the recommended flow; queries never touch it.
- Do not add embeddings yet. Int8 quantization is implemented and tested, but there is no
  caption embedder in the prototype. Add it only after the caption ledger exists.

### Next

1. Caption ledger from a strong offline VLM (30 s/60 s windows), zlib-compressed.
2. Snapshot quality/size sweep: 480/640 px, q70/82, WebP; pick the knee.
3. Coalesce state runs; target 40–60 events/45 s.
4. 30-question paraphrase/negative set + per-category accuracy.
5. Semantic answer cache experiment.
6. 10/30/60-minute concatenated episodes; verify size/latency scaling.

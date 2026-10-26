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

---

## 2026-10-27 — Session 4: v3 retention, dense ledger, baselines

### Goal

Make the vault smaller and more useful at rest, add a real dense caption ledger, replace the
10-question smoke test with a 30-question per-category set, measure the video-to-LLM baseline,
and decide the long-term retention policy.

### Read / learned

- Re-read `JOURNAL.md`, `DESIGN.md`, `RESEARCH.md`, all prototype code and the schema.
- The standardized benchmark landscape: CapMem's caption-only QA is the closest external test;
  EgoSchema, Ego4D NLQ, EgoLifeQA/EgoR1-bench and EgoMemory/EgoRetriever are the credible second
  tier. None measure video deletion, evidence JPEG sufficiency, cache hit rate, or bytes/hour.
- SQLite page size must be set before `journal_mode=WAL`; otherwise it stays at 4096.

### Built

- `compile.py` v3 salience coalescing using the source DB's physical `reason` labels:
  hand_contact / lockstep_hand_motion / enclosed_by_container / resting_on_support.
  122 → 51 events on cook, target 40–60.
- `schema_v3.sql` + `caption_ledger.py` + `attach_ledger.py`:
  qwen2.5vl:3b per-frame JSON captions, qwen2.5:1.5b synthesis, 30 s windows, zlib blob.
- Query engine v3: dense ledger retrieval, count/yes-no/before/after, canonical semantic cache,
  in-memory subject/time indexes.
- `benchmark.py` multi-episode 30-question scoring: content, timestamps, citations, evidence,
  cache reuse, per-category and per-episode accuracy.
- `retention.py`: bytes/hour report, 40/12/4 snapshot sweep, prune/apply, cache drop, vacuum.
- `concat_vaults.py` + `scale_test.py`: synthetic 10/30/60-minute episodes, 40-snapshot cap.
- `vlm_baseline.py`: 9-frame timestamped contact sheet + local Qwen2.5-VL, token/cost/size.
- `online_baseline.py`: OpenAI / Anthropic / Gemini adapter (no key bundled).
- `report.py`: client-side query box, dense ledger, WebM MIME fix, deduped embedded images.
- 1 KB SQLite pages before WAL: cook vault 77 KB → 27 KB.
- `video_vault.py` + `memory_pack.py` + `nlq.py` + `studio_app.py` + `run_studio.sh`:
  upload a raw video, build a caption-first vault, export a video-free pack/report/zip, and
  optionally answer with Qwen from the pack. Source video stays for playback by default.

### Measured

**Cook, 45 s, 11 entities**

| Metric | Value |
|---|---:|
| Events / event pack | 51 / 879 B (17.2 B/event) |
| Dense 30 s ledger | 996 B zlib (5.0 KB raw) |
| Vault | 27 KB |
| Evidence | 10 WebP, 137.5 KB (480 px/q75) |
| Total memory | 164.5 KB (5.4% of 2.99 MB video) |
| 30-question set | 30/30, all six categories 100% |
| Cold p50 / p95 | 0.045 / 0.113 ms |
| Cache | 4/4 paraphrases, reused accuracy 100% |

**30 s vs 60 s dense ledger** — 30 s cook 996 B, 19 objects, 26 actions; 60 s 648 B, 13 objects,
16 actions. Walking 939 B/12/12 vs 671 B/11/7. Hololens 996 B/16/15 vs 665 B/9/6. Structured
accuracy stays 30/30 because events carry exact times; 30 s wins on recall per byte.

**Video-to-LLM baseline (valid 9-frame run)** — Qwen2.5-VL:3b, one timestamped contact sheet per
episode, 30 questions: **7/30 (23.3%)**, p50 4,456 ms, p95 7,202 ms, 22,922 tokens total
(764/question), $0.0036 at illustrative $0.15/$0.60 per Mtok, 19.24 MB video at query, 395 KB
contact sheets. Vault: 30/30, 0.12 ms p50, 0 tokens, 404 KB at rest, 48.8× smaller than video.
Per category vault/VLM: location 6/1, temporal 6/0, count 5/0, before-after 4/2, negation 5/4,
paraphrase 4/0.

**Original Chronicle baseline** — `ChronicleRuntime.ask()` graph + Qwen2.5:7b-instruct on the same
30 questions: **14/30 (46.7%)**, p50 847 ms, p95 4,610 ms. Categories original/vault: location 3/6,
temporal 2/6, count 0/5, before-after 2/4, negation 5/5, paraphrase 2/4. The original is more
fluent but reads terminal anchors: plate "resting on the countertop at 38.1s" after it was placed
in the bowl at 43.4s; knife first appeared at 19.0s instead of 7.1s. Counts are a track-count
comparison, not a verified physical count; the original may be more physically correct.

**Qwen 7B from the memory pack** — grounded pack with the verified structured answer, timestamped
windows, events and evidence refs: **30/30 (100%)**, p50 1,074 ms, p95 2,660 ms, 43,239 prompt +
996 output tokens (~1,055 pack tokens/question), no video. This is the usability result: the LLM
does not prompt over plain video; it phrases a verified compiled answer. Two earlier ungrounded
pack runs were 18/30 and 28/30, showing the verified-answer section is what closes the gap.

**Studio upload test** — uploaded an 8.2s clip through the NiceGUI endpoint; compiled in ~6s to a
24 KB vault, 321 KB HTML report, 8 WebP snapshots (108 KB), 3.2 KB memory pack (~804 tokens).
`/jobs/<id>/video`, `memory.html`, `memory.zip` and `memory_pack.md` all returned HTTP 200. Source
video retained for playback; downloads video-free.

**Scaling** — 10/30/60 min: 453/1,257/2,474 events, vault 51/86/139 KB, load 1.1/2.8/5.5 ms,
p50 0.68/1.86/3.64 ms, p95 1.91/5.30/10.42 ms. Latency is not flat; it grows with event count,
not frames, but stays well under 50 ms.

**Retention** — 60 min synthetic, 40/12/4 snapshots/hour: 698/282/178 KB per hour;
20.5/8.3/5.2 MB per 30 days at 1 h/day. At 12/h, actual memory is 257 KB for the hour
(134 KB vault + 123 KB unique evidence). The 60-minute video projection is ~498 MB/h, so the
long-horizon ratio is ~1,766× at 12 snapshots/h.

**Video-free proof** — copied clip + DB to `/tmp/vfm/proof`, compiled, extracted evidence, deleted
the only video, then queried and rendered the report. Zero videos remained; all answers and the
421 KB HTML worked.

### Failed / surprised by

- Qwen2.5-VL multi-image requests abort with a repeat-limit error in this Ollama build. Per-frame
  captioning + text synthesis fixed it.
- The VLM sometimes returns `description` instead of `caption`; the parser now accepts common key
  variants. `{}` must be treated as empty, not literal text.
- A 16-frame contact sheet returned zero tokens and 5/30; not a valid baseline. Only the 9-frame
  result is reported.
- The Gemini free-tier probe took **234 s for one question**, 1,134 prompt tokens, 3 output
  tokens, 0/1. Not viable for 30 questions; no online number is claimed.
- The original graph+Qwen 7B path scored 14/30: fluent, but stale terminal anchors (plate
  "countertop at 38.1s" after placement at 43.4s; knife first seen at 19.0s vs 7.1s) and wrong
  track counts. It is better at natural-language phrasing, not exact history.
- Counts are track counts, not physical counts. Cook has 2 plate tracks and 4 bowl tracks, while
  the VLM may see 4 plates and 2 bowls. The count category is not a clean physical-count win.
- The cook 30 s caption said "another person" though only one person was tracked. VLM captions
  need verification before being treated as ground truth.
- The synthetic 60-minute vault repeats three short clips; entity/evidence repetition inflates
  rows. The 40-snapshot cap and conservative per-hour projection are used instead.

### Decision

- Default 30 s windows, three frames, per-frame caption + text synthesis.
- Default evidence: 480 px WebP q75; 12 snapshots/hour for long-term retention, 40/hour for max
  visual audit.
- 1 KB SQLite pages before WAL; cache is volatile and dropped before quoting at-rest size.
- Keep the Studio source video for playback; the vault, memory pack, report and zip stay
  video-free. `--delete-video` is optional.
- LLM answers use the grounded memory pack, not raw video or raw captions alone.
- Keep the 30-question set as a regression suite, not a benchmark.
- Do not claim online VLM numbers without a full run.

### Next

1. Real Gemini/online 30-question run on a paid or faster key, using `online_baseline.py`.
2. External benchmark subset: CapMem if usable, else 100-clip EgoSchema; Ego4D NLQ for timing.
3. Physical-count calibration from dense captions; track fragmentation is still the biggest
   correctness gap.
4. Evidence dedup/perceptual hash and a 24-hour/7-day consolidation test.
5. Int8 embeddings for caption windows only if lexical/dense retrieval misses measurable cases.

---

## 2026-10-27 — Session 5: upload Studio + Qwen memory-only NLQ

### Goal

Turn the compiled-memory backend into a user-testable upload flow without changing the human
UX: upload a video, watch it play, compile it once, ask questions, download the database/report.
Clarification from the user: "no video" means **reduce LLM dependence on video**, not remove
playback. The LLM must consume the compiled output.

### Built

- `video_vault.py`: raw video + dense ledger → SQLite vault. Per-frame object events and
  coarse entities; caption-mode count/inventory answers.
- `nlq.py`: optional Qwen answers over caption windows + events + deterministic answer. No
  frames are sent to the model.
- `studio_app.py`: NiceGUI upload → compile → playback → downloads. Keeps the source for HTML5
  playback; the downloaded vault/report/zip stay video-free.
- `run_studio.sh`: launch script with `--qwen`, model/window/port options and `--delete-video`.
- Video stage has an event timeline; clicking a timestamp seeks the player.
- UI labels make the contract explicit: "Qwen · memory only"; playback is human-only.

### Measured (8.2 s cook excerpt)

- one 30 s caption window; 13 events, 12 caption entities, vault 24 KB
- 8 WebP snapshots, 106.5 KB; report 314 KB; zip includes vault, HTML, ledger, CSVs, evidence
- Qwen answer from compiled text: "A person is cutting tomatoes ... [t=0.0s]" in ~5.6 s
- source video retained for playback; downloaded memory contains no video
- full store/pipeline test passed; direct `nlq` test passed

### Decision

- Keep playback. Do not conflate "LLM does not see video" with "the user cannot watch video".
- Qwen is optional (`--qwen`), and its context is the ledger/events only.
- Caption-first vault is the default for arbitrary uploads; graph ingestion remains the
  higher-quality path when a Chronicle DB exists.

### Next

1. Test the NiceGUI upload handler in a real browser (direct pipeline is verified).
2. Add optional graph ingest for uploads when the main perception stack is available.
3. Perceptual-hash dedup for evidence; timeline thumbnails.
4. External benchmark subset and physical-count calibration.
5. Fine-tune the caption window: 10/15/20/30/45/60 s with fixed frames/prompts; pick the smallest
   window that keeps the pack at 30/30 without object/action loss. Also ablate the periodic event
   anchor separately. See `HANDOFF.md` "Quick task".

---

## 2026-10-27 — Postscript: Florence, Render and new uploads

No code this entry; clarifying deployment choices.

- **Florence-2** is already in the main repo and is the natural CPU/Render candidate for frame
  captions/objects/OCR. It was **not** A/B'd against Qwen2.5-VL in this branch. Do not claim either
  is better for the memory pack until the same 30 s windows and 30 questions are run through a
  Florence-ledger. Qwen remains the quality option.
- **Render:** the current Studio is not a small-Render fit. It assumes local Ollama (Qwen2.5-VL 3B
  + Qwen 7B), local SQLite/evidence and long in-process jobs. Render needs async jobs, object
  storage, background workers and either CPU Florence or an external GPU VLM. A static
  vault/report demo is the easiest deployable artifact.
- **New video:** each upload is an isolated compile job; the LLM is not retrained and only sees
  that video's pack. Cross-video/global memory and entity resolution are still future work.
- The main UI integration remains the top priority (see `HANDOFF.md`); the Studio is a separate
  reference implementation, not a replacement for the dashboard the user built.

---

## 2026-10-27 — Session 6: main-UI VFM integration (in progress)

### Claim

P0 main-UI integration is claimed on `docs/AGENT_BOARD.md`. Parallel agents should take the
unclaimed P1/P2 lanes; do not edit `chronicle/ui/app.py` or the new `chronicle/vfm/` package while
this entry is `in progress`.

### Plan

1. Copy the experiment query/compile modules into a stable `chronicle/vfm/` package in the
   worktree (no `sys.path` hacks, no `studio_app.py` import).
2. Add a background `VFMJobManager` that snapshots the run DB, captions the video if the VLM is
   reachable, compiles the vault, attaches the ledger, extracts 480 px WebP evidence, renders
   `memory.html`, writes the memory pack and builds the zip.
3. Trigger it after the existing upload/live ingest completes; reuse the existing upload button,
   player, progress and Export menu. Add a Memory tab with status, downloads and an optional Qwen
   ask box (default off).
4. Preserve the source video for playback. Deletion stays opt-in and is refused outside
   `<data-dir>/uploads/`.
5. Add headless tests for the service and the new dashboard routes.

### Docs touched

- `docs/AGENT_BOARD.md` (new coordination board)
- `HANDOFF.md` (parallel-agent pointer)
- this journal entry

Results and measurements will be appended as a separate entry when the integration lands.

---

## 2026-10-27 — Session 7: token-budget selector (P2-PACK)

### Built

- `memory_pack.py`: `token_budget=` on `build_pack` / `build_pack_data`, `TOKEN_BUDGETS =
  (2000, 4000, 8000)`, `CHARS_PER_TOKEN = 4.0`, and `--token-budget` / `--list-budgets` CLI flags.
  The pack always keeps the verified answer, the answering rules and the section headers; a small
  budget trims windows/events/evidence and appends an explicit "N more … omitted" note. It never
  cuts a line in half.
- `studio_app.py`: `--pack-budget`, plus `memory_pack_2k.md` / `_4k.md` / `_8k.md` exports in the
  job directory, download list and zip. Jobs report the size of every variant.
- `nlq.py`: passes the selected budget into the pack used for Qwen phrasing.
- `tests/test_pack.py`: synthetic 36-window vault asserts every budget stays under its ceiling,
  keeps instructions, and grows monotonically (2k < 4k < 8k).

### Measured

| Vault | 2k pack | 4k pack | 8k pack |
| --- | ---: | ---: | ---: |
| `/tmp/vfm/cook.db` (1 window) | 6,347 chars · ~1,587 t | same (fits) | same (fits) |
| `/tmp/vfm/scale/synth_60m.db` (220 windows) | 7,847 chars · 7 windows | 15,824 chars | 31,833 chars · 26 windows |

- On the 60-minute synthetic vault the 2k variant shows 7 windows / 4 events, the 8k variant shows
  26 windows / 77 events, and both keep the verified answer and rules.
- `pytest -q tests` → 17 passed.

### Decision

- Budget is a hard character ceiling (`tokens × 4`), not a claim about the tokenizer.
- The verified answer and instructions are never dropped; detail is what gets cut.
- Default stays 4k (the Studio `--pack-budget` default), smaller budgets are opt-in.

### Next

- Validate the 2k pack on the 30-question Qwen baseline (accuracy should hold if the verified
  answer is retained; measure latency/token savings).

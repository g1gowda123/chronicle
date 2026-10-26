# Chronicle VFM — Video-Free Memory (experimental branch)

**Branch:** `experiment/video-free-memory`
**Worktree:** `/home/g1/projects/chronicle-vfm`
**Status:** research + prototype, not merged into `main`

## Mission

Stop treating video as the thing we keep. Treat the **compiled memory** as the product.

The current pipeline still depends on the source video for playback, frame extraction and
"re-watch" evidence. That makes queries expensive and forces us to keep raw footage. This
experiment asks a more radical question:

> Can a long egocentric episode be reduced to a small, timestamped, queryable memory plus a
> handful of event snapshots, such that useful questions are answered **without decoding video
> and without running a video LLM at query time**?

The compiler is allowed to be expensive. It runs once, writes a dense ledger, a compact LLM-ready
memory pack and a few WebP snapshots. The video can be kept for playback or deleted; either way,
the answer path and the LLM context never require it. Pixels are an audit/fallback path, never the
default. The product boundary is **memory at rest + a pack an LLM can answer from**, not "video
must be destroyed".

**What "video-free" means here:** the *LLM/query context* never contains video frames. The
Studio may keep the uploaded source for human playback; that is a UX feature, not an LLM
dependency. Qwen receives only the compiled captions/events.

## Core hypothesis

1. **Heavy model once, cheap queries forever.** A local VLM captions sampled frames offline;
   the compressed ledger answers at query time with no model.
2. **Text + graph beats raw video on long horizons.** CapMem, EgoCITE, EgoGraph, MERIT,
   FocusGraph, VL-KnG and R4DSG all point at compiled, timestamped memory.
3. **Pixels on demand.** One 480 px WebP per major event; the video is never required.
4. **Materialise answers.** A canonical answer cache absorbs paraphrases and repeated questions.
5. **Retention is a first-class design constraint.** The ledger grows with events, evidence with
   snapshots. A one-hour episode is designed to stay under ~1 MB at rest.

## Layout

```
docs/
  JOURNAL.md      dated research log (append-only)
  RESEARCH.md     literature review + standardized benchmark options
  DESIGN.md       architecture, schema, query flow, retention policy
benchmark/
  questions.json  30-question curated set with per-category scoring
schema_v2.sql     packed event/caption vault (base)
schema_v3.sql     dense VLM ledger + answer cache additions
run_studio.sh     launch VFM Studio (upload + optional Qwen NLQ)
prototype/
  compile.py          source Chronicle DB -> compressed vault (salience coalescing)
  compact.py          dictionary encoding, event packing, zlib, int8 quantization
  caption_ledger.py   offline VLM -> 30/60 s caption ledger (Ollama + ffmpeg)
  attach_ledger.py    attach compressed ledger to a vault
  query.py            where/when/around/before/after/count/inventory/keyword + cache
  nlq.py              optional Qwen answers from compiled captions/events (no video)
  video_vault.py      raw video + caption ledger -> SQLite vault (upload path)
  studio_app.py       NiceGUI upload → compile → playback/download UI
  benchmark.py        multi-episode 30-question evaluation + legacy smoke test
  original_baseline.py original Chronicle graph + Qwen 7B, same 30 questions
  llm_pack_baseline.py Qwen 7B answering from the compiled memory pack
  vlm_baseline.py     measured video-to-LLM baseline (9-frame contact sheet, Ollama)
  memory_pack.py      LLM-ready pack: captions + events + verified answer + evidence refs
  video_vault.py      caption-first vault builder for arbitrary uploads
  nlq.py              optional Qwen phrasing over the memory pack
  studio_app.py       upload → compile → playback/download NiceGUI app
  run_studio.sh       launcher (`--qwen`, model, port)
  online_baseline.py  same baseline against OpenAI / Anthropic / Gemini (key required)
  extract_evidence.py one-time video pass -> 480 px WebP snapshots, optional delete
  retention.py        bytes/hour report, snapshot-budget sweep, prune/apply
  concat_vaults.py    synthetic long-horizon concatenation
  scale_test.py       10/30/60 min growth + latency test
  report.py           single-file offline HTML browser with a client-side query box
tests/
  test_compact.py     packing/compression/compile->query
  test_retention.py   ledger parsing, cache, retention select/apply, baseline scoring
```

## Quick start

```bash
cd /home/g1/projects/chronicle-vfm/experiments/video-free-memory

# 1. Compile the structured graph into a tiny vault
python3 prototype/compile.py \
  --source-db /home/g1/projects/chronicle/data/real/cook/chronicle.db \
  --out /tmp/vfm/cook.db

# 2. Build a dense caption ledger once (local Ollama VLM; no cloud)
python3 prototype/caption_ledger.py \
  --video /home/g1/projects/chronicle/data/real/cook/aria_cook_45s.mp4 \
  --out /tmp/vfm/cook_ledger_30.json \
  --model qwen2.5vl:3b --window-s 30 --frames-per-window 3
python3 prototype/attach_ledger.py \
  --vault /tmp/vfm/cook.db --ledger /tmp/vfm/cook_ledger_30.json

# 3. One-time evidence pass; chooses 480 px WebP q75 and can delete the video
/home/g1/projects/chronicle/.venv/bin/python prototype/extract_evidence.py \
  --vault /tmp/vfm/cook.db \
  --video /home/g1/projects/chronicle/data/real/cook/aria_cook_45s.mp4 \
  --out /tmp/vfm/evidence-cook --data-root /tmp/vfm \
  --width 480 --quality 75 --format webp --delete-video

# 4. Query with no video and no model
python3 prototype/query.py --db /tmp/vfm/cook.db "where is the cup?"
python3 prototype/query.py --db /tmp/vfm/cook.db "what happened around 20s?"
python3 prototype/query.py --db /tmp/vfm/cook.db "which room is the mug in?"  # cache hit

# 5. 30-question, per-category evaluation
python3 prototype/benchmark.py \
  --questions benchmark/questions.json \
  --vault-dir /tmp/vfm \
  --source-root /home/g1/projects/chronicle/data/real

# 5b. Measured video-to-LLM baseline (9-frame contact sheet, local Ollama)
/home/g1/projects/chronicle/.venv/bin/python prototype/vlm_baseline.py \
  --questions benchmark/questions.json \
  --source-root /home/g1/projects/chronicle/data/real \
  --vault-dir /tmp/vfm --model qwen2.5vl:3b --frames 9

# 5c. Original Chronicle graph + Qwen 7B on the same questions
/home/g1/projects/chronicle/.venv/bin/python prototype/original_baseline.py \
  --questions benchmark/questions.json \
  --source-root /home/g1/projects/chronicle/data/real \
  --model qwen2.5:7b-instruct

# 5c. Optional online baseline (real API key; no key is bundled)
export GEMINI_API_KEY=...
/home/g1/projects/chronicle/.venv/bin/python prototype/online_baseline.py \
  --provider gemini --model gemini-3.8-flash \
  --questions benchmark/questions.json \
  --source-root /home/g1/projects/chronicle/data/real \
  --vault-dir /tmp/vfm --frames 9

# 6. Retention budget and snapshot-policy sweep
python3 prototype/retention.py --vault /tmp/vfm/cook.db --report --sweep 40,12,4

# 7. Single-file offline memory browser (query box runs client-side)
python3 prototype/report.py --db /tmp/vfm/cook.db \
  --out /tmp/vfm/cook.html --data-root /tmp/vfm

# 8. Tests
/home/g1/projects/chronicle/.venv/bin/python -m pytest -q tests
```

## VFM Studio: upload → compiled memory

This is the user-facing converter. The human keeps video playback; the **LLM never receives
frames**. Qwen answers from the caption ledger/events only.

```bash
cd /home/g1/projects/chronicle-vfm/experiments/video-free-memory
./run_studio.sh --qwen --port 8093
# open http://127.0.0.1:8093/
```

Options:

| Flag | Meaning |
|---|---|
| `--qwen` | enable Qwen natural-language answers over the compiled text |
| `--qwen-model` | text model, default `qwen2.5:7b-instruct` |
| `--vlm-model` | compile-time captioner, default `qwen2.5vl:3b` |
| `--window-s` / `--frames-per-window` | caption window policy (default 30 s / 3 frames) |
| `--delete-video` | remove the uploaded source after compile (no playback) |
| `--data-dir` | job workspace (default `/tmp/vfm/studio`) |
| `--port` / `--host` | server bind |

The Studio:

1. saves the upload and shows it in a normal HTML5 player;
2. captions 30 s windows once with the local VLM;
3. builds a SQLite vault + WebP evidence + offline HTML report + CSV exports;
4. lets you download `memory.zip`, `vault.db`, `memory.html`, `ledger.json`, `events.csv`,
   `windows.csv`;
5. if `--qwen` is on, answers questions from the **compiled ledger/events**, not video.

The downloaded memory is video-free. Keeping the source for playback does not change the LLM
context; `nlq.py` sends only captions/events to Qwen. Use `--delete-video` if playback is not
wanted.

## Studio: upload → compile → download

The user-facing app keeps video playback; the compiled vault and downloaded files are video-free.

```bash
cd /home/g1/projects/chronicle-vfm/experiments/video-free-memory
./run_studio.sh --qwen --port 8093
# open http://127.0.0.1:8093/
```

Upload MP4/MOV/WebM/MKV. The Studio:

1. previews the source video with native playback;
2. captions 30 s windows with the local VLM;
3. builds a queryable SQLite vault and LLM-ready memory pack;
4. extracts 480 px WebP evidence and a single-file HTML report;
5. offers downloads: `vault.db`, `memory.html`, `memory_pack.md`,
   `memory_pack.json`, `ledger.json`, `events.csv`, `windows.csv`, `memory.zip`;
6. answers questions from the memory pack with Qwen when `--qwen` is on.

Useful launch options:

```bash
./run_studio.sh --qwen --qwen-model qwen2.5:7b-instruct --vlm-model qwen2.5vl:3b
./run_studio.sh --no-qwen --port 8093          # deterministic answers only
./run_studio.sh --qwen --delete-video          # no playback after compile
./run_studio.sh --help
```

The source video is **kept for playback by default**. The vault, memory pack, report and zip
never reference it. `--delete-video` removes playback and makes the session strictly video-free.

### Why the memory pack is easier for an LLM

Prompting over a plain video forces the model to re-watch/re-sample and still miss exact times.
The compiler now produces `memory_pack.md` / `memory_pack.json` with:

- the **verified structured answer** (authoritative when present),
- ranked timestamped caption windows, objects/actions/zones,
- salient events with `[t=..s]` citations,
- evidence references,
- explicit counts and “say unknown” rules.

Measured on the same 30 questions:

| Input to the LLM | Accuracy | p50 | Tokens/question | Video at answer time |
|---|---:|---:|---:|---|
| 9-frame video contact sheet + Qwen2.5-VL | 7/30 | 4,456 ms | 764 | yes, 19.24 MB |
| Original graph + Qwen 7B | 14/30 | 847 ms | — | yes, DB/video |
| **Compiled memory pack + Qwen 7B (text)** | **30/30** | **1,074 ms** | **~1,055** | **no** |
| Deterministic vault query | 30/30 | 0.045 ms | 0 | no |

The pack is small (~4,000 chars / ~1,000 tokens per question), fits a normal context window, and
Qwen only has to phrase the verified answer. This is the intended usability change: the hard work
moves into compilation, not into prompting a video model.

## Test the two UIs side by side

Original Chronicle dashboard (video + graph + Qwen):

```bash
cd /home/g1/projects/chronicle
.venv/bin/chronicle --data-dir data serve --source cook --host 127.0.0.1 --port 8091
# open http://127.0.0.1:8091/
```

VFM memory browser (single file, no video, client-side query box):

```bash
cd /home/g1/projects/chronicle-vfm/experiments/video-free-memory
python3 prototype/report.py --db /tmp/vfm/cook.db --out /tmp/vfm/cook.html --data-root /tmp/vfm
python3 -m http.server 8092 --directory /tmp/vfm
# open http://127.0.0.1:8092/cook.html
```

Try the same question in both: `where is the plate?`, `what happened around 20s?`,
`how many plates?`, `what happened before the plate was placed in the bowl?`.

## v3 baseline (Aria cook, 45 s, 11 entities)

| Metric | Value |
|---|---:|
| Source video | 2.99 MB |
| Source graph DB | 436 KB |
| Events after salience coalescing | **51** (was 122) |
| Event pack | **879 B, 17.2 B/event** |
| Dense VLM ledger (30 s windows) | **996 B zlib** (5.0 KB raw) |
| Vault (ledger + evidence metadata) | **27 KB** |
| Evidence | **10 WebP, 137.5 KB** (480 px, q75) |
| Memory at rest | **164.5 KB (5.4% of video)** |
| Query cold p50 / p95 | **0.043 / 0.120 ms** |
| 30-question set | **30/30**, all six categories 100% |
| Answer cache | **4/4 paraphrases hit**, reused accuracy 100% |
| HTML report | **421 KB**, no video, no network |

The video-free proof was run by copying the clip and DB to a temp directory, compiling,
extracting evidence, deleting the only video copy, and then querying and rendering the report:
zero video files remained and every answer still worked.

## 30 s vs 60 s caption windows

Three episodes (cook, walking, HoloAssist desk), qwen2.5vl:3b + qwen2.5:1.5b synthesis,
three frames per window.

| Episode | Window | Ledger bytes (zlib) | Objects | Actions | 30-question accuracy |
|---|---:|---:|---:|---:|---:|
| cook | 30 s | 996 B | 19 | 26 | 30/30 |
| cook | 60 s | 648 B | 13 | 16 | 30/30 |
| walking | 30 s | 939 B | 12 | 12 | 30/30 |
| walking | 60 s | 671 B | 11 | 7 | 30/30 |
| hololens | 30 s | 996 B | 16 | 15 | 30/30 |
| hololens | 60 s | 665 B | 9 | 6 | 30/30 |

30 s windows cost roughly 350 B more per episode but carry materially more objects/actions.
The structured questions do not change because events answer exact times; the dense ledger is
for recall and paraphrase/object coverage. **Default: 30 s windows, three frames.**

## Original Chronicle vs VFM (same 30 questions)

Ran the original `ChronicleRuntime.ask()` path (graph + Qwen2.5:7b-instruct) against the
three source DBs, then the vault against the same questions. Same scoring code.

| System | 30-question accuracy | p50 / p95 | Query-time model | Video at query | At-rest footprint |
|---|---:|---:|---|---|---:|
| **Original Chronicle** | **14/30 (46.7%)** | **847 / 4,610 ms** | Qwen2.5 7B (4.7 GB) | yes, for playback/evidence | 436 KB DB + 2.99 MB video (cook) |
| **VFM vault** | **30/30 (100%)** | **0.045 / 0.113 ms** | none | **no** | 164.5 KB (cook) |
| **Qwen 7B from memory pack** | **30/30 (100%)** | **1,074 / 2,660 ms** | Qwen2.5 7B (text only) | **no** | 404 KB + ~1,055 tokens/question |
| Qwen2.5-VL 9-frame sheet | 7/30 (23.3%) | 4,456 / 7,202 ms | Qwen2.5-VL 3B | yes, 19.24 MB | video + 395 KB sheets |

The Qwen pack row is the usability result: a text LLM answers **30/30** from the compiled memory
pack, with no video and no re-captioning. The pack includes the verified structured answer,
timestamped captions, events and evidence references; Qwen phrases it. It uses 43,239 prompt +
996 output tokens for 30 questions (~1,055 pack tokens/question). The deterministic vault remains
the 0.045 ms path; Qwen is the natural-language layer.

Per category (original / vault / Qwen pack):

| Category | Original | Vault | Qwen pack | Total |
|---|---:|---:|---:|---:|
| location | 3 | 6 | 6 | 6 |
| temporal | 2 | 6 | 6 | 6 |
| count | 0 | 5 | 5 | 5 |
| before/after | 2 | 4 | 4 | 4 |
| negation | 5 | 5 | 5 | 5 |
| paraphrase | 2 | 4 | 4 | 4 |

The original is **more fluent** and often explains the scene naturally, but it reads the current
graph/terminal anchors, so it reports stale locations and misses early history: e.g. the plate is
"resting on the countertop at 38.1 s" after it was placed in the bowl at 43.4 s; the knife's first
appearance is 19.0 s instead of 7.1 s. The vault keeps the full event ledger and answers exactly.

Honest caveat on **count** questions: the original says "one cup", "one plate", "at least three
bowls"; the vault says "2 cup tracks", "2 plate tracks", "4 bowl tracks". The tracker splits
physical objects, so the vault's count is a **track count**, not necessarily the physical count.
The count category is not a clean win without visual ground truth. The original also answers
`where is the mug?` with "I don't have that in my memory" while the vault maps mug → cup, and it
only tracks terminal states rather than the full history.

So: **better for retention, speed, video independence, exact temporal lookup and regression
accuracy; not strictly better at natural-language richness or physical object counting.**

## Video-to-LLM baseline (measured)

The comparison is a local Qwen2.5-VL:3b given one 9-frame contact sheet per episode, with
timestamps burned into each tile. The sheet is cached per episode; the VLM is called once per
question. This is a frame-sampling proxy for "send the video to an LLM", not a true video LLM.
A 16-frame sheet was also tried but the Ollama runner returned zero tokens/calls, so it is not
reported as a valid result.

| System | 30-question accuracy | p50 / p95 | Tokens / question | API-equivalent cost / 30 q | Video at query | Memory at rest |
|---|---:|---:|---:|---:|---|---:|
| **Compiled vault** | **30/30 (100%)** | **0.12 / 0.26 ms** | **0** | **$0** | **No** | **404 KB** |
| Qwen2.5-VL, 9-frame sheet | 7/30 (23.3%) | 4,456 / 7,202 ms | 764 (22,922 total) | $0.0036* | Yes (19.24 MB) | video + 395 KB sheets |

\* Illustrative $0.15/M input and $0.60/M output tokens. The local run used 2.33 Wh of GPU energy
(~$0.00035 at $0.15/kWh), but the dominant cost is latency, not electricity. Vault queries are
~36,000× faster at p50 and use no video or model at query time. Video-to-memory compression is
**48.8×** (20.17 MB of source clips vs 404 KB of vault + unique evidence).

Per category (vault vs 9-frame VLM):

| Category | Vault | Video-LLM | Total |
|---|---:|---:|---:|
| location | 6 | 1 | 6 |
| temporal | 6 | 0 | 6 |
| count | 5 | 0 | 5 |
| before/after | 4 | 2 | 4 |
| negation | 5 | 4 | 5 |
| paraphrase | 4 | 0 | 4 |

Caveats:
- The 9-frame baseline often answers `unknown`, misses exact timestamps, and cannot count objects
  it does not see. That is the expected failure mode of sparse frame sampling.
- **Count questions are track counts, not verified physical counts.** The tracker splits objects
  (cook: 2 plate tracks, 4 bowl tracks), while the VLM may count visible physical objects. Do not
  read the 5/0 count split as a clean physical-count win.
- The Gemini free-tier probe (Gemini 3.8 Flash) took **234 s for one question**, used 1,134
  prompt tokens, returned a truncated answer and scored 0/1. A 30-question online run would take
  hours on this quota, so it is not reported as a benchmark. The adapter is real and ready for a
  paid/faster key; no online numbers are claimed here.
- A 16-frame local contact sheet returned zero tokens (Ollama runner failure), so only the valid
  9-frame baseline is reported.

## Data retention

The event ledger is cheap; snapshots dominate. Measured on a synthetic 60-minute episode
(2,474 events, 27 cycles of the three real clips) after capping evidence at 40 rows.

| Snapshot budget | Evidence rows | Vault | Memory/hour | 30 days at 1 h/day | 365 days |
|---:|---:|---:|---:|---:|---:|
| 40 / hour | 40 | 138 KB | **698 KB** | **20.5 MB** | 249 MB |
| 12 / hour | 12 | 134 KB | **282 KB** | **8.3 MB** | 101 MB |
| 4 / hour | 4 | 132 KB | **178 KB** | **5.2 MB** | 63 MB |

At 12 snapshots/hour a full hour fits in **257 KB actual** (134 KB vault + 123 KB unique
evidence); the 282 KB/hour figure is the conservative projection that assumes 12 unique images
per hour. The cache is volatile and can be dropped before quoting at-rest size.

Retention policy:
- **Ledger:** keep all salient events. 2,474 events/hour is only ~42 KB.
- **Dense captions:** keep all 30 s windows. ~1 KB per episode; ~2 KB/hour.
- **Evidence:** 12 snapshots/hour for long-term storage; 40/hour when maximum visual audit is
  wanted. 480 px WebP q75 is the knee (141 KB for 10 cook snapshots vs 457 KB at 640 px/q82
  JPEG; PSNR 35.8 vs 38.2 dB, SSIM 0.954 vs 0.967).
- **Answer cache:** regenerable; drop with `retention.py --drop-cache --vacuum`.

Projection caveat: the 60-minute scaling vault repeats three short clips, so evidence images
repeat. The table uses the conservative "12 unique images/hour" projection, not the smaller
synthetic unique-byte count.

## Long-horizon scaling (synthetic concatenation)

| Minutes | Events | Events/min | Vault | Load ms | p50 ms | p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 10.5 | 453 | 43.0 | 51 KB | 1.1 | 0.68 | 1.91 |
| 30.2 | 1,257 | 41.7 | 86 KB | 2.8 | 1.86 | 5.30 |
| 60.4 | 2,474 | 41.0 | 139 KB | 5.5 | 3.64 | 10.42 |

Size grows linearly with events, not frames. Query latency is **not flat**: it grows with event
count, but in-memory subject/time indexes keep it at 3.6 ms p50 / 10.4 ms p95 for a 60-minute
episode, far below the 50 ms target. Cold open (loading 2,474 events + ledger) is 5.5 ms.

### Does the advantage grow with video length?

Yes — the ratio grows roughly linearly, because video grows in MB/minute while the compiled
memory grows only with events (~42 KB/hour of packed ledger) plus a bounded snapshot budget.

| Episode | Source video | Memory at rest | Video / memory | Vault p50 |
|---|---:|---:|---:|---:|
| cook, 45 s | 2.99 MB | 164.5 KB | 18.6× | 0.045 ms |
| walking, 60 s | 13.18 MB | 168.5 KB | 80.1× | 0.045 ms |
| synthetic 60 min | ~498 MB/h projected | 282 KB/h at 12 snapshots/h | **~1,766×** | 3.6 ms |

At 40 snapshots/hour the 60-minute memory is ~698 KB/hour, still ~713× smaller than the
projected video. A real video LLM must either process more frames as duration grows (tokens,
latency and cost scale with duration) or keep a fixed frame budget and lose coverage. The 9-frame
baseline is already at 23% accuracy on 45–60 s clips; at one hour it would see one frame every
6–7 minutes. That is why the long-horizon advantage is larger than the short-clip numbers suggest.
The caveat: query latency grows with event count (0.045 ms → 3.6 ms), not with video length, and
the 60-minute figures are synthetic projections from real clips, not a new recording.

## Standardized benchmarks

The 30-question set is an internal regression suite, not a public benchmark. The closest
external test is **CapMem's caption-only Episodic Memory Video Caption QA**; **EgoSchema**,
**Ego4D NLQ**, **EgoLifeQA/EgoR1-bench** and **EgoMemory/EgoRetriever** are the other credible
options. See `docs/RESEARCH.md` for fit and caveats. What none of them measure: deleting the
video, evidence-JPEG sufficiency, semantic cache hit rate, or bytes/hour retention.

## Non-goals

- This branch does not replace the demo pipeline. It is a research track.
- It does not require the source video to exist. The prototype is tested with the video deleted.
- It does not claim a new benchmark SOTA. It claims a different **system boundary**: memory at
  rest, not video at query time.

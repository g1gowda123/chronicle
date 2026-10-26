# Chronicle VFM — Video-Free Memory (experimental branch)

**Branch:** `experiment/video-free-memory`
**Worktree:** `/home/g1/projects/chronicle-vfm`
**Status:** research + prototype, not merged into `main`

## Mission

Stop treating video as the thing we keep. Treat the **compiled memory** as the product.

The current pipeline still depends on the source video for playback, frame extraction and
"re-watch" evidence. That makes queries expensive and forces us to keep raw footage. This
experiment asks a more radical question:

> Can a 45-minute egocentric episode be reduced to a few megabytes of structured memory and
> event snapshots, such that every useful question is answered **without decoding video and
> without running a video LLM at query time**?

If the answer is yes, the model is used once, heavily, at ingest (the expensive part), and the
query path becomes a database lookup with optional small-model phrasing. Video becomes a build
artifact — deleted after compilation. Compute is amortised; privacy is structural.

## Core hypothesis

1. **Heavy model once, cheap queries forever.** A large VLM/captioner runs offline over selected
   keyframes/clips and writes a dense, timestamped memory ledger.
2. **Text + graph beats raw video on long horizons.** Recent work (CapMem, EgoCITE, EgoGraph,
   MERIT, VL-KnG, R4DSG) shows caption/graph memory is competitive or better for long-horizon
   episodic questions, while avoiding per-query video tokens.
3. **Pixels on demand.** Store one JPEG per major event. If a query genuinely needs visual
   detail, retrieve 1–3 stored keyframes and call a small VLM — never re-decode the episode.
4. **Materialise the answers.** Cache answers and reasoning traces for repeated/similar
   questions; a semantic cache is a first-class memory tier, not an optimisation afterthought.

## Layout

```
docs/
  JOURNAL.md      dated research log (append-only)
  RESEARCH.md     literature review with links and takeaways
  DESIGN.md       architecture, schema, query flow, radical ideas
schema.sql        compiled-memory schema
prototype/
  compile.py      existing Chronicle SQLite DB -> video-free vault
  query.py        SQL/graph query engine (no video, no LLM)
  benchmark.py    size + latency + coverage report
```

## Quick start

```bash
# Compile an existing Chronicle episode into a compressed vault
python prototype/compile.py \
  --source-db /home/g1/projects/chronicle/data/real/cook/chronicle.db \
  --out /tmp/chronicle-vault.db

# Query it with no video and no model
python prototype/query.py --db /tmp/chronicle-vault.db "where is the cup?"
python prototype/query.py --db /tmp/chronicle-vault.db "what happened around 20 seconds?"

# One-time evidence extraction from the video, then delete it
# (needs OpenCV; use the main Chronicle venv)
/home/g1/projects/chronicle/.venv/bin/python prototype/extract_evidence.py \
  --vault /tmp/chronicle-vault.db \
  --video /home/g1/projects/chronicle/data/real/aria_cook_45s.mp4 \
  --out /tmp/chronicle-evidence --data-root /tmp \
  --delete-video

# Offline, single-file memory browser (timeline + embedded snapshots)
python prototype/report.py --db /tmp/chronicle-vault.db \
  --out /tmp/chronicle-memory.html --data-root /tmp

# Size/latency/accuracy report
python prototype/benchmark.py \
  --source-db /home/g1/projects/chronicle/data/real/cook/chronicle.db \
  --vault /tmp/chronicle-vault.db
```

## v2 baseline (Aria cook, 45 s, 11 entities)

| Metric | Value |
|---|---:|
| Source video | 2.99 MB |
| Vault | **76 KB (2.48% of video)** |
| Event pack | **2,086 B (17.1 B/event)** |
| Caption windows | **429 B (zlib)** |
| Evidence snapshots | 15 JPEGs, 674 KB |
| Query latency | **0.04 ms p50** |
| Single-file HTML report | 915 KB |

The snapshot selector caps at 40 events, so a one-hour episode targets roughly **2 MB of memory
per hour**, versus hundreds of MB of raw video.

## Non-goals

- This branch does not replace the demo pipeline. It is a research track.
- It does not require the source video to exist. In fact, the prototype is tested with the video
  path removed.
- It does not claim a new benchmark SOTA. It claims a different **system boundary**: memory at
  rest, not video at query time.

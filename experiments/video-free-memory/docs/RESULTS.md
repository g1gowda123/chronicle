# Results ready for the demo (2026-10-27)

Short version for the operator/judges. Details: `docs/COUNTS.md`, `docs/JOURNAL.md`
(Sessions 7–9), `docs/count_bounds.benchmark.txt`.

## What changed

1. **Memory-pack token budgets** (`51a09d9`) — `--token-budget` / `--pack-budget` and
   `memory_pack_{2,4,8}k.md` exports. The verified answer and answering rules are never cut.
2. **Object-count bounds** (`f4e915f`) — compile-time `tracks` (upper) + `peak` (lower) per label.
3. **Counts in the offline report** (`99f2273`) — the single-file `memory.html` renders the same
   answer without a server or model.
4. **Cross-label duplicate de-duplication** (`e08cde3`) — the two cook plates were emitted as both
   `bowl` and `plate`; near-identical boxes (IoU 0.91–0.99) are now counted once and one-frame
   transients are dropped.

## Measured

| Check | Result |
| --- | --- |
| 30-question deterministic suite | **30/30 cold, 30/30 warm, 4/4 cache hits** |
| Latency | p50 **0.061 ms** cold, **0.067 ms** warm |
| Experiment tests | **24 passed** |
| Dashboard package tests | **11 passed** (`test_vfm.py`, `test_dashboard_smoke.py`) |
| Full worktree suite | **passes** |
| Cook bowls | 4 tracks → 3 plate duplicates + 1 transient → **physical 0** |
| Cook plates | 2 tracks → **physical 2** (visually confirmed) |
| Walking cars | 2 tracks, peak 1 → **physical 1–2** |
| HoloLens joy-cons | 2 tracks, peak 2 → **physical 2** |
| Memory pack, 60 min synthetic | 2k = 7,844 chars · 4k = 15,824 · 8k = 31,833 |
| Short cook pack | 6,347 chars → fits under 2k |

## Demo answers to expect

```text
how many bowls?
The memory tracked 4 bowl tracks. 3 of them are near-identical duplicate detections of
3 plate tracks and 1 one-frame track is transient, so the physical bowl count is 0.

how many plates?
The memory tracked 2 plate tracks. Up to 2 were visible at the same instant, which is still
a track count, not a verified physical count.

how many cars?
The memory tracked 2 car tracks. At most 1 car track was visible at the same instant, so the
physical count is between 1 and 2: fragmentation or occlusion may inflate the track count.
```

## Reproduce

```bash
cd /home/g1/projects/chronicle-vfm/experiments/video-free-memory
/home/g1/projects/chronicle/.venv/bin/python -m pytest -q tests

# Rebuild the three benchmark vaults, then run the 30 questions
for ep in cook hololens walking; do
  /home/g1/projects/chronicle/.venv/bin/python prototype/compile.py \
    --source-db /home/g1/projects/chronicle/data/real/$ep/chronicle.db \
    --out /tmp/vfm_counts/$ep.db
  /home/g1/projects/chronicle/.venv/bin/python prototype/attach_ledger.py \
    --vault /tmp/vfm_counts/$ep.db --ledger /tmp/vfm/${ep}_ledger_30.json
done
/home/g1/projects/chronicle/.venv/bin/python prototype/benchmark.py \
  --questions benchmark/questions.json --vault-dir /tmp/vfm_counts \
  --source-root /home/g1/projects/chronicle/data/real

# Token budgets
/home/g1/projects/chronicle/.venv/bin/python prototype/memory_pack.py \
  --db /tmp/vfm/scale/synth_60m.db --list-budgets

# Offline report with the count branch
/home/g1/projects/chronicle/.venv/bin/python prototype/report.py \
  --db /tmp/vfm_counts/cook.db --out /tmp/vfm_counts/cook.html --data-root /tmp/vfm
```

## Honest caveats

- The 30 questions are an internal regression suite, not a public benchmark.
- `physical` is a conservative estimate, not hand-labelled ground truth for every episode.
- Cook bowls/plates were visually checked frame by frame; walking cars were not.
- Caption-first uploads (no graph DB) keep caption-coverage wording and must not claim physical
  counts.
- The benchmark still expects the string `4 bowl` because it encodes the raw track count; the
  answer now contains both that and `physical bowl count is 0`.

## UI (zine mission control, merged from main)

- `./chronicle.sh start` now launches the merged zine UI; `./chronicle.sh pipeline` is the
  live-pipeline/VFM operator UI.
- Stage toolbar: **S** saves the current frame, **F** toggles fullscreen. Every grounded answer
  has a `⤓ frame @ <time>` chip that captures the cited frame and opens a fullscreen lightbox
  with save/download.
- A `MEMORY · VIDEO-FREE` card compiles the active clip's DB, opens the report, downloads the
  vault/pack/zip, and the chat answers through the vault when it is ready.
- Clips without a graph DB (replay-only or fresh upload) show a clear note; the Memory card needs
  `<clip>/chronicle.db` (gitignored, copied locally for the demo).

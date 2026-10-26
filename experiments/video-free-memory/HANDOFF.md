# Handoff — Chronicle VFM experiment → main UI

**Branch:** `experiment/video-free-memory`  
**Worktree:** `/home/g1/projects/chronicle-vfm`  
**Experiment dir:** `experiments/video-free-memory`  
**Latest commit at handoff:** `5158a46`  
**Main repo:** `/home/g1/projects/chronicle` — do not destabilise it.

Read first: `README.md`, `docs/DESIGN.md`, `docs/JOURNAL.md`, `docs/RESEARCH.md`.

---

## TL;DR

The experiment builds a **compiled memory** from video: packed timestamped events, dense VLM
caption windows, a few WebP snapshots, a verified structured answer, and an LLM-ready memory pack.
The video is **not required** for query/LLM answers, but the Studio keeps it for playback because
the user experience matters more than deleting the file.

Do **not** rebuild the upload UI. The main dashboard already has the upload button, player, live
progress and export menu. The remaining work is to wire this experiment's VFM compile/export/ask
backend into that existing UI.

Measured on the same 30 questions:

| Path | Accuracy | p50 | Video at answer time |
|---|---:|---:|---|
| Deterministic VFM vault | **30/30** | **0.045 ms** | no |
| **Qwen 7B from memory pack** | **30/30** | **1.07 s** | **no** |
| Original Chronicle graph + Qwen 7B | 14/30 | 0.85 s | yes |
| Qwen2.5-VL, 9-frame contact sheet | 7/30 | 4.46 s | yes, 19.24 MB |

The Qwen pack row is the usability result. It is not prompted over plain video; it receives a
compact pack with the verified structured answer, timestamped captions, events and evidence refs.

---

## 1. What is different / optimised vs `main`

Main is untouched by this branch. The experiment adds:

### Backend / storage

1. **Salience coalescing** using the source DB's physical `reason` labels
   (`hand_contact`, `lockstep_hand_motion`, `enclosed_by_container`, `resting_on_support`).
   Cook: **122 → 51 events**; target 40–60/45 s.
2. **Packed event ledger**: `<IBHHHHBBH>` struct-of-arrays, **17.2 B/event**, zlib captions.
3. **1 KB SQLite pages set before WAL**: cook vault **77 KB → 27 KB**. A 60-minute synthetic
   vault is 139 KB.
4. **Dense VLM caption ledger**: qwen2.5vl:3b per-frame captions + qwen2.5:1.5b synthesis,
   30 s windows, zlib blob. Cook 30 s ledger ≈ **996 B**.
5. **Snapshot knee**: 480 px WebP q75 instead of 640 px JPEG q82.
   10 cook snapshots: **141 KB vs 457 KB**, PSNR 35.8 vs 38.2 dB, SSIM 0.954 vs 0.967.
6. **Retention tooling**: 40/12/4 snapshots per hour; 60-minute synthetic memory:
   **698 / 282 / 178 KB per hour**; 30 days at 1 h/day: **20.5 / 8.3 / 5.2 MB**.

### Query / LLM

7. Deterministic `query.py` expanded to where / when / around / before / after / count /
   inventory / yes-no / keyword, in-memory subject+time indexes, canonical answer cache.
   Cache: **4/4 paraphrases**, reused accuracy 100%.
8. **`memory_pack.py`**: LLM-ready pack = verified structured answer + ranked 30 s caption
   windows + events + evidence refs + counts + rules. **~1,055 tokens/question**, ~4,000 chars.
9. **`nlq.py`**: optional Qwen phrasing over the pack. No video. Qwen pack: **30/30**, p50 1.07 s,
   43,239 prompt + 996 output tokens for 30 questions.
10. **`studio_app.py` + `run_studio.sh`**: upload → caption → vault → evidence → HTML report →
    memory pack → CSV/zip downloads, with video playback and an optional `--qwen` ask box.

### Benchmarks / evaluation

11. **30-question, per-category evaluator** (`benchmark/questions.json`, `benchmark.py`).
12. Baselines: `original_baseline.py` (graph + Qwen 7B), `vlm_baseline.py` (9-frame video VLM),
    `llm_pack_baseline.py` (Qwen from pack), `online_baseline.py` (OpenAI/Anthropic/Gemini).
13. Scaling: 10/30/60 min synthetic episodes, 2,474 events at 60 min, p50 3.64 ms / p95 10.42 ms.
14. Single-file HTML report with timeline, dense captions and a client-side query box.

### Video playback

The **Studio keeps the uploaded source video for playback by default**. The vault, memory pack,
HTML report and zip never reference it. `--delete-video` is only for a strictly video-free demo.

---

## 2. Main UI: use what already exists

The user spent significant time on the main dashboard. **Do not replace it.**

Relevant main files:

- `chronicle/ui/app.py` — NiceGUI dashboard, upload dropzone, video player, export menu.
  Upload handler: `_on_video_upload`; destination helper: `new_upload_target`; source registration:
  `register_upload`.
- `chronicle/ui/uploads.py` — saves uploads under `<data-dir>/uploads/<run-id>/source.<ext>` and
  creates `source.json`; DB is `<run-id>/chronicle.db`.
- `chronicle/integration.py` — `ChronicleRuntime`; `ingest_video()` / `ask()`.
- `dashboard/app.py`, `dashboard/runtime.py`, `dashboard/ui/page.py`, `dashboard/static/*` —
  the other dashboard implementation. **Main currently has uncommitted changes in these files**
  from another agent. Do not revert or overwrite them.

### Integration target

After the existing upload/live ingest finishes, compile VFM memory from the uploaded source and
current DB, then expose downloads in the existing UI. Do not add a second upload button.

Suggested flow:

1. Existing upload saves `source.mp4` and runs `state.request_run(...)`.
2. When ingest reaches a stable DB, enqueue a VFM job with:
   - `video = uploads/<run-id>/source.mp4`
   - `source_db = uploads/<run-id>/chronicle.db`
   - `job_dir = uploads/<run-id>/vfm/`
3. Compile:
   - `compile_vault(source_db, job_dir/"vault.db")`
   - dense ledger: `caption_ledger.build_ledger(video, job_dir/"ledger.json", ...)`
   - `attach_ledger(vault, ledger_json)`
   - `extract_evidence(vault, video, job_dir/"evidence", data_root=job_dir, width=480,
     quality=75, format="webp")`
   - `report.build_report(vault, job_dir/"memory.html", data_root=job_dir)`
   - `memory_pack.build_pack(...)` and `build_pack_data(...)`
   - zip: `vault.db`, `memory.html`, `ledger.json`, `memory_pack.md/json`, `events.csv`,
     `windows.csv`, `evidence/*`
4. Keep the existing video player. Add a **“Memory” / “VFM”** panel or menu item:
   - status/progress (reuse existing log/toast style),
   - “Open memory report” (iframe or new tab),
   - downloads: `vault.db`, `memory.html`, `memory_pack.md`, `memory_pack.json`, zip,
   - optional “Ask with Qwen” input using `nlq.answer_nl(store, question, model=...)`.
5. Add one export menu item next to the existing `Export` menu:
   - `Export VFM memory (.zip)`
   - `Export memory pack (.md)`
   - `Open memory report`
6. Qwen toggle:
   - Default off → deterministic `query.answer`.
   - `--qwen` / UI toggle → `nlq.answer_nl` over the pack.
   - Show method/latency/tokens if available. Do not make Qwen mandatory.
7. Keep `--delete-video` off by default in the main UI. Playback is part of the UX.

### Import path

The experiment modules live under
`/home/g1/projects/chronicle-vfm/experiments/video-free-memory/prototype/`.
For main integration, either:

- copy them into a small `chronicle/vfm/` package (preferred; stable imports), or
- add the experiment prototype dir to `sys.path` in a feature-flagged integration module.

Do not import `studio_app.py` into the main dashboard; build a small service layer and reuse the
existing main UI.

---

## 3. How to run the current experiment

```bash
cd /home/g1/projects/chronicle-vfm/experiments/video-free-memory

# Studio (upload -> compile -> playback/downloads)
./run_studio.sh --qwen --port 8093
# or
./run_studio.sh --no-qwen --port 8093
./run_studio.sh --qwen --delete-video

# Deterministic vault
python3 prototype/query.py --db /tmp/vfm/cook.db "where is the cup?"

# HTML report
python3 prototype/report.py --db /tmp/vfm/cook.db --out /tmp/vfm/cook.html --data-root /tmp/vfm

# Benchmarks
python3 prototype/benchmark.py --questions benchmark/questions.json \
  --vault-dir /tmp/vfm --source-root /home/g1/projects/chronicle/data/real

/home/g1/projects/chronicle/.venv/bin/python prototype/llm_pack_baseline.py \
  --questions benchmark/questions.json --vault-dir /tmp/vfm --model qwen2.5:7b-instruct

/home/g1/projects/chronicle/.venv/bin/python prototype/original_baseline.py \
  --questions benchmark/questions.json \
  --source-root /home/g1/projects/chronicle/data/real --model qwen2.5:7b-instruct

# Retention
python3 prototype/retention.py --vault /tmp/vfm/scale/synth_60m.db --report --sweep 40,12,4

# Tests
/home/g1/projects/chronicle/.venv/bin/python -m pytest -q tests
```

Current local UI processes (may be live):

- Studio: `http://127.0.0.1:8093/`
- VFM report: `http://127.0.0.1:8092/cook.html`
- Original dashboard: `http://127.0.0.1:8091/`

Studio upload has been tested end-to-end through the NiceGUI upload endpoint: 8.2 s clip →
24 KB vault, 321 KB HTML, 8 WebP snapshots, 3.2 KB memory pack (~804 tokens) in ~6 s; video,
report, zip and pack downloads returned 200.

---

## 4. Code map

| File | Purpose |
|---|---|
| `prototype/compile.py` | Chronicle DB → packed vault; salience coalescing |
| `prototype/compact.py` | dictionary, event pack, zlib, int8 quantisation |
| `prototype/caption_ledger.py` | offline VLM frame captions + look/30 s synthesis |
| `prototype/attach_ledger.py` | attach compressed ledger to vault |
| `prototype/video_vault.py` | raw upload → caption-first vault (no graph DB needed) |
| `prototype/query.py` | deterministic intents, indexes, answer cache |
| `prototype/memory_pack.py` | LLM-ready pack (verified answer + captions + events + evidence) |
| `prototype/nlq.py` | optional Qwen phrasing over pack |
| `prototype/studio_app.py` | NiceGUI upload/playback/download/ask app |
| `run_studio.sh` | Studio launcher with `--qwen` and model options |
| `prototype/extract_evidence.py` | 480 px WebP evidence, optional video delete |
| `prototype/report.py` | single-file HTML timeline + query box |
| `prototype/retention.py` | bytes/hour, snapshot sweep, prune/apply |
| `prototype/concat_vaults.py` / `scale_test.py` | synthetic long-horizon scaling |
| `prototype/benchmark.py` | 30-question per-category evaluator |
| `benchmark/questions.json` | 30 questions, six categories |
| `prototype/original_baseline.py` | main graph + Qwen 7B baseline |
| `prototype/vlm_baseline.py` | 9-frame video-to-VLM baseline |
| `prototype/llm_pack_baseline.py` | Qwen from memory pack |
| `prototype/online_baseline.py` | OpenAI/Anthropic/Gemini adapter (key required) |
| `tests/` | 15 tests: pack, cache, retention, studio |

---

## 5. What still needs doing

### P0 — main UI integration

1. Wire VFM compile/export into `chronicle/ui/app.py` after the existing upload ingest.
2. Reuse the existing upload button, player, progress and export menu. No duplicate upload UI.
3. Add downloads and “Open memory report”.
4. Add optional Qwen ask (`--qwen`), default off.
5. Preserve the uploaded video for playback; make deletion optional.
6. Browser E2E: upload → wait → play → seek → ask → download zip → unzip/open report.

### P1 — correctness / evaluation

7. Physical object counts. Current count questions are **track counts**; tracker fragmentation
   (cook: 2 plate tracks vs possibly 4 plates) makes physical counts unverified.
8. External benchmark: CapMem caption-only QA if released; otherwise 100-clip EgoSchema subset;
   Ego4D NLQ for temporal grounding.
9. Improve the ungrounded pack (18/30) only if you want the LLM to reason beyond the verified
   structured answer. The current 30/30 deliberately uses retrieval + phrasing.
10. Gemini/OpenAI online baseline on a paid/faster key. The free Gemini probe took **234 s for one
    question**; do not report it as a benchmark.

### P2 — product polish

11. Snapshot dedup / perceptual hash and a 24 h / 7 day retention test.
12. Add a token-budget selector to the pack (2k/4k/8k).
13. Optional audio/speech ledger (EgoCITE).
14. Int8 embeddings only if lexical/dense retrieval misses measurable cases.

---

## 6. Gotchas / do-nots

- **Do not revert main dashboard changes.** `dashboard/app.py`, `dashboard/runtime.py`,
  `dashboard/ui/page.py`, `dashboard/ui/theme.py`, `dashboard/static/chronicle.js` may have
  uncommitted work from another agent. Inspect before editing.
- **Do not replace the existing upload/player UI.** Add VFM behind it.
- **Do not make video deletion the default in the UI.** Keep playback; `--delete-video` is opt-in.
- **Do not claim 30/30 is a public benchmark.** It is a self-defined regression set.
- **Do not claim physical counts.** Label counts as track counts.
- **Qwen pack 30/30 includes the verified structured answer.** If removed, the same pack scores
  18/30. This is intentional retrieval+phrasing, not a standalone video LLM.
- **The 60-minute scale test is synthetic** (cycles three clips); unique evidence bytes are real,
  repeated rows are not.
- **SQLite size must be checkpointed** with `PRAGMA wal_checkpoint(TRUNCATE)`; `-shm` is not
  at-rest storage.
- **Evidence rel_paths** must be relative to `--data-root` for `report.py` embedding.
- **Qwen2.5-VL multi-image** is unreliable in this Ollama build; caption per frame and synthesize.
- **16-frame contact sheet** failed (0 tokens); do not report it as a baseline.
- **No API key is in the repo.** The Gemini key was only passed at runtime; verify with
  `grep -R "AQ.Ab8"`.

---

## 7. Commit rules

- Work only on `experiment/video-free-memory`; do not push main.
- Run tests: `pytest -q tests` (15 pass at handoff).
- Run `git diff --check`.
- Commit/push after a fix, but coordinate with the other agent if main dashboard files are dirty.
- Current experiment HEAD: `5158a46`.

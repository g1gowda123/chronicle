# Main dashboard ↔ VFM integration

**Status:** implemented on `experiment/video-free-memory` (P0-VFM-UI)
**Date:** 2026-10-27
**Owner:** opencode (AI)
**Scope:** `chronicle/vfm/`, `chronicle/ui/app.py`, `chronicle/cli.py`,
`tests/test_vfm.py`, `tests/test_dashboard_smoke.py`

## What changed

The experiment's compile/query modules now live in a stable package,
`chronicle/vfm/`, so the dashboard never imports `studio_app.py` or adds the
experiment directory to `sys.path`. The existing dashboard is unchanged at the
UX level: same upload button, same player, same progress/log, same Export menu.
VFM is added **behind** them.

### Service

- `chronicle/vfm/service.py`
  - `compile_memory(video, source_db, job_dir, ...)`: snapshot the run DB,
    caption the video (optional), compile the packed vault, attach the dense
    ledger, extract 480 px WebP evidence, render `memory.html`, write the
    memory pack + CSVs and zip everything.
  - `VFMJobManager`: one background worker, per-source job status, persisted
    `job.json`, deterministic and optional Qwen asks, safe artifact lookup.
  - Uploads keep memory at `uploads/<run-id>/vfm/`; bundled sources use
    `<data-dir>/vfm/<source-key>/`.
  - The source DB is copied with SQLite's backup API, so compiling never reads
    a DB that the ingest worker is still mutating.
- Copied modules: `compact`, `compile`, `caption_ledger`, `attach_ledger`,
  `extract_evidence`, `query`, `memory_pack`, `nlq`, `report`, plus the two
  schema SQL files.

### Dashboard

- `DashboardState` gets `on_ingest_done`; the ingest worker calls it **after**
  a successful run, so compilation happens even if the upload tab is closed.
- New routes:
  - `GET /vfm/status/{source_key}`
  - `POST /vfm/compile/{source_key}` (optional `?force=true`)
  - `GET /vfm/report/{source_key}`
  - `GET /vfm/download/{source_key}/{name}`
  - `GET /vfm/evidence/{source_key}/{name}`
  - `POST /vfm/ask/{source_key}` (`{"question": "...", "qwen": false}`)
- New **Memory** tab in the existing right-hand card: status/progress, compile,
  open report, all downloads, and an ask box. Citation chips seek the existing
  video player; the nearest evidence image is shown under the answer. Qwen is a
  checkbox, default off.
- Existing header **Export** menu gains: VFM memory (ZIP), Memory pack (MD),
  Open memory report.
- Playback is preserved. Deletion is opt-in (`--vfm-delete-video`) and refused
  for any video outside `<data-dir>/uploads/`.

### CLI

```bash
chronicle serve --source synthetic
chronicle serve --qwen                  # Qwen over the memory pack
chronicle serve --no-vfm                # disable VFM compiles/asks (panel inert)
chronicle serve --vfm-window-s 20 --vfm-frames-per-window 3
chronicle serve --vfm-max-snapshots 40
chronicle serve --vfm-delete-video      # opt-in; playback is lost
```

## Verification

```bash
cd /home/g1/projects/chronicle-vfm
python -m pytest -q tests/test_vfm.py tests/test_dashboard_smoke.py
python -m pytest -q tests
cd experiments/video-free-memory && python -m pytest -q tests
```

Headless coverage (`tests/test_vfm.py`, 9 tests):

- compile from a generated graph DB + generated video + fake captioner;
- vault answers `where is the cup?` with no video and no VLM;
- zip contains vault, report, pack, CSVs and `evidence/`;
- background manager status/`job.json`/ask/artifact lookup;
- delete-video guard refuses non-upload clips;
- count answers expose track/peak bounds (`vault-count-visibility`);
- dashboard routes: status, report, zip/vault download, evidence image, ask;
- `POST /vfm/compile/{source}` queues a job and reaches done;
- `--no-vfm` disables routes; the ingest callback is wired to the manager;
- CLI flags parse.

Not automated: real browser upload → play → seek. The existing upload handler
is unchanged; the route test covers the compile/ask/download half.

## Measured on this machine

| Case | Result |
| --- | --- |
| cook DB, graph-only, no video | 51 events, 24 KB vault, 1,021-token pack, 38 KB report |
| cook DB + 45 s video + fake 30 s ledger | 10 WebP snapshots (140.8 KB), 4,417-char / ~1,104-token pack, 408 KB report |
| deterministic query | `where is the cup?` → `vault-where`, < 1 ms |
| headless full suite | `tests/` green (all tests) |
| experiment suite | 19 passed |
| **live server, synthetic auto-run** | ingest → auto-compile in ~13 s; 11.9 s clip, 1 caption window (3.2 KB ledger), 26 KB vault, 8 WebP (134 KB), 589-token pack, 377 KB report, 327 KB zip; `where are the keys?` → `vault-where` in 0.38 ms |
| live routes | `/vfm/status` done/100 %, `/vfm/report` 200 (377 KB), `/vfm/download/memory.zip` 200 (327 KB); page contains Memory tab + all Export items |

The numbers are regression evidence, not a new benchmark.

## P1-COUNTS sync

`chronicle/vfm/` is synced with `prototype/` at commit `f4e915f` (P1-COUNTS bounds) and
includes P2-PACK. The dashboard reports count **bounds**: track count (upper bound) plus peak
simultaneous visibility (lower bound), with method `vault-count-visibility`.
`tests/test_vfm.py` covers this.

A later cross-label dedup refinement in `prototype/compile.py` / `query.py` / `report.py` is
being synced by the P1-COUNTS owner. `chronicle/vfm/report.py` already understands the optional
`physical` / `duplicates` / `transient` fields and falls back to the bounds wording until the
compiler copy is updated.

Keep the package in sync when `prototype/` changes. The copy is deliberate:
the dashboard must not import the experiment directory or rely on `sys.path`.

## Guard rails

- Do not replace the upload/player/Export UI.
- Do not delete bundled videos; `--vfm-delete-video` only affects uploads.
- Do not claim physical counts from the dashboard; counts remain track bounds.
- Do not make Qwen mandatory; deterministic queries work with Ollama offline.

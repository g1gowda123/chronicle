# Team status and role assignments

**Updated:** 2026-10-26
**Integration owner:** dashboard/runtime integration (me)
**Branch:** `main`

## TL;DR

The front end is now integrated into the team repo. The prototype dashboard has
been ported onto the real Role A → Role B → Role C stack, so everyone can keep
optimising behind a stable UI instead of waiting for a separate Role D branch.

```bash
uv sync --extra ui
chronicle sources
chronicle serve --source synthetic
```

`chronicle serve` runs the NiceGUI dashboard at <http://127.0.0.1:8080>. The
synthetic fixture works fully offline (colour detector + deterministic
embeddings + template GraphRAG). The HoloLens clip is available as
`--source tabletop`.

## What landed in this push

- `chronicle/integration.py` — `ChronicleRuntime`, the A→B→C façade used by the
  CLI, dashboard and demos.
- `chronicle/ui/` — NiceGUI dashboard: live memory graph, state timeline,
  entity/action tables, video player, GraphRAG chat, pipeline log, LLM status.
- `chronicle/sources.py` — source registry (synthetic, HoloLens tabletop, Aria
  walking/cook/living-room when the clips are present).
- `chronicle/cli.py` — `chronicle sources|run|ask|serve` entry point.
- `chronicle/graph/furniture.py` + memory routing — static furniture (sofa,
  table, TV, …) is associated into persistent background anchors instead of
  becoming duplicate foreground entities.
- `PerceptionPipeline.on_clip` — Role A→Role C clip hand-off with isolated
  callback failures.
- Tests for the runtime seam and furniture association, plus
  `docs/INTEGRATION.md`.

## Recent changes

### TASK-E2 — zone-clustered graph

- The dashboard graph now clusters nodes around their room/zone anchor. Zone
  nodes are fixed on a deterministic ring, each zone gets a stable accent
  colour, entities inherit a matching border, and an invisible force spring
  keeps zone members together without adding false edges to the exported graph.
- Graph/timeline option builders moved to `chronicle/ui/graph_view.py` so the
  payload is testable without NiceGUI/FastAPI (`tests/test_graph_view.py`).
- `visualization_snapshot()` now exposes `zone_id` on entity, furniture and zone
  nodes; foreground snapshots no longer shadow background anchors.

### TASK-E3 — snapshot import from the dashboard

- The header now has an **Import** button that opens a JSON upload dialog and
  calls `ChronicleRuntime.load_snapshot_json()` after stopping ingestion.
- Uploaded files are validated by `chronicle/ui/snapshot_io.py` (UTF-8/BOM,
  JSON object, `nodes`/`entities` lists) with actionable error notifications.
- Import is covered by model-free tests (`tests/test_snapshot_io.py`) and the
  receipt reports entity/action/zone counts.

### TASK-C2 — action-timeline evaluation

- `chronicle/semantic/timeline.py` scores `snapshot()["actions"]` (or a
  predictions export) against a hand-labelled JSON timeline using greedy
  nearest-time matching, tolerance, precision/recall, mean timestamp error and
  entity/container accuracy.
- Labelled files stay outside the code (`docs/action_timeline.example.json`);
  `python -m chronicle.semantic.timeline --labels ... --snapshot ...` exits 0
  only when every label is matched with no extras, so CI can gate the Role C
  "done when" bar.
- The runtime smoke (`chronicle run --source synthetic`) exports a snapshot that
  matches the bundled label at 0.07 s error; `tests/test_timeline.py` covers
  matching, loaders and CLI exit codes without models.

### TASK-F2 — multi-hop "before" questions

- `template_answer()` now recognises temporal phrasing (`before`, `prior to`,
  `previously`, `earlier`) and walks an entity's facts backwards from the
  reference event.
- The reference event is the placement/action the query names (e.g.
  `PLACED table`) or an explicit `t=..s`; the answer cites the latest location
  fact strictly before it, e.g. *"Before it was placed on the table, the cup had
  been resting on the work desk at [t=5.0s]."*
- When no earlier location exists the answer is an ungrounded
  `"I don't have an earlier location..."` fallback, so a multi-hop question can
  never invent a location.
- Added `before-placement` to the GraphRAG evaluation set (now 9/9, 100 %
  citation precision) and model-free coverage in `tests/test_graphrag.py`.

### TASK-F1 — HoloLens tabletop action benchmark

- `load_annotation_actions()` reads HoloAssist `ground_truth.actions`, picks the
  first candidate per row that maps to a Chronicle primitive, and skips
  movement-only rows. The bundled `holoassist_switch_34s.json` yields 7
  scorable labels.
- `chronicle run --snapshot-out PATH` exports the post-ingestion snapshot, and
  `python -m chronicle.semantic.timeline --annotations ... --snapshot ...`
  scores it (`--ignore-entity` drops entity/container agreement from the gate).
- Fallback benchmark (colour detector/hands/embedder; production models are not
  installed here): 19 predictions, 1 match, precision 5.3 %, recall 14.3 %,
  mean |Δt| 0.28 s, FAIL. Documented as fallback-only in
  `docs/PERCEPTION.md`; re-run with the production backends for the real number.

### TASK-F3 — live stream source (superseded)

- Removed per project direction to focus exclusively on reproducible egocentric
  video assets (`synthetic`, `tabletop`, `cook`, `play`, `walking`) with in-UI
  source switching.

### TASK-D5 — service launcher script (`chronicle.sh`)

- Created root `./chronicle.sh` script for zero-friction service management.
- Running `./chronicle.sh` starts the dashboard in the background (using `setsid`
  and detached input) and reports <http://127.0.0.1:8080>.
- Commands: `./chronicle.sh` (or `start`), `./chronicle.sh stop`, `./chronicle.sh restart`,
  `./chronicle.sh status`, `./chronicle.sh logs`, `./chronicle.sh test`.
- All video sources can be switched dynamically directly in the UI dropdown.

### TASK-E1 - Aria clip pipeline benchmark

- `chronicle/benchmark.py` runs any registered source through
  `ChronicleRuntime`, records the backends that actually loaded, checks
  snapshot invariants (unique IDs, known states, ordered transitions, action
  and transition references, summary/snapshot frame agreement), prints a
  Markdown table and writes a JSON report.
- Full-clip offline fallback (colour detector/hands/embedder): `cook` 450
  frames at 23.0 fps, `play` 450 at 22.2 fps, `walking` 600 at 25.3 fps; all
  nine checks pass, but real Aria footage yields zero detections from the
  colour backends.
- Bounded production run (YOLO-World, 30 frames, colour hands/embedder):
  `cook` 74 detections / 3 entities / 4 transitions at 0.4 fps, `play` 167 /
  2 / 4 at 0.6 fps, `walking` 60 / 3 / 7 at 1.0 fps; all 9/9 checks pass.
  YOLO-World tracks bowls, cup, people and a reacquired person across pans.
- No Aria action annotations are bundled, so E1 gates on end-to-end completion
  plus structural invariants rather than claiming action accuracy. The
  HoloLens/HoloAssist clip remains the labelled action benchmark (TASK-F1);
  Aria labels are tracked in `docs/TASK_REAL_FOOTAGE.md`.
- The production run also exposed a duplicate `frame_index` telemetry keyword;
  the sink now merges payload keys explicitly and a model-free regression test
  covers it (`tests/test_integration.py`).

### TASK-D7 / A2 — object appearance attributes

- Role A now runs `chronicle/perception/attributes.py` on every Nth detection
  crop: background-suppressed k-means gives a dominant BGR colour, converted to
  a stable colour word + `#rrggbb` swatch; brightness, saturation, shape/aspect,
  relative size, texture and frame position are recorded too.
- Attributes are merged into tracker metadata (with colour stabilisation across
  noisy frames), flattened onto `ObjectObservation.attributes`, and persisted
  into the foreground graph + SQLite entity `attrs`.
- The snapshot/CSV exports expose `colour`, `colour_hex`, `attributes` and an
  `appearance` summary; graph nodes use the detected swatch as an identity ring
  and include appearance in the tooltip.
- New query paths: "what colour is the cup?", "how big is it?", "describe the
  backpack" are answered from the attributes (grounded on `last_seen`), and
  "what red objects did you see?" filters the inventory by colour.
- Dashboard: the Entities tab gained a live filter box, colour swatch column and
  properties column; suggestions include a colour question for tracked objects.
- Covered by model-free `tests/test_attributes.py` plus extended export and
  dashboard tests.

## Dynamic Task Notice Board (Priority Queue)

> **Working board:** [`NOTICE_BOARD.md`](NOTICE_BOARD.md) is the standalone,
> claimable task board for parallel work. The table below is the historical
> cross-role index; use the notice board for new ownership and status changes.
>
> **Team Notice Board:** Anyone finishing early can claim the next highest-priority `[READY]` ticket or post a new task below.
> - To claim: update status to `[IN PROGRESS (your name)]`.
> - To post a new task: add a row with priority (`P0`/`P1`/`P2`), scope, and acceptance criteria.
> - When completed: update status to `[DONE (commit)]`.
> - Keep team rules: maintain additive APIs in `chronicle/integration.py` and run `.venv/bin/pytest` + `.venv/bin/ruff check chronicle tests` before pushing.

| ID | Pri | Task | Scope / Key Files | Status | Owner |
|---|---|---|---|---|---|
| **TASK-B1** | **P0** | **Dual furniture fixtures & SQLite persistence**<br>Add regression tests for 2 co-visible sofas remaining distinct during consolidation sweep, sequential duplicate merging on pan re-entry, and alias restoration across SQLite reopen blocking anchor re-minting. | `tests/test_consolidation.py`<br>`chronicle/graph/consolidation.py` | `[DONE (0636492)]` | karthiknk7019 |
| **TASK-B2** | **P1** | **Lifecycle & persistence hardening**<br>Enforce exactly one terminal `RESTING_ON`/`INSIDE` placement, non-mutual containment, parent-zone context for contained queries, `exit_dwell_s`, strict-support reacquisition, `OUT_OF_VIEW` archival reachability, multi-frame furniture pan compensation, timestamp clamping, and namespaced furniture promotion. | `chronicle/state_machine.py`<br>`chronicle/graph/{foreground,query,furniture,background,memory}.py`<br>`chronicle/storage/sqlite_store.py`<br>`tests/test_{state_machine,graph_storage,query,furniture,consolidation}.py` | `[DONE]` | Role B |
| **TASK-C1** | **P0** | **Verification queue depth telemetry in snapshot**<br>Expose `verification_stats` (`depth`, `dropped`, `avg_latency_ms`) in `ChronicleRuntime.snapshot()` and `snapshot()["stats"]`. Add queue depth regression test. | `chronicle/integration.py`<br>`tests/test_verification_queue.py` | `[DONE (ee9fed7)]` | karthiknk7019 |
| **TASK-C2** | **P0** | **Action-timeline evaluation harness**<br>Score verified actions against a hand-labelled timeline (tolerance, precision/recall, timestamp error, entity/container agreement) from a snapshot export or predictions file, with a CI-gating CLI. | `chronicle/semantic/timeline.py`<br>`tests/test_timeline.py` | `[DONE (b7f2c70)]` | karthiknk7019 |
| **TASK-D1** | **P0** | **JSON & CSV memory graph export**<br>Add export methods to runtime (`export_snapshot_json`, `export_entities_csv`, `export_actions_csv`, `load_snapshot_json`) and `/export/snapshot.json`, `/export/entities.csv`, `/export/actions.csv` endpoints + UI download buttons in NiceGUI header/card. CSVs use stable headers and JSON-encode nested values. | `chronicle/integration.py`<br>`chronicle/ui/app.py` | `[DONE (f52482b, 5e51c20)]` | Orchestrator / karthiknk7019 |
| **TASK-D2** | **P1** | **Distinct furniture anchor styling in graph**<br>Render background furniture anchors with distinct shape (rounded rectangle) and golden border (`#f59e0b`) so they are immediately visually distinct from episodic entities with lifecycles. | `chronicle/ui/app.py` | `[DONE]` | Orchestrator |
| **TASK-D3** | **P1** | **Queue depth badge & UI error/empty states**<br>Add verification queue depth chip in dashboard header; show friendly empty/stopped/error states when video is missing or query fails. | `chronicle/ui/app.py` | `[DONE]` | Orchestrator |
| **TASK-D4** | **P2** | **Headless dashboard smoke tests**<br>Add Starlette/FastAPI `TestClient` tests for NiceGUI endpoints (`/`, `/source-video`, `/export/snapshot.json`, `/export/entities.csv`, `/export/actions.csv`) verifying zero-GPU, zero-browser CI operation plus attachment headers/media types. | `tests/test_dashboard_smoke.py` | `[DONE (f52482b, 5e51c20)]` | Orchestrator / karthiknk7019 |
| **TASK-D5** | **P1** | **Service runner script (`chronicle.sh`)**<br>Zero-argument `./chronicle.sh` launcher to start dashboard daemon, report URL, tail logs, and stop easily with in-UI source switching. | `chronicle.sh` | `[DONE (7edb9df)]` | Orchestrator |
| **TASK-D6** | **P1** | **Dashboard video upload**<br>Add a video picker to the source card that saves the clip under `<data-dir>/uploads/<run-id>/`, registers and rediscovers it as an `UPLOAD` source, and starts ingestion immediately. | `chronicle/ui/app.py`<br>`chronicle/ui/uploads.py`<br>`tests/test_uploads.py` | `[DONE]` | Orchestrator |
| **TASK-E1** | **P1** | **Aria clip pipeline benchmarking**<br>Benchmark `chronicle run --source cook` / `play` / `walking` offline evaluation using the tracked video assets in `data/real/`; verify state transitions align with ground-truth action annotations. | `chronicle/sources.py`<br>`docs/PERCEPTION.md` | `[DONE (cc250a0)]` | Orchestrator / Likhithvu |
| **TASK-E2** | **P1** | **Graph visualizer force layout tuning & zone clustering**<br>Group nodes by zone in visually distinct clusters or categories in ECharts force layout so multi-room navigation is immediately intuitive. | `chronicle/ui/app.py`<br>`chronicle/ui/graph_view.py` | `[DONE (371f216)]` | Orchestrator |
| **TASK-E3** | **P2** | **Snapshot import UI button**<br>Wire a file upload widget and `/import/snapshot` route in the NiceGUI dashboard so an exported `chronicle_snapshot.json` restores graph state without re-ingesting; validate it with `chronicle/ui/snapshot_io.py`. | `chronicle/ui/app.py`<br>`chronicle/integration.py`<br>`chronicle/ui/snapshot_io.py` | `[DONE (a31d417, 973798a)]` | Orchestrator |
| **TASK-F1** | **P1** | **HoloLens tabletop action verification benchmark**<br>Benchmark `chronicle run --source tabletop` and evaluate action extraction against HoloLens ground truth (`docs/PERCEPTION.md`). | `chronicle/sources.py`<br>`docs/PERCEPTION.md` | `[DONE (feba585)]` | karthiknk7019 |
| **TASK-F2** | **P1** | **GraphRAG question prompt enhancement**<br>Support multi-hop queries like "Where was the cup before it was placed on the table?" using timeline edge walks. | `chronicle/semantic/graphrag.py`<br>`tests/test_graphrag.py` | `[DONE (f224d33)]` | Orchestrator / Likhithvu |
| **TASK-F3** | **P2** | **Live stream source implementation**<br>Implement live video capture ingestion loop for the `live` ClipSource placeholder (`chronicle run --source live`). | `chronicle/sources.py`<br>`chronicle/perception/pipeline.py` | `[SUPERSEDED]` | karthiknk7019 |
| **TASK-A2** | **P0** | **Simple object attributes in perception**<br>Extract dominant colour, swatch, brightness, saturation, shape/aspect, relative size, texture and frame position for every detection; stabilise colour across frames; carry through tracker, graph and SQLite. | `chronicle/perception/attributes.py`<br>`chronicle/perception/tracker.py`<br>`chronicle/graph/{adapter,memory,foreground}.py`<br>`tests/test_attributes.py` | `[DONE]` | Orchestrator |
| **TASK-D7** | **P1** | **Appearance in UI, exports and GraphRAG**<br>Snapshot/CSV colour+attribute columns, graph colour rings/tooltips, entity filter + swatch column, colour/size/describe questions and colour inventory queries. | `chronicle/integration.py`<br>`chronicle/graph/query.py`<br>`chronicle/semantic/graphrag.py`<br>`chronicle/ui/{app,graph_view}.py` | `[DONE]` | Orchestrator |
| **TASK-F4** | **P1** | **Source vocabulary & production-detector benchmark**<br>`chronicle run --source tabletop` currently ignores `ClipSource.vocabulary`, so YOLO-World prompts never include `joy-con`. Apply the source vocabulary, expose `--conf`/`--imgsz`, then benchmark the HoloLens clip with the real detector and record action-timeline numbers. | `chronicle/cli.py`<br>`tests/test_cli.py`<br>`docs/PERCEPTION.md` | `[IN PROGRESS (Role C)]` | karthiknk7019 |

## What I am working on right now

1. **Duplicate consolidation.** Role B now ships `GraphConsolidator` with
   `merge_entities` / `merge_furniture`, dry runs and `[GRAPH]` audit events.
   Legacy `entity:sofa` / `anchor:sofa-*` duplicates are folded conservatively
   while retaining terminal anchors, action edges, history and observations.
   Remaining work is clip-level tuning of the association and merge gates.
2. **Accuracy of the furniture association.** Tuning the confirmation window,
   camera compensation and embedding gate against the living-room clip; adding
   regression fixtures with two co-visible sofas and a re-entry after a pan.
3. **Runtime efficiency.** Deduplicating repeated support updates per frame,
   batching SQLite writes, and keeping UI snapshots cheap enough for 30 fps.

If you need to change `chronicle/integration.py`, `chronicle/ui/app.py` or
`chronicle/graph/furniture.py` while I am in them, leave a note in the PR or
ping me first. Everything else is fair game.

## Jobs by role

### Role A — perception and clips

**Goal:** make the frame stream good enough that Role B does not have to guess.

- [x] Emit support/furniture candidates from Role A (`FrameResult.supports` or
  clip metadata) so furniture anchors are not label-only.
- [x] Guarantee stable `entity_id` across occlusion/track loss; expose the
  re-ID similarity used for the decision.
- [x] Add per-object embedding/confidence to `ObjectObservation` where the
  detector can provide it.
- [x] Put the container name and contact evidence into `InteractionClip.metadata`
  so Role C does not have to infer it from the caption.
- [x] Benchmark YOLO-World and MediaPipe on the real HoloLens clip; record
  precision/recall and latency in `docs/PERCEPTION.md`.
- [x] Keep the palette/colour fallback green: `pytest tests/test_pipeline.py`.

**Status (Role A complete):** All Role A perception tasks completed, tested, and benchmarked. Re-ID stability and decision similarity exposed, container + contact evidence populated in clip metadata, support candidates piped through `FrameResult.supports`, and OpenCLIP shared across runtime. All 150 tests green and ruff clean.


### Role B — lifecycle, graph and persistence

**Goal:** one durable node per real-world object, with a clean audit trail.

- [x] Implement `ChronicleMemory.maybe_consolidate(t)` / a `GraphConsolidator`
  that merges legacy duplicate foreground entities and furniture anchors.
  Preserve terminal anchors, action edges, state history and telemetry.
- [x] Add `merge_entities(keep, remove)` and `merge_furniture(keep, remove)`
  with a dry-run mode and an audit event (`[GRAPH] entity merged`).
- [x] Use spatial-temporal + embedding evidence; never merge two co-visible
  instances of the same label.
- [x] Add regression tests: two sofas in one room stay two anchors; sequential
  duplicates merge; reopening SQLite restores aliases without minting a second
  anchor (`tests/test_consolidation.py`).
- [x] Improve query disambiguation for plural/label aliases and report match
  confidence in `QueryResult` (`match_score`, `confidence`, `match_reason`,
  `candidates`).
- [x] Keep `scripts/demo_role_b.py` passing after every schema change.

**Landed:** `GraphConsolidator`, merge APIs and alias/confidence retrieval;
`pytest`, `unittest discover`, `ruff check chronicle tests` and the demo are
green. Role B checklist complete.

**Hardening note (2026-10-26):** an audit against `spec.txt` found and fixed
concrete correctness gaps: SQLite could retain two terminal placements after a
lift or a `RESTING_ON`→`INSIDE` switch; containment was mutual for a cup inside
a backpack; contained-object queries dropped the container's room context;
`exit_dwell_s` was unused; strict-support reacquisition could stick in
`OCCLUDED`; a still camera demoted `OUT_OF_VIEW` before the 600 s `LOST` path;
multi-frame pans minted duplicate furniture anchors; restored furniture lost
`last_seen`; and a graph-only furniture merge wrote a non-namespaced SQLite id.
Fifteen focused regression tests cover the fixes. Current verification:
`pytest` green (233 model-free tests collected), `ruff check chronicle tests`
clean.

**Done when:** replaying the living-room clip leaves exactly one background
anchor per physical sofa, and `memory.stats()` reports the merges.

### Role C — semantic verification and GraphRAG

**Goal:** fewer VLM calls, higher action precision, answers that never invent a
location.

- [x] Add a bounded verification queue with backpressure and clip de-duplication
  (same `clip_id` must not be verified twice). `chronicle/semantic/queue.py`.
- [x] Calibrate action confidence on labelled clips; reject low-confidence
  `OTHER`/ambiguous captions instead of writing weak actions.
  `calibration.py` + `DEFAULT_ACTION_THRESHOLDS`.
- [x] Extract `container` from the strip/caption reliably; add tests for
  `INSERT_INTO` / `REMOVE_FROM`. `extract_container()` / `resolve_container()`.
- [x] Add a GraphRAG evaluation set: exact location, last-seen time, action,
  temporal "before" questions, and “I don’t know” cases. Track citation
  precision. `evaluation.py` (9 cases, currently 100 % pass / 100 % citation
  precision).
- [x] Answer multi-hop temporal questions ("where was X before it was placed on
  Y?") by walking the entity timeline back to the prior location fact, with an
  ungrounded fallback when no earlier fact exists.
- [x] Keep the offline template path as the reference answer; any LLM answer
  without a verifiable citation must fall back.
- [x] Emit queue depth / verification latency on the telemetry bus so the
  dashboard can show it. `QUEUE_DEPTH`, `QUEUE_BACKPRESSURE`,
  `VERIFICATION_LATENCY`, plus `runtime.verification_stats`.
- [x] Measure the “done when” bar: `chronicle/semantic/timeline.py` matches
  verified actions to a labelled timeline (precision/recall, timestamp error,
  entity/container agreement) and gates CI via its CLI. Remaining input is the
  labelled HoloLens pass (`docs/action_timeline.example.json` shows the format).

**Done when:** the HoloLens clip’s verified actions match the labelled timeline
within tolerance, and a hallucinated timestamp is always rejected.

**Role C note (this push):** all six items landed with model-free tests
(`tests/test_verification_queue.py`, `tests/test_calibration.py`,
`tests/test_evaluation.py`), plus TASK-C2’s action-timeline harness
(`tests/test_timeline.py`) and TASK-F2’s temporal multi-hop answers
(`tests/test_graphrag.py`). Verification on the merged tree: `pytest` 250
collected (249 passed, 1 dashboard test skipped without the `ui` extra),
`unittest discover -s tests` 128 passed, `ruff check chronicle tests` clean,
`chronicle.semantic.demo` / `chronicle.semantic.evaluation` exit 0 (9/9 cases,
100 % citation precision), and `chronicle.semantic.timeline` reports PASS
(1/1 label, 0.07 s error) on a snapshot exported from
`chronicle run --source synthetic --max-frames 360`. `snapshot()` now embeds
`verification_stats` (TASK-C1, `ee9fed7`), so the dashboard can poll queue
depth/dropped/latency without a second call. Labelled captions can be calibrated
from a JSON/JSONL file with
`python -m chronicle.semantic.calibration --labels <file>` (see
`docs/calibration_labels.example.json`), and the timeline scorer takes the same
kind of external label file, so calibrating and gating on the HoloLens pass need
no code changes. The only cross-role edit is the internal `_clip_queue` /
`_drain_clips` seam in `chronicle/integration.py` (bounded queue + latency
marks). Ambient caption phrasing still needs a labelled HoloLens pass before the
“done when” line can be ticked.

### Role D — dashboard and product surface

**Goal:** make the integrated dashboard the team’s daily driver.

- [x] Visual regression pass on the ported UI (graph, timeline, tables, chat,
  pipeline log) at 1280/1440/1920 px.
- [x] In-dashboard video source switching (`synthetic`, `tabletop`, `cook`, `play`, `walking`)
  directly in the UI dropdown without terminal commands.
- [x] Service launcher script (`./chronicle.sh`) for daemon lifecycle management.
- [x] Export/import: download the current snapshot as JSON/CSV, reload it
- [x] Export/import: download the current snapshot as JSON/CSV, reload it
  without re-ingesting (`export_snapshot_json`, `export_entities_csv`, `export_actions_csv`, `load_snapshot_json`),
  including the in-dashboard JSON upload dialog. CSV exports use stable headers and are covered by runtime and endpoint tests.
- [x] Add an explicit error/empty state for missing videos, stopped runs and
  failed GraphRAG queries.
- [x] Add a headless smoke test for `/`, `/source-video` and all three export
  endpoints (Playwright or `TestClient`), and keep the page usable without a GPU
  (`tests/test_dashboard_smoke.py`).
- [x] Show furniture anchors distinctly from episodic entities (they are
  background structure, not objects with a lifecycle, rendered as golden rounded rectangles).
- [x] Cluster the memory graph by zone with fixed room anchors, stable zone
  accents and zone-aware tooltips (`chronicle/ui/graph_view.py`).
- [x] Change the demo video from the dashboard: the source card's video button
  uploads a clip, persists it under `data/uploads/`, adds it as an `UPLOAD`
  source (rediscovered on restart) and starts ingestion immediately.

**Done when:** a teammate can switch sources, watch ingestion, ask a question
and export the graph without touching Python.

## Shared rules

- **Do not block on Role D.** The dashboard now consumes `ChronicleRuntime`; if
  you add a field to `snapshot()`, update the UI in the same PR.
- **Additive APIs only.** Keep `PerceptionPipeline`, `ChronicleMemory` and
  `SemanticEngine` contracts intact; put cross-role glue in
  `chronicle/integration.py`.
- **Model-free tests are mandatory.** CI has no GPU, no model weights and no
  Ollama. Use the colour/deterministic/template fallbacks. Tests that need the
  optional `ui` extra (FastAPI/NiceGUI) skip when it is absent, so the core
  `pytest` run stays green on a plain `pip install -r requirements.txt`.
- **Before pushing:** `pytest`, `unittest discover -s tests`, `ruff check
  chronicle tests`, and one `chronicle run` smoke test.
- **Report status here.** Tick your boxes and add a one-line note with the
  commit/PR. Keep this file current so the next person knows what is safe to
  touch.

## Suggested order

1. Role B consolidation (unblocks the duplicate-node complaint everywhere).
2. Role A support/clip metadata (improves Role B and C accuracy).
3. Role C queue + evaluation (protects the demo from VLM latency).
4. Role D polish/export (makes the work legible to judges and teammates).

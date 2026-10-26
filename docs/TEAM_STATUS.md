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

- [ ] Emit support/furniture candidates from Role A (`FrameResult.supports` or
  clip metadata) so furniture anchors are not label-only.
- [ ] Guarantee stable `entity_id` across occlusion/track loss; expose the
  re-ID similarity used for the decision.
- [ ] Add per-object embedding/confidence to `ObjectObservation` where the
  detector can provide it.
- [ ] Put the container name and contact evidence into `InteractionClip.metadata`
  so Role C does not have to infer it from the caption.
- [ ] Benchmark YOLO-World and MediaPipe on the real HoloLens clip; record
  precision/recall and latency in `docs/PERCEPTION.md`.
- [ ] Keep the palette/colour fallback green: `pytest tests/test_pipeline.py`.

**Done when:** the real clip produces stable tracks for hands + manipulable
objects, and `chronicle run --source tabletop` has no duplicate track IDs in
the entity table.

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
  and “I don’t know” cases. Track citation precision. `evaluation.py`
  (7 cases, currently 100 % pass / 100 % citation precision).
- [x] Keep the offline template path as the reference answer; any LLM answer
  without a verifiable citation must fall back.
- [x] Emit queue depth / verification latency on the telemetry bus so the
  dashboard can show it. `QUEUE_DEPTH`, `QUEUE_BACKPRESSURE`,
  `VERIFICATION_LATENCY`, plus `runtime.verification_stats`.

**Done when:** the HoloLens clip’s verified actions match the labelled timeline
within tolerance, and a hallucinated timestamp is always rejected.

**Role C note (this push):** all six items landed with model-free tests
(`tests/test_verification_queue.py`, `tests/test_calibration.py`,
`tests/test_evaluation.py`). Pushed as `9ae1e5e`; verification before push was
`pytest` 192 passed, `unittest discover -s tests` 100 passed, `ruff check
chronicle tests` clean, and `chronicle.semantic.demo` /
`chronicle.semantic.evaluation` both exit 0 (7/7 cases, 100 % citation
precision). The only cross-role edit is the internal `_clip_queue` /
`_drain_clips` seam in `chronicle/integration.py` (bounded queue + latency
marks); `ChronicleRuntime`'s public API and snapshot schema are unchanged.
Ambient caption phrasing still needs a labelled HoloLens pass before the “done
when” line can be ticked.

### Role D — dashboard and product surface

**Goal:** make the integrated dashboard the team’s daily driver.

- [ ] Visual regression pass on the ported UI (graph, timeline, tables, chat,
  pipeline log) at 1280/1440/1920 px.
- [ ] Add a live/streaming source placeholder next to the pre-recorded sources;
  keep the `ChronicleRuntime` seam, do not import the prototype engine.
- [ ] Export/import: download the current snapshot as JSON/CSV, reload it
  without re-ingesting.
- [ ] Add an explicit error/empty state for missing videos, stopped runs and
  failed GraphRAG queries.
- [ ] Add a headless smoke test for `/` and `/source-video` (Playwright or
  `curl`), and keep the page usable without a GPU.
- [ ] Show furniture anchors distinctly from episodic entities (they are
  background structure, not objects with a lifecycle).

**Done when:** a teammate can switch sources, watch ingestion, ask a question
and export the graph without touching Python.

## Shared rules

- **Do not block on Role D.** The dashboard now consumes `ChronicleRuntime`; if
  you add a field to `snapshot()`, update the UI in the same PR.
- **Additive APIs only.** Keep `PerceptionPipeline`, `ChronicleMemory` and
  `SemanticEngine` contracts intact; put cross-role glue in
  `chronicle/integration.py`.
- **Model-free tests are mandatory.** CI has no GPU, no model weights and no
  Ollama. Use the colour/deterministic/template fallbacks.
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

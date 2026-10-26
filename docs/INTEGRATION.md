# Integration runtime (`chronicle.integration`)

`ChronicleRuntime` is the single façade the dashboard, CLI and demos use. It
wires the three role-owned layers together without changing their public APIs:

```text
video ──► PerceptionPipeline (Role A)
            │ FrameResult
            ▼
       ChronicleMemory (Role B) ◄── InteractionClip ──┐
            │ transitions                              │
            ▼                                          │
       SemanticEngine (Role C) ── verified actions ────┘
            │ GraphContext / QueryAnswer
            ▼
       ChronicleRuntime.snapshot() / .ask() ──► NiceGUI (Role D)
```

## Quick start

```python
from chronicle.integration import ChronicleRuntime

runtime = ChronicleRuntime("data/demo/chronicle.db", zone_label="Synthetic Scene")
summary = runtime.ingest_video("data/demo/chronicle_demo.mp4")
print(summary.as_dict())

answer = runtime.ask("where are my keys?")
print(answer.answer, answer.citations)

snapshot = runtime.snapshot()  # JSON-safe graph + tables + stats
runtime.close()
```

The CLI wraps the same object:

```bash
chronicle sources
chronicle run data/demo/chronicle_demo.mp4 --max-frames 120
chronicle ask "where are my keys?" --db data/demo/chronicle.db
chronicle serve --source synthetic
./chronicle.sh preflight        # demo readiness: PASS/WARN/FAIL table
```

`docs/DEMO_PREFLIGHT.md` explains each preflight check and the stage-day
checklist; `--json` gives the same result to CI.

## Two dashboards, one canonical entry point

The repository ships two deliberately separate surfaces. They share the event
contract but not a process, port, or purpose:

| Entry point | Default port | Purpose | Use for the demo? |
| --- | ---: | --- | --- |
| `./chronicle.sh start [source]` / `chronicle serve` | 8080 | **Operator dashboard.** Live A→B→C ingestion, source dropdown, graph/timeline/tables, GraphRAG chat, export/import. | **Yes — canonical.** |
| `./chronicle.sh mission` / `python run_dashboard.py` | 8081 | **Zine mission control.** Standalone Role D demo/replay surface (`dashboard/`) with demo replay, JSONL replay, SQLite producer or live `POST /api/ingest`. | Optional second screen; never required for the operator walkthrough. |

`chronicle serve` is the single entry point for the spec’s ten-minute script. It
is the only dashboard backed directly by `ChronicleRuntime` and the only one
whose `./chronicle.sh preflight` checks cover. `run_dashboard.py` is a
standalone mission-control view for the bundled demo and for producers that
speak the event contract; use it when you specifically want the zine layout or
live-ingest endpoint, not as a replacement for the operator dashboard.

## Responsibilities

| Concern | Owner | Runtime behaviour |
| --- | --- | --- |
| Frames, detections, hands, clips | Role A | `PerceptionPipeline.iter_results()` |
| Lifecycle, anchors, persistence | Role B | `ChronicleMemory.observe_result()` / `query()` |
| Action verification, GraphRAG | Role C | `SemanticEngine.verify_interaction()` / `answer_query()` |
| Rendering, interaction | Role D | reads `snapshot()` and subscribes to `runtime.bus` |

The runtime does **not** re-implement any role logic. If a behaviour belongs to
Role A/B/C, fix it in that package and keep the façade thin.

## Ingestion lifecycle

1. `PerceptionPipeline` emits a `FrameResult`.
2. `runtime._observe_result()` calls `memory.observe_result()`.
3. The pipeline's `on_clip` callback queues an `InteractionClip`.
4. The queue is drained **after** the frame is observed, so the entity exists
   before `record_action()` resolves `clip.track_id`.
5. `SemanticEngine.verify_interaction()` returns a strict `ActionVerification`.
6. Accepted actions become `USER -> entity` action edges in Role B.
7. `snapshot()` is safe to call from the UI thread at any time.

`on_clip` fires before the matching `FrameResult` is yielded. Do not remove the
queue/drain ordering in `_verify_clip()` unless the pipeline contract changes.

## Snapshot schema

`runtime.snapshot()` returns a plain dictionary:

```python
{
  "nodes": [{"id", "label", "group", "zone_id", "title"}],  # Role B graph
  "edges": [{"from", "to", "label", "arrows"}],
  "entities": [{"id", "label", "state", "zone", "anchor", "colour",
                "colour_hex", "attributes", "appearance", ...}],
  "actions": [{"id", "timestamp", "action", "entity", ...}],
  "transitions": [{"id", "entity_id", "from_state", "to_state", ...}],
  "furniture": [{"anchor_id", "label", "zone_id", "bbox", ...}],
  "stats": {...},
  "summary": {...} | None,
  "running": bool,
  "error": str | None,
  "video": str | None,
  "backends": {"detector", "hands", "embedder", "attributes", "llm"},
}
```

`entities[].attributes` holds the Role A appearance facts (dominant colour,
swatch, shape, size, texture, brightness, position). `nodes[].attributes` and
`nodes[].colour_hex` mirror the same facts for the graph, and the entity CSV
export appends `colour`, `colour_hex` and a JSON `attributes` column.

The dashboard only consumes this schema. Adding a field is backward-compatible;
renaming one requires updating `chronicle/ui/app.py` in the same change.

`nodes[].zone_id` is the namespaced `zone:<id>` anchor for entities, furniture
and zone nodes (a zone points at itself). The dashboard uses it to cluster the
force graph by room; the field is additive and safe for older consumers.

## Graph payload helpers

`chronicle/ui/graph_view.py` owns the ECharts option builders (`graph_options`,
`timeline_options`). It imports no NiceGUI or FastAPI code, so the visual payload
is covered by model-free tests without the optional `ui` extra
(`tests/test_graph_view.py`).

## Snapshot import

The dashboard's **Import** button accepts an exported
`chronicle_snapshot.json`, stops the current ingestion, and calls
`runtime.load_snapshot_json()` with the parsed payload. Validation lives in
`chronicle/ui/snapshot_io.py` (`parse_snapshot`): the file must be UTF-8 (BOM
allowed), a JSON object, and contain `nodes` and `entities` lists. Invalid files
surface a notification and leave the current graph untouched.

## Query normalization

Role B returns `QueryResult.edges` with `src`/`dst`/`t_start`. Role C expects
`GraphContext` facts with `subject`/`relation`/`object_label`/`timestamp`.
`runtime.graph_context()` performs that translation and de-duplicates facts.
Keep this adapter here rather than teaching either role about the other's edge
shape.

Appearance facts are folded into `GraphContext.attributes`, so GraphRAG can
answer "what colour is the cup?", "how big is it?" and "describe the backpack"
without an LLM. The template path cites the entity's `last_seen` timestamp;
`GraphContext.from_mapping` also accepts attributes nested inside the entity
object for older exports. Broad colour filters ("what red objects did you see?")
are answered by `ChronicleRuntime._colour_inventory_answer()` before entity
ranking, so they list every matching object instead of arbitrarily picking one.

## Static furniture

Role A's open-vocabulary detector emits sofas, tables and TVs as ordinary
tracks. `ChronicleMemory._route_furniture()` moves labels in
`STATIC_FURNITURE_LABELS` into `FurnitureRegistry`:

- label + zone prior;
- motion-compensated IoU / centre distance;
- appearance embedding when available;
- a confirmation window (`FurnitureConfig.min_hits`);
- re-entry after a pan or track loss reuses the same `anchor:<label>-<n>`.

The anchor is also offered to the lifecycle engine as a support surface, so
`RESTING_ON` works without a separate surface detector. Two co-visible sofas
stay two anchors. See `chronicle/graph/furniture.py` and
`tests/test_furniture.py`.

Known gap: **legacy** duplicate nodes created before this routing (or by a
detector that changed labels) are not merged yet. Role B owns the graph-level
consolidation sweep; see `docs/TEAM_STATUS.md`.

## Answer evidence frames

A grounded answer can carry a supporting still image. `ChronicleRuntime._answer_frame()`
now resolves it in this order:

1. **Stored evidence** beside the episode database:
   ```
   data/real/hololens/chronicle.db
   data/real/hololens/evidence/index.jsonl
   data/real/hololens/evidence/ev_0001_interaction_strip.jpg
   ```
   Each JSONL row carries `file`, `t`, `kind`, and optional `label`,
   `entity_id`, `zone_id`, `action`, `caption`, `clip_sim` and `bbox`.
   Stored images are ranked by kind, entity/label match, citation time and
   playhead (`chronicle/frames.py: pick_stored_evidence`).
2. **Video-frame extraction** at the newest citation when no stored image fits.
   This keeps working for uploaded clips with no evidence folder.

Stored images are returned as self-contained `data:image/...;base64,...` URLs,
so the CLI and UI need no extra media route. The bundled fixtures cover all five
sources (`5/5` in `./chronicle.sh preflight`) and are regenerated with:

```bash
.venv/bin/python scripts/build_evidence_fixtures.py
```

The synthetic clip reuses the interaction strips/crops in `data/`, HoloAssist
reuses the real interaction strips, and the Aria clips get three zone keyframes
each. The builder is idempotent and only replaces `ev_*.jpg` files; other files
in an evidence folder are left alone.

## Adding a source

Add a `ClipSource` to `chronicle/sources.py` with a unique key, video path,
SQLite path, zone label, vocabulary and suggested questions. Both the CLI and
the dashboard pick it up automatically. Missing videos are shown as unavailable
instead of failing at startup.

## Tests

```bash
python -m pytest
python -m unittest discover -s tests -v
ruff check chronicle tests
```

The runtime/furniture tests are model-free. The dashboard smoke test is:

```bash
chronicle serve --source synthetic --port 8765 &
curl -sf http://127.0.0.1:8765/ >/dev/null
curl -sf http://127.0.0.1:8765/source-video >/dev/null
```

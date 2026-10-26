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
```

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
  "entities": [{"id", "label", "state", "zone", "anchor", ...}],
  "actions": [{"id", "timestamp", "action", "entity", ...}],
  "transitions": [{"id", "entity_id", "from_state", "to_state", ...}],
  "furniture": [{"anchor_id", "label", "zone_id", "bbox", ...}],
  "stats": {...},
  "summary": {...} | None,
  "running": bool,
  "error": str | None,
  "video": str | None,
  "backends": {"detector", "hands", "embedder", "llm"},
}
```

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

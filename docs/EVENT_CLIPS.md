# Event-based clipping and per-event graphs

This is the bounded recorder described in `docs/VFM_ARCHITECTURE.tex`
(§*Event-driven evidence and verification*):

```text
transition / low confidence / label conflict
        -> salience score
        -> 3 s pre + 3 s post clip
        -> per-event replay + graph snapshot
```

It does **not** run a VLM. The graph state machine has already decided that a
moment matters; the clip, the event window and the graph snapshot are the
auditable evidence for that decision.

## Use it

In the zine dashboard, pick a clip and press **✂ events** in the stage toolbar.
The API is also callable directly:

```bash
# build (or rebuild) event clips for the active timeline
curl -s -X POST http://127.0.0.1:8082/api/event-clips \
  -H 'content-type: application/json' \
  -d '{"clip":"events:replays/cslab.events.jsonl","pre":3,"post":3,"limit":12}'

# list an existing index
curl -s 'http://127.0.0.1:8082/api/event-clips?clip=events:replays/cslab.events.jsonl'
```

Every generated clip is added to the clip switcher as an `events:` option, so
selecting it replays the event window and the live graph/boxes/timeline follow
that window.

## Salience rules

The scoring is deliberately readable so the demo can explain a cut:

| Trigger | Score | Reason |
| --- | ---: | --- |
| `PICK_UP`, `INSERT_INTO`, `REMOVE_FROM` interaction | 0.88–0.90 | physical action |
| `HELD` / `INTERACTED` / `CONTAINED` state | 0.80–0.92 | state change |
| `PLACED_ON` / `INSIDE` / `HELD_BY` edge | 0.66–0.70 | relation change |
| low-confidence detection or rejected action | +0.08–0.16 | needs review |
| label conflict / merge | 0.82–0.84 | ambiguity |
| `VISIBLE_STATIC` / `OCCLUDED` | 0.12–0.16 | ignored unless part of a larger moment |

Events within `cooldown` (default **2.5 s**) of each other coalesce into one
clip; the highest-scoring event becomes the anchor. The default window is
**3 s pre + 3 s post** and the default cap is **12 clips** per source.

## Output layout

```text
<data-dir>/event_clips/<source>/
    index.json                 all cuts, scores, reasons and artifact paths
    01_held-keys_t00010900/
        clip.mp4               ffmpeg cut (OpenCV fallback)
        events.jsonl           only this window, rebased to t=0
        graph.json             Vis.js graph snapshot at the anchor time
        event.json             one entry, also present in index.json
```

`clip.mp4` is produced with `ffmpeg -ss <start> -i <source> -t <window>`
(re-encoded, so the first frame is accurate). If ffmpeg is unavailable the
module falls back to OpenCV. A source video that is shorter than its replay
timeline is respected: events past the last frame are not cut.

## What "graph creation" means here

`graph.json` is a real graph snapshot, not a summary string. The builder replays
every event up to the anchor into `dashboard.store.MemoryStore` and stores the
canonical `graph_snapshot()` payload: entity nodes with state, zones, user and
anchor nodes, relation edges, timestamps and tooltips. The same graph is rebuilt
live when the clip is selected because `events.jsonl` is a valid replay
timeline.

The manifest also records `graph_nodes` / `graph_edges` so a reviewer can see
which events produced a legible graph. Clips that only contain a single node
still play; they are simply not strong graph evidence.

## Verification

```bash
pytest -q tests/test_event_clips.py
```

The tests cover salience ordering, coalescing, shifted timelines, graph
snapshots, path-traversal rejection and both API routes.

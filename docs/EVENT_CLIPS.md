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
# build (or rebuild) event clips for the active timeline; verify with a VLM
curl -s -X POST http://127.0.0.1:8082/api/event-clips \
  -H 'content-type: application/json' \
  -d '{"clip":"events:replays/cslab.events.jsonl","pre":3,"post":3,"limit":12,"verify":true,"verify_limit":4}'

# list an existing index
curl -s 'http://127.0.0.1:8082/api/event-clips?clip=events:replays/cslab.events.jsonl'
```

Generated clips live under `<data-dir>/event_clips/` and are **not** added to
the main clip picker. The picker stays limited to real sources/replays; event
artifacts are available through the API and the folder above. The dashboard
button verifies the first four clips by default so the demo gets semantic
captions without waiting for a long queue.

## VLM verification (3 frames per event)

Clipping answers *where* something happened; verification answers *what*
happened. For each event clip the verifier extracts three frames —
**1 s pre / contact / 1 s post** — and sends them to a vision model:

1. **Qwen2.5-VL via Ollama** (`qwen2.5vl:3b` by default). One request carries all
   three images plus the graph candidate (`HELD · Keys`, object, zone).
2. **Florence-2** if Ollama is not reachable and the optional transformers stack
   is installed; the three frame captions are combined.
3. **Rule fallback** if no VLM is available. The clip still gets frames,
   a snapshot, an interaction event and a vault; the record is marked
   `status: "unavailable"` and `accepted: false`.

The model must return strict JSON: `event`, `caption`, `objects`, `actions`,
`zone`, `confidence`, `agrees`, `correction`. The verifier normalises the action
through the same calibrated rule classifier used by the graph pipeline.

**Reconciliation policy.** A VLM answer never blindly overwrites a strong graph
observation. If the graph saw `HELD`, `INTERACTED`, `CONTAINED`, `PICK_UP`,
`INSERT_INTO`, `REMOVE_FROM`, `LOST`, etc. and the VLM disagrees, the graph
action is kept, the VLM caption is retained, and the record gets
`disagreement: true` / `accepted: false` for review. Visibility-only states
(`VISIBLE_STATIC`, `OCCLUDED`, `OUT_OF_VIEW`, …) can be corrected by the VLM.
This is the accuracy win: uncertain moments get a semantic label, strong
moments keep their verified physical state.

## What is written back

| Artifact | Meaning |
| --- | --- |
| `frames/pre.jpg`, `contact.jpg`, `post.jpg` | the three frames sent to the VLM |
| `frames/snapshot.jpg` | one audit snapshot for the short event |
| `events.jsonl` | gains a `kind: "interaction"` event with `source: "vlm-event"`, action, caption, confidence, frames, snapshot, agreement |
| `ledger.json` | one-window dense caption ledger built from the VLM result |
| `vault.db` | queryable per-event VFM vault compiled from that ledger (events + captions) |
| `event.json` / `index.json` | include `vlm` status, provider, action, caption and artifact paths |

Artifacts are also served safely (must live under `event_clips/`):

```bash
curl -OJ 'http://127.0.0.1:8082/api/event-clips/artifact?path=event_clips/cslab/01_interacted-track-7_t00002727/frames/snapshot.jpg'
```

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
        frames/pre.jpg         pre-contact frame sent to the VLM
        frames/contact.jpg     contact frame sent to the VLM
        frames/post.jpg        post-contact frame sent to the VLM
        frames/snapshot.jpg    one audit snapshot for the short event
        ledger.json            one-window VLM caption ledger
        vault.db               queryable per-event VFM vault
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
pytest -q tests/test_event_clips.py tests/test_event_vlm.py
```

The tests cover salience ordering, coalescing, shifted timelines, graph
snapshots, path-traversal rejection, both API routes, the 3-frame VLM write-back,
the no-VLM rule fallback, the strong-graph-vs-VLM disagreement policy, snapshot
files and per-event vault compilation.

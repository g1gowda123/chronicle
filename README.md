# Chronicle

An egocentric, episodic memory engine that parses pre-recorded first-person video into a structured, queryable 4D spatio-temporal knowledge graph.

> **Status:** Role A (perception & kinematics), Role B (state machine & dual-graph
> memory), Role C (semantic verification) and the integrated Role D dashboard are
> implemented and tested. `ChronicleRuntime` wires A → B → C → UI; see
> [`docs/INTEGRATION.md`](docs/INTEGRATION.md) and the live role assignments in
> [`docs/TEAM_STATUS.md`](docs/TEAM_STATUS.md). Integration contracts live in
> [`docs/PERCEPTION.md`](docs/PERCEPTION.md),
> [`docs/ROLE_B.md`](docs/ROLE_B.md) and [`docs/ROLE_C.md`](docs/ROLE_C.md).

## Overview

Chronicle avoids running heavyweight vision-language models on every frame. Instead, deterministic motion and geometric gating decide *when* foundation models are invoked:

1. **Frame ingestion & affine parallax gating** — background-masked rigid motion estimation separates camera pans from head rotation and static pauses.
2. **Open-vocabulary detection & hand landmarks** — YOLO-World tracks user-specified targets while MediaPipe Hands provides 21 egocentric landmarks.
3. **Tracking with visual re-ID** — ByteTrack-style IDs are reconciled against OpenCLIP embeddings to avoid duplicate entities after occlusions.
4. **Containment & interaction evaluation** — Shapely computes Intersection over Smaller Area (IoS) and hand/object containment.
5. **Temporal state machine** — entities move through a 9-state lifecycle driven by wall-clock time, not frame counts.
6. **Event-gated VLM verification** — state transitions trigger composite temporal-strip queries to Florence-2.
7. **Dual-graph ingestion** — events persist to NetworkX and SQLite with monotonic timestamps.
8. **Live telemetry** — a NiceGUI dashboard streams graph mutations, metrics, and logs over WebSockets.

## Architecture

| Layer | Technologies |
| --- | --- |
| Perception & tracking | OpenCV, Ultralytics YOLO-World, MediaPipe Hands, ByteTrack-style association, OpenCLIP |
| Spatial reasoning & state engine | Shapely, NetworkX, SQLite |
| Semantic engine | Florence-2-base, OpenCLIP (ViT-B-32) |
| UI & telemetry | NiceGUI, ECharts, `ui.log` |

### Role A dataflow

```text
video ──► background-masked affine motion ──► STATIONARY / MOVING / PANNING
      ──► YOLO-World + MediaPipe Hands ──► ByteTrack-style tracker + Re-ID
      ──► IoS / fingertip / dwell gate ──► interaction events
      ──► 30-frame ring buffer ──► structured clips + 1×3 composites
      ──► telemetry categories: PERCEPTION, GEOM, TRACK, REID, CLIP, SYSTEM
```

The perception package emits plain dataclasses (`FrameResult`,
`TrackedObject`, `InteractionEvent`, `InteractionClip`, `TelemetryEvent`) so
the state/graph, semantic and UI layers can consume it without importing any
vision code.

### Role B dataflow

```python
from chronicle import ChronicleMemory

with ChronicleMemory("data/chronicle.db") as memory:
    transitions = memory.observe(
        frame,
        on_transition=lambda transition, frame: vlm_queue.put(transition),
    )
    answer = memory.query("where are my cups?").fallback_answer()
```

Role B adds:

- Nine-state lifecycle with wall-clock dwell gates, camera-relative velocity,
  occlusion vs. out-of-view reasoning, re-acquisition, and archival `LOST`.
- Dual NetworkX graphs: persistent background anchors and time-intervaled
  foreground relations.
- SQLite projection with terminal-anchor retention and a 10-minute trajectory
  purge.
- 2-hop GraphRAG retrieval with deterministic fallback answers, an LLM-ready
  prompt, label aliases and ambiguity-aware match confidence.
- Conservative duplicate consolidation for legacy entity/furniture rows with
  dry runs, retained history and `[GRAPH]` audit telemetry.
- Thread-safe `EventBus` telemetry for the Role D dashboard.

### Dual-graph memory

- **Background graph** — invariant topological anchors such as zones (Kitchen, Work Desk) and fixed furniture, relocalized via CLIP place recognition.
- **Foreground graph** — episodic entities (keys, phone, mug) linked by `RESTING_ON`, `INSIDE`, state-transition edges, and action multi-edges.

The terminal relational anchor of an object is retained permanently, so queries about long-lost items still resolve correctly.

### Object lifecycle

`NEW → VISIBLE_STATIC → INTERACTED → VISIBLE_MOVING → HELD → CONTAINED → OCCLUDED → OUT_OF_VIEW → LOST`

Each transition is gated by IoS thresholds, support-surface heuristics, and wall-clock durations. See the specification for the full transition table.

### Query retrieval

Natural-language queries use a two-stage GraphRAG pipeline: retrieve the entity's latest state and spatial edge, serialize the 2-hop subgraph, then have a compact LLM produce a timestamped, location-cited answer. `QueryResult` reports the raw `match_score`, an ambiguity-aware `confidence`, the winning `match_reason`, and ranked `candidates` so duplicate labels resolve predictably.

## Repository layout

```text
.
├── Chronicle Technical Specification.pdf   # Full architecture and algorithms
├── chronicle/
│   ├── types.py                            # Shared Role A/B/C data contract
│   ├── config.py                           # lifecycle thresholds + furniture config
│   ├── geometry.py                         # IoS, support contact, compensated velocity
│   ├── state_machine.py                    # nine-state lifecycle engine
│   ├── telemetry.py                        # Role B event bus
│   ├── text.py                             # normalization, plurals, label aliases
│   ├── integration.py                      # A -> B -> C runtime facade
│   ├── sources.py                          # demo/real source registry
│   ├── cli.py                              # `chronicle sources|run|ask|serve`
│   ├── graph/
│   │   ├── adapter.py                      # FrameResult -> FrameContext bridge
│   │   ├── background.py                   # zones + furniture anchors
│   │   ├── consolidation.py                # duplicate merge sweep + audit reports
│   │   ├── foreground.py                   # entities + time-intervaled edges
│   │   ├── furniture.py                    # persistent static-furniture association
│   │   ├── query.py                        # 2-hop GraphRAG retrieval
│   │   └── memory.py                       # ChronicleMemory facade
│   ├── storage/                            # SQLite schema + persistent store
│   ├── perception/                         # Role A
│   │   ├── motion.py                       # background-masked affine RANSAC
│   │   ├── detector.py                     # YOLO-World + palette fallback
│   │   ├── hands.py                        # MediaPipe Tasks + skin fallback
│   │   ├── tracker.py                      # two-stage tracking + Re-ID
│   │   ├── interactions.py                 # IoS / dwell interaction gate
│   │   ├── clip_buffer.py                  # 30-frame ring + clip assembly
│   │   ├── pipeline.py                     # orchestration / JSONL
│   │   ├── stream.py                       # threaded consumer stream
│   │   ├── telemetry.py                    # perception telemetry bus
│   │   └── cli.py                          # chronicle-perception
│   ├── semantic/                           # Role C
│   │   ├── vpr.py                          # OpenCLIP place recognition
│   │   ├── temporal_strip.py               # 30-frame -> 1x3 composite
│   │   ├── actions.py                      # strict action JSON + containers
│   │   ├── verification.py                 # Florence-2 + classifier
│   │   ├── calibration.py                  # per-action thresholds + metrics
│   │   ├── queue.py                        # bounded verification queue
│   │   ├── graphrag.py                     # cited answers from 2-hop subgraph
│   │   ├── evaluation.py                   # GraphRAG eval set + citation score
│   │   └── engine.py                       # facade + telemetry events
│   └── ui/                                 # Role D NiceGUI dashboard
│       └── app.py                          # graph, timeline, tables, chat, log
├── docs/
│   ├── PERCEPTION.md                       # Role A contract
│   ├── ROLE_B.md                           # Role B integration contract
│   ├── ROLE_C.md                           # Role C contract
│   ├── INTEGRATION.md                      # runtime/UI contract
│   └── TEAM_STATUS.md                      # current work + role assignments
├── scripts/
│   ├── fetch_assets.py                     # download model weights
│   ├── make_demo.py                        # render the offline demo clip
│   └── demo_role_b.py                      # synthetic lifecycle + retrieval demo
├── tests/                                  # model-free unit/integration suite
├── requirements-semantic.txt               # Role C demo/test dependencies
└── requirements-semantic-models.txt        # OpenCLIP + Florence-2 stack
```

Model weights, videos and run outputs are git-ignored.

## Getting started

Requires Python 3.11–3.13 (3.12 recommended).

```bash
uv sync                      # core runtime
uv sync --extra perception --torch-backend cpu   # YOLO-World, MediaPipe, OpenCLIP
python scripts/fetch_assets.py
```

Without `uv`, `pip install -r requirements.txt` installs the core runtime.

### Role B demo (no models)

```bash
python -m pytest
python scripts/demo_role_b.py
```

The demo generates a synthetic pick-up-and-place scene, prints `[STATE]` /
`[GRAPH]` telemetry, shows the dual graph and 2-hop retrieval packet, then
reopens SQLite to answer the query from persisted memory.

### Perception demo (no downloads)

```bash
python scripts/make_demo.py
python -m chronicle.perception data/demo/chronicle_demo.mp4 \
  --detector color --hands color --embedder color \
  --jsonl runs/demo.jsonl --clips-dir runs/demo_clips
```

You get per-frame JSONL, console telemetry, and hand/keys interaction clips
with `PRE | CONTACT | POST` composites.

### Offline semantic demo (Role C)

```bash
pip install -r requirements-semantic.txt   # NumPy + Pillow; demo works offline
python -m chronicle.semantic.demo
python -m chronicle.semantic.evaluation    # GraphRAG eval: 7 cases, citation precision
```

The demo exercises VPR, temporal strips, action verification, the bounded
verification queue (dedup + backpressure + latency), confidence calibration and
the GraphRAG evaluation set, then prints the telemetry the dashboard consumes.
The suite is model-free: `python -m unittest discover -s tests` runs 100 tests
and `python -m pytest` collects 192 across the repo.

For the real OpenCLIP + Florence-2 stack and optional Ollama GraphRAG:

```bash
pip install -r requirements-semantic-models.txt
ollama pull qwen2.5:3b-instruct
```

Role C consumes Role A's `InteractionClip` directly, including its
`contact_index` and BGR `ClipFrame` images. See
[docs/ROLE_C.md](docs/ROLE_C.md) for the Role A/B/D integration contract,
telemetry payloads, action JSON schema, and configuration.

### Dashboard (Role D)

```bash
uv sync --extra ui
chronicle sources                 # list bundled/demo clips
chronicle serve --source synthetic --auto-run
# open http://127.0.0.1:8080
```

The dashboard streams the live dual graph, state timeline, entity/action
tables, pipeline log and grounded GraphRAG chat from the same
`ChronicleRuntime` used by the CLI. It degrades to the colour detector,
deterministic embeddings and template GraphRAG when models are unavailable.
`chronicle serve --source tabletop` uses the real HoloLens clip.

Headless smoke test:

```bash
chronicle serve --source synthetic --port 8765 &
curl -sf http://127.0.0.1:8765/ >/dev/null
curl -sf http://127.0.0.1:8765/source-video >/dev/null
```

### Real footage (production backends)

```bash
python -m chronicle.perception data/real/holoassist_switch_34s.mp4 \
  --labels "joy-con,switch,grip,table" \
  --detector yolo-world --hands mediapipe --embedder openclip \
  --jsonl runs/real.jsonl --clips-dir runs/real_clips \
  --save-video runs/real_overlay.mp4
```

Run `python -m chronicle.perception --help` for flags. The installed
`chronicle-perception` entry point is equivalent.

### Tests

```bash
python -m pytest
python -m unittest discover -s tests -v   # Role C suite without pytest
```

## Role boundaries

- **Role A (implemented):** video ingestion, motion gating, detection, hands,
  tracking/Re-ID, interaction qualification, ring buffer and clips, telemetry.
- **Role B (implemented):** 9-state lifecycle, containment/support reasoning,
  dual graph, persistence and query.
- **Role C (implemented):** OpenCLIP place recognition, 1×3 temporal strips,
  Florence-2 action verification with strict JSON, and cited GraphRAG answers.
- **Role D (implemented):** NiceGUI dashboard, graph visualization, live
  telemetry, source switching and grounded chat over `ChronicleRuntime`.

Role A intentionally does not mutate lifecycle state, call a VLM, persist a
graph, or render a dashboard. Role B consumes `FrameContext` (or a Role A
`FrameResult`) and owns the episodic memory. Role C does not mutate the graph;
it emits validated events and answer payloads for Roles B and D.

## References

The design builds on R4DSG (ACM MM 2026), VL-KnG (2026), ESOM (WACV 2026), hand-object interaction research (WACV 2024), and LightMem-Ego.

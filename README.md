# Chronicle

An egocentric, episodic memory engine that parses pre-recorded first-person video into a structured, queryable 4D spatio-temporal knowledge graph.

> **Status:** Role B (state machine + dual-graph engineering) is implemented and
> tested. Roles A (perception), C (semantic engine), and D (dashboard) are being
> built by the rest of the team. See [`docs/ROLE_B.md`](docs/ROLE_B.md) for the
> integration contract.

## Overview

Chronicle avoids running heavyweight vision-language models on every frame. Instead, deterministic motion and geometric gating decide *when* foundation models are invoked:

1. **Frame ingestion & affine parallax gating** — background-masked rigid motion estimation separates camera pans from head rotation and static pauses.
2. **Open-vocabulary detection & hand landmarks** — YOLO-World tracks user-specified targets while MediaPipe Hands provides 21 egocentric landmarks.
3. **Tracking with visual re-ID** — ByteTrack IDs are reconciled against CLIP embeddings to avoid duplicate entities after occlusions.
4. **Containment & interaction evaluation** — Shapely computes Intersection over Smaller Area (IoS) and hand/object containment.
5. **Temporal state machine** — entities move through a 9-state lifecycle driven by wall-clock time, not frame counts.
6. **Event-gated VLM verification** — state transitions trigger composite temporal-strip queries to Florence-2.
7. **Dual-graph ingestion** — events persist to NetworkX and SQLite with monotonic timestamps.
8. **Live telemetry** — a NiceGUI dashboard streams graph mutations, metrics, and logs over WebSockets.

## Architecture

| Layer | Technologies |
| --- | --- |
| Perception & tracking | OpenCV, Ultralytics YOLO-World, MediaPipe Hands, ByteTrack |
| Spatial reasoning & state engine | Shapely, NetworkX, SQLite |
| Semantic engine | Florence-2-base, OpenCLIP (ViT-B-32) |
| UI & telemetry | NiceGUI, Vis.js, `ui.log` |

### Dual-graph memory

- **Background graph** — invariant topological anchors such as zones (Kitchen, Work Desk) and fixed furniture, relocalized via CLIP place recognition.
- **Foreground graph** — episodic entities (keys, phone, mug) linked by `RESTING_ON`, `INSIDE`, state-transition edges, and action multi-edges.

The terminal relational anchor of an object is retained permanently, so queries about long-lost items still resolve correctly.

### Object lifecycle

`NEW → VISIBLE_STATIC → INTERACTED → VISIBLE_MOVING → HELD → CONTAINED → OCCLUDED → OUT_OF_VIEW → LOST`

Each transition is gated by IoS thresholds, support-surface heuristics, and wall-clock durations. See the specification for the full transition table.

### Query retrieval

Natural-language queries use a two-stage GraphRAG pipeline: retrieve the entity's latest state and spatial edge, serialize the 2-hop subgraph, then have a compact LLM produce a timestamped, location-cited answer.

## Role B implementation

The `chronicle` package exposes a single facade, `ChronicleMemory`, that Role A
feeds and Roles C/D consume:

```python
from chronicle import ChronicleMemory

with ChronicleMemory("data/chronicle.db") as memory:
    transitions = memory.observe(
        frame,
        on_transition=lambda transition, frame: vlm_queue.put(transition),
    )
    answer = memory.query("where are my cups?").fallback_answer()
```

Implemented pieces:

- Nine-state lifecycle with wall-clock dwell gates, camera-relative velocity,
  occlusion vs. out-of-view reasoning, re-acquisition, and archival `LOST`.
- Dual NetworkX graphs: persistent background anchors and time-intervaled
  foreground relations.
- SQLite projection with terminal-anchor retention and 10-minute trajectory
  purge.
- 2-hop GraphRAG retrieval with deterministic fallback answers and an
  LLM-ready prompt.
- Thread-safe `EventBus` telemetry for the Role D dashboard.
- 51 deterministic tests covering frame-rate independence, camera pans, all
  nine states, persistence, retrieval, and multi-entity scenes.

## Repository layout

```text
.
├── Chronicle Technical Specification.pdf   # Full architecture and algorithms
├── README.md
├── requirements.txt
├── pyproject.toml
├── chronicle/
│   ├── config.py            # thresholds and container labels
│   ├── types.py             # FrameContext and transition contracts
│   ├── geometry.py          # IoS, support contact, compensated velocity
│   ├── state_machine.py     # nine-state lifecycle engine
│   ├── telemetry.py         # EventBus tags and replay buffer
│   ├── graph/
│   │   ├── background.py    # zones + furniture anchors
│   │   ├── foreground.py    # entities + time-intervaled edges
│   │   ├── query.py         # 2-hop GraphRAG retrieval
│   │   └── memory.py        # ChronicleMemory facade
│   └── storage/             # SQLite schema and store
├── tests/                   # synthetic, model-free test suite
├── scripts/demo_role_b.py   # end-to-end synthetic demo
└── docs/ROLE_B.md           # integration contract
```

## Getting started

```powershell
pip install -r requirements.txt
python -m pytest -q
python scripts/demo_role_b.py
```

The demo needs no video, GPU, or model downloads. It generates a synthetic
pick-up-and-place scene, prints `[STATE]` / `[GRAPH]` telemetry, shows the dual
graph and 2-hop retrieval packet, then reopens SQLite to answer the query from
persisted memory.

## References

The design builds on R4DSG (ACM MM 2026), VL-KnG (2026), ESOM (WACV 2026), hand-object interaction research (WACV 2024), and LightMem-Ego.

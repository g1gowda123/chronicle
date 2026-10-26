# Chronicle

An egocentric, episodic memory engine that parses pre-recorded first-person video into a structured, queryable 4D spatio-temporal knowledge graph.

> **Status:** Role A (perception & kinematics) and Role C (semantic engine) are
> implemented and tested. The state machine/graph (Role B) and UI (Role D) are
> in progress by the rest of the team. See [docs/PERCEPTION.md](docs/PERCEPTION.md)
> and [docs/ROLE_C.md](docs/ROLE_C.md) for the integration contracts.

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
| UI & telemetry | NiceGUI, Vis.js, `ui.log` |

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

### Dual-graph memory

- **Background graph** — invariant topological anchors such as zones (Kitchen, Work Desk) and fixed furniture, relocalized via CLIP place recognition.
- **Foreground graph** — episodic entities (keys, phone, mug) linked by `RESTING_ON`, `INSIDE`, state-transition edges, and action multi-edges.

The terminal relational anchor of an object is retained permanently, so queries about long-lost items still resolve correctly.

### Object lifecycle

`NEW → VISIBLE_STATIC → INTERACTED → VISIBLE_MOVING → HELD → CONTAINED → OCCLUDED → OUT_OF_VIEW → LOST`

Each transition is gated by IoS thresholds, support-surface heuristics, and wall-clock durations. See the specification for the full transition table.

### Query retrieval

Natural-language queries use a two-stage GraphRAG pipeline: retrieve the entity's latest state and spatial edge, serialize the 2-hop subgraph, then have a compact LLM produce a timestamped, location-cited answer.

## Repository layout

```text
.
├── Chronicle Technical Specification.pdf   # Full architecture and algorithms
├── chronicle/
│   ├── types.py                            # Shared Role A data contract
│   ├── perception/                         # Role A
│   │   ├── motion.py                       # Background-masked affine RANSAC
│   │   ├── detector.py                     # YOLO-World + palette fallback
│   │   ├── hands.py                        # MediaPipe Tasks + skin fallback
│   │   ├── tracker.py                      # Two-stage tracking + Re-ID
│   │   ├── interactions.py                 # IoS / dwell interaction gate
│   │   ├── clip_buffer.py                  # 30-frame ring + clip assembly
│   │   ├── pipeline.py                     # Orchestration / JSONL
│   │   ├── stream.py                       # Threaded consumer stream
│   │   ├── telemetry.py                    # Telemetry bus + printer
│   │   └── cli.py                          # chronicle-perception
│   └── semantic/                           # Role C
│       ├── vpr.py                          # OpenCLIP place recognition
│       ├── temporal_strip.py               # 30-frame -> 1x3 composite
│       ├── actions.py                      # strict action JSON boundary
│       ├── verification.py                 # Florence-2 + classifier
│       ├── graphrag.py                     # cited answers from 2-hop subgraph
│       └── engine.py                       # facade + telemetry events
├── docs/
│   ├── PERCEPTION.md                       # Role A contract
│   └── ROLE_C.md                           # Role C contract
├── scripts/
│   ├── fetch_assets.py                     # Download model weights
│   └── make_demo.py                        # Render the offline demo clip
├── tests/                                  # Role A + Role C pytest/unittest suite
├── requirements-semantic.txt               # Role C demo/test dependencies
├── requirements-semantic-models.txt        # OpenCLIP + Florence-2 stack
└── README.md
```

Model weights, videos and run outputs are git-ignored.

## Getting started

Requires Python 3.11–3.13 (3.12 recommended).

```bash
uv sync                      # core runtime
uv sync --extra perception --torch-backend cpu   # YOLO-World, MediaPipe, OpenCLIP
python scripts/fetch_assets.py
```

### Offline synthetic demo (Role A)

No model downloads, no network:

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
```

For the real OpenCLIP + Florence-2 stack and optional Ollama GraphRAG:

```bash
pip install -r requirements-semantic-models.txt
ollama pull qwen2.5:3b-instruct
```

Role C consumes Role A's `InteractionClip` directly, including its
`contact_index` and BGR `ClipFrame` images. See
[docs/ROLE_C.md](docs/ROLE_C.md) for the Role A/B/D integration contract,
telemetry payloads, action JSON schema, and configuration.

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

- **Role A (implemented):** video ingestion, motion gating, detection,
  hands, tracking/Re-ID, interaction qualification, ring buffer and clips,
  telemetry.
- **Role B:** 9-state lifecycle, containment/support reasoning, dual graph,
  persistence and query.
- **Role C (implemented):** OpenCLIP place recognition, 1×3 temporal strips,
  Florence-2 action verification with strict JSON, and cited GraphRAG answers.
- **Role D:** NiceGUI dashboard, graph visualization and live telemetry UI.

Role A intentionally does not mutate lifecycle state, call a VLM, persist a
graph, or render a dashboard. Role C does not mutate the graph; it emits
validated events and answer payloads for Roles B and D.

## References

The design builds on R4DSG (ACM MM 2026), VL-KnG (2026), ESOM (WACV 2026), hand-object interaction research (WACV 2024), and LightMem-Ego.

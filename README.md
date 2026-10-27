# Chronicle

An egocentric, episodic memory engine that parses pre-recorded first-person video into a structured, queryable 4D spatio-temporal knowledge graph.

> **Status:** Role A (perception & kinematics), Role B (state machine & dual-graph
> memory), Role C (semantic verification) and the integrated Role D dashboard are
> implemented and tested. `ChronicleRuntime` wires A → B → C → UI; see
> [`docs/INTEGRATION.md`](docs/INTEGRATION.md) and the live role assignments in
> [`docs/TEAM_STATUS.md`](docs/TEAM_STATUS.md). Integration contracts live in
> [`docs/PERCEPTION.md`](docs/PERCEPTION.md),
> [`docs/ROLE_B.md`](docs/ROLE_B.md) and [`docs/ROLE_C.md`](docs/ROLE_C.md).

## Why Chronicle

Most video-QA systems re-watch the video for every question. Chronicle moves the
expensive work to **compile time** and leaves a tiny, portable memory for
**query time**:

| Advantage | What it means |
| --- | --- |
| **Compress once, on the server** | A GPU or workstation runs perception and the VLM once, then writes a compact SQLite vault: graph index, packed event ledger, timestamped captions and a few evidence stills. The pitch measurement is **249 MB / 15 min → ~95 KB**; the 45 s cook clip compiles to **164.5 KB (5.4% of the video)**, and a 60 min episode projects to **~282 KB/hour** at 12 snapshots/hour. |
| **Ask on the glasses, locally** | The vault is a single portable file. Copy it to a phone or smart-glasses-class device and answer where/when/count/around questions **on-device**, with no server round-trip and no GPU. The deterministic path needs no LLM; a small local text model is optional phrasing. |
| **Privacy by construction** | Raw video is used only for compilation and optional playback. The query context is text + graph, never frames. Compile locally to keep footage on-device, or upload once to a server and delete the source after compile (`--delete-video`). |
| **Answers you can audit** | Every answer carries a timestamp and citation; a `frame @ t` chip opens the exact evidence frame. If the memory does not contain the answer, Chronicle says so instead of inventing one. |
| **Compute only when it matters** | Motion and geometry gating skips static frames, and the VLM runs on state transitions and salient events rather than every frame. |
| **Graceful degradation** | Missing Ollama, Florence-2 or OpenCLIP falls back to the colour detector, deterministic embeddings and template GraphRAG. The demo and tests run fully offline. |
| **Storage grows with events, not frames** | The packed ledger is ~42 KB/hour plus a bounded snapshot budget; a 60 min synthetic episode is a 139 KB vault. |
| **Open and portable** | MIT-licensed code, a single-file SQLite vault, JSON/CSV/ZIP import-export, local models and no vendor lock-in. |

The repository ships the vault, the deterministic query engine and a
browser-side query path (`memory.html`). The glasses/phone client is a
deployment target for that same vault, not a separate AR app in this repo. The
30-question numbers in this README are an internal regression suite, not a
public benchmark; count answers are track/visibility bounds, not verified
physical counts.

## Overview

Chronicle avoids running heavyweight vision-language models on every frame. Instead, deterministic motion and geometric gating decide *when* foundation models are invoked:

1. **Frame ingestion & affine parallax gating** — background-masked rigid motion estimation separates camera pans from head rotation and static pauses.
2. **Open-vocabulary detection & hand landmarks** — YOLO-World tracks user-specified targets while MediaPipe Hands provides 21 egocentric landmarks.
3. **Tracking with visual re-ID** — ByteTrack-style IDs are reconciled against OpenCLIP embeddings to avoid duplicate entities after occlusions.
4. **Appearance attributes** — a cheap crop pass records dominant colour (with swatch), brightness, saturation, shape/aspect, relative size, texture and frame position. The graph, exports and GraphRAG answers use these facts to disambiguate look-alike objects.
5. **Containment & interaction evaluation** — Shapely computes Intersection over Smaller Area (IoS) and hand/object containment.
6. **Temporal state machine** — entities move through a 9-state lifecycle driven by wall-clock time, not frame counts.
7. **Event-gated VLM verification** — state transitions trigger composite temporal-strip queries to Florence-2.
8. **Dual-graph ingestion** — events persist to NetworkX and SQLite with monotonic timestamps.
9. **Live telemetry** — a NiceGUI dashboard streams graph mutations, metrics, and logs over WebSockets.

## Architecture

| Layer | Technologies |
| --- | --- |
| Perception & tracking | OpenCV, Ultralytics YOLO-World, MediaPipe Hands, ByteTrack-style association, OpenCLIP |
| Spatial reasoning & state engine | Shapely, NetworkX, SQLite |
| Semantic engine | Florence-2-base, OpenCLIP (ViT-B-32) |
| UI & telemetry | NiceGUI, ECharts, vis-network, `ui.log` |

### Split deployment: compile on the server, query on the glasses

```text
┌──────────────────────────────────┐        ┌──────────────────────────────────┐
│ Server / workstation (once)      │        │ Smart glasses / phone (per query)│
│                                  │        │                                  │
│ video ─► perception ─► graph     │        │  vault.db  (KB–MB, SQLite)       │
│       ─► VLM captions ─► vault   │───────►│  ├─ deterministic lookup         │
│       ─► evidence stills         │ copy   │  ├─ timestamp + citation         │
│                                  │        │  └─ optional local text LLM      │
└──────────────────────────────────┘        └──────────────────────────────────┘
```

The compile side can be a cloud GPU, a home server or a laptop. The query side
only needs SQLite and, for natural-language phrasing, a small local text model.
`memory.html` is a single-file offline report whose query box runs in the
browser — the same split without any server.

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
│   ├── text.py                              # normalization, plurals, label aliases
│   ├── attributes.py                       # public appearance-attribute helpers
│   ├── integration.py                      # A -> B -> C runtime facade
│   ├── benchmark.py                        # reproducible source benchmark + JSON report
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
│   │   ├── attributes.py                   # dominant colour + simple properties
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
│       ├── app.py                          # graph, timeline, tables, chat, log
│       ├── graph_view.py                   # zone-clustered ECharts payloads
│       └── snapshot_io.py                  # validated snapshot import
├── docs/
│   ├── PERCEPTION.md                       # Role A contract
│   ├── ROLE_B.md                           # Role B integration contract
│   ├── ROLE_C.md                           # Role C contract
│   ├── INTEGRATION.md                      # runtime/UI contract
│   ├── CALIBRATION.md                      # real-label action thresholds + report
│   ├── DEMO_PREFLIGHT.md                   # pre-stage readiness checklist
│   ├── NOTICE_BOARD.md                     # claimable, prioritised team tasks
│   └── TEAM_STATUS.md                      # current work + role assignments
├── scripts/
│   ├── fetch_assets.py                     # download model weights
│   ├── make_demo.py                        # render the offline demo clip
│   ├── build_evidence_fixtures.py          # stored answer-frame index + images
│   ├── demo_preflight.py                   # PASS/WARN/FAIL demo readiness check
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

### Source benchmark (throughput + snapshot checks)

```bash
python -m chronicle.benchmark --sources cook,play,walking \
  --detector color --hands color --embedder color
```

`chronicle/benchmark.py` ingests each tracked clip through the full
`ChronicleRuntime`, records the backends that actually loaded, prints a Markdown
table and writes `runs/benchmark/sources.json`. It validates entity/state/
transition invariants rather than claiming accuracy on unlabelled clips; pass
`--max-frames` for a bounded production-backend run. See
[`docs/PERCEPTION.md`](docs/PERCEPTION.md#aria-clip-pipeline-benchmark).

### Offline semantic demo (Role C)

```bash
pip install -r requirements-semantic.txt   # NumPy + Pillow; demo works offline
python -m chronicle.semantic.demo
python -m chronicle.semantic.evaluation    # GraphRAG eval: 13 cases, citation precision
python -m chronicle.semantic.calibration \
  --labels docs/calibration_labels.holoassist.json   # per-action thresholds from real labels
python -m chronicle.semantic.timeline \
  --labels docs/action_timeline.example.json \
  --predictions docs/action_predictions.example.json   # verified actions vs labels
python -m chronicle.semantic.timeline \
  --annotations data/real/holoassist_switch_34s.json \
  --snapshot runs/hololens_snapshot.json \
  --tolerance 2.2 --ignore-entity                   # HoloAssist ground truth
```

The demo exercises VPR, temporal strips, action verification, the bounded
verification queue (dedup + backpressure + latency), confidence calibration and
the GraphRAG evaluation set, then prints the telemetry the dashboard consumes.
The calibrated thresholds come from the real HoloAssist clip; see
[`docs/CALIBRATION.md`](docs/CALIBRATION.md) for the label file, per-action
report and reproduction commands.
The suite is model-free: `python -m unittest discover -s tests` and
`python -m pytest` run the full suite without downloads; the dashboard smoke
test skips when the `ui` extra is not installed.

For the real OpenCLIP + Florence-2 stack and optional Ollama GraphRAG:

```bash
pip install -r requirements-semantic-models.txt
ollama pull qwen2.5:7b-instruct
```

Both `chronicle ask` and the dashboard query console answer through the local
Ollama model when it is reachable, then fall back to the deterministic
GraphRAG answer. Select another model with `CHRONICLE_LLM_MODEL` or
`--llm-model`; `./chronicle.sh start --no-llm` runs the dashboard without
Ollama.

Role C consumes Role A's `InteractionClip` directly, including its
`contact_index` and BGR `ClipFrame` images. See
[docs/ROLE_C.md](docs/ROLE_C.md) for the Role A/B/D integration contract,
telemetry payloads, action JSON schema, and configuration.

### Dashboard (Role D)

```bash
uv sync --extra ui
./chronicle.sh preflight          # PASS/WARN/FAIL demo readiness check
chronicle sources                 # list bundled/demo clips
chronicle serve --source synthetic --auto-run
# open http://127.0.0.1:8080
```

`./chronicle.sh preflight --warm-llm` also loads Qwen before the demo. See
[`docs/DEMO_PREFLIGHT.md`](docs/DEMO_PREFLIGHT.md) for what each check means
and the stage-day checklist.

The dashboard streams the live dual graph, state timeline, entity/action
tables, pipeline log and grounded GraphRAG chat from the same
`ChronicleRuntime` used by the CLI. It degrades to the colour detector,
deterministic embeddings and template GraphRAG when models are unavailable.
`chronicle serve --source tabletop` uses the real HoloLens clip. The memory
graph clusters entities and furniture around fixed room anchors, so switching
between zones stays visually legible. Snapshots can be exported as JSON/CSV or
re-imported from the header without re-running the pipeline.

The header download buttons expose the same data as files, and the endpoints
can be scripted directly:

```bash
curl -sf http://127.0.0.1:8080/export/snapshot.json -o chronicle-snapshot.json
curl -sf http://127.0.0.1:8080/export/entities.csv -o chronicle-entities.csv
curl -sf http://127.0.0.1:8080/export/actions.csv -o chronicle-actions.csv
```

`snapshot.json` is the exact dashboard snapshot (graph, entities, actions,
transitions, furniture, verification telemetry, stats and backends). Both CSVs
use stable headers so they can be diffed between runs.

The default mission control is the zine dashboard — warm paper cards, the
head-cam feed with evidence overlays, a grounded chat with citation chips, the
memory graph, entity dossiers, state-change timelines and pipeline vitals.
Launch it with ``./chronicle.sh start`` or ``python run_dashboard.py``.

**This worktree's control script.** `chronicle.sh` works without a local
`.venv` by using the sibling main checkout's interpreter and importing this
tree (`PYTHONPATH`). Because main already owns `:8080`, the experiment worktree
defaults to **:8082** (`CHRONICLE_PORT=...` overrides it):

```bash
./chronicle.sh start            # zine UI in background on :8082
./chronicle.sh status           # PID, URL, recent log lines
./chronicle.sh logs             # tail the log
./chronicle.sh restart
./chronicle.sh stop             # stops only this worktree's PID
./chronicle.sh pipeline         # foreground live-pipeline/VFM UI
./chronicle.sh test             # pytest + ruff + GraphRAG evaluation
```
 It
drives the event contract from ``dashboard/``; ``--source live`` waits for
``POST /api/ingest`` instead of replaying the bundled demo. The live-pipeline
demo (real perception, clip switching, exports) stays available via
``./chronicle.sh pipeline`` (a.k.a. ``chronicle serve``).

The zine dashboard can replay any recorded clip timeline (`.events.jsonl`)
instead of the bundled demo:

```bash
# the HoloLens tabletop timeline ships with the repo
python run_dashboard.py --replay data/real/holoassist_switch_34s.events.jsonl
# record one for another source, then replay it
python -m chronicle.event_export --source cook --out data/real/cook.events.jsonl
python run_dashboard.py --replay data/real/cook.events.jsonl
```

The session header in the event file supplies the video, title and duration, so
the feed, overlays, timeline and chat all follow that clip.

**Screenshot + fullscreen.** The stage toolbar has a frame button (`⤓`) and a
fullscreen button (`⛶`); keyboard shortcuts are `S` and `F`. Every grounded chat
answer also gets a `⤓ frame @ 12.3s` chip that seeks to the answer's cited time,
captures that exact frame (stamped with clip and timestamp), and opens it in a
lightbox with **save PNG** and **fullscreen**. The lightbox itself can go
fullscreen, so judges can blow up an evidence frame without leaving the page.
When the VFM vault has a stored still **within 8 s of the question time**, one
evidence thumbnail is attached to the reply; otherwise the reply stays text-only
and the explicit `frame @ t` button remains the way to inspect a frame.

**Export / import graphs.** The stage toolbar has `⤒ export` and `⤓ import`.
Export downloads `chronicle_graphs.zip` with a manifest, the current full event
timeline (`current/events.jsonl`), a readable `snapshot.json` built from that
whole timeline (not just the playhead), entity/edge/zone CSVs, every replay
timeline under `clips/`, each per-event graph/timeline under `event_clips/`, and
the active VFM vault/report/pack/evidence when compiled. Import accepts that ZIP
and reloads the timeline into the dashboard without re-ingesting video; the
imported graph appears first in the clip
switcher. Endpoints: `GET /api/export/graphs.zip?clip=<active-clip>`,
`POST /api/import/graphs`.

**Event clips + per-event graphs.** The toolbar's **✂ events** button cuts the
active timeline into bounded event windows (3 s pre + 3 s post by default):
state transitions, interactions, low-confidence/rejected actions and label
conflicts become short `clip.mp4` files, a rebased `events.jsonl` replay and a
`graph.json` snapshot with entity/zone/edge nodes at the trigger time. The first
four cuts are then verified by **Qwen2.5-VL**
(Ollama) or **Florence-2**: three frames (pre/contact/post) identify the event,
the result is appended to the timeline as a `vlm-event` interaction, one
`frames/snapshot.jpg` is kept, and a queryable per-event `vault.db` is compiled
from the caption ledger. If no VLM is available the clip still gets frames,
snapshot, rule-based action and vault. Strong graph states are never overwritten
by a disagreeing VLM answer; the disagreement is flagged for review.
Endpoints: `POST /api/event-clips`, `GET /api/event-clips`.
See [`docs/EVENT_CLIPS.md`](docs/EVENT_CLIPS.md) for the salience table,
reconciliation policy and artifact layout.

**Memory card / VLM answers.** When the active clip has a graph DB, the zine
chat uses the compiled VFM vault: dense VLM caption windows + event ledger +
evidence, answered deterministically in **~0.05 ms** without touching the video.
The first time a clip is opened the memory is compiled **automatically in the
background** (VLM captions, one-time cost); the `MEMORY · VIDEO-FREE` card shows
progress and then the report/downloads (`vault.db`, `memory.html`,
`memory_pack.md`, `memory.zip`). The card's **Compile memory** button forces a
rebuild; after the first build it reads **Rebuild memory**. If a vault was
compiled while an upload was still being perceived (an empty "thin" vault), it
is detected and rebuilt once the graph DB has entities. Disable auto-compile
with `CHRONICLE_VFM_AUTO=0`, disable the whole panel with `CHRONICLE_VFM=0`.
Clips with no graph DB (fresh uploads, long street walks) use the **caption-first**
path: VLM captions become the ledger and entities, so they still answer
"what happened around 10 s?" without running perception. The zine manager now
defaults to **20 s caption windows** (`CHRONICLE_VFM_WINDOW_S=20`) so a 10 s
question gets a frame at 10 s, not a 0–30 s caption; set
`CHRONICLE_VFM_WINDOW_S=30` for long videos where compile time matters more.

With a compiled vault, answers show `provider: vfm-memory`; count questions get
the track/visibility bounds and deduped physical estimate. "Where is X?"
answers use the current playhead: if X is no longer visible it says so and
reports the last seen timestamp instead of pretending X is still in frame.
Clips that only have a replay timeline and no DB show a clear "no graph DB"
note; the graph fallback still answers state/edge questions instead of claiming
there were no events.

The **canonical demo surface is `./chronicle.sh start`** (zine mission control,
port 8080); the live-pipeline/VFM operator UI is `./chronicle.sh pipeline`
(a.k.a. `chronicle serve`). See
[`docs/INTEGRATION.md`](docs/INTEGRATION.md#two-dashboards-one-canonical-entry-point).

Headless smoke test:

```bash
chronicle serve --source synthetic --port 8765 &
curl -sf http://127.0.0.1:8765/ >/dev/null
curl -sf http://127.0.0.1:8765/source-video >/dev/null
curl -sf http://127.0.0.1:8765/export/snapshot.json >/dev/null
curl -sf http://127.0.0.1:8765/export/entities.csv >/dev/null
curl -sf http://127.0.0.1:8765/export/actions.csv >/dev/null
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

## Video-free memory (VFM)

Architecture note (LaTeX/PDF, 16 references): [`docs/VFM_ARCHITECTURE.pdf`](docs/VFM_ARCHITECTURE.pdf)
— what is stored, compile/query flow, count bounds, UI integration and the papers behind it.
Rebuild with `docs/build-vfm-architecture.sh`.


The dashboard can compile a small, queryable memory from the current run or an uploaded video:
packed events, dense VLM captions, 480 px WebP evidence, a single-file HTML report and an
LLM-ready pack. The source video is kept for playback; it is not required to answer questions.

```bash
# deterministic memory panel, Qwen off
chronicle serve --source synthetic

# Qwen answers over the memory pack (no frames)
chronicle serve --source synthetic --qwen

# control the compile; --no-vfm disables compiles
chronicle serve --vfm-window-s 20 --vfm-frames-per-window 3 --vfm-max-snapshots 40
chronicle serve --no-vfm
```

A successful ingest triggers a background compile; the **Memory** tab shows progress, opens the
report, downloads `vault.db` / `memory.html` / `memory_pack.md` / `memory.zip`, and answers
questions. Citation chips seek the existing player. Counts are track/visibility bounds (and a
deduped physical estimate where the compiler can compute one), never a claim of verified physical
counts.

Files: `chronicle/vfm/`, tests `tests/test_vfm.py`, full details
`experiments/video-free-memory/docs/MAIN_UI_INTEGRATION.md`.

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

## Papers, prior art & open source

Chronicle is an integration of prior work: the memory model, perception stack
and query path all build on published research and open-source projects. Links
are provided for attribution and further reading.

### Memory, retrieval and compression

| Work | Venue | What Chronicle borrows |
| --- | --- | --- |
| [R4DSG](https://arxiv.org/abs/2608.11017) | ACM MM 2026 | Anchor-relative 4D scene memory; compact queryable entries without a global 3D reconstruction. |
| [VL-KnG](https://arxiv.org/abs/2510.01483) | 2025/2026 | Persistent spatio-temporal knowledge graph; query latency decoupled from video length. |
| [EgoGraph](https://arxiv.org/abs/2602.23709) | 2026 | Training-free temporal knowledge graph with an egocentric entity/location/event schema. |
| [CapMem](https://arxiv.org/abs/2609.17688) | EMNLP 2026 | Caption-only episodic-memory QA; 30 s / 60 s caption-window study and retrieve-and-verify. |
| [EgoCITE](https://arxiv.org/abs/2608.12627) | 2026 | Dense visual captions and transcripts as memory entries; time as a first-class retrieval key. |
| [MERIT](https://arxiv.org/abs/2608.07663) | ECCV 2026 | Query-agnostic memory construction; high-recall keys and query-time temporal expansion. |
| [FocusGraph](https://arxiv.org/abs/2603.04349) | 2026 | Text-first, pixels-second routing; sparse keyframes only when the text answer is insufficient. |
| [VideoMem](https://arxiv.org/abs/2512.04540) | 2025/2026 | Adaptive global memory buffer; compression and retention-policy inspiration. |
| [EM-LLM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c05144b635df16ac9bbf8246bbbd55ca-Abstract-Conference.html) | ICLR 2025 | Event segmentation with similarity plus temporal-contiguity retrieval. |
| [LightMem](https://arxiv.org/abs/2510.18866) | ICLR 2026 | Sensory → short-term → long-term tiers; offline consolidation. |
| [HippoRAG](https://arxiv.org/abs/2405.14831) | NeurIPS 2024 | Knowledge-graph retrieval as a cheap alternative to iterative model calls. |
| [EgoMemory / EgoRetriever](https://aclanthology.org/2026.findings-acl.362.pdf) | ACL Findings 2026 | Personal memory bank and training-free retrieval over long egocentric video. |
| [Continuous Semantic Caching](https://research.ece.cmu.edu/lions/Papers/Semantic_LLM_Caching.pdf) | 2026 | Semantic answer-cache reuse and eviction under uncertainty. |
| [Titans](https://arxiv.org/abs/2501.00663) | 2024/2025 | Neural long-term memory as a north star for model-native episodic memory. |
| [MemoryLLM](https://arxiv.org/abs/2402.04624) | ICML 2024 | Fixed-size latent memory pool inside a transformer. |
| [Caption-once, Frames-on-Demand](https://aclanthology.org/2026.emnlp-main.422.pdf) | EMNLP 2026 | Cache captions once; fetch bounded pixels only when confidence is low. |

The specification also cites ESOM (WACV 2026) for streaming object memory and
hand-object contact research (WACV 2024) for contact-triggered interaction.

### Perception and semantic methods

| Work | Venue | What Chronicle uses |
| --- | --- | --- |
| [YOLO-World](https://arxiv.org/abs/2401.17270) | CVPR 2024 | Real-time open-vocabulary object detection with a user-supplied label set. |
| [Florence-2](https://arxiv.org/abs/2311.06242) | 2023 | Unified vision foundation model for action/caption verification. |
| [MediaPipe Hands](https://arxiv.org/abs/2006.10214) | 2020 | On-device 21-point hand landmarks. |
| [ByteTrack](https://arxiv.org/abs/2110.06864) | ECCV 2022 | High/low-confidence association and Hungarian assignment for tracking. |
| [CLIP](https://arxiv.org/abs/2103.00020) | ICML 2021 | Contrastive image-text representations behind visual re-ID. |
| [OpenCLIP](https://arxiv.org/abs/2212.07143) | CVPR 2023 | Reproducible CLIP training/scaling; the OpenCLIP implementation used here. |
| [GraphRAG](https://arxiv.org/abs/2404.16130) | 2024 | Graph-based retrieval and grounded answer generation. |
| [Qwen2.5](https://arxiv.org/abs/2412.15115) | 2024 | Optional local text model for phrasing verified answers. |
| [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) | 2025 | Optional local VLM for captions and event-clip verification. |
| [HoloAssist](https://arxiv.org/abs/2309.17024) | ICCV 2023 | Real HoloLens 2 egocentric interaction dataset used for the demo clip and labels. |

### Open-source projects

| Project | Used for | Licence |
| --- | --- | --- |
| [Python](https://www.python.org/) | Runtime | PSF-2.0 |
| [FastAPI](https://github.com/fastapi/fastapi) | HTTP/JSON API | MIT |
| [NiceGUI](https://github.com/zauberzeug/nicegui) | Dashboard | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | ASGI server | BSD-3-Clause |
| [Apache ECharts](https://github.com/apache/echarts) | Operator graph/timeline charts | Apache-2.0 |
| [vis-network](https://github.com/visjs/vis-network) | Zine dashboard memory graph | Apache-2.0 / MIT |
| [SQLite](https://sqlite.org/) | Episode DB and VFM vault | Public domain |
| [NetworkX](https://github.com/networkx/networkx) | Dual-graph memory | BSD-3-Clause |
| [Shapely](https://github.com/shapely/shapely) | IoS and containment geometry | BSD-3-Clause |
| [NumPy](https://github.com/numpy/numpy) | Numerical arrays | BSD-3-Clause |
| [SciPy](https://github.com/scipy/scipy) | Optimisation / linear algebra | BSD-3-Clause |
| [OpenCV](https://github.com/opencv/opencv) | Video I/O and image operations | Apache-2.0 |
| [Pillow](https://github.com/python-pillow/Pillow) | Image encode/decode | MIT-CMU (HPND) |
| [PyTorch](https://github.com/pytorch/pytorch) / [torchvision](https://github.com/pytorch/vision) | Model runtime | BSD-3-Clause |
| [Ultralytics YOLO-World](https://github.com/ultralytics/ultralytics) | Open-vocabulary detection | **AGPL-3.0** (optional) |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Hand landmarks | Apache-2.0 |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | Place recognition and re-ID | MIT |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | Tracking design reference | MIT |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) / [Accelerate](https://github.com/huggingface/accelerate) / [timm](https://github.com/huggingface/pytorch-image-models) / [einops](https://github.com/arogozhnikov/einops) | Florence-2 stack | Apache-2.0 / Apache-2.0 / Apache-2.0 / MIT |
| [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) | Action/caption verification | MIT |
| [Ollama](https://github.com/ollama/ollama) | Local LLM/VLM serving | MIT |
| [Qwen2.5](https://github.com/QwenLM/Qwen2.5) / [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) | Local text/VLM models | Apache-2.0 |
| [FFmpeg](https://ffmpeg.org/) | Frame extraction for VFM captions | LGPL-2.1+ / GPL-2.0+ (build-dependent) |
| [pytest](https://github.com/pytest-dev/pytest) / [Ruff](https://github.com/astral-sh/ruff) / [uv](https://github.com/astral-sh/uv) | Tests, lint, environment | MIT / MIT / Apache-2.0 OR MIT |

### Datasets and media

| Asset | Used for | Licence |
| --- | --- | --- |
| [HoloAssist](https://holoassist.github.io/) | Real HoloLens 2 egocentric demo clip and action labels | CDLA-Permissive-2.0 |
| [Meta Project Aria Gen 2](https://www.projectaria.com/datasets/) | Real Aria cook/play/walk research clips | CC BY-NC 4.0 |

The Aria clips are locally trimmed research windows; the full source data
remains with Meta. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the
full dependency/licence notes, including the optional AGPL-3.0 YOLO-World
backend and the model-weight terms.

## License

Chronicle is released under the [MIT License](LICENSE).

Copyright (c) 2026 Chronicle contributors.

The MIT licence covers Chronicle's source code only. Third-party libraries,
model weights and datasets keep their own licences. In particular, the optional
Ultralytics YOLO-World backend is **AGPL-3.0**, and the bundled real clips come
from HoloAssist (**CDLA-Permissive-2.0**) and Meta Aria Gen 2
(**CC BY-NC 4.0**). Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before redistributing Chronicle or using it commercially.

# Role B - State Machine & Graph Engineering

This document is the integration contract for the Role B slice of Chronicle. It
covers the nine-state lifecycle engine, the dual NetworkX graph, the SQLite
projection, duplicate consolidation, and the 2-hop retrieval API.

Role B owns **when** something happened and **where it was last anchored**. It
does not run detectors or VLMs. Role A produces `FrameContext` objects, Role C
validates semantic events, and Role D renders the graph and telemetry.

> The dashboard/CLI use `ChronicleRuntime` rather than calling `ChronicleMemory`
> directly. See [`INTEGRATION.md`](INTEGRATION.md) for the A→B→C seam, snapshot
> schema and furniture routing.

## Quick start

```powershell
pip install -r requirements.txt
python -m pytest
python scripts/demo_role_b.py
```

The demo is completely synthetic (no video, no models). It runs a cup through
`VISIBLE_STATIC -> INTERACTED -> HELD -> VISIBLE_MOVING -> VISIBLE_STATIC`,
prints `[STATE]` / `[GRAPH]` telemetry, retrieves "where are my cups?", closes
SQLite, reopens it, and answers the same question from disk.

## Module map

| Module | Responsibility |
| --- | --- |
| `chronicle/types.py` | Shared Role A/B dataclasses: `FrameResult`, `BBox`, `FrameContext`, `ObjectObservation`, `HandObservation`, `SupportSurface`, `StateTransition` |
| `chronicle/config.py` | Every pixel / wall-clock threshold (`Thresholds`) |
| `chronicle/geometry.py` | IoS, support contact, camera-relative velocity, time-aware EMA |
| `chronicle/state_machine.py` | Nine-state lifecycle, dwell gates, occlusion / pan / reacquisition logic |
| `chronicle/graph/background.py` | Zones + fixed furniture; CLIP place relocalization |
| `chronicle/graph/foreground.py` | Entities, terminal anchors, time-intervaled relations, actions |
| `chronicle/graph/consolidation.py` | Conservative duplicate merge sweep, dry-run reports, alias audit |
| `chronicle/graph/query.py` | Entity resolution, 2-hop serialization, prompt/table/fallback answer |
| `chronicle/graph/memory.py` | `ChronicleMemory` facade used by every other role |
| `chronicle/storage/sqlite_store.py` | Persistent schema, snapshots, retention purge |
| `chronicle/text.py` | Normalization, plural handling, label aliases, match reasons |
| `chronicle/telemetry.py` | Thread-safe `EventBus` with replayable history |

## Integration contract

### 1. Role A -> Role B: build one frame

```python
from chronicle import (
    CameraMotion,
    FrameContext,
    HandObservation,
    ObjectObservation,
    SupportSurface,
)

frame = FrameContext(
    t=presentation_time_s,  # wall-clock timestamp, not frame index
    objects=[ObjectObservation(entity_id="cup", label="cup", bbox=(x1, y1, x2, y2))],
    hands=[HandObservation.from_landmarks(landmarks_21)],
    supports=[SupportSurface(anchor_id="dining_table", label="dining table", polygon=table_poly)],
    camera=CameraMotion(dx=affine_dx, dy=affine_dy, dtheta_deg=affine_theta),
    frame_size=(width, height),
    zone_id="kitchen",
    attributes={"zone_label": "kitchen"},
)
```

Requirements:

- `t` is monotonic in wall-clock seconds. Frame counts are never used by any
  gate, so 15 fps and 60 fps produce the same event times.
- `entity_id` is stable across track loss. Role A must re-bind ByteTrack IDs via
  CLIP re-ID (threshold `reid_cosine = 0.82`) rather than minting a new entity.
- `camera` is the rigid background motion for this frame
  (`cv2.estimateAffinePartial2D`). Object velocity is compensated over the same
  history span, so a pan does not look like object movement.
- `supports` may be omitted while the surface detector is unavailable. The
  engine then reports `support_verified=False` instead of dropping stability.
  Set `strict_support=True` to require verified surfaces.
- `zone_id` / `zone_label` drive background graph placement. Missing zones fall
  back to an `unmapped` zone.

### 2. Advance the engine

```python
transitions = memory.observe(
    frame,
    on_transition=lambda transition, frame: vlm_queue.put(transition),
)
```

`observe()` returns only the transitions that fired on this timestamp. Each
`StateTransition` carries:

```python
transition.tag  # "VISIBLE_STATIC->INTERACTED"
transition.t_event  # presentation timestamp
transition.evidence  # IoS, velocity, support, occluder, dwell_s, reason, ...
```

`on_transition` is called after the graphs and SQLite have been updated, so Role
C may enqueue a Florence-2 job from inside the hook. Exceptions in the hook are
isolated and reported on the `[VLM]` telemetry channel.

A re-detected entity whose last sample was missing gets `reacquired=True` and
one forced transition on the first frame back. Its centroid history is cleared,
so stale samples cannot fabricate motion.

### 2b. Consuming Role A directly

If Role A hands you a `FrameResult` instead of a `FrameContext`, use the
built-in bridge:

```python
transitions = memory.observe_result(
    result,
    supports=(table_surface,),   # Role A does not detect support surfaces
    zone_id="kitchen",
    zone_label="kitchen",
)
```

The bridge:

- copies `result.timestamp` into the presentation clock (not `wall_time`),
- maps `MotionEstimate` to `CameraMotion` (`theta` radians -> degrees),
- maps each `TrackedObject` to an `ObjectObservation` with a stable
  `track_<id>` entity id (override with `metadata["entity_id"]`),
- carries CLIP embeddings and `metadata["reid_score"]` into evidence,
- forwards `HandObservation` objects unchanged (the type is shared by both
  roles), and
- infers `frame_size` from `result.image` when present.

Call `frame_context_from_result(...)` directly if you want to inspect or patch
the frame before handing it to `observe()`.

### 3. Role C hooks

```python
memory.record_action(
    "cup",
    action="pick_up",
    t=frame.t,
    confidence=0.91,
    source="florence-2",
    clip_ref="clip_0042.mp4",
    evidence={"vlm_answer": "person picks up a white cup"},
)

zone_id = memory.relocalize_zone(clip_embedding, t=frame.t)  # None below 0.85
memory.enter_zone("work desk", t=frame.t, embedding=desk_clip)
```

- Actions become `ACTION` multi-edges from `agent:user` to the entity.
- `relocalize_zone` uses cosine similarity at `place_cosine = 0.85`. A miss
  returns `None`; it never creates a duplicate zone.
- To rank entity candidates with CLIP instead of string matching, pass an
  embedder: `memory.query("where are my keys?", embedder=clip_text_embedder)`.

### 4. Role D dashboard

```python
snapshot = memory.visualization_snapshot()
# {"nodes": [...], "edges": [...], "stats": {...}}

rows = memory.state_table()  # compact per-entity table
stats = memory.stats()  # persisted counts + active zone + camera
anchor = memory.terminal_anchor("cup")
```

Nodes and edges already use the public `<kind>:<id>` namespacing
(`entity:cup`, `anchor:dining_table`, `zone:kitchen`, `agent:user`), so Vis.js
can render the payload directly. Every entity, furniture and zone node also
carries `zone_id` (the namespaced `zone:<id>` it belongs to), which the
dashboard uses to cluster the force graph by room.

Telemetry is pushed to every subscriber of `memory.bus`:

| Tag | Examples |
| --- | --- |
| `[STATE]` | `cup: VISIBLE_STATIC -> INTERACTED`, re-acquisition, entity minted |
| `[GRAPH]` | zone/anchors minted, `RESTING_ON` opened/closed, foreground node minted, `entity merged`, `consolidation sweep` |
| `[VLM]` | transition-hook failures (Role C queue health) |
| `[QUERY]` | entity resolved, subgraph serialized, answer context ready |
| `[STORE]` | schema ready, persistence issues |
| `[SYSTEM]` | memory online, restored from disk, closed |

Subscribe before constructing `ChronicleMemory` to capture startup events:

```python
bus = EventBus()
bus.subscribe(lambda event: print(event.format()))
memory = ChronicleMemory("data/chronicle.db", bus=bus)
```

### 5. Query resolution

```python
result = memory.query("where are my cups?")
result.fallback_answer()  # deterministic answer, no LLM required
result.to_table()  # markdown table of the 2-hop subgraph
result.to_prompt()  # complete prompt for Role C's compact LLM
result.as_dict()  # JSON-safe payload for the dashboard
```

Resolution is plural- and punctuation-tolerant. The retrieval window is two
hops from the entity node, which deliberately includes the parent zone anchor.
The terminal `RESTING_ON` / `INSIDE` edge is always retained, so an object that
has been `LOST` for hours still resolves to its last known surface.

`QueryResult.match_score` is the raw label/embedding score. `confidence`
additionally accounts for competing candidates and current visibility, so a
duplicate label can resolve to the visible instance while reporting lower
certainty:

```python
result = memory.query("where is my purse?")
result.entity_id      # surviving entity id
result.match_score    # raw score, e.g. 0.85
result.confidence     # ambiguity-aware score in [0, 1]
result.match_reason   # "exact label", "label alias", "embedding similarity", ...
result.candidates     # ranked alternatives with ids, states and reasons
```

Label aliases are symmetric and deliberately narrow (`tv`/`television`,
`sofa`/`couch`, `wallet`/`purse`, `keys`/`keychain`). Look-alike categories that
the detector keeps distinct (`cup` vs. `coffee mug`) are not aliased.

## Lifecycle reference

All transitions are evaluated against presentation timestamps. Geometry gates
come from `Thresholds`; every target state also has a dwell gate.

| From | Target | Gate (summary) | Dwell |
| --- | --- | --- | --- |
| `NEW` | `VISIBLE_STATIC` | relative speed <= 6 px/s and supported | 0.50 s |
| `NEW` | `INTERACTED` | `IoS(hand, object) >= 0.60` | 0.30 s |
| `VISIBLE_STATIC` | `INTERACTED` | hand contact | 0.30 s |
| `INTERACTED` | `HELD` | `IoS >= 0.50`, hand/object velocity cosine >= 0.80, speed > 25 px/s | 0.20 s |
| any visible | `VISIBLE_MOVING` | camera-relative speed > 25 px/s and hand IoS < 0.60 | 0.20 s |
| any visible | `CONTAINED` | `IoS(object, container) >= 0.85` | 0.15 s |
| `CONTAINED` | `VISIBLE_STATIC` / `VISIBLE_MOVING` | container no longer encloses object | 0.15 s |
| visible | `OCCLUDED` | missing and `IoS(last bbox, occluder) >= 0.50` (or line of sight lost) | 0.20 s |
| visible | `OUT_OF_VIEW` | missing while the camera pans, or the object center leaves the frame | 0.20 s |
| `OUT_OF_VIEW` | `LOST` | 600 s archival window elapses (spec section 4) | 0 s |

Notes:

- A missing entity is classified `OCCLUDED` unless the camera is panning or its
  center has left the frame. The occluder evidence (id + IoS) is attached for
  Role C.
- `LOST` is reached only from `OUT_OF_VIEW` by default. Set
  `lost_from_occluded=True` if the team wants the archival timeout to apply to
  long-lived occlusions too.
- `CONTAINED` objects are expected to be undetectable while their container is
  visible, so they are not demoted to `OCCLUDED`.
- `INTERACTED` is a transient state: the engine will move to `HELD` if the hand
  and object move in lock-step, or back to a visible state when contact ends.
- The spec treats `IoS(hand, object) < 0.10` as fully hand-free. The engine also
  allows independent motion up to `0.60` when the hand is near but not grasping,
  because otherwise the lifecycle can stall in the 0.10-0.60 ambiguity band. The
  evidence `reason` distinguishes `independent_motion` from
  `independent_motion_relaxed`.
- Re-acquisition forces the next transition immediately (`force=True`) instead
  of waiting for a dwell window — including when `strict_support=True` and the
  first frame back has no surface geometry (reason
  `reacquired_without_support_geometry`).
- A candidate container must be strictly larger than the object. IoS is
  symmetric, so without that guard a cup would also mark its backpack as
  `CONTAINED` and create a mutual `INSIDE` edge.
- Leaving `CONTAINED` for a visible state uses `exit_dwell_s` (0.15 s) instead
  of the full static/moving dwell.
- Once a missing object is classified `OUT_OF_VIEW` during a pan, a later still
  camera does not demote it to `OCCLUDED`; the state is held until re-detection
  or the 600 s archival timeout.
- Non-monotonic presentation timestamps are clamped to the last accepted time
  and the frame timestamp is rewritten, so camera/velocity windows never
  rewind.

## Dual graph

### Background graph (`memory.background`)

- `zone:<slug>` nodes: label, CLIP embedding, first/last seen, visit count.
- `anchor:<slug>` nodes: fixed furniture / support surfaces.
- `zone -[CONTAINS]-> anchor` edges.
- `ensure_zone()` and `relocalize()` are idempotent. The 0.85 threshold prevents
  a new zone from being minted for a known place.

### Foreground graph (`memory.foreground`)

- `entity:<id>` nodes: label, state, timestamps, bbox, zone, container.
- `agent:user` singleton representing the wearer.
- Time-intervaled edges with `t_start` / `t_end`:
  - `entity -[RESTING_ON]-> anchor`
  - `entity -[INSIDE]-> anchor`
  - `entity -[STATE_TRANSITION]-> entity` (one per transition, keyed by `t_event`)
  - `agent:user -[ACTION]-> entity`
- Closed placements are retained; only the newest placement is marked
  `is_terminal=True`. `RESTING_ON` and `INSIDE` are mutually exclusive, so an
  object that moves from a surface into a container (or back) still has exactly
  one terminal placement. The terminal anchor is the one queries read first.
- The 2-hop query context explicitly hydrates each entity's parent zone, so a
  `CONTAINED` object (whose terminal anchor is the container entity) still
  reports `zone:<room>` in `terminal_anchor`, `state_table()` and `to_table()`.

### SQLite

`SQLiteStore` mirrors the graphs in `zones`, `furniture`, `entities`,
`state_transitions`, `edges`, `anchors`, and `observations`.

Persistence invariants:

- Exactly one `RESTING_ON` / `INSIDE` row per entity is terminal; every older
  placement is closed. `SQLiteStore` normalizes this on terminal-edge insert,
  on merge, and once on `initialize()` so databases written by older builds are
  repaired on reopen. Foreground placement intervals are normalized again when
  the memory graph is restored.
- `anchors` rows are never deleted, even after the placement closes.
- `observations` (bbox trajectories) are the only rows purged, after
  `trajectory_retention_s = 600` seconds.
- `ChronicleMemory(path)` restores zones, entities, transition history, edges,
  terminal anchors, and furniture timestamps on construction. Persisted
  furniture `last_seen` / `first_seen` survive reopen; missing values fall back
  to `created_at`.
- `ChronicleMemory(":memory:")` is the zero-artifact mode used by the test suite.
- Merging a graph-only furniture keeper promotes its SQLite row to the same
  `anchor:<id>` namespacing used by normal persistence.
- SQLite is a durable projection, not a per-frame mirror. Entity and support
  rows persist on a state change or every `persistence_interval_s = 0.5`;
  observations keep their `0.1 s` trajectory cadence. On a 180-frame `walking`
  run this cut entity upserts from 172 to 59 with the in-memory snapshot
  unchanged (`tests/test_graph_storage.py::test_entity_and_support_writes_are_throttled`).

### Duplicate consolidation

`GraphConsolidator` (`memory.consolidator`) cleans up legacy duplicates without
weakening the one-node-per-object invariant:

```python
planned = memory.merge_entities("cup-a", "cup-b", dry_run=True)
report = memory.merge_entities("cup-a", "cup-b")
memory.merge_furniture("sofa-1", "sofa-old")
memory.maybe_consolidate(t=frame.t)  # periodic sweep
```

The sweep is conservative:

- labels must canonicalize to the same alias group;
- zones must agree (an unknown zone is not a mismatch);
- two co-visible instances of the same label are never merged (furniture may
  only merge when their boxes overlap at `consolidation_covisible_iou`);
- temporal, spatial and embedding evidence must clear `consolidation_min_score`;
- the automatic sweep runs at most once per `consolidation_interval_s` and is
  capped at `consolidation_max_merges` per pass.

A merge re-points SQLite transitions, observations, anchors and edges at the
keeper before deleting the removed entity row. Anchors are retained and exactly
one placement remains terminal. In-memory state/placement intervals are
normalized, action edges are preserved, and the lifecycle engine is re-seeded
from the keeper's latest state. The removed id is registered as an alias; later
Role A frames carrying it are canonicalized before entity creation, so the
duplicate is not minted again. Aliases are rebuilt from persisted `merged_from`
attributes when the database is reopened.

Every mutation is auditable: dry runs emit `[GRAPH] entity merge planned`, real
merges emit `[GRAPH] entity merged` (with keep/remove ids and preserved row
counts), and each automatic pass that changes something emits
`[GRAPH] consolidation sweep`.

### Furniture association across pans

`ChronicleMemory._route_furniture()` advances the registry's virtual camera once
per frame, including frames with no furniture detections. Each anchor stores the
camera position at which its bbox was last updated, so re-entry prediction uses
the accumulated displacement since *that* anchor was seen, not just the current
frame's `camera.dx/dy`. A multi-frame pan therefore re-binds to the existing
anchor instead of minting a duplicate.

### Dual furniture fixtures and persistence validation

Dual furniture fixtures (e.g. `sofa-left` and `sofa-right` in the same room) are validated under
full SQLite disk persistence (`tests/test_consolidation.py::test_dual_furniture_consolidation_sweep_and_sqlite_persistence`):
- Co-visible instances of static furniture remain distinct background anchors during periodic consolidation sweeps.
- Sequential re-detections (e.g., re-entry after camera pan) merge cleanly into the keeper.
- Reopening the database from SQLite disk file restores `merged_from` aliases into the `FurnitureRegistry` and blocks re-minting of retired track anchors.

## Configuration

All tunables live in `chronicle/config.py` as frozen dataclasses. Override
thresholds at construction time:

```python
from dataclasses import replace
from chronicle import ChronicleMemory, DEFAULT_THRESHOLDS
from chronicle.config import FurnitureConfig

fast = replace(DEFAULT_THRESHOLDS, static_dwell_s=0.25, lost_after_s=120.0)
memory = ChronicleMemory(
    "data/demo.db",
    thresholds=fast,
    furniture_config=FurnitureConfig(consolidation_enabled=False),
)
```

Spec-fixed values are marked in the source. `FurnitureConfig` also owns the
consolidation gates (`consolidation_enabled`, `consolidation_interval_s`,
`consolidation_min_score`, `consolidation_embedding_cosine`,
`consolidation_max_gap_s`, `consolidation_covisible_iou`,
`consolidation_max_merges`). Everything else is an engineering default chosen
to be stable on 15-60 fps egocentric video.

## Tests

The suite is synthetic and deterministic; no video or model downloads:

```powershell
python -m pytest -q
```

Coverage includes:

- frame-rate independence (15 / 30 / 60 fps produce the same transition order),
- camera-pan compensation and re-acquisition,
- all nine states, dwell gates, occlusion vs. out-of-view,
- terminal-anchor retention and SQLite restore,
- CLIP place relocalization,
- duplicate consolidation: dry runs, history/action/anchor retention, co-visible
  refusal, alias persistence across reopen, and furniture merges,
- 2-hop retrieval, label aliases, match confidence and duplicate disambiguation,
- multi-entity independent lifecycles,
- telemetry tags and transition hooks.

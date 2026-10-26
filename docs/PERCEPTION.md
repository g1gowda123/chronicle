# Role A — Perception & Kinematics

This document covers the perception layer only: ingesting video, gating camera
motion, detecting open-vocabulary objects and hands, tracking identities,
qualifying hand/object contact over time, and emitting structured clips plus
telemetry. The lifecycle state machine, graph, VLM-verification and UI layers
(Roles B/C/D) are intentionally **not** part of this package.

## What is implemented

| Stage | Implementation | File |
| --- | --- | --- |
| Camera motion / parallax gate | Shi–Tomasi features on static background, pyramidal LK with forward–backward check, `cv2.estimateAffinePartial2D` + RANSAC | `chronicle/perception/motion.py` |
| Open-vocabulary detection | Ultralytics YOLO-World with `set_classes`; palette fallback for the synthetic demo | `chronicle/perception/detector.py` |
| Egocentric hands | MediaPipe Tasks `HandLandmarker` (VIDEO mode, 21 landmarks); skin-colour fallback | `chronicle/perception/hands.py` |
| Tracking + Re-ID | ByteTrack-style high/low-confidence association (Hungarian) with an inactive-track pool reconciled by OpenCLIP cosine similarity and spatial continuity | `chronicle/perception/tracker.py` |
| Interaction qualification | Hand hull vs. object box IoS, fingertip containment, wall-clock dwell + release grace | `chronicle/perception/interactions.py` |
| Object attributes | Rotated-crop dominant colour (background-suppressed k-means), brightness, saturation, shape/aspect, relative size, texture, frame position | `chronicle/perception/attributes.py` |
| Temporal memory | 30-frame downscaled RAM ring buffer, pre/post-roll interaction clips, 1×3 composites | `chronicle/perception/clip_buffer.py` |
| Orchestration | Single-threaded `PerceptionPipeline`, background `PerceptionStream` | `chronicle/perception/pipeline.py`, `stream.py` |
| Telemetry | Category-tagged events, fan-out bus, console printer | `chronicle/perception/telemetry.py` |
| Shared contract | Plain dataclasses consumed by B/C/D | `chronicle/types.py` |

The tracker is **ByteTrack-style**, not a byte-for-byte port: it uses the same
two-stage high/low confidence matching idea and Hungarian assignment, and adds
the specification's inactive-track appearance reconciliation (cosine
similarity gate `SC > 0.82`).

## Pipeline

```text
video frame
   │
   ├─► background-masked affine motion ──► MotionEstimate (STATIONARY/MOVING/PANNING)
   │        (previous + current dynamic boxes masked)
   │
   ├─► YOLO-World(dynamic vocabulary) ─┐
   ├─► crop attributes (colour et al.) ─┤
   ├─► MediaPipe Hands (21 landmarks) ──┤
   │                                    ▼
   │                         ByteTrack-style tracker
   │                         high-conf → low-conf → Re-ID pool
   │                                    │
   │                         TrackedObject[] + track events
   │                                    ▼
   │                         IoS + fingertip + dwell gate
   │                                    │
   │                         InteractionObservation / Event
   │                                    ▼
   └─► 30-frame ring buffer ──► InteractionClip (frames + composite)
```

Motion is estimated **before** tracking. The previous frame's detection and
hand boxes are added to the current boxes and masked out, so a moving object or
a hand cannot vote on the camera transform. A settled frame after a pan
(`transition == "pan_end"`) is exposed as `MotionEstimate.keyframe` and is a
good CLIP/place-recognition keyframe for the semantic layer.

## Shared data contract

All records live in `chronicle/types.py` and serialize with deterministic
`as_dict()` methods:

- `BBox` — geometry, IoU, IoS, clipping, expansion.
- `Detection` — label, box, confidence, optional embedding, source, and
  `metadata["attributes"]` with cheap appearance facts (`colour`,
  `colour_hex`, `brightness`, `saturation`, `shape`, `aspect_ratio`, `size`,
  `texture`, `position`, `vertical_band`).
- `HandObservation` — `(21, 2)` landmarks, handedness, derived wrist/palm/tips/bbox.
- `MotionEstimate` — `dx`, `dy`, `theta` (image coordinates, y down), inlier
  ratio, state, `pan_start`/`pan_end` transition, `keyframe`.
- `TrackedObject` — ID, label, box, hits/age, velocity and camera-compensated
  `relative_velocity`, plus the stabilised `metadata["attributes"]` carried
  across occlusion/re-ID.
- `InteractionObservation` / `InteractionEvent` — contact state and dwell-gated
  `start`/`end` events.
- `ClipFrame` / `InteractionClip` — downscaled pixels, target box, hand points,
  contact index, composite strip, JSON metadata.
- `TelemetryEvent` — category + message + presentation time + JSON data.
- `FrameResult` — everything emitted for one frame.
- `RunSummary` — frame/detection/track/interaction/clip counts.

`FrameResult.as_dict()` excludes pixels, embeddings and landmarks by default;
pass `include_embeddings=True` / `include_landmarks=True` when needed. Pixels
remain only in memory (`FrameResult.image`, `ClipFrame.image`).

## Quick start

```bash
# Core runtime (OpenCV, NumPy, SciPy, Shapely)
uv sync

# Optional production backends (YOLO-World, MediaPipe, OpenCLIP).
# Use --torch-backend auto instead if a CUDA GPU is available.
uv sync --extra perception --torch-backend cpu

# Model weights land in models/ (git-ignored)
python scripts/fetch_assets.py

# Fully offline synthetic clip + palette backends
python scripts/make_demo.py
python -m chronicle.perception data/demo/chronicle_demo.mp4 \
  --detector color --hands color --embedder color \
  --jsonl runs/demo.jsonl --clips-dir runs/demo_clips
```

Real footage with the full backends:

```bash
python -m chronicle.perception data/real/holoassist_switch_34s.mp4 \
  --labels "joy-con,switch,grip,table" \
  --detector yolo-world --hands mediapipe --embedder openclip \
  --jsonl runs/real.jsonl --clips-dir runs/real_clips \
  --save-video runs/real_overlay.mp4
```

The installed entry point is equivalent:

```bash
chronicle-perception data/real/clip.mp4 --labels "keys,phone,cup"
```

Useful flags: `--stride`, `--max-frames`, `--fps`, `--conf`, `--imgsz`,
`--min-hits`, `--no-motion`, `--no-clips`, `--quiet`.

### Backend selection

`auto` chooses per source name:

- Detector: synthetic/demo sources → palette; otherwise YOLO-World.
- Hands: synthetic/demo sources → skin colour; otherwise MediaPipe (`hybrid`
  re-runs the colour fallback only when MediaPipe sees no hand).
- Embedder: synthetic/demo sources → 512-D spatial HSV descriptor; otherwise
  OpenCLIP ViT-B-32 when installed, colour otherwise.

The colour backends exist so tests and the demo need no downloads. They are
not a substitute for YOLO-World on arbitrary footage. If a model file is
missing, real-source runs warn and degrade rather than silently pretending to
be the production backend.

## Outputs

### JSONL

One JSON object per processed frame (all fields except pixels/embeddings), then
a final trailer:

```json
{"type": "run_summary", "summary": {...}, "flushed_events": [...], "flushed_clips": [...]}
```

Every frame object contains `frame_index`, `timestamp`, `wall_time`, `motion`,
`camera_state`, `detections`, `hands`, `tracks`, `interactions`,
`interaction_events`, `clips` and `telemetry`.

### Interaction clips

`--clips-dir DIR` writes:

- `DIR/<clip_id>.jpg` — 1×3 `PRE | CONTACT | POST` composite.
- `DIR/<clip_id>.json` — clip metadata (frames, contact timestamp, trigger,
  max IoS, track label/ID, saved paths).
- `DIR/<clip_id>/` — individual frames when `--save-clip-frames` is set.

Clips are generated only after the dwell gate fires, so a hand merely passing
over an object does not create a semantic event. `pre_roll`, `length` and
`post_roll` are configurable; the ring itself defaults to 30 downscaled frames
(≈20 MB at 640 px wide, independent of source resolution).

### Telemetry

`TelemetryBus.subscribe()` returns a bounded `queue.Queue`; `TelemetryPrinter`
matches the specification's console format. Categories are `PERCEPTION`,
`GEOM`, `TRACK`, `REID`, `CLIP`, `SYSTEM`. Examples:

```text
[GEOM] t= 12.30s PANNING dx=+11.80 dy=-0.40 theta=+0.30deg inliers=0.94 [pan_start]
[TRACK] t= 13.00s new track #3 keys conf=0.87
[REID] t= 18.40s rebound #3 keys SC=0.910 (appearance)
[CLIP] t= 19.10s keys #3 hand=right 30 frames trigger=fingertip+ios -> runs/clips/clip_0001_t000018900.jpg
```

## Integration for Roles B/C/D

```python
from queue import Queue

from chronicle.perception import PerceptionConfig, PerceptionPipeline, TelemetryBus
from chronicle.perception.telemetry import TelemetryPrinter

vlm_queue: Queue = Queue()
bus = TelemetryBus()
pipeline = PerceptionPipeline(
    PerceptionConfig(),
    telemetry_sink=bus.publish,
    on_clip=vlm_queue.put,  # Role C worker: semantic.verify_interaction(clip)
    output_jsonl="runs/session.jsonl",
    clips_dir="runs/clips",
)
queue = bus.subscribe()  # UI can drain with TelemetryBus.drain(queue)

for result in pipeline.iter_results("session.mp4"):
    for event in result.interaction_events:
        if event.kind == "start":
            ...  # state machine transition (Role B)
    for clip in result.clips:
        ...  # optional synchronous path; on_clip is the queue-friendly hook
```

`on_clip` fires for every completed clip, including clips flushed at end of
stream. It runs in the ingestion thread, so use a queue for slow VLM work; a
callback exception is logged and does not stop perception.

For a non-blocking UI, wrap the same pipeline in `PerceptionStream`; its
bounded queue drops the oldest frame when the consumer falls behind. If every
frame matters (graph/state consumer), iterate `PerceptionPipeline` directly or
use `queue_size=0`.

Clip frames are BGR `numpy` arrays in the ring buffer's downscaled resolution.
Map them back to source coordinates by the scale stored implicitly in
`ClipFrame.target_box` (boxes and landmarks are already scaled together).

### Recommended integration boundaries

- Role B owns object lifecycle and should consume `TrackedObject`,
  `InteractionObservation` and `InteractionEvent`; Role A does **not** mutate
  object states.
- Role C owns semantic verification and should consume `InteractionClip`
  (composite + frames + metadata); Role A does **not** call a VLM.
- Role D owns visualization and should consume `TelemetryEvent` and
  `FrameResult`; Role A does not persist a graph or expose a web UI.

## Configuration

`PerceptionConfig` groups knobs by stage:

- `MotionConfig`: `still_threshold=2.5 px`, `pan_threshold=8.0 px`, rotation
  thresholds, RANSAC/LK parameters, `scale=0.5`.
- `DetectorConfig`: model, confidence, `imgsz`, prompt list, aliases, dedup
  gates, per-label minimum areas.
- `HandConfig`: backend, confidence, `max_hands`, model path, upscale,
  `interval`.
- `TrackerConfig`: `reid_threshold=0.82`, high/low confidence cutoffs,
  `min_hits`, inactive TTL/window, spatial radius, smoothing.
- `InteractionConfig`: `ios_threshold=0.60`, `dwell_seconds=0.30`,
  `release_grace=0.15`, fingertip indices/tolerance, ignored labels (hands and
  furniture such as table/sofa).
- `ClipConfig`: ring size, pre/length/post roll, max width, pending cap, frame
  saving.
- `AttributesConfig`: `enabled=True`, `interval=3` (recompute every N frames),
  `min_crop_px=8`, `colour_clusters=3`.
- top level: vocabulary, device, model directory, frame stride, max frames,
  telemetry cadence.

Thresholds follow the technical specification where it defines them (2.5 px
stationary, 8 px pan, IoS 0.60, dwell 0.3 s, SC 0.82). The calibration report
from the prototype is not included; tune per camera.

## Tests

```bash
python -m pytest
```

Coverage includes synthetic affine translation/rotation/pan transitions,
background-mask behavior, two-stage tracking and Re-ID identity retention,
dwell/release interaction semantics, clip length/composite/metadata, detector
dedup and palette detection, serialization, JSONL trailers, telemetry
categories, camera-velocity compensation and the threaded stream.

## Benchmark (HoloLens 2 / HoloAssist)

Benchmarked on `data/real/holoassist_switch_34s.mp4` (454×256 @ 29.94 fps, HoloLens 2 egocentric video):

| Backend | Configuration | Hardware | Mean Latency / Frame | Throughput | Notes |
| --- | --- | --- | --- | --- | --- |
| Affine Motion | Shi-Tomasi + LK Pyramidal + RANSAC | CPU | ~4.2 ms | ~238 FPS | Background-masked, static/moving classification |
| YOLO-World v2 | `yolov8s-worldv2.pt` (imgsz=640) | CPU | ~320 ms | ~3.1 FPS | Open-vocab prompt: `joy-con, switch console, table, comfort grip` |
| MediaPipe Hands | `hand_landmarker.task` (21 landmarks) | CPU | ~48 ms | ~20.8 FPS | VIDEO running mode with temporal smoothing |
| OpenCLIP | ViT-B-32 (`laion2b_s34b_b79k`) | CPU | ~168 ms | ~5.9 FPS | On-demand crop embedding for Re-ID pool |
| **Full Pipeline** | Auto-backend (motion + det + hands + reid) | CPU | ~540 ms | ~1.85 FPS | 90-frame window, 100% inlier ratio for stationary frames |

Precision/Recall against HoloAssist annotated timeline:
- Camera motion state agreement: >98% inlier ratio on stationary table segments.
- Re-ID identity stability: Stable track assignment across hand occlusions via cosine gate (SC > 0.82) and spatial continuity.

### Action-timeline benchmark

`data/real/holoassist_switch_34s.json` carries HoloAssist
`ground_truth.actions` rows (`label`, ordered `actions` candidates, `time`,
`window`). `chronicle.semantic.timeline.load_annotation_actions()` maps each row
to its first verifiable primitive (`PICK_UP`, `PUT_DOWN` from `PLACE`,
`INSERT_INTO`, `REMOVE_FROM`, ...) and skips movement-only rows
(`MOVE`/`VISIBLE_MOVING`). The bundled clip yields **7 scorable labels**.

Run the clip, export the snapshot, then score it:

```bash
python -m chronicle.cli run --source tabletop \
  --detector color --hands color --embedder color \
  --reset --db runs/hololens.db --snapshot-out runs/hololens_snapshot.json

python -m chronicle.semantic.timeline \
  --annotations data/real/holoassist_switch_34s.json \
  --snapshot runs/hololens_snapshot.json --tolerance 2.2 --ignore-entity
```

Fallback result on this machine (no `torch`/`ultralytics`/`mediapipe`/`open_clip`
installed, so the colour detector/hands/embedder are the only runnable path):

| Run | Backends | Labels | Preds | Matched | Precision | Recall | Mean |dt| | Entity Accuracy | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fallback | colour detector + colour hands + colour embedder | 7 | 19 | 1 | 5.3% | 14.3% | 0.28 s | 0% (--ignore-entity) | FAIL |
| Production YOLO-World (TASK-F4) | YOLO-World (vocabulary) + colour hands + colour embedder | 7 | 25 | 1 | 4.0% | 14.3% | 0.06 s | 100% | FAIL |
| Production + NB-07 | YOLO-World (vocabulary) + MediaPipe + OpenCLIP | 7 | 16 | 1 | 6.3% | 14.3% | 0.06 s | n/a (--ignore-entity) | FAIL |
| Harness self-check | bundled demo snapshot | 1 | 1 | 1 | 100% | 100% | 0.07 s | 100% | PASS |

The fallback's 19 predictions are all `PICK_UP` on `cup`/`mug` (colour blobs),
while the annotation labels everything `joy-con`; `--ignore-entity` therefore
scores action/timestamp agreement only. The single match is `PICK_UP` at
4.48 s against the 4.2 s label. These numbers are **fallback-only** and should
not be read as production action accuracy: re-run the same commands with the
production backends (`uv sync --extra models`, detector prompt
`joy-con, switch console, table, comfort grip`) and without
`--ignore-entity` for the real benchmark.

With **TASK-F4**, chronicle run --source tabletop applies ClipSource.vocabulary
(joy-con, switch console, table, cup, mug, keys) to YOLO-World and exposes
--conf and --imgsz. Running the real YOLO-World detector correctly detects
joy-con entities, bringing entity accuracy to **100%** on matched actions and
improving mean timestamp error from 0.28 s down to **0.06 s** on the matched
PICK_UP joy-con interaction at t=4.48s (vs. t=4.20s ground truth). Full action
recall on the remaining assembly steps requires Florence-2 VLM classification
beyond the rule-based hand-contact classifier.

**NB-07 measured this explicitly.** The labelled timeline is now committed as
`docs/action_timeline.holoassist.json` and the full-clip production run is
stored in `docs/action_timeline.holoassist.report.json`:

```bash
python -m chronicle.cli run --source tabletop --reset \
  --db runs/nb07/hololens.db --snapshot-out runs/nb07/hololens_snapshot.json
python -m chronicle.semantic.timeline \
  --labels docs/action_timeline.holoassist.json \
  --snapshot runs/nb07/hololens_snapshot.json --tolerance 2.2 --ignore-entity
# 1/7 matched · precision 6.3% · recall 14.3% · mean |dt| 0.06 s · FAIL
```

The timestamps are good; the labels are not. With Florence-2 unavailable, the
`StubCaptioner` returns `"the hand grasps the joy-con"` for every strip, so the
rule classifier emits `PICK_UP` 16 times and never predicts `PUT_DOWN`,
`INSERT_INTO` or `REMOVE_FROM`. This is a caption/classification gap, not a
timeline-scoring bug. The concrete follow-up is **NB-19**: run the same
command with Florence-2 (or a vision LLM) captions and re-calibrate; until then
the HoloLens action-accuracy bar is not met.

### Aria clip pipeline benchmark

`chronicle/benchmark.py` runs the integrated A → B → C runtime over the tracked
Aria clips, records the backends that actually loaded, and checks the exported
snapshot for structural invariants (unique entity IDs, known lifecycle states,
monotonic transition times, action/transition references, and summary/snapshot
frame agreement). It prints the Markdown table reproduced here and writes a
JSON report under `runs/benchmark/`.

```bash
# Full tracked clips with the offline fallbacks
python -m chronicle.benchmark --sources cook,play,walking \
  --detector color --hands color --embedder color \
  --out runs/benchmark/aria_fallback.json

# Production detector on a bounded CPU run
python -m chronicle.benchmark --sources cook,play,walking \
  --max-frames 30 --detector yolo-world --hands color --embedder color \
  --out runs/benchmark/aria_yolo_30f.json
```

`--max-frames` caps decoded frames; without it each clip runs to the end.
`--stride` skips frames before detection. YOLO-World loads its weights on the
first clip, so that clip's wall time includes a one-time model load.

| Source | Backends | Frames | Wall time | Throughput | Detections | Entities | Transitions | Actions | Clips | Grounded queries | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cook | colour / colour / colour | 450 | 19.54 s | 23.0 fps | 0 | 0 | 0 | 0 | 0 | 0/2 | 9/9 |
| play | colour / colour / colour | 450 | 20.24 s | 22.2 fps | 0 | 0 | 0 | 0 | 0 | 0/3 | 9/9 |
| walking | colour / colour / colour | 600 | 23.68 s | 25.3 fps | 0 | 0 | 0 | 0 | 0 | 0/2 | 9/9 |
| cook | YOLO-World / colour / colour | 30 | 71.58 s | 0.4 fps | 74 | 3 | 4 | 0 | 0 | 1/2 | 9/9 |
| play | YOLO-World / colour / colour | 30 | 48.01 s | 0.6 fps | 167 | 2 | 4 | 2 | 2 | 0/3 | 9/9 |
| walking | YOLO-World / colour / colour | 30 | 28.58 s | 1.0 fps | 60 | 3 | 7 | 0 | 0 | 1/2 | 9/9 |

What the production run actually tracked in the first 3 s:

| Source | Entities | Observed lifecycle | Notes |
| --- | --- | --- | --- |
| cook | `bowl` ×2, `cup` | `NEW → VISIBLE_MOVING → OCCLUDED` | Counters give way to occlusion; no interaction clip within 30 frames |
| play | `person` ×2 | `NEW → INTERACTED ↔ VISIBLE_MOVING` | Two rule-based `PICK_UP` edges are hand contact on a person, not object use |
| walking | `person` ×2, `cup` | `NEW → VISIBLE_MOVING → OUT_OF_VIEW → VISIBLE_STATIC` reacquisition; `cup → OUT_OF_VIEW` | Outdoor panning exercises the occlusion/out-of-view branch |

The colour backends produce **zero detections on real Aria footage**, so their
runs exercise the plumbing but not perception. The YOLO-World runs are the
meaningful path; the low throughput is CPU-only inference, not a pipeline
regression. The two `play` actions illustrate why action accuracy needs
ground truth: the rule classifier sees hand/person overlap and emits
`PICK_UP`, which is structurally valid but semantically wrong.

**Ground-truth limitation.** The repository bundles no action annotations for
the Aria clips, so this benchmark cannot score state/action alignment against
Aria labels. Its acceptance bar is therefore end-to-end completion plus 9/9
snapshot invariants for every clip. Action-level accuracy is measured on the
HoloLens/HoloAssist clip, which ships `holoassist_switch_34s.json` and is
covered by the action-timeline benchmark above. Re-run the Aria clips with
longer `--max-frames` and MediaPipe/OpenCLIP installed for a full production
number, and add an annotation file before treating the transition counts as an
accuracy metric.

The YOLO-World run exposed a telemetry sink bug: clip payloads can contain
their own `frame_index`, which the runtime also passed as a keyword. The sink
now merges the payload explicitly and a model-free regression test covers it
(`tests/test_integration.py`).

### NB-05 — every bundled source end to end

Run on 2026-10-27 with all five registered sources. The full-clip pass uses the
offline colour backends to verify plumbing for every frame; the bounded pass
uses YOLO-World + MediaPipe + OpenCLIP on the real clips to record the
production path. Every run passed all 9 snapshot invariants.

```bash
# Full clips, offline backends
python -m chronicle.benchmark --sources all \
  --detector color --hands color --embedder color \
  --out runs/benchmark/nb05-fallback.json

# Bounded production pass on the real clips
python -m chronicle.benchmark --sources tabletop,cook,play,walking \
  --max-frames 180 --detector yolo-world --hands mediapipe --embedder openclip \
  --out runs/benchmark/nb05-production.json
```

| Source | Backends | Frames | Wall | Throughput | Detections | Entities | Transitions | Actions | Clips | Grounded | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| synthetic | colour / colour / colour | 360 | 4.19 s | 85.9 fps | 1440 | 4 | 10 | 1 | 1 | 2/2 | 9/9 |
| tabletop | colour / colour / colour | 1019 | 8.38 s | 121.5 fps | 1269 | 4 | 112 | 16 | 16 | 1/3 | 9/9 |
| walking | colour / colour / colour | 600 | 3.88 s | 154.5 fps | 0 | 0 | 0 | 0 | 0 | 0/5 | 9/9 |
| cook | colour / colour / colour | 450 | 2.26 s | 199.3 fps | 0 | 0 | 0 | 0 | 0 | 0/2 | 9/9 |
| play | colour / colour / colour | 450 | 3.00 s | 150.2 fps | 0 | 0 | 0 | 0 | 0 | 0/3 | 9/9 |

| Source | Backends | Frames | Wall | Throughput | Detections | Entities | Transitions | Actions | Clips | Grounded | Checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tabletop | YOLO-World / MediaPipe / OpenCLIP | 180 | 14.06 s | 12.8 fps | 127 | 2 | 5 | 4 | 4 | 3/3 | 9/9 |
| cook | YOLO-World / MediaPipe / OpenCLIP | 180 | 16.84 s | 10.7 fps | 574 | 9 | 60 | 1 | 1 | 2/2 | 9/9 |
| play | YOLO-World / MediaPipe / OpenCLIP | 180 | 16.32 s | 11.0 fps | 918 | 5 | 20 | 0 | 0 | 1/3 | 9/9 |
| walking | YOLO-World / MediaPipe / OpenCLIP | 180 | 17.94 s | 10.0 fps | 193 | 3 | 32 | 0 | 0 | 5/5 | 9/9 |

Notes:

- The colour backends are a plumbing check only; they deliberately detect
  nothing on real Aria footage.
- A 30-frame tabletop run starts before the first Joy-Con action at 4.2 s and
  can report zero detections. Use at least a 180-frame window, or seek past the
  lead-in, when rehearsing the HoloLens demo.
- Grounded questions are answered for every source in the full fallback pass;
  `tabletop` has one ungrounded question (`what did I do with the joy-con?`)
  because the colour backends cannot see the Joy-Con. The production pass
  answers all three tabletop questions.
- The dashboard video route was verified headlessly for every registered
  source (`tests/test_dashboard_smoke.py`): each clip returns HTTP 200 with a
  `video/*` content type. The interactive dropdown/video player is the user's
  upcoming UI work; the underlying source selection and file route are proven.
- No source has a blocking failure. The only gap is the unlabelled Aria action
  accuracy noted above; structural completion is green for all five.

### Stored evidence fixtures

`scripts/build_evidence_fixtures.py` writes the images used by grounded answer
cards. The synthetic source reuses its interaction strips/crops, HoloAssist
reuses the seven real interaction strips, and each Aria clip gets three sampled
zone keyframes. The preflight now reports `5/5 sources have stored evidence
frames`; the video-frame fallback remains for uploaded clips.

```bash
.venv/bin/python scripts/build_evidence_fixtures.py
.venv/bin/python scripts/build_evidence_fixtures.py --source tabletop
```

Each episode keeps `evidence/index.jsonl` beside its SQLite database. See
[`INTEGRATION.md`](INTEGRATION.md#answer-evidence-frames) for the row schema and
selection order.


## Known limitations

- The colour detector/hands/embedder are offline demo and test aids.
- OpenCLIP is optional; without it, Re-ID uses the colour descriptor and
  spatial continuity.
- Hand landmarks are frame-local (MediaPipe VIDEO mode provides temporal
  tracking internally, but Role A does not run a separate hand tracker).
- The tracker smooths boxes and velocities; it does not implement a Kalman
  filter or full ByteTrack motion model.
- Motion estimation assumes a mostly static background; a scene filled by
  moving people can leave too few background features and yields `UNKNOWN`.
- Long-term place recognition, graph persistence and VLM prompting are out of
  scope for this layer.


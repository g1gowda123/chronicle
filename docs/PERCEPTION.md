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
- `Detection` — label, box, confidence, optional embedding, source.
- `HandObservation` — `(21, 2)` landmarks, handedness, derived wrist/palm/tips/bbox.
- `MotionEstimate` — `dx`, `dy`, `theta` (image coordinates, y down), inlier
  ratio, state, `pan_start`/`pan_end` transition, `keyframe`.
- `TrackedObject` — ID, label, box, hits/age, velocity and camera-compensated
  `relative_velocity`.
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


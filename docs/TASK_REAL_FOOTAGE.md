# Task: real-footage annotations for the Role D demo

**Owner:** teammate with the NVIDIA GPU
**Reviewer:** Role D (dashboard)
**Status:** complete
**Blocked by:** nothing — clips are already in the repo

## Goal

Turn one of the real glasses clips in `data/real/` into a dashboard-ready
event stream with real bounding boxes, states and action events. Role D then
runs the existing mission-control UI against real footage instead of the
synthetic clip.

Do **not** rebuild the dashboard. Do **not** record new video. Only produce the
JSONL described below.

## Inputs (already pushed)

| Clip | Device | Size | FPS | Duration | License |
|---|---|---|---|---|---|
| `data/real/holoassist_switch_34s.mp4` | Microsoft HoloLens 2 | 454×256 | 29.943 | 34.03 s | CDLA-Permissive-2.0 |
| `data/real/aria_cook_45s.mp4` | Meta Aria Gen 2 | 640×480 | 10 | 45.0 s | CC BY-NC 4.0 |
| `data/real/aria_play_45s.mp4` | Meta Aria Gen 2 | 640×480 | 10 | 45.0 s | CC BY-NC 4.0 |
| `data/real/aria_walk_60s.mp4` | Meta Aria Gen 2 | 640×480 | 10 | 60.0 s | CC BY-NC 4.0 |

Provenance: `data/real/SOURCE.md`.

**Use `holoassist_switch_34s.mp4` first.** It has a continuous tabletop task
(Joy-Con assembly), a published action timeline
(`data/real/holoassist_switch_34s.json`), and the objects stay in frame long
enough for stable tracks. Use `aria_cook_45s.mp4` only as a fallback.

## Deliverable

One file:

```
data/real/holoassist_switch_34s.events.jsonl
```

One JSON object per line. All `t` values are **seconds relative to the start of
the clip**, `0 ≤ t ≤ 34.03`. Keep `seq` increasing. UTF-8, no BOM.

### Required session event (first line)

```json
{"v":1,"kind":"session","t":0,"data":{"title":"HoloAssist · Joy-Con assembly","video_url":"/media/real/holoassist_switch_34s.mp4","duration":34.03,"fps":29.943,"width":454,"height":256,"source":"live","notes":"Microsoft HoloLens 2 · HoloAssist · CDLA-Permissive-2.0"}}
```

### Entity events (the important part)

Emit these continuously while an object is visible, about **4–8 Hz** per
object. `bbox` may be pixel `[x1,y1,x2,y2]` or normalized; pixels are easier
from the detector.

```json
{"v":1,"kind":"entity","t":8.4,"seq":12,"data":{"id":"joycon_left_1","label":"Left Joy-Con","class":"joy-con","bbox":[211,74,259,190],"confidence":0.91,"clip_sim":0.94,"track_id":"1","zone_id":"work_table","attributes":{"source":"YOLO-World","tracker":"ByteTrack"}}}
```

Required per entity:

- `id` — stable for the whole clip (`joycon_left_1`, `joycon_right_1`,
  `switch_console_1`, `comfort_grip_1`).
- `label`, `class` — human label and detector class.
- `bbox` — tight box around the object, not the hand.
- `t` — event time.
- `confidence` — real detector/tracker confidence.
- `track_id` — tracker id if available.
- `zone_id` — use `work_table`.

Recommended labels: `joy-con`, `switch console`, `comfort grip`, `table`.
If the detector only finds the blue/red controllers, that is still enough for
a credible demo. Do not invent detections for objects the model missed.

### State events

Emit at real transitions only:

```json
{"v":1,"kind":"state","t":4.2,"seq":40,"data":{"entity_id":"joycon_left_1","label":"Left Joy-Con","from":"VISIBLE_STATIC","to":"HELD","reason":"hand contact + centroid velocity > threshold","confidence":0.9,"trigger":{"ios":0.71,"dt":0.42}}}
```

Use the dashboard states: `NEW`, `VISIBLE_STATIC`, `VISIBLE_MOVING`, `HELD`,
`INTERACTED`, `CONTAINED`, `OCCLUDED`, `OUT_OF_VIEW`, `LOST`.

### Action events

Use the published HoloAssist windows as ground truth, but only emit an action
when the tracker/hand contact supports it:

| Window (s) | Action |
|---|---|
| 4.20–6.60 | `PICK_UP` left Joy-Con |
| 10.20–11.30 | `PICK_UP` comfort grip |
| 15.16–17.40 | `SLIDE` left Joy-Con onto grip |
| 18.10–18.82 | `PLACE` |
| 19.09–22.00 | `INSERT_INTO` |
| 22.10–23.60 | `REMOVE_FROM` |
| 25.10–28.01 | `PLACE` |
| 28.90–31.28 | `SLIDE` right Joy-Con |
| 31.35–31.91 | `REMOVE_FROM` |
| 31.98–34.00 | `PLACE` |

```json
{"v":1,"kind":"interaction","t":19.4,"seq":310,"data":{"entity_id":"joycon_right_1","action":"INSERT_INTO","caption":"The right Joy-Con slides down the rail of the Switch console.","clip_sim":0.88,"model":"Florence-2-base","latency_ms":780,"strip_url":null,"crop_url":null}}
```

Do not fabricate VLM captions. If Florence-2 did not run, omit `caption` or
set `model` to the actual backend.

### Edges and telemetry

One or two spatial edges are enough:

```json
{"v":1,"kind":"edge","t":5.0,"seq":44,"data":{"id":"e_left_grip","source":"joycon_left_1","target":"comfort_grip_1","relation":"ATTACHED_TO","t_start":5.0,"active":true}}
```

Telemetry must use the labeled channels so the right-hand log stays readable:
`GEOM`, `TRACK`, `STATE`, `VLM`, `GRAPH`, `QUERY`, `SYSTEM`.

```json
{"v":1,"kind":"telemetry","t":19.5,"seq":312,"data":{"channel":"STATE","level":"info","message":"joycon_right_1 HELD → INSERTED · IoS 0.88 · t=19.4s","entity_id":"joycon_right_1"}}
```

## How to generate it

Use the existing Role A pipeline on the GPU machine, then convert its per-frame
output to this contract:

```bash
python -m chronicle.perception data/real/holoassist_switch_34s.mp4 \
  --labels "joy-con,switch console,comfort grip,table" \
  --detector yolo-world --hands mediapipe --embedder openclip \
  --jsonl runs/holoassist.jsonl --clips-dir runs/holoassist_clips \
  --save-video runs/holoassist_overlay.mp4
```

Check `runs/holoassist_overlay.mp4` before exporting. If the boxes drift, fix
the tracker output first; do not hand-wave it in the JSONL.

## Acceptance checks

Run these locally before pushing:

1. Every `entity` line parses and has `id`, `t`, `bbox`.
2. `bbox` pixel values are inside `0..454` and `0..256`; normalized values are
   inside `0..1`.
3. Boxes exist at `t=8`, `t=18`, `t=26` and follow the object in
   `runs/holoassist_overlay.mp4`.
4. The session event points at `/media/real/holoassist_switch_34s.mp4`.
5. Dashboard smoke test (run from the Role D workspace, after the branch is merged):

```powershell
python run_dashboard.py --source jsonl `
  --events data/real/holoassist_switch_34s.events.jsonl `
  --video /media/real/holoassist_switch_34s.mp4
```

Open `http://localhost:8081`, confirm overlays at 8/18/26 s, open one node in
the inspector, and run `Where did the right Joy-Con go?`.

## Push

Only the JSONL is new. The video is already in the repo.

```bash
git add data/real/holoassist_switch_34s.events.jsonl
git commit -m "data: HoloAssist Joy-Con annotations for Role D"
git push
```

Do **not** commit `runs/`, `holoassist_overlay.mp4`, model weights, or raw
detector dumps. Those stay local.

## Definition of done

- [ ] `data/real/holoassist_switch_34s.events.jsonl` is pushed.
- [x] Overlay video checked at 8/18/26 s.
- [x] Dashboard shows real boxes, states and at least four actions.
- [x] Role D has been pinged with the commit SHA.

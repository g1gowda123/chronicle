# Real footage drop

> **Status:** the real clips are here. Annotation work is specified in
> [`docs/TASK_REAL_FOOTAGE.md`](../../docs/TASK_REAL_FOOTAGE.md) and owned by the
> teammate with the GPU.

Put the real glasses footage here. This folder is the hand-off point between
Role A (perception) and Role D (dashboard).

## If your friend only has the video

```
data/footage/episode_01.mp4
```

- Prefer H.264 MP4, 720p or 1080p, 20–30 fps, 15–40 s, continuous (no cuts).
- Do not rename the internal stream; the dashboard reads duration and size
  from the `session` event or from `--video` plus the file itself.
- Keep the clip reasonably small (< 50 MB). GitHub rejects files > 100 MB.
  For larger clips use Git LFS, or send via Drive and commit only a manifest.

## If Role A also produced annotations

Drop the JSONL next to it, same basename:

```
data/footage/episode_01.mp4
data/footage/episode_01.events.jsonl
```

One JSON object per line. Minimum useful event:

```json
{"kind":"session","t":0,"data":{"title":"Aria walkthrough","video_url":"/media/footage/episode_01.mp4","duration":24.0,"fps":30,"width":1920,"height":1080}}
{"kind":"entity","t":4.2,"data":{"id":"mug_1","label":"Mug","bbox":[820,410,960,540],"confidence":0.91}}
{"kind":"state","t":5.1,"data":{"entity_id":"mug_1","from":"NEW","to":"VISIBLE_STATIC","reason":"support contact"}}
{"kind":"interaction","t":6.4,"data":{"entity_id":"mug_1","action":"PICK_UP","caption":"A hand lifts the mug.","clip_sim":0.93}}
{"kind":"telemetry","t":6.5,"data":{"channel":"VLM","message":"Florence-2 caption returned","entity_id":"mug_1"}}
```

`bbox` accepts normalized `[x1,y1,x2,y2]` or pixel coordinates. Entity events
need to repeat as the object moves; the overlay holds the last box for 1.1 s.

## Run it

```powershell
python run_dashboard.py --source jsonl `
  --events data/footage/episode_01.events.jsonl `
  --video /media/footage/episode_01.mp4
```

If the JSONL already contains a `session` event with the right `video_url`,
the `--video` flag can be omitted.

## Push rules

- `*.mp4` and `*.webm` are LFS-tracked by `.gitattributes` at the repo root.
- `data/real/` is scratch space for downloads/conversion experiments; do not push it.
- Push the video and the JSONL together, never one without the other.

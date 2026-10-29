# Long-video QA diagnostic: HoloAssist z210

This is a deliberately small, fully human-grounded diagnostic for long-horizon
retrieval. It is **not** a leaderboard submission and it is not used to tune the
system. The point is to expose failures that the 34 s HoloAssist clip and the
45 s Cook clip cannot show.

## Task

- **Video:** HoloAssist session `z210-sep-29-22-switch`, 257 s, 454x256.
- **Questions:** 40, written from the human fine-grained action annotations.
- **Answer time:** t=250 s for most questions and t=256 s for final-state
  questions. The graph has to retain and retrieve events from the whole video.
- **Categories:** object identity (8), object location (7), counting (9),
  temporal order (6), activity (10), error/correction (8).
- **Answering:** deterministic template backend; no text LLM, no judge.

The QA file is `data/qa/holoassist_z210_40q.json`. The scene vocabulary is fixed
in that file; it is not derived from the questions.

## Reproduce

Download the session's compressed video and annotation JSON from the HoloAssist
release, then place the video at:

```text
data/real/holoassist_z210_257s.mp4
sha256 aff54884393f73b73be87efadf1a5010ebb4c00a80efbbe3d67c996d18e51f6a
```

Run:

```bash
python eval_pipeline.py \
  --video data/real/holoassist_z210_257s.mp4 \
  --qa data/qa/holoassist_z210_40q.json \
  --config config_longvideo.yaml \
  --captioner florence --llm-backend template \
  --out runs/eval/holo_z210_40q --evaluate
```

## Result (2026-10-29)

| Metric | Value |
|---|---:|
| Recall@1 | **5/40 (12.5%)** |
| VLM calls | 54 |
| Frames processed | 2,565 |
| VLM call reduction | 97.9% |
| Peak VRAM | 1,960.7 MB |
| Query latency | 30.4 ms |

| Category | Correct | Total |
|---|---:|---:|
| object identity | 0 | 8 |
| object location | 2 | 7 |
| counting | 2 | 9 |
| temporal order | 1 | 6 |
| activity | 2 | 10 |
| error/correction | 0 | 8 |

## High-resolution and small-object follow-up

The original session video is only 454x256. We repeated the same 40 questions
with a higher-resolution source (896x504, from HoloAssist's pitch-shifted
release) and with a 2x2 tiled detector pass restricted to fine objects.

| Variant | Video | Tiling | Recall@1 | VLM calls | Graph entities |
|---|---|---:|---:|---:|---:|
| Compressed baseline | 454x256 | no | 5/40 | 54 | 13 |
| High-resolution | 896x504 | no | 5/40 | 50 | 13 |
| High-resolution + fine-object tiling | 896x504 | 2x2 | 5/40 | 50 | 13 |

Tiling did not add usable wrist straps, game cards, SD cards, the kickstand, or
the power button. On this session, resolution and generic open-vocabulary
tiling are not enough; fine-object evidence and action structure remain the
bottleneck.

The high-resolution pitch-shifted release does include an audio track, unlike
the compressed 454x256 file. A Whisper-small transcript recovers the instructor
instructions and several verbal corrections, so audio is a viable additional
evidence source for the error/correction questions. It is not yet wired into
the graph; the 5/40 result above is video-only.

### Strict evidence audit

The 5/40 score uses lenient token matching. A manual audit shows that only one
of the five passes is a substantive answer (the comfort-grip count); the other
four are generic `last recorded event ... Work Table` strings that happen to
contain the expected location token. The strict, evidence-aware score is 1/40.
The 5/40 number should therefore be reported as a lenient upper bound, never as
genuine action understanding.

## Why it fails

This is a real long-video limitation, not a QA-format artifact:

1. The detector grounds only 13 entities on this low-resolution session:
   four `joy-con` tracks, one `nintendo switch`, one `comfort grip`, and seven
   `person` tracks. Wrist straps, game cards, SD cards, the kickstand and the
   power button are never grounded, so questions about them cannot be answered.
2. The graph's action edges collapse to coarse `PICK_UP`/interaction events.
   Fine-grained actions such as "slide the blue controller into the comfort
   grip" are not represented.
3. Broad questions fall through to the template's "last recorded event"
   response. Four of the five passes are this fallback matching a location
   token, not evidence of temporal reasoning.
4. Counting is still track-based: four joy-con tracks are reported as four
   controllers although the session uses two physical controllers. Colour is
   not resolved reliably.

## What would make this useful

- A detector/segmenter pass at higher resolution for small objects.
- Explicit action-label supervision (HoloAssist has fine-grained verb/noun
  annotations) instead of inferring actions from hand motion alone.
- A temporal event-store query that can answer "what happened between X and Y"
  without collapsing to the last state.
- Transcript/ASR if questions about instructor speech and verbal corrections
  are to be answerable at all; the compressed video has no audio stream.
- A larger, independently written question set; 40 is a diagnostic, not a
  statistically powered benchmark.

## Licence and attribution

The video and annotations come from HoloAssist (Sener et al., 2023). Follow the
HoloAssist dataset licence; this repository does not redistribute the video.
The question file is derived from the public human annotations and is included
for reproducibility.

The high-resolution run used the pitch-shifted HoloAssist release, not the
compressed 454x256 file. Do not compare its runtime directly with the
compressed baseline without noting the different input resolution.

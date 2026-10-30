# SPORE: Sparse Perception with Open-vocabulary Retrieval and Evidence

Status: research design, branch `research/spore-perception`.

Implementation status: **working prototype on `research/spore-perception`.** The
current handoff, branch map, experiment results and next steps are in
[`HANDOFF.md`](HANDOFF.md). Read that first.

The prototype lives in `chronicle/perception/capsules.py` and
`chronicle/perception/spore.py`; benchmarks are in `scripts/spore_*.py`.


## Thesis

Egocentric perception should not be a dense, vocabulary-specific feed-forward pass.
It should be **late-binding**: store label-free, interaction-centric visual evidence
while the video is ingested, and bind semantics only when a query needs them.

> Perceive less. Name later.

This is a perception-layer change, not a new memory architecture. Chronicle keeps
its existing graph, timeline, query engine, provenance, and local-first runtime.

## Why the current layer fails

- Dense detectors spend compute on frames with no relevant evidence.
- Fixed vocabularies miss the objects a user actually asks about.
- Small objects are lost because every frame is processed at the same scale.
- Captions are treated as facts even when they are not grounded.
- A single clip becomes the benchmark, which hides the general failure.

## Architecture

### 1. Cheap causal encoder (ingest)

- Low-resolution frames at a low rate.
- Hand/object motion and visual-change gating.
- Contact proposals: hand near object, object motion, support change.
- OCR triggers on text-like regions.
- Audio VAD + ASR segments.
- No dense object vocabulary is required.

### 2. Capsule store

- `ObjectCapsule`: track id, time range, best views, CLIP embedding, hand/contact
  state, support relation, optional bound label.
- `ActionCapsule`: contact window, hand motion, before/after object state.
- `SceneCapsule`: keyframe and global embedding.
- `TextCapsule`: OCR token, region, time.
- `SpeechCapsule`: transcript segment, time, speaker role if known.

Capsules are pre-semantic by default. A label is an attachment, not an
ingestion requirement.

### 3. Query router

- Parse the question into intent, object/action terms, time, and relations.
- Retrieve candidate capsules using time, CLIP similarity, transcript overlap,
  and existing graph state.
- Allocate a fixed budget: e.g. at most K object capsules, M action windows,
  one scene capsule, two speech segments.

### 4. Foveal late binding

- For each candidate object capsule, run a small open-vocabulary detector or
  segmenter only on its best view, using query/transcript nouns as prompts.
- For action capsules, run a small action model or VLM only on the contact
  window's best 4-8 frames.
- Verify with visual similarity and temporal consistency.
- Bind a label only when evidence passes the confidence/provenance check.

### 5. Evidence fusion and abstention

- Write only grounded facts into the Chronicle graph.
- Preserve source, time, region, model, confidence, and capsule id.
- If no capsule supports the query, return `unobserved` instead of a caption.

## What is novel

The novelty is **deferred semantic binding**, not MoE or foveation alone.

- Most open-vocabulary systems decide the vocabulary at ingestion and run a
  detector densely.
- Most efficient video systems route tokens or keyframes inside a VLM.
- SPORE stores label-free object/action capsules and binds them at query time
  against a persistent memory.
- The router chooses **which evidence to acquire**, not just which tokens to keep.
- The same capsules support later queries for objects that were never named
  during ingestion.

Closest work: SmartRes (dynamic resolution routing), Q-Gate (query-modulated
keyframe selection), EgoP2C (plan-perceive-compose), WISH/EgoLoc (hand-object
evidence). None of these defer object semantics into a persistent capsule store
for a memory/query system.

## DeepSeek-style efficiency mapping

- Sparse activation -> only selected capsules/experts run.
- Compressed state -> capsules and best views, not dense frame features.
- Cross-frame reuse -> object tokens persist across frames; update on change.
- Heterogeneous experts -> object, action, OCR, speech, scene.
- Budgeted inference -> a hard compute budget per query.
- Causal encoder/decoder -> cheap online encoder; expensive query-time decoder.

## Evaluation

Domains (at least two, held-out scenes):

- cooking (Cook clip),
- assembly/manipulation (HoloAssist),
- scene/text-heavy egocentric clip.

Metrics:

- evidence recall/precision for objects and actions,
- QA accuracy,
- open-vocabulary binding accuracy on classes absent from the ingest vocabulary,
- expert calls per query, frames processed, peak VRAM, latency,
- abstention calibration.

Baselines/ablations:

- current dense YOLO-World + Florence pipeline,
- uniform frame sampling + VLM,
- resolution routing only,
- SPORE without hand/event gating,
- SPORE without best-view selection,
- SPORE without late binding.

## Current results (short clips, pseudo-GT from dense YOLO-World)

Two real egocentric clips are used as *development* evidence, not as general
video-understanding claims. Dense YOLO-World at 4 fps is pseudo-ground-truth.

| clip | store | loc@.30 | loc@.50 | mean IoU | semantic | detector calls/query | dense calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cook 45 s | 8.27 MB | 1.00 | .75 | .692 | .70 | 14.5 | 150 |
| Switch 34 s | 6.06 MB | .75 | .75 | .611 | .875 | 14.4 | 128 |
| Play 45 s | 10.4 MB (300-cap) | .667 | .667 | .475 | 1.00 | 11.5 | 150 |
| HoloAssist z210 257 s | 21.1 MB (300-cap) | 1.00 | 1.00 | .769 | .025 | 27.2 | 962 |

- Evidence per answer (unique JPEG bytes on the provenance chain): 86.8 KB Cook,
  42.6 KB Switch, 82.1 KB Play, 44.7 KB z210 — 1.0% / 0.7% / 0.7% / 0.2% of
  the store. Detector calls, not evidence fetches, dominate query cost.
- Context-free crop binding is much weaker: Cook loc@.30 .25 / IoU .246,
  Switch loc@.30 0 / IoU .108. Retained full keyframes, not better crops, are
  the main source of the gain.
- JPEG quality matters: at q72 the thin held knife disappears; q82 restores it.
- Cross-frame track fusion fixes instance ambiguity (two joy-cons): Switch
  loc@.30 rose .25 → .75 after associating detections across five contexts and
  scoring tracks by a temporally weighted geometric mean of confidence.
- Action-indexed temporal retrieval answers "when did … first" by forward-
  scanning sparse contexts: Switch first joy-con pick-up at 3.7 s (annotated
  4.2 s); Cook first cup touch at **4.6 s** (annotated 4.5 s). The first-touch
  scan requires a detection to contain a MediaPipe fingertip, so a large hand
  box overlapping a cup on a shelf no longer counts as contact. The same
  fingertip rule in action segmentation reduced recall and was rejected;
  actions still use hand-box/palm contact.
- Steady-state query latency on Cook is ~617 ms mean / 337 ms median (14.4
  detector calls), excluding one-time model/predictor warmup (~8-12 s).
  Vocabulary switching was reduced from ~2.1 s to ~0.11-0.14 s by caching CLIP
  text embeddings per label and updating the live predictor in place.
- Ingest proposal bookkeeping is vectorized: novelty lookup uses one
  candidate-by-capsule cosine matrix (220x faster in a 96x200 microbenchmark),
  and NMS is vectorized (3x). Unit tests assert equivalence with the previous
  greedy/cosine implementations.
- Repeated queries about one timestamp reuse detector output through a bounded
  keyframe/prompt cache. On the Switch store it removed 13 of 110 detector
  calls (12%) and cut warm query time 3.97 s -> 3.33 s (16%), with semantics
  unchanged.
- Session-level prefetch (`--batch-prefetch`) runs one union-vocabulary pass
  over retained contexts before a known question batch. On z210 40 Q it cut
  calls 898 -> 599 (-33%) and query time 24.05 s -> 14.03 s (-42%), plus 3.49 s
  prefetch; semantics stayed .05 and IoU .768 vs .77. It only pays when
  contexts are fewer than per-question calls (264 vs 22.45/Q here; Cook would
  not benefit).
- Negative result: adaptive context density (0.25 s while a hand action is
  open, 1.5 s idle) increased store size and *hurt* Switch localisation
  (loc@.30 0, IoU .087), because changed context placement altered track
  association. It remains an opt-in ablation, not the default.
- Hand-only context storage saves nothing: hands are present in ~all keyframes.
- Raising the CLIP-only margin threshold to .22 correctly abstains on
  undetectable objects (rice cooker, cutting board, switch console) but also
  suppresses counting candidates. CLIP-only labels need corroboration-aware
  calibration, not a single global threshold.
- Counting uses three signals: simultaneous detections (peak occupancy),
  persistent detector tracks across action-start contexts, and spatial
  clustering of track mean boxes. Peak counting fixes the two joy-con case
  (`2 joy-con controllers`); clustering the six fragmented cup tracks into two
  places fixes the Cook cup count. Detection-to-capsule attribution now weights
  coverage by crop tightness, so count provenance points at the cup capsule
  rather than one background capsule.
- Attribute questions use retained pixels, not another model: the central 70%
  of the bound box is converted to HSV, low-saturation pixels are discarded,
  and coarse hue bins vote. Switch `What colour is the joy-con?` answers `cyan`
  from the blue joy-con; this raises semantic accuracy from .75 to .875.
- Third-domain evidence: on `aria_play_45s.mp4` (no QA, dense pseudo-GT), SPORE
  retains 2/3 dense-detectable objects (book, chair) with loc@.30 .667 and mean
  IoU .475. The moving play scene creates 407 capsules; the 300-capsule budget
  keeps the store at 10.4 MB with identical metrics, so contexts absorb the
  loss rather than object evidence.
- Latency on z210 (7,695 decoded frames) is 298 s ingest at 4 fps sampling; the
  first attempt stalled because compaction was O(n^2) and ran on every frame
  once the capsule cap was exceeded. Compaction is now throttled and tests the
  cheap spatial gate before the appearance similarity; `CapsuleStore.match` is
  vectorised over prototypes.
- Memory is bounded by interaction evidence, not appearance alone. Play-domain
  analysis found only one near-duplicate prototype pair at cosine >= .95 among
  406 capsules, so merging cannot control growth. `CapsuleStore.evict_to`
  ranks capsules by contact window, hand overlap, motion, view quality, then
  recency. On z210 this cuts 1,947 capsules to 300 and the JSON store from
  44.2 MB to 21.1 MB with localisation unchanged (semantic remains detector-
  limited); Cook, Switch and Play keep their headline metrics, and Play drops
  from 11.5 MB to 10.4 MB. Context frames, not capsules, carry most answer
  evidence.
- Counting is stable across serialization: action-start sampling now uses
  temporal bins instead of `starts[::stride]`, which changed every sampled
  context when a 4-decimal timestamp shifted. On fresh and reloaded Cook
  stores the cup count is 2 and semantic accuracy is .60.
- Action capsules are a usable temporal index even without verbs. Against the
  real 12-event HoloAssist switch timeline, a still-fragment pass (same hand,
  no path length, duration <= .8 s, gap <= .25 s) folds tracking fragments into
  their neighbour and leaves longer holds alone. The resulting 24 actions give
  recall 12/12 at temporal IoU .1, 11/12 at .2/.3, 97.2% timeline coverage,
  and precision .917 at IoU .1 / .792 at .2 / .708 at .3 (38 actions before
  the pass: .684/.526/.447). QA is unchanged. Reproduce with
  `scripts/spore_action_eval.py`; verb naming is still open, and thresholds
  were validated on one clip. Merging across capsule identities by hand-box
  proximity still over-merges (7 actions, recall .58/.25/.17/.17), so identity
  remains the merge gate.
- Sampling is not monotonically better. On Cook, semantic/loc@.3/IoU are
  .60/.60/.632 at 2 fps, .70/1.00/.692 at 4 fps, and .40/.80/.524 at 8 fps;
  Switch is .75/.75/.649, .875/.75/.590, .75/.75/.573. More frames change the
  temporal phase of retained contexts and can skip a short interaction. Default
  stays 4 fps. Opt-in `rescan=True` re-decodes a 1 s window of the source at
  12 fps only when retained evidence is weak; on the 8 fps Cook store it
  recovers loc@.3 .80->1.00 / IoU .626->.739 at 20.3 calls,
  but adds no accuracy at 4 fps, so it is an escalation path, not a default.
- Fine-object limits are detector-bound, not prompt-bound: over all 264 z210
  contexts, YOLO-World-S returned zero detections for game card, SD card, wrist
  strap, and comfort grip, including 1.5x and 2.6x hand-zoomed crops (1,372+
  calls). A query-time YOLO-World-L pass raises z210 semantic accuracy only
  .025 to .05 (loc/IoU .77 to .83 on the one dense-detectable object, calls
  27.2 to 22.2) and uses 2.2 GB peak VRAM; at 1280 px it still finds none of
  the four fine objects. CLIP top crops for those terms have cosines of only
  .27-.36 and are mostly background. These counts require contact/action
  evidence, narration, or a stronger detector.
- Held-out automatic questions (`scripts/spore_autoeval.py`) track dense
  detections and ask only about repeatedly seen objects. With 10 questions per
  clip: Cook localisation .90 / mean IoU .781 / 4.1 calls, Play .90 / .710 /
  5.9 calls, Switch .667 / .475 / 8.0 calls (any-instance .889 / .619). Labels
  are always present in the answer; the remaining error is choosing the wrong
  instance of a repeated object. Question generation is automatic, so these
  numbers are not tuned to a hand-written QA list.
- The store now has two tiers. `CapsuleStore.save(path, split=True)` writes a
  compact index plus a sibling `*_evidence/` directory; `load` resolves the
  references transparently. Index sizes: Cook 1.02 MB (vs 8.22 MB monolithic),
  Switch 1.34 MB (vs 5.99), Play 1.33 MB (vs 11.45), z210-300 2.40 MB (vs
  21.13). Evidence blobs are 3.99 / 2.43 / 4.65 MB on Cook/Switch/Play with
  WebP crops (z210 stays 14.08 MB until re-ingested). An LLM can read
  the index without loading image payloads; query-time code loads evidence only
  when it needs pixels.
- Capsule/scene crops use WebP q82; context keyframes stay JPEG. This cuts
  monolithic stores from 8.22 to 6.41 MB (Cook), 6.16 to 4.69 MB (Switch), and
  11.45 to 7.68 MB (Play) with identical QA. Recoding contexts to WebP is not
  safe: Cook semantic/loc@.3 fall .70/1.00 to .50/.50, so only the narrower
  crops use it. Lowering context JPEG quality to 70 saves only 6% but drops
  Cook semantic .70 to .60 and IoU .692 to .573. Split evidence is 3.99 /
  2.43 / 4.65 MB. Downscaling contexts is also unsafe: 512 px wide saves 13.6%
  of Cook split bytes but drops semantic .70 -> .40, loc@.3 1.00 -> .50, and
  IoU .692 -> .425; 576 px still drops semantic to .40.
- View pixels are optional (`include_view_images=False`): views keep boxes and
  embeddings for routing, while contexts keep the pixels used for detection.
  Cook drops to 3.48 MB, Switch to 2.37 MB, Play to 4.55 MB, with unchanged QA
  and fewer detector calls (8.3 / 8.6 / 4.5 per query).
- Capsule eviction is not a free memory knob. Evicting Cook 196 -> 50 preserves
  all answers (.70 / 1.00 / .692) and 6.46 -> 4.86 MB, but calls rise 13.7 ->
  14.9; z210 300 -> 75 cuts 17.07 -> 12.35 MB, but calls rise 22.5 -> 24.0 and
  semantic falls .05 -> .025 (one question). Switch 300 -> 150 drops semantic
  .875 -> .75 because routing chooses a different context and loses the
  console's support phrase, despite unchanged loc/IoU. Keep capsules for
  routing and drop view pixels instead. Context frames are the floor: ~4 MB
  Cook, ~12 MB z210.
- Sparse contexts are not a free win: with view pixels dropped, 1.0 s and 1.5 s
  context gaps save only 3-25% but reduce Cook loc@.30 from 1.00 to .75 (and
  1.0 s also Switch .75 -> .50). Keep `context_min_gap=0.75`; detector phase
  matters more than raw frame count.
- Query-time context sweep is also non-monotonic. On the Cook store, 1/2/3/5
  frames give loc@.30 .75/.75/.75/1.00 at 10.9/11.8/12.5/13.7 calls; on Switch
  the same sweep gives semantic .75/.75/.75/.875 and loc@.30
  .75/.25/.25/.75. Two or three frames fuse competing same-class instances into
  an unstable track, so fewer frames save calls but not reliably. Keep
  `context_sweep_frames=5`; use `--context-sweep-frames` only for ablations.
- Prefetch makes denser sweeps cheap in detector calls but not in accuracy:
  z210 at 9 frames/+/-4 s gives IoU .834 vs .768 and 586 vs 599 calls; Play
  .545 vs .493 and 60 vs 73 calls; Switch is unchanged; Cook drops .692 -> .560
  and semantic .70 -> .60 because a wider window fuses competing instances.
  Default stays 5 frames/+/-2 s; `--context-sweep-window` is an ablation knob.



## Non-goals

- Do not tune to a single HoloAssist session.
- Do not replace Chronicle's graph or query semantics.
- Do not add a large VLM to every frame.
- Do not treat unverified captions as facts.

## First milestone

1. Implement `chronicle/perception/capsules.py` and a standalone `SporePerceiver`.
2. Build capsules from low-res frames, hand/motion proposals, and best views.
3. Implement query-time binding for object-location and object-identity questions.
4. Evaluate on Cook and HoloAssist 34 s first; use the 257 s z210 only as a
   diagnostic, not the objective.
5. Report compute and evidence metrics alongside QA.

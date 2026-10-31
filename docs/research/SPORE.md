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
- An opt-in inventory path answers "what is on the table?" by sampling support
  contexts across the clip (12, expanding to 24 or 48 when the first batch
  finds nothing), detecting a candidate vocabulary, and keeping objects whose
  centres lie on the support box. A generic 43-label household/office bank is
  usable with confidence-gated persistence (low-confidence detections need two
  contexts) plus alias and duplicate-box suppression. On Switch it yields
  "joy-con on the work table" instead of "joy-con, remote, phone"; caller-label
  inventory remains 8/8. It is a grounded listing operator, not a general
  captioner; the main results table keeps the base Switch score and reports
  the inventory variant separately.
- Support surfaces are also queryable targets: when a support term routes no
  object capsule, the engine runs context fusion and accepts an unattributed
  support track, with aliases mapping table/work table/desk/countertop. On
  Play, a fresh engine now answers "table at 2.0s" instead of abstaining, at
  4.75 vs 5.0 mean calls and unchanged semantic/loc/IoU; there is no dense
  table target, so this is routing robustness, not a localisation gain.
- A detector-free scene-recall path ranks explicit "when did I see X?" /
  "when was X visible?" questions by cosine-scoring text against retained
  `SceneCapsule` embeddings. On a 60 s outdoor walk it retrieved 4/6
  informative landmarks at top-1 (stairs, person, building, cup) with zero
  detector calls and 13-14 ms warm query time, but missed a painted curb sign
  and a glass door. A twelve-term absent probe scored .145-.259 and eleven
  terms exceeded the .15 gate, so absolute CLIP scores cannot verify absence;
  the natural-language path is opt-in and only ranks a term assumed present.

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
- eating, cleaning, social, and outdoor windows from the Aria Gen 2 Pilot
  release (see [`DOMAINS.md`](DOMAINS.md)),
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

Four real egocentric clips are used as *development* evidence, not as general
video-understanding claims. Dense YOLO-World at 4 fps is pseudo-ground-truth.

| clip | compact store | loc@.30 | loc@.50 | mean IoU | semantic | detector calls/query | dense calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cook 45 s | 3.11 MB | 1.00 | 1.00 | .826 | .70 | 7.9 | 150 |
| Switch 34 s | 1.95 MB | .75 | .75 | .611 | .875 | 7.4 | 128 |
| Play 45 s | 3.11 MB (300-cap) | .667 | .667 | .475 | 1.00 | 5.0 | 150 |
| HoloAssist z210 257 s | 7.55 MB (300-cap) | 1.00 | 1.00 | .77 | .05 | 12.7 | 962 |

Compact stores are split index plus evidence blobs with capsule view pixels
and scene thumbnails omitted; context frames remain. Full split stores with
both pixel sets are 5.00/3.75/5.98/13.35 MB; monolithic stores are
6.46/4.69/7.74/17.07 MB.

### Held-out Aria domains (automatic pseudo-GT)

Four windows selected before any SPORE run: eating, cleaning, social, and
outdoor. Each is 45 s at 4 fps; questions are generated from dense YOLO-World
tracks, 12 per clip, budget 8.

| domain | clip | semantic | loc@.30 | mean IoU | calls/query | dense calls |
|---|---|---:|---:|---:|---:|---:|
| eating | `aria_eat_0_45s` | 1.000 | .917 | .711 | 3.33 | 150 |
| cleaning | `aria_clean_0_45s` | 1.000 | .833 | .678 | 4.50 | 150 |
| social | `aria_play_3_45s` | 1.000 | .917 | .719 | 4.75 | 150 |
| outdoor | `aria_walk_0_45s` | 1.000 | .833 | .617 | 5.25 | 150 |
| **all** | | **1.000** | **.875** | **.681** | **4.46** | **150/clip** |

This is a retention/routing result, not independent open-world accuracy:
the same dense detector supplies the questions and reference boxes. Support
relations remain noisy (`sofa on the basket`). One general parser fix came out
of the outdoor clip: bare “Where is the person?” is now a valid people query
instead of being dropped as a stopword.

On the same hand-written QA used by the research branch, SPORE scores Cook
7/10, Switch 7/8, and Play 4/4; the corresponding `main` numbers were Cook
8/10 and Switch 7/8. SPORE therefore does **not** beat `main` on accuracy in
this comparison. Its measured advantage is efficiency: ~5–8 detector calls per
query versus 128–150 dense calls, with 1.95–3.11 MB compact stores. The Cook
regressions are a shelf/countertop support error and a plate-destination
tracking error; they are general failure modes, not reasons to tune on Cook.
See `docs/research/DOMAINS.md` for the comparison table.

- Ablation call counts below predate disabling the redundant crop fallback;
  absolute calls are higher there, while accuracy/storage trends are unchanged.
  Cook IoU values of .692 in older ablations predate adaptive fusion; the
  current default is .826, and the direction of each effect is unchanged.
  The gain is phase-conditional: at 4.4 fps adaptive/mean is .826/.692, but at
  3.6 fps both are .371 because that ingest loses the cup evidence; absolute
  scores are ingest-phase sensitive.
- The z210 aggregate is 2/40 and is dominated by task mix. One object-identity
  and one counting answer are correct; object location is 0/6 because the
  questions ask for place labels such as “Work Table”, not object boxes, while
  error correction needs instructor speech and temporal/activity questions
  need verbs. Only identity and counting are in the current object-memory
  scope (2/14). z210 has no audio stream.
- Evidence per answer (unique bytes on the provenance chain): 66.3 KB Cook,
  34.6 KB Switch, 65.2 KB Play, 35.9 KB z210 — 2.1% / 1.8% / 2.1% / 0.5% of
  the compact store. Detector calls, not evidence fetches, dominate query cost.
- Context-free crop binding is much weaker: Cook loc@.30 .25 / IoU .246,
  Switch loc@.30 0 / IoU .108. Retained full keyframes, not better crops, are
  the main source of the gain.
- JPEG quality matters: at q72 the thin held knife disappears; q82 restores it.
- Cross-frame track fusion fixes instance ambiguity (two joy-cons): Switch
  loc@.30 rose .25 → .75 after associating detections across five contexts and
  scoring tracks by a temporally weighted geometric mean of confidence.
- Adaptive fusion box: tracks whose five context boxes agree (mean pairwise IoU
  ≥ .4) keep the smoother weighted mean box; spatially inconsistent tracks use
  the detection nearest the question time. Cook mean IoU rises .692 → **.826**
  and loc@.50 .75 → **1.00** with semantic unchanged (.70); Switch and Play are
  unchanged. Unconditional representative fusion is unsafe (Switch .611 →
  .533, Play .475 → .217). Defaults: `context_fusion_box="adaptive"`,
  `context_fusion_spread=.4`.
- Action-indexed temporal retrieval answers "when did … first" by forward-
  scanning sparse contexts: Switch first joy-con pick-up at 3.7 s (annotated
  4.2 s); Cook first cup touch at **4.6 s** (annotated 4.5 s). The first-touch
  scan requires a detection to contain a MediaPipe fingertip, so a large hand
  box overlapping a cup on a shelf no longer counts as contact. The same
  fingertip rule in action segmentation reduced recall and was rejected;
  actions still use hand-box/palm contact.
- Capsule crops are attention records, not detector inputs. With keyframes
  retained, the crop detector fallback changed zero answer strings on
  Cook/Switch/Play and only two wrong timestamps on z210. Disabling it by
  default cuts calls 13.7 -> 7.9 (Cook), 12.9 -> 7.4 (Switch), 11.5 -> 5.0
  (Play), 22.5 -> 12.7 (z210), with identical accuracy/localization/IoU. It
  remains available for context-less stores or via `view_binding=True`.
- Steady-state query latency on Cook is now ~201 ms mean / 151 ms median (7.9
  detector calls), down from 617 / 337 ms at 14.4 calls with the redundant
  crop fallback; excluding one-time model/predictor warmup (~8-12 s).
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
  over retained contexts before a known question batch. On z210 40 Q, with
  crop fallback off, it cut calls 506 -> 264 (-48%) and query time 11.47 s ->
  5.37 s (-53%), plus 3.28 s prefetch; session 11.47 -> 8.65 s (-25%),
  semantics .05, IoU .768 vs .77. It only pays when contexts are fewer than
  per-question calls (264 vs 12.7/Q here; Cook would not benefit). Prefetching
  all 989 capsule views as well costs 24.79 s and makes the session 33.4 s vs
  8.7 s despite zero query calls; blind view prefetch needs query-aware
  selection.
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
  keeps the store compact with identical metrics, so contexts absorb the
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
  recency. On z210 this cuts 1,947 capsules to 300; the monolithic JSON store
  falls 44.2 -> 17.1 MB and the compact split store is 7.55 MB, with
  localisation unchanged (semantic remains detector-limited); Cook, Switch and
  Play keep their headline metrics. Context frames, not capsules, carry most
  answer evidence.
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
  were validated on one clip. A duration sweep (.4-5.0 s) spans 26-20 actions
  with a broad 1.2-2.5 s plateau (mean IoU .514, precision@.2 .818, recall@.2
  .917), so .8 s is a conservative default rather than a tuned point.
  Merging across capsule identities by hand-box
  proximity still over-merges (7 actions, recall .58/.25/.17/.17), so identity
  remains the merge gate.
- Sampling is not monotonically better. On Cook, semantic/loc@.3/IoU are
  .60/.60/.632 at 2 fps, .70/1.00/.826 at 4 fps, and .40/.80/.524 at 8 fps;
  Switch is .75/.75/.649, .875/.75/.611, .75/.75/.573. More frames change the
  temporal phase of retained contexts and can skip a short interaction. Default
  stays 4 fps. Opt-in `rescan=True` re-decodes a 1 s window of the source at
  12 fps only when retained evidence is weak; on the 8 fps Cook store it
  recovers loc@.3 .80->1.00 / IoU .626->.739 at 20.3 calls. On Play it recovers
  the shoe (loc .667->1.00, IoU .475->.717) at 12.75 calls, but adds no
  semantic accuracy at 4 fps Cook/Switch; on a 3.6 fps Cook store it raises IoU
  .371->.587 while semantic stays .40. It is an escalation path, not a default.
- Fine-object limits are detector-bound and source-resolution-bound, not
  prompt-bound: over all 264 z210 contexts, YOLO-World-S returned zero
  detections for game card, SD card, wrist strap, and comfort grip, including
  1.5x and 2.6x hand-zoomed crops (1,372+ calls). A query-time YOLO-World-L
  pass leaves z210 semantic accuracy at .05 (2/40), moves loc/IoU .77 to .83
  on the one dense-detectable object, changes calls 22.45 to 22.2, and uses
  2.2 GB peak VRAM; at 1280 px it still finds none of
  the four fine objects. CLIP top crops for those terms have cosines of only
  .27-.36 and are mostly background. The z210 source is 454x256, so these
  objects occupy only a few pixels; a 2x2 overlapping tile pass with 2x
  upsampling over all 264 contexts yielded only six detections at conf .05,
  all false positives on Joy-Con/console/screen. A direct six-term probe over
  all 264 contexts returned zero detections for comfort grip, wrist straps,
  game card, SD card, kickstand, and power button at conf .10; only joy-con
  (76) and Switch console (16) were visible. The limit is source resolution
  before it is detector capacity; higher-resolution capture is required, not
  just a stronger detector, and z210 temporal/activity questions cannot be
  grounded by this capture.
- Held-out automatic questions (`scripts/spore_autoeval.py`) track dense
  detections and ask only about repeatedly seen objects. With 10 questions per
  clip: Cook localisation .90 / mean IoU .781 / 4.1 calls, Play .90 / .710 /
  5.9 calls, Switch .667 / .475 / 8.0 calls (any-instance .889 / .619). Labels
  are always present in the answer; the remaining error is choosing the wrong
  instance of a repeated object. Question generation is automatic, so these
  numbers are not tuned to a hand-written QA list.
- The four held-out Aria windows are now downloaded and scored; see the
  dedicated table above and `docs/research/DOMAINS.md`. The harness remains
  separate from the original four-clip development table so future changes can
  be checked for domain transfer. The current headline is 48/48 semantic
  agreement, loc@.30 .875, mean IoU .681, 4.46 calls/query, but semantic
  agreement is against the same dense detector that generated the questions.
- The store now has two tiers. `CapsuleStore.save(path, split=True)` writes a
  compact index plus a sibling `*_evidence/` directory; `load` resolves the
  references transparently. Current compact indexes are Cook .99 MB, Switch
  1.29 MB, Play 1.29 MB, z210-300 2.34 MB; evidence blobs are 2.12 / .67 /
  1.82 / 5.21 MB, for compact totals of 3.11 / 1.95 / 3.11 / 7.55 MB. An LLM
  can read the index without loading image payloads; query-time code loads
  evidence only when it needs pixels. Scene thumbnails are omitted because
  nearly every scene (188/189 on z210; all on the short clips) is within 0.5 s
  of a retained context; scene evidence resolves from that context.
- Capsule/scene crops use WebP q82; context keyframes stay JPEG. This cuts
  monolithic stores from 8.22 to 6.41 MB (Cook), 6.16 to 4.69 MB (Switch), and
  11.45 to 7.68 MB (Play) with identical QA. Recoding contexts to WebP is not
  safe: Cook semantic/loc@.3 fall .70/1.00 to .50/.50, so only the narrower
  crops use it. Lowering context JPEG quality to 70 saves only 6% but drops
  Cook semantic .70 to .60 and IoU .692 to .573. The view-bearing split
  evidence was 3.99 / 2.43 / 4.65 MB; compact evidence is 2.12 / .67 / 1.82
  MB. Downscaling contexts is also unsafe: 512 px wide saves 13.6%
  of Cook split bytes but drops semantic .70 -> .40, loc@.3 1.00 -> .50, and
  IoU .692 -> .425; 576 px still drops semantic to .40.
- View and scene pixels are optional (`include_view_images=False`,
  `include_scene_images=False`): views keep boxes and embeddings for routing,
  scenes keep embeddings for ranking, and contexts keep the pixels used for
  detection and evidence. This is the benchmark dump default now. It cuts the
  current split stores 38-48% (Cook 5.00 -> 3.11 MB, Switch 3.75 -> 1.95, Play
  5.98 -> 3.11, z210 13.35 -> 7.55) with identical QA and detector calls.
  `--view-pixels` and `--scene-pixels` keep those pixels; `--dump-split`
  writes index plus evidence blobs.
- Capsule eviction is not a free memory knob. Evicting Cook 196 -> 50 preserves
  all answers (.70 / 1.00 / .692) and 6.46 -> 4.86 MB, but calls rise 13.7 ->
  14.9; z210 300 -> 75 cuts 17.07 -> 12.35 MB, but calls rise 22.5 -> 24.0 and
  semantic falls .05 -> .025 (one question). Switch 300 -> 150 drops semantic
  .875 -> .75 because routing chooses a different context and loses the
  console's support phrase, despite unchanged loc/IoU. Keep capsules for
  routing and drop view and scene pixels instead. In the compact store,
  context evidence is the floor: 2.12 MB Cook, 5.21 MB z210.
- Sparse contexts are not a free win: in an earlier pre-WebP no-view ablation,
  1.0 s and 1.5 s
  context gaps save only 3-25% but reduce Cook loc@.30 from 1.00 to .75 (and
  1.0 s also Switch .75 -> .50). Keep `context_min_gap=0.75`; detector phase
  matters more than raw frame count.
- Query-time context sweep is also non-monotonic. On the Cook store, 1/2/3/5
  frames give loc@.30 .75/.75/.75/1.00 at 10.9/11.8/12.5/13.7 calls; on Switch
  the same sweep gives semantic .75/.75/.75/.875 and loc@.30
  .75/.25/.25/.75. Two or three frames fuse competing same-class instances into
  an unstable track, so fewer frames save calls but not reliably. Keep
  `context_sweep_frames=5`; use `--context-sweep-frames` only for ablations.
- Adaptive early exit does not work: a confidence/margin gate on 3 frames
  either expands every query (calls unchanged) or cuts calls only 13.7 -> 13.6
  while dropping Cook loc@.30 1.00 -> .75 and IoU .692 -> .664. Detector
  confidence is not calibrated enough to choose fusion depth safely.
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

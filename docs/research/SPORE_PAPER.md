# SPORE: Sparse Perception with Open-vocabulary Retrieval and Evidence

Status: paper draft / experiment log. Branch `research/spore-perception`.
Everything below is measured on the current prototype unless marked as a
hypothesis. Numbers are pseudo-GT numbers, not human-annotated benchmark
numbers; see Threats to Validity.

## Abstract

Always-on egocentric memory systems face a tension: dense perception is
expensive and vocabulary-bound, while storing raw video is cheap only until a
user asks a question. We present **SPORE**, a label-free, interaction-centric
perception layer that stores sparse visual evidence during ingest and binds
semantics only at query time. SPORE keeps frames when motion, hands, or visual
change make them informative; proposes object-agnostic crops; embeds them with
CLIP; and stores compressed full keyframes as query-time context. A query
routes question terms to capsules, runs an open-vocabulary detector only on a
few retained keyframes, and fuses detections across time into instance tracks.
The store separates a compact index (metadata, embeddings, provenance) from
JPEG evidence blobs, so an LLM can read the index without loading pixels. On
three short egocentric clips, SPORE localises 0.90/0.90/0.667 of
dense-detectable objects with 4-8 detector calls per query versus 128-150 for
dense sampling; a compact 300-capsule store cuts the 257 s HoloAssist z210
store from 44.2 MB to 17.1 MB while preserving localisation, and an answer
touches only 43-87 KB of evidence. Fine-object counting remains source-
resolution-bound as well as detector-bound.

## 1. Introduction

Egocentric assistants need to answer questions about objects, actions, and
locations long after the video was recorded. Two common designs fail in
opposite directions:

1. Run a detector or VLM on every frame. Accurate on seen classes, expensive,
   and still blind to objects outside the detector vocabulary.
2. Store video and retrieve frames at query time. Cheap at ingest, but the
   retrieval problem is pushed to query time and raw video is not LLM-readable.

SPORE's thesis is **perceive less, name later**. Ingest should record evidence
of *where something changed or was touched* without deciding what it is. A
query supplies the vocabulary and pays for semantic perception only when it is
needed.

The contribution is a working system and a measurement protocol:

- A label-free capsule memory built from motion, hands, saliency, and grid
  proposals, with bounded views and interaction-aware eviction.
- Retained full keyframes as late-binding context; this is the largest single
  accuracy gain in our ablations.
- Cross-frame detection fusion and tightness-aware capsule attribution for
  instance disambiguation and provenance.
- A two-tier store: compact JSON index plus lazily loaded JPEG evidence.
- Self-supervised evaluation questions generated from dense pseudo-GT tracks,
  to reduce overfitting to a handful of hand-written QA items.

## 2. Related Work

- **SmartRes / Q-Gate / EgoP2C / WISH / EgoLoc** motivate query-conditioned,
  sparse, or interaction-centric egocentric perception.
- **Open-vocabulary detectors** (YOLO-World) provide query-time naming but have
  a fixed prompt set per call and miss very small objects.
- **CLIP** provides cheap image-text routing but is weak on fine-grained
  egocentric crops; we use it for routing and as a fallback, not as ground
  truth.
- **Video-language models** can answer questions directly but are too expensive
  for always-on ingest and obscure provenance.

SPORE is complementary: it is a memory layer that can feed a VLM or graph
builder later, not a replacement for Chronicle's graph or query engine.

## 3. Method

### 3.1 Sparse causal ingest

Decode at a low rate (4 fps in all reported runs). Estimate affine camera
motion and compare motion-compensated frames. Retain a frame when:

- it is the first frame;
- change exceeds `change_threshold`;
- a hand is present and large enough;
- camera pan ends;
- `keyframe_max_gap` elapses.

This yields roughly 0.3-0.5 keyframes per second on the test clips, instead of
processing every frame.

### 3.2 Label-free capsules

On each keyframe, propose boxes from:

- hand boxes at three padding scales plus a polar hand lattice;
- motion connected components;
- saliency peaks at multiple scales and aspect ratios;
- high-detail regions;
- a 48 px grid.

Candidates are scored by detail, hand overlap, motion, saturation, a
lower-centre prior, and size. A diversity pass enforces one candidate per grid
cell and suppresses near-duplicate HSV descriptors. The best crops are embedded
with OpenCLIP ViT-B/32 and matched to existing capsules by appearance plus a
spatial gate. No class label is required at ingest.

Each capsule keeps at most five best views, contact windows, and a running
prototype. A hard budget (`max_capsules=300` by default) evicts capsules by
interaction priority: contact, hand overlap, motion, view quality, recency.

### 3.3 Context frames and two-tier storage

A small crop is often insufficient: a knife held under a hand may be invisible
in the crop. SPORE stores compressed full keyframes (`context_quality=82`,
minimum 0.75 s apart) for query-time re-detection. This is method, not a debug
artifact.

To keep the store LLM-friendly, `CapsuleStore.save(path, split=True)` writes:

- `index.json`: capsules, views, embeddings, actions, contacts, scene/text
  records, and `file:` references;
- `<stem>_evidence/`: JPEG crops and context frames.

`CapsuleStore.load` resolves references transparently. A monolithic store is
still available for compatibility.

Views are routing indices, not the primary evidence.  Setting
`include_view_images=False` keeps view boxes and embeddings but drops view
pixels; context frames remain the localisation evidence.  In the pre-WebP
ablation this reduced Cook from 8.27 to 3.48 MB, Switch from 6.06 to 2.37 MB,
Play from 11.53 to 4.55 MB, and the 1,947-capsule z210 store from 44.15 to
17.05 MB, with Cook/Switch/Play answers unchanged and z210 at .025 in that
run.  Detector calls per query fell to 8.3 / 8.6 / 4.5 / 16.8 because the
redundant crop detector fallback disappeared.  We therefore recommend storing
view pixels only when crop-level binding must work without contexts.

### 3.4 Query routing and late binding

A question is parsed into object terms. Each term is embedded and compared with
capsule prototypes; routing adds contact, recency, and timestamp priors, then a
greedy diversity pass. For the top capsules:

1. Select a small number of views, ranked by quality, text similarity, and
   temporal proximity to the question timestamp.
2. Decode the nearest retained context frame.
3. Run YOLO-World with the query term plus support labels.
4. Fuse detections across up to five nearby contexts into tracks using IoU,
   centre distance, and area ratio.
5. Score tracks by a temporally weighted geometric mean of confidence, so a
   one-frame false positive loses to a persistent instance.
6. Attach the track to the capsule with the best combination of coverage,
   proximity, and crop tightness.
7. Verify with CLIP when the detector is weak; abstain below a confidence
   threshold.

### 3.5 Answer operators

- **Location:** best binding's box and timestamp, with support surface.
- **Counting:** peak simultaneous occupancy, persistent tracks, and spatial
  clustering of fragmented tracks.
- **Colour:** HSV vote over the central 70% of the bound region, ignoring
  low-saturation pixels.
- **Support:** detector pass with support labels only.
- **Action / first touch:** action capsules and forward context scans. The
  first-touch scan requires a detected object box to contain a MediaPipe
  fingertip (within a small padding), falling back to hand-box overlap when
  fingertips are unavailable. This distinguishes touching from an object that
  merely appears behind a large, merged hand box.
- **Abstention:** no binding, count zero with no evidence, or confidence below
  threshold.

## 4. Experimental Protocol

### 4.1 Clips

| Clip | Length | Domain | Notes |
|---|---:|---|---|
| `aria_cook_45s.mp4` | 45 s | kitchen | knife, toaster, cup, plate, rice cooker, cutting board |
| `holoassist_switch_34s.mp4` | 34 s | tabletop repair | two joy-cons, switch console, fine accessories |
| `aria_play_45s.mp4` | 45 s | play | book, chair, shoe, table; held-out third domain |
| `holoassist_z210_257s.mp4` | 257 s | long tabletop session | 40 questions, long-video stress test |

### 4.2 Pseudo-ground truth

We run YOLO-World densely at 4 fps on 640 px frames as pseudo-GT. This is not
human annotation. It under-detects fine objects and sometimes emits duplicate
boxes; all retention and localisation numbers should be read as
**detector-relative**.

### 4.3 Metrics

- **Store retention:** at least one capsule view covers the dense target box
  (intersection over target area >= .50 or IoU >= .30) within 0.35 s.
- **Localisation@0.3 / @0.5:** IoU of the tight binding box with the dense
  target near the answer timestamp.
- **Semantic accuracy:** expected answer token (or numeric/temporal constraint)
  appears in the composed answer.
- **Detector calls per query:** includes routing, context binding, fusion,
  counting, and support inference.
- **Store bytes:** serialized JSON, with a separate index/evidence breakdown.
- **Evidence bytes per answer:** unique JPEG bytes attached to the answer's
  provenance chain (best capsule view plus retained context), not the whole
  store. This approximates what an LLM-side reader would fetch from the split
  evidence directory for one question.

### 4.4 Held-out automatic questions

`scripts/spore_autoeval.py` tracks dense detections and asks only about labels
seen in at least three frames. Questions are generated automatically, so the
question set is not hand-tuned per clip. When several instances of a label are
visible, the prompt becomes "Where is a X?" and any instance is accepted;
otherwise the exact track is scored. We report both target-instance and
any-instance localisation.

## 5. Results

### 5.1 Main results

Hand-written QA sets:

| Domain | Semantic | loc@.3 | loc@.5 | mean IoU | Store | Calls/query | Dense calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cook 45 s | .70 | 1.00 | .75 | .692 | 6.46 MB | 7.9 | 150 |
| Switch 34 s | .875 | .75 | .75 | .611 | 4.69 MB | 7.4 | 128 |
| Play 45 s | 1.00 | .667 | .667 | .475 | 7.74 MB | 5.0 | 150 |
| z210 257 s (300 caps) | .05 | 1.00 | 1.00 | .77 | 17.07 MB | 12.7 | 962 |

Calls/query use the default context-first binding, which does not re-detect
retained crops; batch prefetch is a separate session-level optimisation
(Section 5.7).

The z210 aggregate is dominated by task mix, not by a single perception
failure. In the timed 40-question baseline, semantic accuracy is 2/40: one
object-identity answer and one count. Per category:

| Category | Questions | Correct | What the question needs |
|---|---:|---:|---|
| object identity | 8 | 1 | object capsule / detector |
| object location | 6 | 0 | place label ("Work Table"), not an object box |
| counting | 6 | 1 | contact or instance evidence |
| temporal order | 6 | 0 | action sequence / verbs |
| activity | 6 | 0 | action recognition |
| error correction | 8 | 0 | instructor speech / narration |

Six location questions ask for a place label, eight error-correction questions
require speech, and the temporal/activity questions require verbs. Only the
identity and counting questions are inside the current object-memory scope,
where the system scores 2/14. z210 has no audio stream, so a narration/ASR
route cannot be tested on this file.

Automatic held-out questions (10 per short clip):

| Domain | Semantic | loc@.3 | mean IoU | any-instance loc@.3 | Calls/query |
|---|---:|---:|---:|---:|---:|
| Cook | 1.00 | .90 | .781 | .90 | 4.1 |
| Play | 1.00 | .90 | .710 | .90 | 5.9 |
| Switch | 1.00 | .667 | .475 | .889 | 8.0 |

The automatic numbers are higher than the hand-written numbers because the
questions are simpler ("Where is X?") and because the dense tracker only asks
about objects it can see. They are useful for measuring routing and instance
ambiguity, not for claiming general video understanding.

### 5.2 Ablations and negative results

Unless noted, these ablations were run before the redundant crop fallback was
disabled.  Their absolute call counts are therefore higher than the current
default; the accuracy and storage trends are unaffected.

| Intervention | Effect | Decision |
|---|---|---|
| Remove context frames | Cook loc@.3 .25, IoU .246; Switch 0 / .108 | keep contexts |
| Adaptive context density | Switch loc@.3 0, IoU .087; larger store | reject |
| Prune to 3-4 views/capsule | Switch semantic .875 -> .75 | keep 5 views |
| Contact-area gate | cleaner actions but Cook .60 -> .50, Switch .875 -> .75 | revert |
| Fold short still fragments | actions 38 -> 24, precision@.2 .526 -> .792, recall .917 | keep |
| Context sweep 5 -> 1 | calls 13.7 -> 10.9, but Cook loc@.3 1.00 -> .75; Switch at 2-3 frames loc@.3 .25 | keep 5 |
| Denser sweep + prefetch | z210 IoU .768 -> .834, Play .493 -> .545, Switch flat, Cook .692 -> .560 | keep 5 +/-2 s |
| Adaptive context early exit | calls 13.7 -> 13.6, but Cook loc@.3 1.00 -> .75, IoU .692 -> .664 | reject |
| CLIP margin .22 | suppresses false switch console, but Cook .40 -> .30 | keep .16 |
| Count clustering | Cook cups 6 -> 2, semantic .50 -> .60 | keep |
| Colour attribute | Switch semantic .75 -> .875 | keep |
| Capsule eviction, 1,947 -> 300 | z210 44.2 -> 17.1 MB, loc unchanged | keep |
| Split index/evidence | 8.22 -> 1.02 MB index on Cook | keep |
| Drop view pixels, keep contexts | Cook 8.27 -> 3.48 MB, metrics unchanged | keep optional |

### 5.3 Storage breakdown

These split-storage measurements predate WebP view crops; current full-store
sizes are in Table 1.

| Store | Full JSON | Index | Evidence | Embeddings (in index) |
|---|---:|---:|---:|---:|
| Cook | 8.22 MB | 1.02 MB | 5.42 MB | .66 MB |
| Switch | 5.99 MB | 1.34 MB | 3.50 MB | .81 MB |
| Play | 11.45 MB | 1.68 MB | 7.34 MB | 1.00 MB |
| z210 (300 caps) | 21.13 MB | 2.40 MB | 14.08 MB | ~1.5 MB |

The index is 5-8x smaller than the monolithic store and contains the
LLM-readable structure: timestamps, boxes, labels, contacts, actions, and
provenance.

A single answer touches far less evidence than the store: counting the unique
JPEG bytes attached to its provenance chain gives **86.8 KB/query** on Cook,
**42.6 KB** on Switch, **82.1 KB** on Play, and **44.7 KB** on z210.  These are
1.0%, 0.7%, 0.7%, and 0.2% of the corresponding stores, respectively.  The
number is a fetch estimate for an LLM-side reader, not a claim that decoding
cost is negligible: detector calls dominate query latency.

Capsule eviction is not a free memory knob.  Offline eviction of the Cook store
from 196 to 50 capsules preserves every answer (.70 / 1.00 / .692) while
shrinking the serialized store 6.46 to 4.86 MB, but calls rise 13.7 to 14.9.  On z210, 300 to 75
capsules shrinks 17.07 to 12.35 MB, yet calls rise 22.5 to 24.0 and semantic
accuracy falls .05 to .025 (one question).  Switch from 300 to 150 capsules
drops semantic .875 to .75: the console support phrase disappears because
routing selects a different context (18.2 s instead of 20.8 s), even though
localization and IoU are unchanged.  Capsule views are routing priors, not just
payload.  (Sizes here are the JSON serialization reported by the query engine;
split-on-disk stores are smaller because image bytes leave the index.)  Keep
the capsule count and drop view pixels instead; the no-view
store keeps 300-capsule routing at 3.48 / 2.37 / 4.55 MB.  Retained context
frames set the memory floor: about 4 MB for Cook and 12 MB for z210.

Context density is not monotonic with accuracy.  With view pixels dropped, a
0.75 s gap gives Cook 3.48 MB / Switch 2.37 MB and Cook loc@.3 1.00; a 1.0 s
gap gives 3.37 / 2.22 MB but Cook loc@.3 .75 and Switch .50; a 1.5 s gap gives
2.61 / 2.08 MB, Cook loc@.3 .75 and Switch .75.  The sampled keyframe phase
changes which detections survive, so saving 25-30% of context bytes is not
worth losing small static objects.  Keep `context_min_gap=0.75`.

The same phase/instance problem appears in query-time context sweep.  Reducing
from five to one frame cuts Cook calls 13.7 to 10.9, but loc@.3 falls 1.00 to
.75; on Switch, two or three frames give semantic .75 and loc@.3 .25 versus
.875/.75 at five.  The intermediate counts are worse than either endpoint
because competing same-class instances are fused into one track over a short
window.  Keep `context_sweep_frames=5`; the CLI flag is for ablation only.

Prefetching (Section 5.7) removes the detector-call cost of a denser sweep,
but not its semantic cost.  On z210, nine frames over a +/-4 s window raised
mean IoU from .768 to .834 while total calls fell 599 to 586; on Play it raised
IoU from .493 to .545 and total calls fell 73 to 60.  Switch was unchanged,
but Cook fell from .692 to .560 (semantic .70 to .60): the wider window merges
competing same-class instances, and a sharper temporal kernel did not recover
the answer.  Keep the five-frame, +/-2 s default.  Denser sweeps are an opt-in
ablation, not a safe universal default.

### 5.4 Fine-object limits

Over all 264 retained z210 contexts, YOLO-World returned zero detections for
game card, SD card, wrist strap, and comfort grip, including 1.5x and 2.6x
hand-zoomed crops (1,372+ calls). CLIP top crops for those terms have cosines
of .27-.36 and are mostly background. The z210 source is only 454x256, so the
detail needed to count these objects is absent before the detector sees it;
higher-resolution capture, not just a stronger detector, is the prerequisite.

### 5.5 Sampling rate and query-time foveal rescan

Sampling is not monotonic in accuracy.  These runs predate WebP view crops, but
the relative ordering is unchanged.  On Cook, 4 fps is the best operating
point; 8 fps stores more keyframes but changes the temporal phase of retained
contexts, so a short held-knife interval can fall between two contexts:

| Domain | fps | Semantic | loc@.3 | loc@.5 | Mean IoU | Store | Ingest | Calls/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cook | 2 | .60 | .60 | .60 | .632 | 7.86 MB | 32.5 s | 13.5 |
| Cook | 4 | .70 | 1.00 | .75 | .692 | 8.27 MB | 24.6 s | 14.5 |
| Cook | 8 | .40 | .80 | .40 | .524 | 8.44 MB | 73.0 s | 14.6 |
| Switch | 2 | .75 | .75 | — | .649 | 4.50 MB | 8.0 s | 10.0 |
| Switch | 4 | .875 | .75 | .75 | .590 | 6.06 MB | 14.3 s | 14.25 |
| Switch | 8 | .75 | .75 | — | .573 | 5.83 MB | 15.6 s | 12.1 |

A query-time **foveal rescan** can recover a short interaction without storing
every frame: when retained contexts yield no detector evidence, re-decode a
one-second window of the original video at 12 fps.  On the 8 fps Cook store,
rescan raises loc@.3 .80→1.00 and mean IoU .626→.739 at
20.3 calls/query (vs 12.7 without).  On 4 fps stores it changes no headline
metric and roughly doubles calls, so it remains opt-in (`rescan=False`) and is
reported as a cost/accuracy escalation rather than a default.

### 5.6 Action intervals against real annotation

The HoloAssist switch clip also ships a real 12-event interaction timeline.
SPORE ingests without those labels and records only hand-object contact
geometry.  A second consolidation pass folds short stationary fragments (same
hand, no path length, duration $\leq .8$ s) into the neighbouring action;
longer holds are preserved.  Comparing the resulting 24 merged action capsules
to the timeline:

| Temporal IoU | Recall | Precision | Timeline coverage |
|---|---:|---:|---:|
| .1 | 1.00 | .917 | .972 |
| .2 | .917 | .792 | — |
| .3 | .917 | .708 | — |
| .5 | .417 | .208 | — |

Every annotated interaction is recovered at IoU .1, and 92% at IoU .2-.3,
while 97% of annotated interaction time is covered.  The still-fragment pass
roughly halves the action count (38 to 24) and raises precision at IoU .2 from
.526 to .792 without changing any QA answer.  The action stream is still
coarser than the 12-event annotation, and the threshold was validated on one
annotated clip; it is a temporal-index improvement, not evidence that SPORE
understands verbs.  Verb naming remains future work.

A threshold sweep on the same 38-action stream checks that the gain is not a
single-point fit.  With still duration .4/.6/.8/1.0/1.2/2.5/3.0/5.0 s, the
merged action counts are 26/25/24/24/22/22/21/20, mean best IoU is
.475/.475/.488/.488/.514/.514/.510/.493, and precision at IoU .2 is
.769/.800/.792/.792/.818/.818/.810/.850; recall at IoU .2 stays .917
throughout.  The 1.2-2.5 s range is a broad plateau, and .8 s is the
conservative default.  The still-gap parameter (.15-.45 s) made no difference
on this clip, so the result depends mainly on the duration rule.

A more permissive merge that joined actions across capsule identities when
their hand boxes were spatially close (IoU $\geq .10$ or centre distance
$\leq .75\times$ the hand-box scale) collapsed the stream to 7 actions and
cut recall to .58/.25/.17/.17 at IoU .1-.5 (mean best IoU .194).  Hand
proximity is not an identity signal: different objects handled in the same
place chain together.  The primary-capsule gate is retained.

### 5.7 Query latency and vocabulary switching

Capsule crops are attention records, not detector inputs.  Once keyframes are
retained, running the detector again on crops is redundant: on Cook, Switch,
and Play it changed **zero answer strings**, and on z210 it changed only two
wrong timestamps.  Disabling that fallback by default cuts detector calls from
13.7 to **7.9** on Cook (-42%), 12.9 to **7.4** on Switch (-43%), 11.5 to
**5.0** on Play (-57%), and 22.5 to **12.7** on z210 (-44%), with identical
semantic, localization, and IoU numbers.  The crop path remains available for
context-less stores and as an explicit `view_binding=True` ablation.

Steady-state query latency on the Cook saved store, measured on the RTX 4060
Laptop with the detector and CLIP already resident, is **201 ms mean / 151 ms
median** across the nine non-warmup questions (7.9 detector calls per query),
down from 617 ms / 337 ms at 14.4 calls with the redundant crop fallback.
The slowest query is counting, which scans 32 contexts. The first query after
engine construction is not representative: loading YOLO-World and building the
predictor costs roughly 8-12 s once.

Open-vocabulary switching was the dominant avoidable cost. Ultralytics
`YOLOWorld.set_classes` rebuilds its CLIP text encoder on every call (~2.1 s on
this machine). SPORE caches one CLIP text embedding per label, encodes only
new labels, updates the live predictor's features in place, and avoids
invalidating the predictor. Vocabulary changes drop to **~0.11-0.14 s** after
the first swap, and per-frame detection is **~20 ms** once warm. This is a
general systems optimisation: it does not change any benchmark answer.

Repeated questions about the same moment also reuse detector output. A bounded
cache keyed by keyframe and exact prompt set serves location, colour and
activity questions from one pass. On the Switch saved store, answering all
eight questions used 110 detector calls without the cache and 97 with it (13
hits, 12% fewer calls); warm query time fell from 3.97 s to 3.33 s (16%).
Cook semantics stay at .70 and Switch at .875. The cache is implementation
transparent and is cleared per engine, so it cannot leak across clips.

Question batches admit a larger optimisation. When the vocabulary is known
before answering, `SporeQueryEngine.prefetch` runs one detector pass over every
retained context with the union of question terms and support labels; each
question then filters the superset instead of paying one prompt-specific pass
per context. On the z210 40-question set, with the redundant crop fallback
disabled, this reduced detector calls from **506 to 264** (-48%), query time
from **11.47 s to 5.37 s** (-53%; mean 287 ms to 134 ms), at the cost of a
**3.28 s** prefetch pass; session time fell from 11.47 s to **8.65 s** (-25%).
Semantic accuracy stayed at .05 and mean IoU at .768 (vs .77). The trade pays
only when the number of contexts is small relative to per-question calls: z210
has 264 contexts and 12.7 calls per question, whereas Cook has 140 contexts
and 7.9 calls per question, so prefetching there would add work. The API is
opt-in (`--batch-prefetch`); a new prefetch replaces the previous table, and
the table is released with the engine.

Extending the same pass to capsule views is not a free win. Prefetching all 989
retained view crops raises prefetch cost to 24.79 s and total calls to 1,253;
query calls fall to zero and query time to 8.56 s, but the session takes 33.4 s
versus 8.7 s with context-only prefetch. Blind view prefetch overpays because
most views are never queried; it needs query-aware selection rather than a
global vocabulary union.

Ingest remains the other cost. The 45 s Cook clip took ~25 s (225 sampled
frames, 140 keyframes) when the GPU was idle; the 257 s z210 clip takes ~298 s.
Compaction is throttled to once per second and checks box overlap before
appearance similarity, which removed the earlier quadratic blow-up. Proposal
bookkeeping is now vectorized as well: in a 20-iteration microbenchmark with 96
candidates and 200 stored descriptors, novelty lookup fell from 2.42 s to
0.011 s (~220x) and NMS from 167 ms to 57 ms (~3x). Unit tests check that both
vectorized paths match the original greedy/cosine semantics exactly.

### 5.8 Evidence codec

Capsule and scene crops are stored as WebP at quality 82; context keyframes
stay JPEG.  Re-encoding or fresh ingest with WebP leaves every headline answer
unchanged: Cook .70 / loc@.3 1.00 / IoU .692, Switch .875 / .75 / .611, Play
1.00 / .667 / .475.  Monolithic stores shrink from 8.22 to 6.41 MB (Cook), 6.16
to 4.69 MB (Switch), and 11.45 to 7.68 MB (Play), a 22-33% reduction.  With
split storage the Cook index is 1.02 MB and its evidence blobs 3.99 MB;
Switch is 1.33 / 2.43 MB, Play 1.33 / 4.65 MB.  The remaining context bytes
are not free to recompress: recoding context frames alone to WebP quality 80
dropped Cook semantic accuracy .70 to .50 and loc@.3 1.00 to .50.  Small
objects on the wider keyframe are codec-sensitive, so the narrower capsule
crops are the safe target.  Lowering context JPEG quality is not a useful
alternative: quality 70 saves only 6% of Cook bytes but drops semantic accuracy
.70 to .60 and mean IoU .692 to .573.

Downscaling context images is not a workaround.  Storing Cook contexts at
512 px wide and resizing them back to frame space before detection saves 13.6%
of split-store bytes (4.32 vs 5.00 MB), but drops semantic accuracy .70 to .40,
loc@.3 1.00 to .50, and mean IoU .692 to .425.  A 576 px version saves 7.9% and
still drops semantic accuracy to .40.  Because the boxes stay in frame
coordinates and the images are upscaled before detection, this is image-
information loss rather than the earlier box-scaling bug: the wide keyframe
needs its native resolution for small objects.

### 5.9 Detector capacity

A larger query-time detector does not unlock the fine objects.  Re-running the
z210 questions with YOLO-World-L (binding confidence .15) leaves semantic
accuracy at .05 (2/40), moves localisation on the single dense-detectable
object from IoU .77 to .83, and changes calls from 22.45 to 22.2.  The cost is
2.2 GB peak VRAM, roughly double the small model.  At 1280 px the large model
finds no
game card, SD card, kickstand, or power button in any of the 264 retained
contexts.  The z210 source is only 454x256, so a game card or SD card occupies
a handful of pixels; the missing detail is absent before the detector sees it.
Tiling cannot recover it: a 2x2 overlapping tile pass with 2x upsampling over
all 264 contexts produced only six detections at confidence .05, and visual
inspection showed every one was a false positive on a Joy-Con, console back,
or screen.  A direct probe of the objects named by z210's temporal and
activity questions is equally stark: across all 264 contexts at confidence
.10, YOLO-World-S returned zero detections for comfort grip, wrist straps,
game card, SD card, kickstand, and power button; only joy-con controllers
(76 detections) and the Switch console (16) were visible.  Fine-object QA is
therefore source-resolution-bound as well as detector-bound, not
retrieval-bound, and the temporal/activity questions cannot be grounded by
this capture regardless of the question parser.  A higher-resolution capture
is required before a stronger detector can help.

## 6. Discussion

SPORE shows that **retained full context**, not better crops, is the dominant
factor for late-binding localisation. Capsule crops alone are too tight and too
occluded; the same detector on a retained keyframe recovers objects that are
invisible in the crop. The second factor is cross-frame fusion: two joy-cons
produce identical text queries, and only temporal association and geometric
mean scoring reliably pick a consistent instance.

Storage remains evidence-heavy. The compact index solves the LLM-readability
problem, but evidence bytes still scale with clip length; after WebP crops,
contexts are the largest component on Cook, while capsule crops still dominate
Switch and Play. A production system should add an explicit evidence budget,
adaptive context placement around interactions, and possibly a context codec
or a small learned compressor. Context WebP is not a drop-in win because small
objects are sensitive to its smoothing.

## 7. Threats to Validity

- Pseudo-GT is produced by the same detector family used at query time; it
  under-detects fine objects and can reward detector-specific behaviour.
- The hand-written QA sets are small (8-10 questions) and were developed
  alongside the system. The automatic questions reduce but do not eliminate
  this risk.
- All experiments use one GPU (RTX 4060 Laptop, 8 GB) and one model stack
  (YOLO-World, OpenCLIP ViT-B/32, MediaPipe). Latency and memory will differ
  elsewhere.
- The synthetic world validates retention mechanics but uses vector drawings
  that CLIP cannot recognise; it is not evidence of semantic ability.
- Same-label instance ambiguity is not solved in general; the any-instance
  metric exposes the gap.
- No human study, no comparison to a full VLM baseline, and no multi-seed
  variance are reported yet.

## 8. Reproducibility

```bash
# Ingest + hand-written QA
python scripts/spore_real_bench.py \
  --clip data/real/aria_cook_45s.mp4 \
  --qa data/qa/cook_45s.json \
  --out research/spore/results/cook.json

# Held-out automatic questions
python scripts/spore_autoeval.py \
  --clip data/real/aria_play_45s.mp4 \
  --store research/spore/results/play_store/store.json \
  --out research/spore/results/autoeval_play.json \
  --max-questions 10

# Compact index + evidence
python - <<'PY'
from chronicle.perception.capsules import CapsuleStore
store = CapsuleStore.load("research/spore/results/gap_cook/store.json")
store.save("research/spore/results/gap_cook/index.json", split=True)
PY
```

Models: `yolov8l-worldv2.pt` / `yolov8s-worldv2.pt`, OpenCLIP
`ViT-B-32`/`laion2b_s34b_b79k`, MediaPipe hand landmarker. Hardware: RTX 4060
Laptop 8 GB, CUDA, PyTorch, Python 3.12.

## 9. Conclusion

SPORE is a working sparse perception layer for Chronicle: it stores
interaction-centric evidence without committing to a vocabulary, binds names
only when asked, and keeps a compact index that an LLM can read while evidence
stays in cheap sidecar blobs. The current limits are clear and measurable:
fine objects are detector-bound, same-label instances remain ambiguous, and
long-video evidence storage still needs an explicit budget.

## Appendix A: Default configuration

`sample_fps=4.0`, `max_width=640`, `change_threshold=.035`,
`keyframe_max_gap=1.2`, `max_views=5`, `max_capsules=300`,
`context_quality=82`, `context_min_gap=.75`, `context_max_gap=1.5`,
`context_fusion=True`, `context_sweep_frames=5`, `query_capsules=8`,
`query_views=2`, `count_contexts=32`, `count_track_gap=20`,
`count_peak_confidence=.45`, `clip_margin_center=.16`.

## Appendix B: What not to claim

- Do not claim 60/60 or 7/40 as general video understanding.
- Do not claim human-level object naming.
- Do not claim fine-object recall from the current detector.
- Do not claim the pseudo-GT is exhaustive.

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
store from 44.2 MB to 21.1 MB while preserving localisation, and an answer
touches only 43-87 KB of evidence. Fine-object counting remains detector-bound.

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
pixels; context frames remain the localisation evidence.  This reduces Cook
from 8.27 to 3.48 MB, Switch from 6.06 to 2.37 MB, Play from 11.53 to 4.55 MB,
and the 1,947-capsule z210 store from 44.15 to 17.05 MB.  Answers are unchanged
(Cook semantic .70, Switch .875, Play auto-eval 1.00, z210 .025), while detector
calls per query fall to 8.3 / 8.6 / 4.5 / 16.8 because the redundant crop
detector fallback disappears.  We therefore recommend storing view pixels only
when crop-level binding must work without contexts.

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
| Cook 45 s | .70 | 1.00 | .75 | .692 | 8.27 MB | 14.5 | 150 |
| Switch 34 s | .875 | .75 | .75 | .611 | 6.06 MB | 14.4 | 128 |
| Play 45 s | 1.00 | .667 | .667 | .475 | 10.4 MB | 11.5 | 150 |
| z210 257 s (300 caps) | .025 | 1.00 | 1.00 | .769 | 21.1 MB | 27.2 | 962 |

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

| Intervention | Effect | Decision |
|---|---|---|
| Remove context frames | Cook loc@.3 .25, IoU .246; Switch 0 / .108 | keep contexts |
| Adaptive context density | Switch loc@.3 0, IoU .087; larger store | reject |
| Prune to 3-4 views/capsule | Switch semantic .875 -> .75 | keep 5 views |
| Contact-area gate | cleaner actions but Cook .60 -> .50, Switch .875 -> .75 | revert |
| CLIP margin .22 | suppresses false switch console, but Cook .40 -> .30 | keep .16 |
| Count clustering | Cook cups 6 -> 2, semantic .50 -> .60 | keep |
| Colour attribute | Switch semantic .75 -> .875 | keep |
| Capsule eviction, 1,947 -> 300 | z210 44.2 -> 21.1 MB, loc unchanged | keep |
| Split index/evidence | 8.22 -> 1.02 MB index on Cook | keep |
| Drop view pixels, keep contexts | Cook 8.27 -> 3.48 MB, metrics unchanged | keep optional |

### 5.3 Storage breakdown

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

Context density is not monotonic with accuracy.  With view pixels dropped, a
0.75 s gap gives Cook 3.48 MB / Switch 2.37 MB and Cook loc@.3 1.00; a 1.0 s
gap gives 3.37 / 2.22 MB but Cook loc@.3 .75 and Switch .50; a 1.5 s gap gives
2.61 / 2.08 MB, Cook loc@.3 .75 and Switch .75.  The sampled keyframe phase
changes which detections survive, so saving 25-30% of context bytes is not
worth losing small static objects.  Keep `context_min_gap=0.75`.

### 5.4 Fine-object limits

Over all 264 retained z210 contexts, YOLO-World returned zero detections for
game card, SD card, wrist strap, and comfort grip, including 1.5x and 2.6x
hand-zoomed crops (1,372+ calls). CLIP top crops for those terms have cosines
of .27-.36 and are mostly background. Counting these objects requires contact
or action evidence, narration, or a stronger detector.

### 5.5 Sampling rate and query-time foveal rescan

Sampling is not monotonic in accuracy.  On Cook, 4 fps is the best operating
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
geometry.  Comparing its 38 merged action capsules (from 54 raw contacts) to
the timeline:

| Temporal IoU | Recall | Precision | Timeline coverage |
|---|---:|---:|---:|
| .1 | 1.00 | .684 | .972 |
| .2 | 1.00 | .526 | — |
| .3 | 1.00 | .447 | — |
| .5 | .33 | .105 | — |

Every annotated interaction is recovered at IoU .1-.3, and 97% of annotated
interaction time is covered.  The action stream is still over-segmented: 38
merged capsules for 12 events, so precision falls as the IoU requirement
tightens.  This is evidence that contact geometry is a usable *temporal index*
before any verb labels, not evidence that SPORE understands verbs.  Verb
naming remains future work.

A more permissive merge that joined actions across capsule identities when
their hand boxes were spatially close (IoU $\geq .10$ or centre distance
$\leq .75\times$ the hand-box scale) collapsed the stream to 7 actions and
cut recall to .58/.25/.17/.17 at IoU .1-.5 (mean best IoU .194).  Hand
proximity is not an identity signal: different objects handled in the same
place chain together.  The primary-capsule gate is retained.

### 5.7 Query latency and vocabulary switching

Steady-state query latency on the Cook saved store, measured on the RTX 4060
Laptop with the detector and CLIP already resident, is **617 ms mean / 337 ms
median** across the nine non-warmup questions (14.4 detector calls per query).
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

Ingest remains the other cost. The 45 s Cook clip took ~25 s (225 sampled
frames, 140 keyframes) when the GPU was idle; the 257 s z210 clip takes ~298 s.
Compaction is throttled to once per second and checks box overlap before
appearance similarity, which removed the earlier quadratic blow-up. Proposal
bookkeeping is now vectorized as well: in a 20-iteration microbenchmark with 96
candidates and 200 stored descriptors, novelty lookup fell from 2.42 s to
0.011 s (~220x) and NMS from 167 ms to 57 ms (~3x). Unit tests check that both
vectorized paths match the original greedy/cosine semantics exactly.

## 6. Discussion

SPORE shows that **retained full context**, not better crops, is the dominant
factor for late-binding localisation. Capsule crops alone are too tight and too
occluded; the same detector on a retained keyframe recovers objects that are
invisible in the crop. The second factor is cross-frame fusion: two joy-cons
produce identical text queries, and only temporal association and geometric
mean scoring reliably pick a consistent instance.

Storage remains context-heavy. The compact index solves the LLM-readability
problem, but evidence bytes still scale with clip length. A production system
should add an explicit evidence budget, adaptive context placement around
interactions, and possibly a keyframe codec or a small learned compressor.

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

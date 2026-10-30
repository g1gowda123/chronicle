# SPORE: Sparse Perception with Open-vocabulary Retrieval and Evidence

Status: research design, branch `research/spore-perception`.

Implementation status: **design only, no perceiver code yet.** The current
handoff, branch map, experiment results and next steps are in
[`HANDOFF.md`](HANDOFF.md). Read that first.

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

# Chronicle evaluation results (corrected)

All numbers below were produced locally on an NVIDIA RTX 4060 Laptop (8 GB)
with the corrected, leak-free pipeline. Raw outputs live in `runs/research/`;
the statistical reports are `runs/research/stats_*/stats.md`. The earlier
model-swap matrix is retained in `paper/table_matrix.md` with a clear legacy
warning.

## Headline

| System | Cook 45 s | HoloAssist 34 s | Pooled 18 q | VLM calls (C/H) | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Chronicle (final, stride 4) | **80.0%** (8/10) | **87.5%** (7/8) | **83.3%** (15/18) | 26 / 13 | 1960 MB |
| Dense 1 FPS baseline | 40.0% (4/10) | 12.5% (1/8) | 27.8% (5/18) | 45 / 34 | 819 MB |
| Leak-free Chronicle baseline | 40.0% (4/10) | 62.5% (5/8) | 50.0% (9/18) | 53 / 35 | 1830 MB |

The final system uses **fewer VLM calls than the dense baseline** and is
+33.3 points on the pooled suite. The improvement is largest on event-centric
questions (temporal, counting, inventory, activity): 8/10 vs 4/10 pooled.
Location questions are discussed separately in **Metric audit** below because
the official matcher credits zone names.

Two caveats are important and are not hidden:

1. The suites are tiny (10 and 8 questions). Per-dataset differences of one
   question move Recall@1 by 10-12.5 points. The pooled McNemar test is
   p = 0.070 before the faithfulness audit and p = 0.016 after it (see
   **Statistical analysis**).
2. The official matcher accepts zone aliases ("Kitchen", "Work Table"). Under a
   strict support check, Cook location accuracy is 1/6, not 6/6. The reliable
   gains are in event questions, not in support grounding.

## Final configuration

| Item | Value |
|---|---|
| Detector | YOLO-World-L (`yolov8l-worldv2.pt`), confidence 0.30 |
| Detector vocabulary | fixed scene list from QA metadata (12-15 classes); **not** derived from questions |
| Prompt synonyms | fixed a priori: `cup`←mug/coffee cup; `cutting board`←chopping board/wooden cutting board/cutting board wooden; `rice cooker`←instant pot/rice pot; `bowl`←white bowl/ceramic bowl |
| Tracker / graph | ByteTrack + lifecycle state machine, unchanged |
| Captioner | Florence-2-base, caption every 4th event trigger (`caption_stride: 4`) |
| Answer backend | deterministic template / planner (no text LLM at answer time) |
| QA suites | Cook 45 s (10 silver questions); HoloAssist 34 s (8 gold questions) |
| Matcher | deterministic token/number matching, `tolerance_s` for temporal rows |

`semantic.action_llm_enabled: false`; the text LLM is not on the ingestion or
answer path.

## Research-integrity fixes

The original pipeline had three weaknesses that inflated or obscured results.
All are now fixed by default, with explicit ablations where useful.

1. **Question leakage into perception.** The detector vocabulary used to be
   derived from QA nouns. It is now off (`derive_vocabulary_from_qa: false`);
   the scene vocabulary is declared in the QA metadata, independently of the
   questions. The leakage path is kept only as an ablation and is *worse*
   (see below).
2. **Primitive query answering.** A deterministic planner
   (`chronicle/graph/planner.py`) answers `when did ... first ...`,
   `how many ...`, `where did ... go`, and activity questions from timestamped
   action/placement edges instead of returning the latest entity state.
3. **Unfaithful entity resolution.** Fuzzy matching used a 3-character prefix;
   "switch console" resolved to `joy-con` and scored a false pass. The resolver
   now requires a 4-character prefix and returns an explicit
   "I don't have that in my memory." when a specific object is absent.
4. **Counting.** Instances are counted by union-find over near-duplicate CLIP
   crop embeddings (`INSTANCE_SIMILARITY = 0.90`), with alias-aware grouping
   (`controller` and `joy-con` are one class). The answer reports both the
   resolved count and the raw track count, so the reader can see the merge.
5. **Prompt dilution.** Broad 111-class prompts were replaced by a scene list;
   YOLO-World grounds more target objects with 12-15 prompts than with 111.
6. **Statistical reporting.** Exact/Wilson intervals, paired bootstrap and
   exact McNemar are produced by `scripts/analyze_results.py`; a strict
   location audit is produced by `scripts/audit_location.py`.

## Cumulative ablation

Each row adds one change to the previous row. `*` marks the two pre-fix
HoloAssist 100% rows: they include the false "switch console → joy-con" pass
that strict resolution later removed. The final row is the configuration in
`config.yaml`.

| Stage | Detector | Planner | Scene vocab | Strict resolution | Cook | Holo |
|---|---|:--:|:--:|:--:|---:|---:|
| Leak-free baseline | S | no | no | no | 4/10 (40.0%) | 5/8 (62.5%) |
| + deterministic planner | S | yes | no | no | 4/10 (40.0%) | 8/8* (100%) |
| + scene vocabulary | S | yes | yes | no | 5/10 (50.0%) | 8/8* (100%) |
| + strict resolution, fixed synonyms, bowl alias | S | yes | yes | yes | 7/10 (70.0%) | 7/8 (87.5%) |
| + large detector (final) | L | yes | yes | yes | 8/10 (80.0%) | 7/8 (87.5%) |
| + two-scale ensemble (rejected) | S+L | yes | yes | yes | 8/10 (80.0%) | 5/8 (62.5%) |

The largest single gain is the planner on HoloAssist (5/8 → 8/8 before the
strict-resolution correction; 7/8 after). The large detector adds the cutting
board on Cook, but that answer is a zone-token pass, not a support-grounded
one; see the strict audit.

The early rows ran at caption stride 1; the final rows use stride 4. This does
not confound the comparison because the budget curve below shows recall is
identical at strides 0, 2, 4, 8 and 1 on both suites.

## Detector-scale ablation (final synonyms, stride 4)

| Detector | Cook | Holo | Pooled | VLM calls (C/H) | Wall time (C/H) | Peak VRAM | Strict Cook location |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO-World-S | 7/10 (70.0%) | 7/8 (87.5%) | 14/18 (77.8%) | 25 / 9 | 27.5 / 25.2 s | 1830 MB | 2/6 |
| YOLO-World-L (final) | 8/10 (80.0%) | 7/8 (87.5%) | 15/18 (83.3%) | 26 / 13 | 35.8 / 38.5 s | 1960 MB | 1/6 |
| S + L ensemble | 8/10 (80.0%) | 5/8 (62.5%) | 13/18 (72.2%) | 30 / 14 | 43.7 / 46.3 s | 2009 MB | 1/6 |

The large model is the best official-recall choice, but it costs ~130 MB and
~8-13 s of wall time, and its extra Cook pass is a zone-token location answer.
The small model is the efficient choice (25/9 VLM calls) and has slightly
better strict support grounding. The ensemble is rejected: extra detections
destabilise HoloAssist instance counting (5 joy-con tracks sometimes collapse
to one instance) and attribute resolution (the red rather than cyan controller
is selected). The ensemble code remains available via `detector_models` for
reproduction, but it is not the default.

## Question-leakage ablation

Running the same final config with `derive_vocabulary_from_qa: true` adds QA
nouns ("keys", "switch console", "colour", "person", ...) to the detector
vocabulary:

| Detector vocabulary | Cook | Holo | Observed failure |
|---|---:|---:|---|
| Scene vocabulary (final) | 8/10 | 7/8 | — |
| QA-derived vocabulary | 7/10 | 5/8 | corrupted classes such as "person joy-con" and "colour joy-con"; Holo count becomes "6 person joy-con" from 13 tracks |

Leakage is therefore not only methodologically invalid here; it also *lowers*
accuracy by diluting the prompt set and creating spurious classes. This is a
useful negative result for the paper.

## VLM budget curve

`--caption-stride N` captions every Nth event trigger; 0 disables captions.
The curve is flat: on these grounded questions the captions are not on the
critical path, because action/placement edges come from the deterministic
geometry and tracker.

| Dataset | off | every 8th | every 4th | every 2nd | every trigger |
|---|---:|---:|---:|---:|---:|
| Cook 45 s | 80.0% (0 calls) | 80.0% (13) | 80.0% (26) | 80.0% (51) | 80.0% (102) |
| HoloAssist 34 s | 87.5% (0 calls) | 87.5% (7) | 87.5% (13) | 87.5% (26) | 87.5% (51) |

The recommended default is stride 4 (26/13 calls): it preserves a caption for
salient events at roughly one quarter of the every-trigger cost and stays below
the dense baseline's 45/34 calls. The flat curve is evidence for the paper's
efficiency claim on *grounded* questions; open-ended caption QA is not covered
by these suites. Full table: `paper/table_budget.md`.

## Statistical analysis

Exact (Clopper-Pearson) 95% intervals. Paired bootstrap resamples questions;
McNemar uses only discordant questions. With 18 pooled questions the intervals
are still wide.

| Comparison | Baseline | Final | Delta | Bootstrap 95% CI | Discordant (final only / baseline only) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Cook | 40.0% (4/10) | 80.0% (8/10) | +40.0 | [10.0, 70.0] | 4 / 0 | 0.125 |
| HoloAssist | 62.5% (5/8) | 87.5% (7/8) | +25.0 | [-25.0, 75.0] | 3 / 1 | 0.625 |
| Pooled (official) | 50.0% (9/18) | 83.3% (15/18) | +33.3 | [5.6, 61.1] | 7 / 1 | 0.070 |
| Pooled (faithfulness-corrected) | 44.4% (8/18) | 83.3% (15/18) | +38.9 | [16.7, 61.1] | 7 / 0 | **0.016** |

The one baseline-only pooled question is "Where is the switch console?".
The baseline answered **"Joy-con was last seen resting on the floor in the Work
Table"** and was scored correct because "Work Table" is an accepted alias. The
console is a different object and the answer is wrong; the final system returns
"I don't have that in my memory." because the console is not detected. The
faithfulness-corrected row removes that one demonstrably wrong pass. Both rows
are reported; the correction is documented rather than silently applied.

The event-centric subset (activity, attribute, counting, inventory, temporal,
unknown) is 8/10 for the final system vs 4/10 for the baseline, with all four
discordant questions favoring the final system (McNemar p = 0.125). This is the
most defensible capability gain because it does not depend on zone aliases.

## Metric audit: location answers

The official matcher accepts every alias in the QA file. For Cook, the aliases
for a "countertop" question include "Kitchen", so a sentence that only names
the zone passes. `scripts/audit_location.py` adds a strict check: when the gold
answer is not the zone itself, the prediction must contain the gold support
noun.

| Run | Official Cook location | Strict Cook location | Official Holo location | Strict Holo location |
|---|---:|---:|---:|---:|
| Leak-free baseline | 3/6 | 2/6 | 2/2 | 2/2 |
| Final | 6/6 | 1/6 | 1/2 | 1/2 |

The final system's extra official location passes are:
"rice cooker … OUT_OF_VIEW in the Kitchen", "cutting board … INTERACTED in the
Kitchen", and "knife … inside the cabinet in the Kitchen". These name the zone
or a state, not the queried support. The baseline's two strict passes are the
knife and rice cooker "resting on the countertop" answers; the final system
answers them with states because the large detector's state transitions differ.
**Support grounding remains the bottleneck**, and the paper should not claim
that location QA is solved. The gold labels themselves are coarse: at t=12 the
knife is being used to slice, and the cup is in the cabinet, so "countertop"
is not always the visually correct answer.

The HoloAssist gold location is the zone ("Work Table"), so the strict audit
cannot distinguish a correct zone answer from a wrong entity that happens to
mention the zone. The switch-console row above is exactly that failure in the
baseline.

## Failure analysis

### Cook 45 s (8/10)

- **"What is the person slicing?"** fails. The interaction recogniser emits
  `PICK_UP` edges only; there are no `SLICE`/`CUT` edges. The tomato in the
  person's hand at t=12 is not tracked as an entity, and the three tomato
  tracks are on the counter. Tool-target inference (instrument + co-visible
  food) is future work, not implemented.
- **"When did the person first touch the cup?"** fails. The graph reports the
  first `PICK_UP` action at t=0.1 s (large model) or 0.8 s (small model); the
  silver ledger's `TOUCHED` event is at 4.5 s (tolerance 1.0 s). The first cup
  tracks are mugs inside the cabinet; the fingertip/IoS gate fires while the
  cabinet is open. This is an action-grounding/label disagreement, not a query
  planning error.
- **Counting cups** passes (2 from 6 tracks) but is fragile. The identical
  white mugs merge at cosine similarity 0.97; the two round bread rolls at
  t=27 are sometimes detected as mugs. The gold count (2) comes from two
  ledger appearances, not a physical inventory. Lowering
  `INSTANCE_SIMILARITY` to 0.88 would force this case to 2 but was rejected as
  overfitting the gold label.
- **Locations** pass officially but not strictly (see Metric audit).

### HoloAssist 34 s (7/8)

- **"Where is the switch console?"** is the intended failure: the console is
  not detected, so the system abstains. The baseline's "joy-con on the floor"
  answer was wrong but scored; the final system is more faithful and scores
  one point lower on the official metric.
- **Colour** passes ("cyan"), but the clip contains one red and one cyan
  Joy-Con. The question is ambiguous; which controller is resolved depends on
  the detector/tracker run. The ensemble picked the red one and failed. This is
  not a stable capability.
- **Counting** passes (2 joy-con from 5-7 tracks). The ensemble over-merged to
  one instance and failed; this is why the ensemble is not the default.
- **Activity/temporal** pass ("picked up" at 3.8 s in the final run, 5.0 s in an
  earlier run, vs gold 4.2 s and tolerance 1.5 s). The planner returns the
  dominant action for the object, not an arbitrary action at an arbitrary time;
  the latter remains a limitation. The small timing jitter between identical
  configs is model/tracker nondeterminism, not a different answer path.

## Legacy model matrix

`paper/table_matrix.md` and `paper/table_matrix.tex` contain the original
16-run captioner/answerer matrix. Those runs predate the leakage fix, the
planner, strict resolution and the detector upgrade, so they are **not**
comparable with the corrected numbers above. They are retained to document the
Florence-2 vs Qwen2.5-VL and template vs Qwen2.5 swaps:

| Captioner / answerer | Cook Chronicle | Cook dense | Holo Chronicle | Holo dense |
|---|---:|---:|---:|---:|
| Florence-2 / template | 40.0% | 40.0% | 62.5% | 12.5% |
| Florence-2 / Qwen2.5 | 40.0% | 30.0% | 50.0% | 25.0% |
| Qwen2.5-VL / template | 40.0% | 30.0% | 62.5% | 12.5% |
| Qwen2.5-VL / Qwen2.5 | 40.0% | 20.0% | 62.5% | 12.5% |

Qwen2.5-VL remains a drop-in captioner with no accuracy gain on this suite and
a large memory/time cost (~5.5 GB device-wide, ~2x wall time). The
deterministic template answerer is more conservative than Qwen2.5 and scores
higher; Qwen sometimes invents counts ("10 joy-con controllers").

## Limitations

- **Tiny suites:** 10 and 8 questions over one clip each; pooled n=18. The
  intervals are wide by construction and the per-dataset McNemar tests are not
  significant at 0.05.
- **Silver Cook labels:** Cook answers come from a Qwen2.5-VL event ledger;
  HoloAssist uses human annotations and is the stronger evidence.
- **Lenient official matcher:** zone aliases inflate location scores; strict
  support accuracy is low (1/6 Cook). The paper reports both.
- **Action recognition is coarse:** only `PICK_UP`/placement transitions are
  emitted reliably; slicing/cutting and first-touch semantics are not solved.
- **Instance counting is appearance-based:** identical objects merge and
  cross-category false positives (bread → mug) can enter the count.
- **VLM budget result is for grounded QA:** the flat curve does not show that
  captions are useless for open-ended or attribute-rich questions outside this
  suite.
- **One clip per domain:** no cross-video or cross-session generalisation is
  measured.
- **Large detector cost:** +130 MB and ~8-13 s wall time vs the small model;
  the small model is a better efficiency/grounding trade-off.

## Reproduction

```bash
export PATH=/home/g1/projects/chronicle/.venv/bin:$PATH

# Final runs (config.yaml: YOLO-World-L, scene vocab, fixed synonyms,
# deterministic planner, strict resolution, caption_stride: 4)
python eval_pipeline.py \
  --video data/real/cook/aria_cook_45s.mp4 \
  --qa data/qa/cook_45s.json --config config.yaml \
  --captioner florence --llm-backend template \
  --out runs/research/final_cook --evaluate

python eval_pipeline.py \
  --video data/real/holoassist_switch_34s.mp4 \
  --qa data/qa/holoassist_switch.json --config config.yaml \
  --captioner florence --llm-backend template \
  --out runs/research/final_holo --evaluate

# Statistics and audits
python scripts/pool_results.py \
  --run runs/research/leakfree_cook/results.json \
  --run runs/research/leakfree_holo/results.json \
  --out runs/research/pooled/baseline.json --name pooled
python scripts/pool_results.py \
  --run runs/research/final_cook/results.json \
  --run runs/research/final_holo/results.json \
  --out runs/research/pooled/final.json --name pooled
python scripts/analyze_results.py \
  --run baseline=runs/research/pooled/baseline.json \
  --run final=runs/research/pooled/final.json \
  --compare-to baseline --out runs/research/stats_pooled
python scripts/audit_location.py \
  --run runs/research/final_cook/results.json \
  --qa data/qa/cook_45s.json --out runs/research/audit_final_cook.md

# VLM budget curve (one command per stride; see paper/table_budget.md)
python eval_pipeline.py \
  --video data/real/cook/aria_cook_45s.mp4 \
  --qa data/qa/cook_45s.json --config config.yaml \
  --captioner florence --llm-backend template \
  --caption-stride 0 --out runs/research/budget_cook_s0 --evaluate
```

## Long-video diagnostic (HoloAssist z210, 257 s)

A separate 40-question suite tests whether the graph survives a video roughly
7.5x longer than the 34 s HoloAssist clip. Questions are asked at t=250 s (and
t=256 s for final-state cases), so the answer must be retrieved from the full
history. Ground truth comes from human fine-grained annotations; the scene
vocabulary is fixed and not derived from the questions.

**Result: 5/40 (12.5%)**, with 54 VLM calls over 2,565 frames (97.9% fewer than
one call per frame), 1,960.7 MB peak VRAM, and 30.4 ms average query latency.

| Category | Correct | Total |
|---|---:|---:|
| object identity | 0 | 8 |
| object location | 2 | 6 |
| counting | 1 | 6 |
| temporal order | 1 | 6 |
| activity | 1 | 6 |
| error/correction | 0 | 8 |

The failure is structural, not a QA-format artifact. The detector grounds only
13 entities (four joy-con tracks, one Switch, one comfort grip, seven person
tracks); wrist straps, game cards, SD cards, the kickstand and the power button
are never grounded. Action edges collapse to coarse pick-up/interaction events,
and the template falls back to the last recorded event for broad questions.
Three of the five passes are that fallback matching a location token. Counting
reports four tracked joy-cons although the session uses two physical
controllers.

This diagnostic is intentionally kept on the research branch and is not used to
claim general long-video competence. It motivates higher-resolution small-object
grounding, explicit action supervision, and a temporal event store that can
answer interval questions. See `benchmarks/long_video/README.md`.

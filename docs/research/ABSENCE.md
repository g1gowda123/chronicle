# Presence, absence, and state-at-time calibration

Status: experiment complete (2026-11-27), branch `research/spore-perception`.
Code: `scripts/spore_absence.py`. Probe set:
`data/qa/eat_1_absence.json`. Raw report (gitignored):
`research/spore/results/eat1_absence/report.{json,md}`.

## Question

SPORE has no absence verifier. A zero count and a failed detector scan are both
reported as "not enough evidence", but the paper does not measure how often
"not found" is wrong, or whether scanning more retained contexts fixes it. This
experiment measures both, plus a small set of state-at-time questions, on a
clip that was never used to develop a default.

## Pre-registration

- Clip: Aria Gen 2 Pilot `eat_1`, window 30--75 s, 640x480, 10 fps after trim.
  Added to `data/aria_gen2pilot.json` before download. This is the fifth Aria
  window; the first four (`eat_0`, `clean_0`, `play_3`, `walk_0`) were used in
  earlier experiments.
- Probe bank: 26 terms, fixed in `data/qa/eat_1_absence.json` before the first
  ingest. 14 annotated present (with coarse visibility intervals), 12 annotated
  absent. Terms mix tabletop objects, people, and background appliances.
- Annotation: human visual inspection of a 2 s contact sheet plus
  full-resolution frames at 0, 8, 16, 24, 32, 40 and 44 s. The detector was not
  consulted while writing the labels. One label error was found afterwards and
  is audited below.
- The store, scan, query path, and state questions were then run with shipped
  defaults. No threshold or clip-specific rule was changed after looking at the
  probe set.

## Setup

| item | value |
|---|---|
| Ingest | default `SporeConfig`, 4 fps, 640 px, 31.0 s |
| Store | 47 contexts, 300 capsules (cap), 52 actions, 15 scenes |
| Store bytes | 1.38 MB split index + 2.04 MB evidence = **3.42 MB** |
| Scan | every context x every term, detector floor .05, thresholds applied post hoc |
| Scan cost | 1,222 YOLO-World-S calls in 13.5 s, peak 2.69 GB VRAM |
| Query path | `binding_mode=detector` and `hybrid`, budget 8, 26 questions each |
| State probes | 5 questions with an explicit `at_time` |

The low scan floor is deliberate: thresholds are applied to the recorded
per-context scores afterwards, so one scan serves .22/.30/.40/.50 without
re-running the detector. The shipped query path uses its own floors.

## Result 1: not found is not absence

At the query-time detector floor (.22), with **all 47 contexts** scanned:

| | human present | human absent |
|---|---:|---:|
| detected somewhere | 10 | 5 |
| never detected | 4 | 7 |

Headline: **10/14 present terms found anywhere; 7/12 absent terms clean.**
Max-confidence AUC is **.685**. Four present terms are never found: `glass`
(named `cup` by the human, so this is a vocabulary mismatch), `window`
(max .08), `pitcher` (max confidence never above .05) and `fork` (max .14,
visible for only 2.5 s). Five absent terms receive detections: `bowl`, `mug`,
`bottle`, `spoon`, `television`.

The split by object scale is informative:

| group | found anywhere |
|---|---:|
| background / large (`person`, `refrigerator`, `microwave`, `oven`, `cabinet`, `window`, `television`) | 6/7 |
| tabletop / small (`plate`, `cup`, `glass`, `napkin`, `sandwich`, `table`, `pitcher`, `fork`) | 5/8 |

The false detections on absent terms are not random noise. The best-scoring
box for each was inspected visually:

- `mug` .77 on a tall pink-drink **glass**;
- `bottle` .62 on the same kind of **glass**;
- `bowl` .60 on a **plate** piled with chips;
- `spoon` .32 on a low-confidence, unaudited box (4/47 contexts);
- `television` .91 on a **wall-mounted TV** that the human annotation missed
  (see audit).

So the dominant absence failure is *category substitution*: the prompt-
conditioned detector finds a nearby object of a confusable category and scores
it highly. Terms such as `laptop`, `bicycle`, `umbrella` and `book` are clean
because the scene contains nothing close to them, not because the detector
verifies absence.

## Result 2: more contexts do not fix it

Term-level rule: object present in the clip but never detected in the sampled
contexts = false negative; object absent but detected in at least one sampled
context = false positive. Averages over evenly spaced phases:

| k contexts | phases | false negative | false positive | per-context sensitivity | per-context false positive |
|---:|---:|---:|---:|---:|---:|
| 4 | 286 | .351 | .364 | .666 | .198 |
| 8 | 130 | .343 | .367 | .663 | .195 |
| 12 | 78 | .333 | .361 | .663 | .202 |
| 16 | 52 | .321 | .417 | .668 | .195 |
| 24 | 26 | .286 | .333 | .658 | .198 |
| 32 | 26 | .286 | .417 | .667 | .205 |
| 48 | 26 | .286 | .417 | .667 | .198 |

Scanning 12x more contexts moves the term-level false-negative rate from .351
to .286, and the false-positive rate does not improve at all. Per-context
sensitivity is flat at about .66. The bottleneck is the detector's prompt-
conditioned semantics, not the sampling budget: a context either contains a
confidently mislabelled look-alike or it does not, and adding contexts cannot
turn a miss into a hit for a term the detector never grounds (`glass`,
`pitcher`, `fork`, `window`).

A second observation: at .22 the per-context false-positive rate is ~.20 even
for terms that are absent in the whole clip. If absence were a per-context
decision, one in five contexts would claim the object is there.

## Result 3: the shipped query path cannot be an absence operator

"Where is the X?" for the same 26 terms, budget 8:

| mode | present terms abstained | absent terms answered | absent answers with a support label |
|---|---:|---:|---:|
| detector | **5/14** | **4/12** | 2/4 |
| hybrid | **2/14** | **8/12** | 5/8 |

Detector-only mode is conservative: it misses five present terms, including the
table, but only answers four absent terms. Hybrid mode buys recall with the
CLIP fallback and then fabricates confident locations for two thirds of the
absent terms (`laptop on the shelf`, `knife on the shelf`, `sofa on the desk`).
Neither mode can be used as a negative test. This is the measured version of
the paper's existing caveat that the hybrid score is not pure detector
accuracy.

Support attribution is worse than "occasional". Of the 13 detector-mode
answers, nine carry a support label and **eight of those say `shelf`**; the
hybrid run produces 12 `shelf`, one `table`, one `desk`. `person on the shelf`,
`refrigerator on the shelf` and `microwave on the shelf` are the same 2-D
containment failure: the kitchen cabinets are behind the subjects in the image,
so a high-confidence `shelf` box overlaps them and wins `_support_with_box`.
The dining table itself is detected in the raw scan (34/47 contexts, max .57)
but `where is the table?` abstains in detector mode, because `table` is routed
as a support target rather than an object. This replicates the clean-clip table
ambiguity on a second, unseen clip; it is not tuned.

## Result 4: state-at-time is not usable yet

Five questions were asked with an explicit `at_time` and human-checked
expected outcomes. A question counts as found when the engine returns a
non-abstained binding.

| question | expected | found | reported support | found | support |
|---|---|---|---|---|---|
| where is the cup at 24 s? | found, table | yes | `shelf` | ok | **wrong** |
| where is the pitcher at 24 s? | not found | abstained | — | ok | ok (trivial) |
| where is the pitcher at 8 s? | found, held | **abstained** | — | **wrong** | ok (trivial) |
| where is the fork at 0 s? | found, held | **abstained** | — | **wrong** | ok (trivial) |
| where is the fork at 24 s? | not found | abstained | — | ok | ok (trivial) |

Found/abstained matches 3/5. The only non-trivial support check is wrong: the
cup is on the table, not on a shelf. The pitcher (8 s) and fork (0 s) are both
in-hand objects the detector never grounds. A "held" state cannot be reported
today: the answer path has no support value for it, and a missing support is
indistinguishable from an ungrounded detection.

## Annotation audit (post hoc)

Inspecting the `television` boxes showed a real wall-mounted TV in the upper
left of the frame, visible from about 35 s. The pre-registered label was
wrong. Correcting only that label gives:

| | original | audited |
|---|---:|---:|
| present terms found anywhere | 10/14 | **11/15** |
| absent terms clean | 7/12 | **7/11** |
| term-level false negative at full scan | .286 | **.267** |
| absent terms answered, detector mode | 4/12 | **3/11** |
| absent terms answered, hybrid mode | 8/12 | **7/11** |

The audit does not change the conclusion: the false-positive set shrinks from
five to four terms, and the detector still answers three absent terms and
misses four present ones. As with the Cook QA audit, the pre-registered numbers
stay the headline; the audited numbers are a diagnostic.

## Post-hoc: fixed-label detector check

The false positives above are prompt-conditioned: YOLO-World is asked for
`mug` and scores a glass. A fixed-label detector cannot substitute arbitrary
categories, so the same 26 probes were scanned once with COCO YOLOv8-S
(`yolov8s.pt`, 80 classes, one 47-context pass, floor .05, thresholds applied
post hoc). This was run after the main result and is a diagnostic, not part of
the pre-registered protocol. Six probes have no COCO class (`plate`, `napkin`,
`cabinet`, `window`, `pitcher`, `mug`), so the comparison is restricted to the
20 covered terms, with `television` moved to present after the audit:

| detector (floor) | present found (of 10) | absent clean (of 10) |
|---|---:|---:|
| YOLO-World-S, .22 | 8/10 | 7/10 |
| YOLO-World-S, .30 | 8/10 | 7/10 |
| YOLO-World-S, .50 | 7/10 | 8/10 |
| COCO YOLOv8-S, .25 | 9/10 | 8/10 |
| COCO YOLOv8-S, .50 | 8/10 | **10/10** |

At .50 the fixed-label detector returns no detections for any of the ten
absent terms, while still finding 8/10 present terms. Its remaining misses are
`glass` (the label is `wine glass`; the tumblers fire only once, at .27) and
`fork` (max .21). The world model keeps `bowl` (.59 on the plate of chips) and
`bottle` (.62 on a glass) even at .50. COCO also confirms the audit: it detects
`tv` at .73.

The gain is not only precision. On the same frames COCO finds `sandwich` in
47/47 contexts (max .82) and `dining table` in 47/47 (max .89), versus 1/47
(max .31) and 34/47 (max .57) for YOLO-World-S. Fixed-label supervision is
much better on the classes it knows, and it has a usable high-precision
operating point for negative answers. The cost is coverage: a user term outside
the 80 classes cannot be checked at all. A practical system could therefore
allow absence claims only from a fixed-label detector on covered classes, and
use the open-vocabulary path for presence/exploration -- the opposite of the
current single-model design.

## Interpretation

1. **Absence is bounded by detector semantics, not by memory size or sampling
   budget.** Across 4 to 48 contexts, false negatives stay at .29--.35 and
   false positives at .33--.42. Exhaustive scanning is cheap (13.5 s for 1,222
   calls) but buys little.
2. **A negative answer must name its evidence source.** "No plate on the
   filing cabinet" is supportable only as "the query detector found no plate in
   47 retained contexts"; it is not a fact about the world. The measured
   per-term error rates should travel with the claim.
3. **Hybrid CLIP fallback is unsafe for negative or state answers.** It raises
   present-term recall from 9/14 to 12/14 but converts 4/12 false answers into
   8/12. Any absence-aware product path should default to detector-only
   abstention.
4. **State answers need a support relation that does not exist yet.** The
   current 2-D containment heuristic collapses to the background surface behind
   the object. An explicit "held / on <surface>" state, backed by hand contact
   and depth, is the missing capstone; it cannot be recovered by prompt tuning.
5. **A fixed-label detector is a viable absence substrate on its own classes.**
   The post-hoc COCO check reaches zero false positives on 10/10 covered absent
   terms at .50 while finding 8/10 present terms, and it fixes the `sandwich`
   and `table` recall holes. It cannot answer the six probes outside its 80
   classes. This points to a two-detector design -- fixed-label for negative
   claims, open-vocabulary for exploration -- rather than a single model.
6. **What would still change the conclusion:** independent human labels per
   context (rather than intervals), a second fixed-label detector with a
   different class list, or a prompt-free objectness/verifier model evaluated
   on the full 26-term bank. Until then, the honest claim is "SPORE stores
   evidence that can confirm presence, but cannot certify absence with its
   open-vocabulary detector".

## Reproduce

```bash
python scripts/fetch_aria_clip.py --sequence eat_1
python scripts/spore_absence.py \
  --clip data/footage/aria_eat_1_45s.mp4 \
  --probes data/qa/eat_1_absence.json \
  --dump research/spore/results/eat1_absence/store.json \
  --out research/spore/results/eat1_absence/report.json

# Post-hoc fixed-label check (downloads yolov8s.pt if absent)
python scripts/spore_absence_fixed.py \
  --store research/spore/results/eat1_absence/store.json \
  --probes data/qa/eat_1_absence.json \
  --world-report research/spore/results/eat1_absence/report.json \
  --out research/spore/results/eat1_absence/fixed_label.json
```

The first script writes a JSON report and a markdown summary; the second adds
the fixed-label comparison. Store dumps and run artifacts are gitignored; only
this note and the probe file are committed.

## Threats to validity

- One 45 s clip and 26 terms, annotated by the same agent that knows the
  detector stack. One label error was found and audited; others may remain.
- The term bank deliberately includes near-synonyms (`cup`/`glass`) and large
  background appliances. Both are reported separately where they matter.
- The detector floor for the scan is .05, below the query floor, to record
  scores; the query-path result uses shipped defaults. This is not a tuned
  comparison.
- The absence labels are clip-level, not per-context facts: a term can be
  genuinely absent at a context even if present elsewhere. Per-context
  sensitivity/specificity use the annotated intervals, which are coarse
  (±2 s).
- The COCO check is post hoc, uses a different label space and operating
  points, and covers only 20 of the 26 probes; the `glass -> wine glass`
  mapping is loose. It is evidence that class competition matters, not a
  calibrated second benchmark.
- Pseudo-GT is not used here: ground truth is human presence labels, not
  YOLO-World. That is the point of the experiment, but it is still a small
  human sample.
- The 300-capsule cap was reached. The store still retained 47 contexts, so the
  scan covers the full window; the cap affects capsule crops, not context
  coverage.

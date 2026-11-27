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

## Held-out validation: fixed-label verifier on eat_2

The fixed-label check above was post hoc on the same clip that motivated it, so
the verifier (`chronicle/perception/absence.py`, default threshold .50) and the
same 26-term bank were run on a fully unseen seventh window, `eat_2`
(30--75 s). The probe set was committed as `3653fbd` before the first ingest;
the verifier threshold and aliases were fixed on `eat_1`. The scene is another
dining room with a different layout: `bowl` and `bottle` are genuinely present
(on the table and kitchen island), `television` is visible only around 22--27 s,
and `mug`, utensils, `pitcher`, electronics and outdoor objects are absent.

World detector on this clip (all 26 terms, floor .22): **13/15 present found,
8/11 absent clean**, AUC .864 -- noticeably better than on `eat_1` (10/14,
7/12, .685), so clip-to-clip variance is large. The misses are `glass` (the
vocabulary mismatch again) and `window` (.085); the false positives are `mug`
(.79, the dining glasses), `spoon` (.31) and `sofa` (.45). The shipped query
path abstains on 3/15 present terms and answers 1/11 absent terms in
detector-only mode; hybrid abstains on 0/15 and answers 4/11.

Fixed-label verifier on the 20 COCO-covered terms:

| operating point | TP | FP | FN | TN | positive precision | positive recall | false-presence rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| verifier, threshold .50 (pre-registered) | 8 | **0** | 3 | 9 | **1.00** | .727 | **0.00** |
| verifier, threshold .25 (post hoc) | 11 | 2 | 0 | 7 | .846 | **1.00** | .222 |

At .50 the verifier makes no false-presence claim on any of the nine covered
absent terms, but it misses `oven` (max .36), `bottle` (.48) and `television`
(.33) -- all genuinely present, all below threshold. The world detector found
all three (`.47`/.55/.77). Lowering the verifier to .25 recovers every present
term, but `fork` (.41) and `knife` (.32) become false presences; the at-time
questions stay at 4/5 either way (`tv at 24 s` is threshold-sensitive, `fork
at 12 s` flips to a false present at .25).

The fair comparison is on the same 20 covered terms:

| detector (floor) | present found (of 11) | absent clean (of 9) |
|---|---:|---:|
| YOLO-World-S, .22 | 10/11 | 7/9 |
| YOLO-World-S, .50 | 8/11 | 9/9 |
| COCO YOLOv8-S, .25 | **11/11** | 7/9 |
| COCO YOLOv8-S, .50 | 8/11 | **9/9** |

The held-out result is more mixed than the post-hoc `eat_1` check. Fixed-label
detection does not dominate the world detector at matched precision here: at
.50 both reach 9/9 clean, and at the permissive end COCO trades two false
presences for one extra true positive over YOLO-World. What transfers is the
**existence of a clean high-precision point** and the much better recall on
unambiguous classes (`sandwich` 30/47, `table` 47/47, `cup` 47/47 versus
5/47, 42/47, 45/47 for the world model). What does not transfer is calibrated
confidence across classes: `television` .33, `oven` .36 and `bottle` .48 are
clearly visible yet score below the same .50 that a `sandwich` clears at .81.
A single global threshold leaves recall on the table; per-class calibration or
a two-band verdict (confident present / weak evidence / not observed) is the
next step.

Coverage is the other blocker: 6/26 terms have no COCO class, and they include
`plate` -- the most common object in this scene, which the world detector finds
in 47/47 contexts and COCO cannot represent at all -- plus `napkin`, `cabinet`,
`window`, `mug` and `pitcher`. A practical absence operator cannot be built on
COCO alone; it needs a tableware/appliance label bank or a fixed-label detector
with a wider class list. The honest conclusion is that fixed-label detection is
the right *shape* for negative claims, but this particular detector and
threshold are not yet a deployable absence verifier.

### Model scale on the same held-out clip

The verifier accepts any fixed-label model, so the same `eat_2` store was
re-scanned with YOLOv8-S/M/L at the pre-registered .50 threshold (models
downloaded on demand; 47 calls each, 2-12 s):

| model (.50) | TP | FP | FN | TN | precision | recall | at-time |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8-S | 8 | 0 | 3 | 9 | 1.00 | .727 | 4/5 |
| YOLOv8-M | 8 | 1 | 3 | 8 | .889 | .727 | 5/5 |
| YOLOv8-L | **10** | **0** | **1** | **9** | **1.00** | **.909** | **5/5** |

Scale is not monotonic: M gains `bottle` and `television` but loses `bowl` and
`glass` and adds a `fork` false presence at .56; L recovers all four and keeps
zero false presences. The remaining miss is `oven` (zero detections at every
scale) -- the object is a range under a hood, and COCO has no `stove`/`range`
class. The score shifts confirm that per-class confidence is model-dependent,
so a threshold calibrated on one backbone does not transfer unchanged to
another. L at .50 was the best fixed-label operating point seen on `eat_2`, so
it was fixed as the validation model and run on the next unseen window below.

### Cross-clip validation: eat_3

A further unseen window, `eat_3` (30--75 s, probe set committed as `dbb5c56`
before ingest), moves to a living/dining area: sofa, wooden hutch and windows,
no kitchen appliances, and a pitcher held only around 27--35 s. The world
detector does worse here: **8/11 present found, 9/15 absent clean** (AUC .715;
misses `glass`, `window`, `pitcher`; false positives `bowl`, `mug`, `bottle`,
`spoon`, `television`, `book`). The verifier covers the same 20 terms (6
present, 14 absent) and at .50 with YOLOv8-L gives **TP 6, FP 2, FN 0, TN 12**
-- precision .75, recall 1.0, false-presence rate .143.

The two false presences are confident and instructive:

- `bowl` .73 at 24.6 s: a plate piled with chips, the same look-alike that the
  world detector substitutes;
- `television` .70 at 5.4 s: the dark rectangle between the window blinds. The
  human annotation inspected it at full resolution and called it a
  window/opening, not a TV. A closed label set still chooses the plausible
  class on an ambiguous dark panel.

At-time questions score 4/5; the one non-match is `pitcher` at 28 s, which is
**unsupported** because COCO has no pitcher class -- a coverage gap, not a
wrong verdict. On the same covered terms the world detector finds 5/6 present
and leaves 9/14 absent clean at .22 (4/6 and 11/14 at .50), so L at .50
dominates the world model on both axes on this clip.

Aggregating the two held-out windows (`eat_2` + `eat_3`, same 20 covered
terms):

| detector (.50 unless noted) | TP | FP | FN | TN | precision | recall | false-presence rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO-World-S, .22 | 15 | 7 | 2 | 16 | .682 | .882 | .304 |
| YOLOv8-L fixed-label, .50 | **16** | **2** | **1** | **21** | **.889** | **.941** | **.087** |

This is the strongest evidence for the two-detector design so far: on the
classes it covers, a fixed-label detector at a fixed .50 threshold is both more
complete and much less likely to claim a false presence than the
open-vocabulary path. It is still not an absence verifier in the strict sense:
two confident false presences remain, 6/26 terms are unsupported, and both
windows come from the same household/domain.

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
5. **A fixed-label detector is the right substrate for negative claims, but a
   negative verdict is still bounded.** Across two held-out windows and the 20
   COCO-covered terms, YOLOv8-L at .50 is both more complete and much safer
   than YOLO-World-S at .22 (precision .889 vs .682, recall .941 vs .882,
   false-presence .087 vs .304). It still makes two confident look-alike
   errors (a plate as `bowl`, a dark window as `television`), covers only 20/26
   terms (`plate` has no class), and both windows are the same
   household/domain. The design is validated; the verifier is not. The next
   step is a wider fixed label bank plus per-class calibration or a two-band
   verdict (confident present / weak evidence / not observed), not a global
   threshold on an 80-class detector.
6. **What would still change the conclusion:** a fixed-label detector with a
   tableware/appliance class list and per-class calibration, independent human
   labels per context (rather than intervals), or a prompt-free
   objectness/verifier model evaluated on the full 26-term bank. Until then,
   the honest claim is "SPORE stores evidence that can confirm presence, but its
   open-vocabulary detector cannot certify absence, and its fixed-label
   fallback covers only 20/26 terms".

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

# Held-out fixed-label verifier on eat_2
python scripts/spore_absence.py \
  --clip data/footage/aria_eat_2_45s.mp4 \
  --probes data/qa/eat_2_absence.json \
  --dump research/spore/results/eat2_absence/store.json \
  --out research/spore/results/eat2_absence/report.json
python scripts/spore_absence_verify.py \
  --store research/spore/results/eat2_absence/store.json \
  --probes data/qa/eat_2_absence.json \
  --out research/spore/results/eat2_absence/verify.json

# Cross-clip validation of the L verifier on eat_3
python scripts/spore_absence.py \
  --clip data/footage/aria_eat_3_45s.mp4 \
  --probes data/qa/eat_3_absence.json \
  --dump research/spore/results/eat3_absence/store.json \
  --out research/spore/results/eat3_absence/report.json
python scripts/spore_absence_verify.py \
  --store research/spore/results/eat3_absence/store.json \
  --probes data/qa/eat_3_absence.json \
  --model yolov8l.pt \
  --out research/spore/results/eat3_absence/verify_l.json
```

The first script writes a JSON report and a markdown summary; the second adds
the fixed-label comparison; the third evaluates the fixed-label verifier. Store
dumps and run artifacts are gitignored; only this note and the probe files are
committed.

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
- The COCO check is post hoc on `eat_1`, uses a different label space and
  operating points, and covers only 20 of the 26 probes; the `glass -> wine
  glass` mapping is loose. The held-out `eat_2` verifier run repeats the clean
  high-precision point but misses 3/11 present terms and shows per-class score
  miscalibration; it is evidence for the shape of a negative path, not a
  calibrated benchmark.
- `eat_1` and `eat_2` are both dining scenes, likely from the same household
  and capture setup. The verifier result is a within-domain held-out check, not
  evidence of transfer to a new room or activity; the state work already shows
  that a living-room replication can fail on support detection.
- Pseudo-GT is not used here: ground truth is human presence labels, not
  YOLO-World. That is the point of the experiment, but it is still a small
  human sample.
- The 300-capsule cap was reached. The store still retained 47 contexts, so the
  scan covers the full window; the cap affects capsule crops, not context
  coverage.

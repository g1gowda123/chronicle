# Grounded presence, absence, and state answers

Status: experiment complete (2026-11-27), branch `research/spore-perception`.
Code: `chronicle/perception/spore.py` (`SporeConfig.state_queries`, opt-in) and
`scripts/spore_state_eval.py`. Probe set: `data/qa/cook_0_state_45s.json`.
Raw report (gitignored): `research/spore/results/cook0_state/report.{json,md}`.

This experiment is the query-path complement to [`ABSENCE.md`](ABSENCE.md).
`ABSENCE.md` measures per-context detector sensitivity and finds that absence is
bounded by detector semantics. Here the question is different: if the system is
allowed to answer explicit yes/no and state questions, how often does it claim a
present object that is not there, and what does it miss?

## Answer semantics

The opt-in path adds a three-way verdict instead of forcing every question
through "where is X?" binding:

- **present** - the noun was detected on the asked surface, with a timestamp
  and bounding box;
- **moved** - it was detected near the queried time on a different surface;
- **stayed** - for "did X move?", its dominant surface did not change;
- **not observed** - no supported detection in the scanned contexts;
- **unknown** - too little repeated evidence to decide.

A negative verdict is always **not observed**, never "absent from the world".
This is the same epistemic rule as `ABSENCE.md`, made operational.

## Protocol and provenance

**This is not a strict pre-registration.** The 19-question file was written and
labelled before the first SPORE run, but it was committed with the analysis, not
before the run. `ABSENCE.md` uses the stricter standard (probe file committed
before the first ingest); by that standard the `cook_0` result is post-hoc and
the 19-question number is a protocol-defined diagnostic, not a held-out
pre-registered estimate. The `play_2` replication was committed before its
first run and is the clean pre-registered check; it fails (**5/14**), which
bounds every positive claim below. The `play_0` surface-head replication was
also committed before its first run; it fails (**5/12**, **1/5 false
presence**) and is the reason the surface head is not presented as a fix.

- Clip: Aria Gen 2 Pilot `cook_0`, window 50--95 s, 640x480, 10 fps after trim.
  This is the fifth Aria window after the four in `DOMAINS.md`; it was not used
  in any earlier SPORE experiment or default.
- Window selection: sampled frames from the full preview were inspected to pick
  a stable counter scene. The manifest entry was added after that inspection and
  before the first SPORE ingest.
- Probe set: 19 questions written and labelled before the first SPORE run on
  this clip: 6 positive presence, 6 absent-object presence, 4 "still" questions
  and 3 movement questions. Positive controls are visible in all 42 sampled
  frames; absent terms are not visible anywhere in the 45 s window.
- Annotation: visual inspection of a 2 s contact sheet plus full frames at 0, 3,
  10, 20, 30, 40, 43 and 44 s. The annotation was assisted by an image-analysis
  model; it is independent of the YOLO-World query detector but is not a human
  annotator. See threats below.
- No threshold, place rule, or question was changed after the first run. The
  later probes (kettle detections, competing-label check, support geometry) were
  post-hoc diagnostics and are reported as such; the original 19-question result
  is the headline.
- A second, **post-hoc** transfer check uses nine questions on `eat_0`, a
  dining room that was already part of the held-out location QA. It is reported
  separately and is not counted as a pre-registered result.

## Setup

| item | value |
|---|---|
| Ingest | `SporeConfig` defaults, 4 fps, 640 px, 29–31 s (real hand landmarks) |
| Store | 47 contexts, 249 capsules, 3.3 MB split store on disk (1.2 MB index + 2.1 MB evidence) |
| Scan | 32 of 47 contexts per clip-level presence term, 1 detector call each |
| State path | `state_queries=True`, `state_contexts=32`, floor .30, 2 observations |
| Verification | generic 43-label bank, margin .05, one extra call per positive answer |
| Peak VRAM | 1.34 GB (state run), 0.73 GB on the reused-store ablation |

A parallel run with the colour-fallback hands (the MediaPipe landmarker was
temporarily missing; 220 capsules, 24.1 s ingest) produced the identical 16/19
and 0/6, so the verdicts here are not sensitive to the hand backend.

Presence is a clip-level question unless the question itself names a time; the
harness's evaluation timestamp is not a state constraint. "Still" uses a 1.5 s
window at the query time plus a whole-clip persistence check. Movement compares
the dominant support in the two halves of the clip and treats detector aliases
(table/work table/desk/countertop) as one surface.

## Result 1: the state path is safe on absent probes, the default path cannot answer state

| mode | questions scored | correct | false presence | missed presence |
|---|---:|---:|---:|---:|
| state (opt-in) | 19 | **16 (0.842)** | **0/6** | 3/10 |
| default binding | 12 | 11 (0.917) | 0/6 | 1/6 |

The default score is not comparable: it counts a non-abstained location answer
with the right support as a pass, and it has no representation for "no",
"still" or "moved". On the four "still" and three movement questions it is
unscored. The state path answers all 19 with an explicit verdict.

State examples:

```text
Is there a toaster on the countertop?      yes — toaster on the countertop at 35.6s
Is there a kettle on the countertop?       ambiguous evidence: toaster outscores kettle ...
Is the plate still on the countertop?      yes — plate still on the countertop at 44.2s
Did the toaster move?                      no — toaster stayed on the countertop across 25 observations
Did the kettle move?                       not enough evidence to tell whether kettle moved
```

The three misses are detector-bound:

- `rice cooker`: one .32 detection in 47 contexts, below the two-observation
  persistence rule; when a competing bank is present the same box is labelled
  `toaster` at .57. The system correctly refuses.
- `cutting board`: zero detections at any context with the term + support
  vocabulary.
- `still rice cooker`: same detector miss near the end.

## Result 2: transfer check on a second room (`eat_0`, post-hoc)

The same harness was then run on the earlier eating window (`eat_0`, 30–75 s)
with nine state questions. This clip was already used for the held-out location
QA in Section 5.14, so the check is post-hoc rather than pre-registered; the
questions were written from fresh frame inspection before the state run and the
labels were not produced by the detector. The dining table is the target
surface; the room also contains a shelving unit with a kettle and coffee maker,
so "no kettle on the table" would be a place-qualified absence.

| mode | questions | lenient correct | strict support correct | false presence |
|---|---:|---:|---:|---:|
| state | 9 | **9/9 (1.000)** | **6/9 (0.667)** | **0/4** |
| default binding | 6 | 5/6 (0.833) | 5/6 | 1/4 |

The verdict behaviour transfers: the state path claims no absent object on the
table (laptop, hammer, fork, bottle), and the default binding path answers the
bottle question (false presence). The strict-support column exposes the
remaining bottleneck: at the clip-level scan the detector labels the wooden
dining table as `desk`, so `plate on the desk` and `cup on the desk` are lenient
passes only; the end-of-clip `still` questions happen to land on frames where
the same surface is labelled `table`. This is the same support-attribution
failure reported in Section 5.17, not a new one.

## Result 3: pre-registered replication on `play_2` is negative

The `play_2` probe (14 questions, committed as `032b2dd` before the first
ingest) is a living-room card-game scene: a round coffee table with an orange
game box and a book, a gray sofa, a leather armchair and a woven basket on the
floor.

| mode | scored | correct | strict support | false presence | missed presence |
|---|---:|---:|---:|---:|---:|
| state (pre-registered) | 14 | **5/14 (0.357)** | 5/14 | **0/5** | **7/7** |
| default binding | 10 | 6/10 (0.600) | 6/10 | 0/5 | 4/5 |

All five absent-object probes are answered correctly, so the safety result
survives. **Every positive is missed** by the state path: box, book, sofa,
chair, basket, still-box and still-basket. The default path finds the book but
misses the other four positives.

Post-hoc diagnosis (no default, threshold, or question was changed):

- Across 48 contexts the support detections are `desk` 66, `table` 14,
  `basket` 12, `shelf` 1, and **`floor` 0**. The living-room floor is never
  detected, and the round coffee table is usually labelled `desk`.
- Raw term hits at floor .30: `box` 0, `book` 8 (1 supported), `sofa` 85 (62
  supported, all `desk`), `chair` 43 (21 supported, `desk`/`basket`), `basket`
  12 (all supported by its own `basket` label), and 0 for every absence term.
- So the failure is **not retention**: sofa and chair are detected and even
  supported, but never on `floor`; `desk` cannot satisfy a floor query. The
  basket is its own support because `basket` is both an object and a support
  label. The orange game box is never detected at all. The book has one .35
  hit, rejected by the two-observation persistence rule.

This refutes the assumption that the `cook_0`/`eat_0` state behaviour transfers
to a new room. False-presence safety transfers; positive recall does not,
because it depends on a support label that the scene's detector does not emit.
The state path is tabletop/kitchen-specific until the support vocabulary and
detector cover the room's actual surfaces.

## Result 4: competing labels remove the one false presence

The first run without label competition produced **15/19 (0.789)** with
**1/6 false presence**: `Is there a kettle on the countertop?` answered "yes"
at 16.0 s with confidence .51. Re-inspecting the winning box with the generic
43-label bank showed the same box is a **toaster** at .50 while `kettle` drops
to .35; with a competing bank the answer is downgraded to
`ambiguous evidence: toaster outscores kettle`. True positives survive the same
check (`toaster` .60, `plate` .75, `knife` .40, `mug` .70). The ablation is
therefore:

| variant | correct | false presence |
|---|---:|---:|
| no competing-label check | 15/19 (0.789) | 1/6 |
| competing-label check | **16/19 (0.842)** | **0/6** |

This is consistent with the paper's prompt-set finding: the detector's
text-conditioned layers let a target-only prompt inflate a confusable label, and
the competing pass is a cheap way to expose it. It is a verification filter,
not a calibrated absence test.

## Result 5: support geometry matters; missing support detections remain

The original `_support_with_box` assigned the **shelf above the toaster** as its
support at the final frame, turning `still_toaster` into a false "moved" answer.
A stricter `_support_under` rule now rejects a support whose centre is above the
object's centre unless the support box contains the object centre. On this clip
it removes that false move; the toaster is then answered from a context 1.2 s
later where the countertop is detected. The same 2-D containment failure is what
`ABSENCE.md` reports on `eat_1` (`person on the shelf`); the geometry rule helps
only when the correct support is detected somewhere in the window.

At 43.2 s the detector finds `shelf` and `toaster` but **no countertop** at all
(verified with a support-only pass), so the answer cannot be grounded on that
frame. The ±1.5 s still window is what recovers the 44.2 s countertop detection.
This is an evidence-coverage limit, not a selection bug.

## Result 6: post-hoc surface-head fallback repairs support, not detection

`state_surface_support=True` reuses the existing ADE20K SegFormer-B0 head when
the box detector cannot name the surface under a state placement.  It is a
post-hoc diagnostic: the pre-registered `play_2` number (5/14) stands, the flag
is off by default, and the pre-registered `play_0` test below **does not
replicate** the gains shown here.

| clip | without head | with head | strict support | false presence |
|---|---:|---:|---:|---:|
| `cook_0` | 16/19 | 16/19 | 16/19 | 0/6 |
| `eat_0` | 9/9 (6/9 strict) | **9/9 (9/9 strict)** | 9/9 | 0/4 |
| `play_2` | 5/14 | **7/14** | 7/14 | 0/5 |

The head fixes exactly the support-label failures it should:

- `eat_0`: the dining table changes from `desk` to `table`, so the plate, cup
  and moved-plate answers become strict passes.
- `play_2`: the sofa and chair receive `floor` (the box detector never emitted
  a floor box), recovering 2 of the 7 positives.
- `cook_0`: no change, no regression.

It does **not** fix the remaining `play_2` misses: the orange box is never
detected as an object, the book has one low-confidence hit, the basket is
supported by its own label, and the living-room table is segmented as
`rug`/`floor` rather than `table`.  The head repairs *where*, not *what*.

Cost is one cached segmentation per scanned context: roughly +0.2–0.3 s per
query (play_2 0.72 s vs 0.39 s; eat_0 0.88 s vs 0.61 s; cook_0 0.63 s vs
0.43 s) and about +0.18 GB peak VRAM (0.91 GB on play_2).  It stays opt-in.

## Result 7: pre-registered surface-head test on `play_0` is negative

The `play_0` probe (12 questions, committed as `aa9c72b` before the first
ingest, with `state_surface_support=true` in the run config) was the clean test
of the surface-head idea: a stable living-room seating area with a round coffee
table and a red/white game box, a gray sofa, a beige pouf and a navy blanket on
the rug.

| mode | scored | correct | strict support | false presence | missed presence |
|---|---:|---:|---:|---:|---:|
| state + surface head (pre-registered) | 12 | **5/12 (0.417)** | 5/12 | **1/5** | 4/5 |
| default binding | 8 | 5/8 (0.625) | 5/8 | 0/5 | 3/3 |

The surface-head gain did **not** replicate.  The only correct positives are
`still_sofa` (floor at 40 s); `presence_box`, `presence_sofa`,
`presence_blanket`, `still_box`, `moved_box` and `moved_sofa` are wrong.  The
false presence is a cup at .31-.44 on `desk`/`shelf` (five observations, so the
persistence rule accepts it), and the sofa is reported as moved because the
surface head assigns different supports across the clip.

Post-hoc diagnosis (no default or probe changed):

- Raw term hits across 48 contexts at floor .30: `box` 0, `blanket` 0, `sofa`
  96 (78 supported, mostly `desk`), `cup` 6 (5 supported, a false positive),
  `laptop` 1 (rejected by persistence), and 0 for plate/bottle/hammer.
- The box and blanket are never detected, so no support head can recover them.
- The sofa is detected and supported, but the box detector says `desk`; the
  surface head only sometimes overrides it to `floor`, and the disagreement
  between frames creates the false "moved" verdict.
- The cup false positive is object-side: YOLO-World hallucinates a cup at low
  confidence across several frames, and the competing-label bank does not
  outscore it.

This is the pre-registered boundary result: the surface head repairs an
individual support label in `eat_0` and a post-hoc rerun in `play_2`, but it
does **not** make state answers safe or accurate in a new living room.  The
state path remains an experimental tabletop operator, not a general state
tracker, and the 0/5 false-presence results on earlier clips should not be
generalised.

A post-hoc fixed-label veto (`state_fixed_verify=True`, COCO YOLOv8-S at .50,
using `chronicle.perception.absence.FixedLabelVerifier`) removes the cup false
presence on `play_0`: the score becomes **6/12 with 0/5 false presence**, and
`still_sofa` is preserved because the veto is applied only to presence
questions.  `cook_0` stays 16/19 and `eat_0` stays 9/9.  The false move
(`moved_sofa`) remains; the veto checks object presence, not surface
consistency.  This is a post-hoc mechanism check, not a re-run of the
pre-registered result; `play_1` is the pre-registered test, and it does **not**
replicate the gain (Result 8).

## Result 8: the pre-registered fixed-label veto does not replicate the `play_0` gain

The `play_1` probe (13 questions, committed as `13ca794` before the first
ingest, with `state_surface_support=true` and `state_fixed_verify=true` at .50)
is a fourth living-room scene: round coffee table with an orange game box and
cards, gray sofa, gray armchair, tan pouf, navy blanket on the rug.

| mode | scored | correct | strict support | false presence | missed presence |
|---|---:|---:|---:|---:|---:|
| state + surface head + fixed veto (pre-registered) | 13 | **7/13 (0.538)** | 7/13 | **0/5** | 4/6 |
| state + surface head, veto off (post-hoc ablation) | 13 | **8/13 (0.615)** | 8/13 | **0/5** | 3/6 |
| default binding | 9 | 5/9 (0.556) | 5/9 | 0/5 | 4/4 |

The pre-registered run does **not** reproduce the post-hoc `play_0` benefit.
On `play_1` the open-vocabulary path already withholds every absent probe
(`cup`, `bottle`, `laptop`, `plate`, `hammer`), so the veto has no false
presence to remove; its only measurable effect is to downgrade the true
`presence_chair` at `floor`, because COCO YOLOv8-S finds no chair at .50.  The
post-hoc ablation on the same store scores **8/13 with 0/5 false presence**.
The `play_0` cup removal therefore remains a single post-hoc observation, not a
replicated safety mechanism, and the veto is off by default.

The remaining misses are the same artifacts as `play_0`: `box` is supported by
its own `box` label (3 hits at .36-.51), `blanket` is supported as `desk`/`shelf`
(34 hits, no floor), and the sofa is reported as moved because its support
flips across contexts.  Positive recall remains support- and detector-bound.

## Result 9: excluding an object as its own support recovers the basket

`box` and `basket` are both object terms and support labels.  `_support_under`
now excludes the queried object's own class from its support candidates, so a
basket cannot be "on the basket".  This is a correctness fix, not a threshold
change.

Post-hoc on the existing stores, with the surface head enabled:

- `play_2`: **7/14 -> 8/14**; `presence_basket` becomes present on `floor`
  (the self-support artifact was the only reason it was missed).  Box/book and
  the still/moved basket questions remain wrong, and false presence stays 0/5.
- `play_1`: unchanged at 7/13; the box has no non-self support in this clip.
- `cook_0`/`eat_0`: unchanged (no queried term is also a support label).

The fix removes a real bug, but it does not create support evidence where the
detector has none; positive recall is still detector- and surface-bound.

## Result 10: the soft-floor rule does not fire on a large area rug

The `play_3` probe (13 questions, committed as `a550347` before the first state
ingest, surface head on and fixed veto off) is a held-out board-game living
room: a large patterned area rug, round coffee table with an orange box, gray
sofa and brown armchair.

| mode | scored | correct | strict support | false presence | missed presence |
|---|---:|---:|---:|---:|---:|
| state + surface head (pre-registered) | 13 | **8/13 (0.615)** | 8/13 | **1/5** | 3/6 |
| state + surface head + fixed veto (post-hoc) | 13 | **9/13 (0.692)** | 9/13 | **0/5** | 3/6 |
| default binding | 9 | 5/9 (0.556) | 5/9 | 0/5 | 4/4 |

The soft-floor rule did **not** fire. The open-vocabulary detector finds the rug
(14 hits at .31-.48), but the surface head's floor fraction in the bands around
the rug box is only .15-.30, below the .35 override, because the bands include
the surrounding furniture and walls; the rug therefore stays supported by
`desk` or is unsupported. The box is detected only twice with no support, and
`still_rug` has no evidence at 40 s. This is a threshold/band-layout limit, not
evidence that the rule is wrong; it is a negative pre-registered test and the
rule is not validated.

Post hoc, the fixed veto removes the one false presence (`cup` on `desk` at
.31-.54) and changes nothing else: 9/13 with 0/5. This is the second clip where
the veto removes a cup false positive (after the post-hoc `play_0` rerun), but
it did not replicate on the pre-registered `play_1` test (Result 8), so the
evidence remains clip-dependent.

## Cost

- 525 detector calls for 19 questions (27.6/query) versus 71 (3.7/query) for the
  default binding path. The difference is the explicit 32-context clip scan;
  a windowed question uses about 5 contexts.
- Mean query latency 0.43 s versus 0.19 s (detector calls dominate; warm GPU).
- One extra detector call per positive answer for label competition.
- The state path is opt-in (`state_queries=False`); the default answer path and
  all existing benchmark scores are unchanged.

## Threats to validity

- One protocol-defined clip (19 questions) plus one post-hoc transfer clip (9
  questions), one detector stack. The 0/6 and 0/4 false-presence results are
  bounded safety checks on those clips, not calibrated absence guarantees.
- The pre-registered `play_2` replication is negative on positives (0/7
  found). Its 0/5 false-presence result therefore shows only that the path
  stays silent when it cannot ground a support, not that it can answer state
  questions in a new room. Positive recall depends on the support detector
  emitting the scene's surfaces; the living room's floor is never detected.
- The pre-registered `play_0` surface-head test is also negative, and it breaks
  the safety claim: 5/12 with **1/5 false presence** (a low-confidence cup
  hallucination on the desk) and a false move (sofa).  The surface head only
  sometimes overrides `desk` to `floor`, so the state path is neither accurate
  nor safe in this room.  Do not generalise the earlier 0/5 results.
- Annotation was vision-assistant-assisted, not by an independent human
  annotator. It is independent of YOLO-World but shares the usual
  image-understanding biases of the assistant. The probe file was written
  before the first run but committed with the analysis, so it is not a
  strict pre-registration commit.
- The clip-level scan samples 32 of 47 retained contexts. `ABSENCE.md` shows
  that more contexts do not fix the detector's category substitutions; the same
  limit applies here.
- The competing bank is a household/office prior. It can reject a valid label
  that is genuinely absent from the bank, and it adds one call per positive.
- "Still" uses a 1.5 s window. A fast state change inside that window could be
  reported from the wrong frame; the answer includes the evidence timestamp so
  the caller can judge it.
- The clip is a kitchen; the support-under rule is geometric, but the label bank
  and object vocabulary are not domain-neutral.

## Reproduce

```bash
python scripts/fetch_aria_clip.py --sequence cook_0
python scripts/spore_state_eval.py \
  --clip data/footage/aria_cook_0_45s.mp4 \
  --qa data/qa/cook_0_state_45s.json \
  --dump research/spore/results/cook0_state/store.json \
  --out research/spore/results/cook0_state/report.json \
  --compare

# Ablation: no competing-label check
python scripts/spore_state_eval.py \
  --clip data/footage/aria_cook_0_45s.mp4 \
  --qa data/qa/cook_0_state_45s.json \
  --store research/spore/results/cook0_state/store.json \
  --out research/spore/results/cook0_state/report_noverify.json \
  --no-verify

# Post-hoc transfer check (eating room)
python scripts/spore_state_eval.py \
  --clip data/footage/aria_eat_0_45s.mp4 \
  --qa data/qa/eat_0_state_45s.json \
  --dump research/spore/results/eat0_state/store.json \
  --out research/spore/results/eat0_state/report.json \
  --compare

# Pre-registered replication (living room; negative on positives)
python scripts/fetch_aria_clip.py --sequence play_2
python scripts/spore_state_eval.py \
  --clip data/footage/aria_play_2_45s.mp4 \
  --qa data/qa/play_2_state_45s.json \
  --dump research/spore/results/play2_state/store.json \
  --out research/spore/results/play2_state/report.json \
  --compare
```

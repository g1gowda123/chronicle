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

## Pre-registration

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

## Setup

| item | value |
|---|---|
| Ingest | `SporeConfig` defaults, 4 fps, 640 px, 24.1 s |
| Store | 47 contexts, 220 capsules, 3.91 MB split store |
| Scan | 32 of 47 contexts per clip-level presence term, 1 detector call each |
| State path | `state_queries=True`, `state_contexts=32`, floor .30, 2 observations |
| Verification | generic 43-label bank, margin .05, one extra call per positive answer |
| Peak VRAM | 1.34 GB (state run), 0.73 GB on the reused-store ablation |

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

## Result 2: competing labels remove the one false presence

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

## Result 3: support geometry matters; missing support detections remain

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

## Cost

- 525 detector calls for 19 questions (27.6/query) versus 71 (3.7/query) for the
  default binding path. The difference is the explicit 32-context clip scan;
  a windowed question uses about 5 contexts.
- Mean query latency 0.43 s versus 0.19 s (detector calls dominate; warm GPU).
- One extra detector call per positive answer for label competition.
- The state path is opt-in (`state_queries=False`); the default answer path and
  all existing benchmark scores are unchanged.

## Threats to validity

- One clip, 19 questions, one detector stack. The 0/6 false-presence result is a
  lower bound on this clip, not a calibrated absence guarantee.
- Annotation was vision-assistant-assisted, not by an independent human
  annotator. It is independent of YOLO-World but shares the usual
  image-understanding biases of the assistant.
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
```

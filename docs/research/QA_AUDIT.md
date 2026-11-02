# Cook QA audit (post-hoc)

This note records a small, post-hoc audit of the hand-written Cook QA set. It is
**not** an independent annotation set: the two errors were found by inspecting
the frames after the original run. The original silver file remains the
comparable benchmark; the audited file is a diagnostic.

## Corrections

| Question | Timestamp | Original | Audited | Evidence |
|---|---:|---|---|---|
| Where is the cup? | 20.0 s | countertop | shelf | At 19.6 s the binding bbox `(319.6, 49.5, 350.2, 84.6)` is a mug inside the cabinet/shelf. |
| Where did the plate go? | 44.8 s | bowl | countertop | At 44.2 s the plate is visibly on the countertop; the 44.8 s frame also shows it there. |

The cup answer at 19.6 s is `shelf`; the plate answer at 44.2 s is
`countertop`. Both are supported by the saved frames, not by a new model run.

## Files

- Original: `data/qa/cook_45s.json` (unchanged).
- Audited: `data/qa/cook_45s_audited.json` (3,615 bytes). Only the two
  `answer`/`answers` entries above differ; `audit.corrections` records the
  evidence and the post-hoc caveat.

## Result

Compact Cook store (`research/spore/results/cook_300_split/store.json`),
hybrid binding, support remeasurement:

| QA set | Semantic | Calls/query | loc@.30/.50 | IoU |
|---|---:|---:|---:|---:|
| Silver (original) | .70 (7/10) | 8.1 | 1.00 / 1.00 | .826 |
| Audited (post-hoc) | .90 (9/10) | 8.1 | 1.00 / 1.00 | .826 |

The remaining audited failure is:

- `What is the person slicing?` → `slicing on the countertop` instead of
  `tomatoes`. The action answer names the activity/support but not the object
  argument. This is a genuine action-grounding gap, not a label correction.

Command:

```bash
python scripts/spore_real_bench.py \
  --clip data/real/aria_cook_45s.mp4 \
  --qa data/qa/cook_45s_audited.json \
  --budget 8 --mode hybrid \
  --store research/spore/results/cook_300_split/store.json \
  --support-remeasure
```

## Interpretation

Do not report .90 as the headline or as held-out accuracy. Report the original
.70 for comparability, then note that a post-hoc audit found two silver-label
errors and that the corrected diagnostic is .90. The audit does not justify
tuning the model to Cook, and the original labels should not be silently
overwritten.

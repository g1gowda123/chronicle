# Action confidence calibration

The action layer must never turn a weak caption into a graph fact. Calibration
turns labelled captions into per-action acceptance thresholds, so a confident
`HOLD` is not accepted at the same score as an uncertain `OPEN`, and negative
examples are rejected instead of inflating recall.

This page is the reproducible record for **NB-08** (`docs/NOTICE_BOARD.md`).
The shipped defaults in `chronicle/semantic/actions.py` are generated from
`docs/calibration_labels.holoassist.json`, not from the old 16-caption seed set.

## TL;DR

```bash
.venv/bin/python -m chronicle.semantic.calibration \
  --labels docs/calibration_labels.holoassist.json \
  --target-precision 0.9 \
  --write docs/calibration.holoassist.report.json
```

Result on the real HoloAssist clip (19 labels, 5 correct rejections):

| Metric | Value |
| --- | --- |
| Labels | 19 |
| Accepted | 14 |
| Correct rejections | 5 |
| Precision | **1.000** |
| Recall | **1.000** |
| F1 | **1.000** |

The command exits `0` because precision is at or above the `0.9` target.

## Label file

`docs/calibration_labels.holoassist.json` is built from the published
HoloAssist Joy-Con timeline in `data/real/holoassist_switch_34s.json` and the
real interaction strips in `runs/real_clips/`.

- One label per published event: grab, hold, slide, place, insert, withdraw,
  align, with a few nearby rest/hold frames from the same clip.
- Scene, pre-grasp, rest and adjust rows are included as `OTHER` negatives so
  false positives are measured.
- Each row keeps its `timestamp` and the original HoloAssist `event` string so
  a reviewer can trace it back to the clip.
- Captions are **hand-authored labels**, not Florence-2 output. No model
  caption was invented or copied into the file.

The file is a normal calibration label file; metadata keys are ignored by the
loader and the `labels` list is consumed:

```json
{
  "clip": "data/real/holoassist_switch_34s.mp4",
  "dataset": "HoloAssist",
  "labels": [
    {
      "timestamp": 19.09,
      "event": "insert joy_con_controller",
      "caption": "The hand inserts the left Joy-Con into the rail of the Switch console.",
      "action": "INSERT_INTO"
    }
  ]
}
```

## Recommended thresholds

These are now the defaults in `DEFAULT_ACTION_THRESHOLDS`:

```text
CHRONICLE_ACTION_THRESHOLDS=CLOSE=0.6,HAND_OFF=0.6,HOLD=0.62,INSERT_INTO=0.8,OPEN=0.6,OTHER=1,PICK_UP=0.74,PULL=0.55,PUSH=0.68,PUT_DOWN=0.76,REMOVE_FROM=0.78
```

| Action | Threshold | Source |
| --- | --- | --- |
| `PICK_UP` | 0.74 | real labels |
| `PUT_DOWN` | 0.76 | real labels |
| `INSERT_INTO` | 0.80 | real labels |
| `REMOVE_FROM` | 0.78 | real labels |
| `HOLD` | 0.62 | real labels |
| `PUSH` | 0.68 | real labels |
| `OPEN` | 0.60 | seed set (no real label yet) |
| `CLOSE` | 0.60 | seed set (no real label yet) |
| `HAND_OFF` | 0.60 | seed set (no real label yet) |
| `PULL` | 0.55 | seed set (no real label yet) |
| `OTHER` | 1.00 | always rejected |

Set the environment variable only when overriding a deployment; the calibrated
values are already the code defaults.

## Per-action report

| Action | Support | Correct | Accepted | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `PICK_UP` | 3 | 3 | 3 | 1.000 | 1.000 |
| `HOLD` | 3 | 3 | 3 | 1.000 | 1.000 |
| `PUSH` | 2 | 2 | 2 | 1.000 | 1.000 |
| `PUT_DOWN` | 3 | 3 | 3 | 1.000 | 1.000 |
| `INSERT_INTO` | 1 | 1 | 1 | 1.000 | 1.000 |
| `REMOVE_FROM` | 2 | 2 | 2 | 1.000 | 1.000 |
| `OTHER` | 5 | 5 | 0 | — | — |

**NB-18** removed the pre-grasp false positive. The caption
`"The hand reaches toward the Joy-Con but does not grasp it."` now maps to
`OTHER` because the rule classifier rejects negated action verbs; the three
true pickups remain accepted. Precision is now 1.00 with recall 1.00, and the
per-action thresholds are unchanged (the fix is phrasing, not a cutoff).

## Why `OTHER` does not lower recall

`OTHER` is never accepted. Counting a *correct* rejection as a false negative
made the aggregate recall collapse whenever negative examples were present, so
`evaluate_calibration()` ignores correct `OTHER` rows in the recall denominator
and reports them separately as `rejected_correct`. This keeps the score focused
on actions while still measuring the rejection behaviour.

## Adding labels

Accepted shapes are documented in `docs/ROLE_C.md`:

```json
[
  {"caption": "The hand holds the bottle.", "action": "HOLD"}
]
```

```json
{"labels": [{"caption": "The hand holds the bottle.", "action": "HOLD"}]}
```

```json
{"HOLD": ["The hand holds the bottle."]}
```

`.jsonl` / `.ndjson` files are read one object per line. Actions accept
`REMOVE_FROM`, `remove from` or `remove-from`.

Suggested loop for a new clip:

1. Export the interaction strips and note the published event times.
2. Add rows to a new `docs/calibration_labels.<clip>.json`.
3. Run the calibration command with `--target-precision 0.9` and inspect the
   per-action table.
4. Add a regression test that loads the file and asserts the target is met.
5. Commit the label file and its report JSON together.

## Limitations

- The rule classifier emits a fixed confidence per action, so this calibration
  mostly measures coverage and rejection, not a smooth confidence curve.
- Only six action classes have real labels. `OPEN`, `CLOSE`, `HAND_OFF` and
  `PULL` still use seed thresholds.
- The labels are one hand-labelled pass over one 34 s clip. Add a second
  annotator / clip before claiming generalisation.
- This does not calibrate the Florence-2 + Qwen path. To do that, export real
  model confidences and run `suggest_thresholds()` on
  `CalibrationSample(action, confidence, correct)` rows.
- The HoloAssist labels use `joy-con` for both controllers. Entity agreement is
  a separate concern handled by `chronicle.semantic.timeline`.

## Verification

```bash
.venv/bin/python -m chronicle.semantic.calibration \
  --labels docs/calibration_labels.holoassist.json --target-precision 0.9

.venv/bin/python -m pytest -q tests/test_calibration.py tests/test_actions.py
.venv/bin/python -m unittest tests.test_calibration tests.test_actions
.venv/bin/ruff check chronicle tests
```

## Related

- [`NOTICE_BOARD.md`](NOTICE_BOARD.md) — task card NB-08 and follow-ups.
- [`ROLE_C.md`](ROLE_C.md) — calibration API, label shapes and timeline scorer.
- [`TEAM_STATUS.md`](TEAM_STATUS.md) — Role C status and benchmark history.
- `chronicle/semantic/calibration.py` — `CalibrationSample`,
  `suggest_thresholds()`, `evaluate_calibration()` and the CLI.

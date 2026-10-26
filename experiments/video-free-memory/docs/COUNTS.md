# Object counts: tracks, peaks and physical estimates

**Task:** P1-COUNTS (physical object counts / track fragmentation)  
**Status:** implemented and measured on the bundled episodes  
**Where:** `prototype/compile.py` (`visibility_counts`), `prototype/query.py`
(`answer_count`), `prototype/report.py`, `tests/test_counts.py`

## The problem

A multi-object tracker assigns an id per *track*, not per *physical object*. Occlusion, a
re-detection, or a momentary label flip can split one object into two tracks. The vault used to
answer "how many bowls?" with the raw track count and a caveat:

> The memory tracked 4 bowl tracks. … this is a track count, not a verified object count.

That is honest but not useful: the user cannot tell whether four bowls were on the table or one
bowl was split into four tracks. On the cook clip it is worse than fragmentation:

| Physical object | Tracker output |
| --- | --- |
| empty plate by the sink | `bowl:track_2`, then `plate:track_8` |
| plate with tomato slices | `bowl:track_3`, then `plate:track_11` / `bowl:track_12` |
| one-frame false positive | `bowl:track_20` |

The same two plates are emitted under **both** `bowl` and `plate`, with near-identical boxes.
The old answer "4 bowls" was therefore wrong in the most damaging way: there are no bowls in the
scene.

## The method

At compile time we still have the source `observations` table. For every label we compute three
numbers:

| Number | Definition | Interpretation |
| --- | --- | --- |
| `tracks` | number of entity ids with that label | raw tracker count |
| `peak` | max number of same-label tracks present in one observation frame | lower bound before de-duplication |
| `physical` | tracks minus cross-label duplicates, minus one-frame transients | conservative physical estimate |

`peak` is computed by grouping observations at the 10 fps cadence (`round(t, 2)`) and counting
distinct entity ids per group:

```text
peak[label] = max over frames t of |{entity_id : label(entity_id) = label, observed at t}|
```

### Cross-label duplicate detection

Two tracks with **different labels** are the same physical object when the same frame contains
boxes with:

- IoU ≥ **0.90** and centre distance ≤ **5 px** (the same pixels, two labels), and
- the other label has more total observations across the episode.

The label with more observations wins; the other track is recorded as a duplicate. This is decided
per *label pair*, not per track, so a single long-lived bowl track cannot drag a plate track into
the bowl label.

Measured on cook:

| Duplicate pair | Strong frames | Max IoU | Result |
| --- | ---: | ---: | --- |
| `plate:track_8` ← `bowl:track_2` | 3 | 0.99 | bowl track is a duplicate |
| `plate:track_8` ← `bowl:track_3` | 1 | 0.91 | bowl track is a duplicate |
| `plate:track_11` ← `bowl:track_12` | 47 | 0.95 | bowl track is a duplicate |
| `bowl:track_20` | 0 | — | one-frame transient |

So cook has **4 bowl-labelled tracks → 0 physical bowls**, and **2 plate tracks → 2 physical
plates**. A visual check of frames at 0 s, 12 s, 20 s, 30 s, 40 s and 44 s confirms two plates and
no bowls on the counter.

The same gate does **not** fire on walking: the closest cross-label pair is backpack/person at
max IoU 0.63, well below the 0.90 threshold, so the backpack is not absorbed into the person.

### Why not merge same-label boxes?

Merging same-label co-visible boxes would be wrong for this data. On cook the maximum IoU between
co-visible same-label boxes is **0.01** (mean ≈ 0.00 over 69 plate pairs): co-visible plate/plate
and bowl/bowl pairs are spatially separate, so they are different objects at that moment. Only
**cross-label** near-identical boxes are duplicates.

### What is stored

One JSON value in the vault's `meta` table:

```json
{
  "bowl": {
    "tracks": 4, "peak": 2, "physical": 0,
    "duplicates": 3, "transient": 1,
    "merged_into": {"plate": 3}
  },
  "plate": {"tracks": 2, "peak": 2, "physical": 2, "duplicates": 0, "transient": 0}
}
```

Query time reads only that JSON. It never opens the video or the raw observations, so the
video-free contract and the ~0.05 ms deterministic latency are unchanged.

## Measured on the bundled episodes (30-question vaults)

| Episode | Label | Tracks | Peak | Physical | Count question |
| --- | --- | ---: | ---: | ---: | --- |
| cook | bowl | 4 | 2 | **0** | C3 |
| cook | plate | 2 | 2 | 2 | C2 |
| cook | cup | 2 | 2 | 2 | C1 |
| cook | knife | 2 | 1 | 2 | — |
| walking | car | 2 | 1 | 2 | C4 |
| walking | person | 4 | 3 | 4 | — |
| walking | backpack | 1 | 1 | 1 | — |
| hololens | joy-con | 2 | 2 | 2 | C5 |

Cars are still the fragmentation case: two tracks exist but only one is visible at a time, so the
answer remains a **range** (1–2) rather than a false single count. Bowls are the label-confusion
case: the physical count collapses to **0** because all four tracks are duplicates or transient.

## Answer examples

```
how many bowls?
→ The memory tracked 4 bowl tracks. 3 of them are near-identical duplicate detections of
  3 plate tracks and 1 one-frame track is transient, so the physical bowl count is 0.

how many plates?
→ The memory tracked 2 plate tracks. Up to 2 were visible at the same instant, which is still
  a track count, not a verified physical count.

how many cars?
→ The memory tracked 2 car tracks. At most 1 car track was visible at the same instant, so the
  physical count is between 1 and 2: fragmentation or occlusion may inflate the track count.

how many joy-cons?
→ The memory tracked 2 joy-con tracks. Up to 2 were visible at the same instant, which is still
  a track count, not a verified physical count.
```

The first sentence always keeps the raw track count. That preserves the existing 30-question
regression strings (`4 bowl`, `2 plate`, …) while adding the physical estimate. Methods:
`vault-count` (no observations) → `vault-count-visibility` (peak only) → `vault-count-dedup`
(cross-label duplicates/transients).

## Where it shows up

| Surface | Behaviour |
| --- | --- |
| `query.py` / CLI | Count answers use the same three-tier wording; `--stats --json` includes `counts`. |
| Studio (deterministic ask) | Uses `query.answer`, so the estimate is shown without Qwen. |
| Studio (`--qwen`) | The verified answer is in the memory pack; Qwen phrases the same estimate. |
| Single-file `memory.html` | `DATA.counts` is embedded and the client-side count branch renders the same estimate offline; no server or model needed. |
| Caption-first uploads | No observations, so they keep the caption-coverage wording and must not claim physical counts. |

`tests/test_counts.py` covers the compile contract, cross-label de-duplication, singular grammar,
the legacy-vault fallback, and the HTML report embedding.

## Verification

```bash
cd /home/g1/projects/chronicle-vfm/experiments/video-free-memory
/home/g1/projects/chronicle/.venv/bin/python -m pytest -q tests

# Rebuild the three benchmark vaults and re-attach the 30 s ledgers
python3 prototype/compile.py --source-db /home/g1/projects/chronicle/data/real/cook/chronicle.db --out /tmp/vfm_counts/cook.db
python3 prototype/attach_ledger.py --vault /tmp/vfm_counts/cook.db --ledger /tmp/vfm/cook_ledger_30.json
python3 prototype/benchmark.py --questions benchmark/questions.json \
  --vault-dir /tmp/vfm_counts --source-root /home/g1/projects/chronicle/data/real
```

Result after the change: **30/30 (100 %)**, cold p50 **0.061 ms**, warm p50 **0.067 ms**,
4/4 paraphrase cache hits. Same accuracy and latency as before; strictly more information in
count answers.

> **Benchmark nuance:** C3 still expects the string `4 bowl` because the regression suite encodes
> the raw track count. The answer now contains both `4 bowl tracks` and `physical bowl count is 0`.
> Do not change the expectation until the physical labels are added to the benchmark itself.

## Limits (do not overclaim)

- `physical` is a conservative **estimate**, not ground truth. It removes only near-identical
  cross-label boxes and tracks shorter than 2 observations (0.2 s at 10 fps).
- `peak` is a lower bound from sampled observations. A short flicker may be missed at 10 fps.
- `tracks` is an upper bound for a single label when label confusion is present.
- Visual verification exists for cook plates/bowls only; the other episodes are not hand-labelled.
- Label quality remains upstream: an armchair labelled `sofa`, or a bowl that is never co-detected
  with a plate, still contributes to the wrong label.
- Caption-first vaults (`video_vault.py`, arbitrary uploads with no graph DB) have no
  `observations`, so they keep the caption-coverage wording and must not claim physical counts.

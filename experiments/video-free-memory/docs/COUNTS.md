# Object counts: tracks, peaks and physical bounds

**Task:** P1-COUNTS (physical object counts / track fragmentation)  
**Status:** implemented and measured on the bundled episodes  
**Where:** `prototype/compile.py` (`visibility_counts`), `prototype/query.py`
(`answer_count`), `tests/test_counts.py`

## The problem

A multi-object tracker assigns an id per *track*, not per *physical object*. Occlusion, a
re-detection, or a momentary label flip can split one object into two tracks. The vault used to
answer "how many bowls?" with the raw track count and a caveat:

> The memory tracked 4 bowl tracks. … this is a track count, not a verified object count.

That is honest but not useful: the user cannot tell whether four bowls were on the table or
whether one bowl was split into four tracks. The question is also not answerable from the
compiled vault alone, because the vault deliberately drops per-frame observations to stay small.

## The method

At compile time we still have the source `observations` table. For every label we compute two
numbers:

| Number | Definition | Interpretation |
| --- | --- | --- |
| `tracks` | number of entity ids with that label | **upper** bound on physical objects |
| `peak` | max number of same-label tracks present in one observation frame | **lower** bound on physical objects |

So the physical count is reported as `peak … tracks`. The peak is computed by grouping
observations at the 10 fps cadence (`round(t, 2)`) and counting distinct entity ids per group:

```text
peak[label] = max over frames t of |{entity_id : label(entity_id) = label, observed at t}|
```

The two numbers are stored once in the vault's `meta` table:

```json
{"bowl": {"tracks": 4, "peak": 2}, "cup": {"tracks": 2, "peak": 2}}
```

Query time reads only that JSON. It never opens the video or the raw observations, so the
video-free contract and the ~0.04 ms deterministic latency are unchanged.

### Why not merge co-visible boxes?

Merging same-label boxes that overlap would be the obvious de-duplication test, but it is wrong
for this data. On the cook episode the maximum IoU between co-visible same-label boxes is
**0.01** (mean ≈ 0.00 over 69 plate pairs): co-visible plate/plate and bowl/bowl pairs are
spatially separate, so they are genuinely different objects at that moment. Merging them would
under-count real objects. The peak is therefore a lower bound, not a claim that tracks are
duplicates.

## Measured on the bundled episodes (30-question vaults)

| Episode | Label | Tracks | Peak visible | Physical bound | Count question |
| --- | --- | ---: | ---: | --- | --- |
| cook | bowl | 4 | 2 | 2–4 | C3 |
| cook | cup | 2 | 2 | 2 | C1 |
| cook | plate | 2 | 2 | 2 | C2 |
| walking | car | 2 | 1 | 1–2 | C4 |
| walking | person | 4 | 3 | 3–4 | — |
| walking | cup | 1 | 1 | 1 | — |
| hololens | joy-con | 2 | 2 | 2 | C5 |

The interesting rows are **bowls (4 tracks, at most 2 visible)** and **cars (2 tracks, at most 1
visible)**. Those are exactly the cases where the old answer implied a physical count it could
not support. The vault now says the count is a range.

## Answer examples

```
how many bowls?
→ The memory tracked 4 bowl tracks. At most 2 bowl tracks were visible at the same instant,
  so the physical count is between 2 and 4: fragmentation or occlusion may inflate the track count.

how many cars?
→ The memory tracked 2 car tracks. At most 1 car track was visible at the same instant, so the
  physical count is between 1 and 2: fragmentation or occlusion may inflate the track count.

how many joy-cons?
→ The memory tracked 2 joy-con tracks. Up to 2 were visible at the same instant, which is still
  a track count, not a verified physical count.
```

The first sentence still contains the track count, so the existing 30-question regression suite
(which expects `2 plate`, `4 bowl`, …) is unchanged. Method changes from `vault-count` to
`vault-count-visibility` when the bounds are available.

### Where it shows up

| Surface | Behaviour |
| --- | --- |
| `query.py` / CLI | `how many bowls?` reports the range; `--stats --json` includes the `counts` map. |
| Studio (deterministic ask) | Uses `query.answer`, so the range is shown without Qwen. |
| Studio (`--qwen`) | The verified answer is in the memory pack; Qwen phrases the same range. |
| Single-file `memory.html` | `DATA.counts` is embedded and the client-side count branch renders the same range offline; no server or model needed. |
| Caption-first uploads | No observations, so they keep the caption-coverage wording and must not claim physical counts. |

`tests/test_counts.py` covers the compile contract, the range wording, singular grammar, the
legacy-vault fallback, and that the HTML report embeds `counts` and the visibility branch.

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

Result after the change: **30/30 (100 %)**, cold p50 **0.041 ms**, warm p50 **0.048 ms**,
4/4 paraphrase cache hits. Same accuracy and latency as before; strictly more information in
count answers.

## Limits (do not overclaim)

- `peak` is a **lower bound** from sampled observations, not ground truth. A short flicker may be
  missed at 10 fps.
- `tracks` is an **upper bound**. The true count can be anywhere in the range.
- Caption-first vaults (`video_vault.py`, arbitrary uploads with no graph DB) have no
  `observations`, so they keep the caption-coverage wording. The Studio path should not claim
  physical counts.
- The bounds are computed per label, so an armchair labelled `sofa` still contributes to the
  `sofa` count. Label quality is upstream of this fix.

# Chronicle evaluation results

All numbers below were produced locally on an NVIDIA RTX 4060 (8 GB) with the
final scorer and with Ollama models unloaded before each run. Raw outputs live
in `runs/bench/` (Cook) and `runs/bench_holo/` (HoloAssist); paper tables are in
`paper/`.

## Setup

| Item | Value |
|---|---|
| Chronicle | event-gated state machine, tracker and graph unchanged |
| Captioners | `microsoft/Florence-2-base`; Ollama `qwen2.5vl:3b` |
| Answer backends | deterministic template GraphRAG; Ollama `qwen2.5:1.5b` |
| Cook clip | 45 s, 450 frames, 10 questions, silver labels from the Qwen2.5-VL ledger |
| HoloAssist clip | 34 s, 1019 frames, 8 questions, gold human annotations |
| Baseline | dense captioning at 1 FPS (45 and 34 captions) |
| Matcher | deterministic token/number matching; `tolerance_s` on temporal rows |

`semantic.action_llm_enabled: false`, so during ingestion the text LLM is not
used for action verification. Text-LLM call counts in the tables are answer
calls only.

## Primary comparison: Florence-2 + template answers

### Cook 45 s

| Method | Recall@1 | VLM Calls | Avg Latency (ms) | Peak VRAM (MB) |
|---|---:|---:|---:|---:|
| Chronicle (event-gated) | 40.0% | 53 | 21.2 | 2071.2 |
| Dense 1 FPS | 40.0% | 45 | 262.9 | 819.2 |

### HoloAssist Joy-Con 34 s

| Method | Recall@1 | VLM Calls | Avg Latency (ms) | Peak VRAM (MB) |
|---|---:|---:|---:|---:|
| Chronicle (event-gated) | **62.5%** | 37 | 17.7 | 2095.2 |
| Dense 1 FPS | 12.5% | 34 | 228.8 | 817.2 |

Chronicle matches the dense baseline on Cook and improves HoloAssist from
1/8 to 5/8 correct, using 3 more VLM calls (37 vs 34).

## Full model matrix

`paper/table_matrix.tex` and `paper/table_matrix.md` contain all 16 runs.
Summary of Recall@1:

| Captioner / answerer | Cook Chronicle | Cook dense | Holo Chronicle | Holo dense |
|---|---:|---:|---:|---:|
| Florence-2 / template | 40.0% | 40.0% | 62.5% | 12.5% |
| Florence-2 / Qwen2.5 | 40.0% | 30.0% | 50.0% | 25.0% |
| Qwen2.5-VL / template | 40.0% | 30.0% | 62.5% | 12.5% |
| Qwen2.5-VL / Qwen2.5 | 40.0% | 20.0% | 62.5% | 12.5% |

## What happened to Qwen

Qwen appears in two independent roles:

1. **Qwen2.5-VL (`qwen2.5vl:3b`) as the captioner.** Implemented in
   `chronicle/semantic/qwen_vl.py` and selectable with `--captioner qwen`. It
   is a drop-in replacement for Florence-2, so the state machine is unchanged.
   - Cook: identical 40% recall for Chronicle, but the dense baseline drops
     from 40% (Florence) to 30% (Qwen-VL).
   - HoloAssist: identical 62.5% for Chronicle, same 12.5% for dense.
   - Cost: ~5.5 GB device-wide instead of ~2.1 GB, and roughly 2x the wall time.
   - Conclusion: no accuracy gain on this suite, a clear memory/time cost. It is
     a valid swap-in when a natural-language captioner is preferred or Florence-2
     is unavailable.

2. **Qwen2.5 (`qwen2.5:1.5b`) as the answer/GraphRAG backend.** Selected with
   `--llm-backend ollama --llm-model qwen2.5:1.5b`.
   - Cook Chronicle: 40%, unchanged from template; dense drops 40% -> 30%.
   - Holo Chronicle: 62.5% -> 50%; dense rises 12.5% -> 25%.
   - Qwen answers are fluent but sometimes add unsupported detail, e.g. it
     claimed "10 joy-con controllers" and "first picked up at t=4.1" from
     context that did not support either. The deterministic template backend is
     more conservative and scores higher on this suite.
   - `qwen2.5:7b-instruct` was tested first but was too slow and left too little
     VRAM beside the captioner on 8 GB, so the reported runs use the 1.5B model.

Qwen is therefore fully wired in, but the evidence here does not justify making
it the default. Florence-2 + template remains the primary configuration.

## Cost analysis (honest version)

- **VLM calls:** Chronicle is *not* cheaper than a 1 FPS baseline on these
  clips. It makes 53 vs 45 calls on Cook (+18%) and 37 vs 34 on HoloAssist
  (+9%). The `vlm_call_reduction_pct` field (88% and 96%) compares against
  captioning *every decoded frame*, not against the 1 FPS baseline, and should
  not be read as a saving over that baseline.
- **Wall time:** Chronicle processes every frame (detection, hands, tracking),
  so it is slower end-to-end: 38.4 s vs 15.0 s on Cook and 40.3 s vs 10.9 s on
  HoloAssist. It sustains 11.7 FPS (Cook) and 25.3 FPS (HoloAssist) on the
  4060; the dense pipeline sustains ~3 FPS because it captions every sampled
  frame.
- **Latency column:** the reported average is per processed sample, so the two
  systems have different denominators (decoded frame vs 1 FPS caption). The
  ~12x gap measures per-sample cost, not end-to-end speed. Actual GraphRAG
  query latency is sub-millisecond for Chronicle (0.6-0.9 ms) and ~1.1 ms for
  the dense baseline.
- **VRAM:** Chronicle costs ~2.5x the dense baseline with the same captioner
  (2.1 GB vs 0.8 GB) because it keeps detector, hands, embedder and captioner
  resident. Qwen-VL raises this to 5.5 GB (template) or 6.8 GB (Qwen text),
  still inside 8 GB but with little headroom.

The defensible claim is **accuracy at a similar VLM-call budget**, not compute
savings: on HoloAssist, Chronicle gets 5/8 vs 1/8 correct for 3 extra captions.

## Failure analysis

Cook 45 s (both systems miss the same four questions):

- The cup and cutting board are not grounded as entities at the QA timestamp,
  so both return "I don't have that in my memory."
- "What is the person slicing?" expects `tomatoes`; neither system binds the
  knife/board activity to that noun.
- "When did the person first touch the cup?" expects 4.5 s; Chronicle reports
  the last cup event (41.4 s). The memory supports "last seen", not "first
  seen".
- "Where did the plate go?" expects `bowl` (silver label). Chronicle answers
  `countertop`, which matches the ledger but not the silver label.
- The unknown question is handled correctly by both.

HoloAssist 34 s:

- Chronicle answers activity questions with tracker states (`OCCLUDED`,
  `VISIBLE_MOVING`) rather than the annotated verb (`picked up`).
- Temporal "first pick up" expects 4.2 s, but the answer reports the last
  `pick joy-con` event at 33.9 s.
- Counting "2 joy-con controllers" is not answered from a verified count; the
  template returns an event sentence, and Qwen hallucinated 10.
- Dense 1 FPS fails almost everything because 34 independent captions do not
  bind objects to zones or preserve event identity across seconds.

## Limitations

- **Tiny suites:** 10 and 8 questions over one clip each. Differences of one
  question move Recall@1 by 10-12.5 points; treat the matrix as a sanity check,
  not a leaderboard.
- **Silver Cook labels:** Cook answers are derived from a Qwen2.5-VL event
  ledger, so the Qwen-VL captioner is partly evaluated against labels from the
  same model family. HoloAssist uses human annotations and is the stronger
  evidence.
- **Zone-label location answers:** HoloAssist location questions expect the zone
  name (`Work Table`). An answer that names the zone passes even if it anchors
  the wrong tracked object; both HoloAssist location questions pass this way.
- **Answer style:** template answers are terse and match short gold strings;
  Qwen answers are verbose and can fail token recall even when semantically
  close (e.g. colour "red" vs gold "cyan"/"blue").
- **Scoring fix:** a bare numeric answer no longer matches a digit inside a
  timestamp. Before this fix one HoloAssist counting question was marked
  correct for the wrong reason; the reported 62.5% already reflects the fix.

## Reproduction

```bash
export PATH=/home/g1/projects/chronicle/.venv/bin:$PATH

for cfg in "florence template _ft" "florence ollama _fq" \
           "qwen template _qt" "qwen ollama _qq"; do
  set -- $cfg
  ollama stop qwen2.5vl:3b >/dev/null 2>&1 || true
  ollama stop qwen2.5:1.5b >/dev/null 2>&1 || true
  CAPTIONER=$1 LLM_BACKEND=$2 LLM_MODEL=qwen2.5:1.5b SUFFIX=$3 \
    scripts/run_paper_eval.sh \
    data/real/cook/aria_cook_45s.mp4 data/qa/cook_45s.json runs/bench
done

python generate_matrix_table.py \
  --run-dir runs/bench --run-dir runs/bench_holo \
  --out paper/table_matrix.tex
```

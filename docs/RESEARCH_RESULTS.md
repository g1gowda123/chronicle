# Research results and audit

This note records the controlled evaluation behind the README’s research table.
It is deliberately conservative: the suites are small, the official matcher is
lenient, and the standard-benchmark work is not finished.

## Setup

| Item | Value |
|---|---|
| Clips | Cook 45 s, HoloAssist switch 34 s |
| Questions | 10 + 8 = 18 |
| Detector | YOLO-World-L, fixed scene vocabulary from QA metadata |
| Vocabulary preset | `none` (scene vocabulary only) |
| Prompt aliases | `cup` ← `mug`, `coffee cup`; `cutting board` ← `chopping board`; `rice cooker` ← `instant pot`; `bowl` ← `white bowl` |
| Captioner | Florence-2, `caption_stride: 4` |
| Answer path | deterministic planner/template, no cloud LLM |
| Hardware | RTX 4060 Laptop, 8 GB |
| Branch | `research/vfm-merged` |

The vocabulary is derived from the clip’s scene metadata, never from the
questions. The QA-derived-vocabulary ablation below shows why this matters.

## Headline results

| System | Cook 45 s | HoloAssist 34 s | Pooled 18 q | VLM calls |
|---|---:|---:|---:|---:|
| **Chronicle, event planner** | **80.0%** (8/10) | **87.5%** (7/8) | **83.3%** (15/18) | 26 / 13 |
| Dense 1 FPS baseline | 40.0% (4/10) | 12.5% (1/8) | 27.8% (5/18) | 45 / 34 |
| Leak-free baseline | 40.0% (4/10) | 62.5% (5/8) | 50.0% (9/18) | 53 / 35 |

Paired comparisons:

| Comparison | Delta | Bootstrap 95% CI | McNemar |
|---|---:|---:|---:|
| Cook, planner vs leak-free | +40.0 pts | [10.0, 70.0] | 4/0, p = 0.125 |
| Holo, planner vs leak-free | +25.0 pts | [-25.0, 75.0] | 3/1, p = 0.625 |
| Pooled, planner vs leak-free | +33.3 pts | [5.6, 61.1] | 7/1, p = 0.070 |
| Pooled, after removing one false baseline pass | +38.9 pts | [16.7, 61.1] | 7/0, p = 0.0156 |

The last row removes a baseline “switch console” answer that resolved to
`joy-con` on the floor. The final system abstains on that question. It is a
faithfulness correction, not a change to the system’s predictions.

## Strict location audit

The official matcher accepts a zone word such as “Kitchen” as support. That is
too lenient for a support-grounding claim. The strict audit requires the
predicted support to match the gold support surface, not just the room.

| Run | Official location | Strict support | Reading |
|---|---:|---:|---|
| Final, Cook | 6/6 | **1/6** | Most passes are zone/state answers |
| Baseline, Cook | 3/6 | 2/6 | Fewer passes, similar grounding weakness |
| Final, HoloAssist | 1/2 | 1/2 | The one pass is genuinely grounded |
| Baseline, HoloAssist | 2/2 | 2/2 | Baseline’s switch-console pass is false |

The honest claim is: **the final system improves event/temporal answering; its
location answers are not yet support-grounded at the strict level.**

## VLM budget

Accuracy as a function of caption calls:

| Dataset | 0 calls | 7 | 13 | 26 | 51 | 102 |
|---|---:|---:|---:|---:|---:|---:|
| Cook | 80% | — | 80% | 80% | 80% | 80% |
| HoloAssist | 87.5% | 87.5% | 87.5% | 87.5% | 87.5% | — |

`caption_stride=0` disables Florence captions. The deterministic semantic rule
verifier still runs, producing ACTION edges whose evidence caption is a
fallback such as `"the hand grasps the cup"`. The predictions are identical to
the caption-on runs.

A stronger control, `--no-semantic` on Cook, drops accuracy from **8/10 to
5/10** and peak VRAM from ~1960 MB to 861 MB. This proves the semantic verifier
matters; it does **not** prove that Florence caption content matters. The
correct claim is:

> Florence captioning is optional for this grounded QA suite; the deterministic
> semantic verifier is still on the critical path.

A caption-necessary suite (open-ended description, OCR/brand/material, social
context) and a shuffled-caption control are the missing experiments.

## Ablations

| Ablation | Result | Decision |
|---|---|---|
| QA-derived vocabulary (leakage) | Cook 7/10, Holo 5/8; corrupted labels such as “person joy-con”; count 6 from 13 tracks | Rejected |
| Broad 111-class prompts | Diluted YOLO-World; Cook location weak | Rejected |
| Scene vocabulary | Cook 40% → 50% before planner; fewer VLM calls | Adopted |
| Detector S | 7/10, 7/8; pooled 14/18 | Efficient fallback |
| Detector L | 8/10, 7/8; pooled 15/18 | Adopted |
| S + L ensemble | 8/10, 5/8; pooled 13/18 | Rejected (Holo counting/attribute instability) |
| Detector confidence 0.20 | Cook 5/10, Holo 6/8 | Rejected |
| Counting similarity 0.88 | Would merge two cup clusters but overfits the gold count | Not adopted |
| Prompt aliases | Cup/cutting-board/rice-cooker/bowl recall improved | Adopted |
| Strict entity resolution | Removed false “switch console → joy-con” pass | Adopted |

## Failure analysis

- **Cook, rice cooker:** not detected in the clip.
- **Cook, cutting board at t=12:** chopping-board confidence is only ~0.087;
  even confidence 0.20 does not ground it in time.
- **Cook, slicing:** no fine-grained action edge; the system abstains.
- **Cook, first touch:** reports 0.8 s (gold 2.6 s); the `PICK_UP` edge is early.
- **Cook, cup count:** reports 3 clusters (gold 2); appearance threshold is
  conservative.
- **Holo, switch console:** not detected in frames; the system now abstains
  instead of resolving to `joy-con`.

## Reproduce

```bash
# run on research/vfm-merged
python eval_pipeline.py \
  --video data/real/cook/aria_cook_45s.mp4 \
  --qa data/qa/cook_45s.json \
  --config config.yaml \
  --captioner florence --llm-backend template \
  --out runs/research/final_cook --evaluate

python eval_pipeline.py \
  --video data/real/holoassist_switch_34s.mp4 \
  --qa data/qa/holoassist_switch.json \
  --config config.yaml \
  --captioner florence --llm-backend template \
  --out runs/research/final_holo --evaluate
```

Statistics:

```bash
python scripts/analyze_results.py ...   # CIs, paired bootstrap, McNemar
python scripts/pool_results.py ...      # pooled 18-question table
python scripts/audit_location.py ...    # official vs strict support audit
```

## Standard benchmarks

The next credibility step is **EgoTempo** (CVPR 2025): 500 open-ended
questions, 367 clip IDs, 222 Ego4D source videos, ten temporal categories. The
harness is on `experiment/egotempo-benchmark`; it is blocked on Ego4D license
credentials. Ego4D NLQ is the secondary target. These results should not be
claimed until the runs exist.

## What not to claim

- Do not call the 18-question suites a benchmark.
- Do not report Cook location as 6/6 without the strict 1/6 audit.
- Do not say “VLMs are unnecessary”; say “Florence captions are optional for
  this grounded suite.”
- Do not claim fine-grained action recognition; the pipeline only emits the
  actions the geometric/rule verifier supports.

# Chronicle Research Handoff

This file is the durable handoff for the next agent. It records the current
state, the user's constraints, the experiments that have already been run, and
the next concrete work. Read this before touching any branch.

## 1. User objective and constraints

The user wants Chronicle improved as a **general-purpose egocentric memory
system**, not tuned to one benchmark clip.

Hard requirements:

- Work autonomously for long stretches; do not ask for confirmation.
- Keep `main` usable at all times. Only stable, tested changes go to `main`.
- Experimental work stays on experimental branches; failures are acceptable
  there.
- The perception layer must be novel, general across egocentric video, and fit
  consumer hardware (RTX 4060 Laptop, 8 GB VRAM).
- Preserve what Chronicle is: structured temporal memory + query engine +
  provenance + local-first operation. Do not replace it with a generic VLM.
- Do not optimize around the single HoloAssist `z210` clip. It is one
  diagnostic, not the objective.
- Search papers/web, compare against current work, and experiment rigorously.

## 2. Repositories, worktrees and branches

| Path | Branch | HEAD | State |
|---|---|---|---|
| `/home/g1/projects/chronicle` | `main` | `1200f84` | clean, pushed, CI green |
| `/home/g1/projects/chronicle-hires` | `research/high-res-longvideo` | `803eed7` | pushed; one untracked config |
| `/home/g1/projects/chronicle-longvideo` | `research/long-video-qa` | `990013e` | clean |
| `/home/g1/projects/chronicle-spore` | `research/spore-perception` | `1200f84` | design doc only, uncommitted |
| `/home/g1/projects/chronicle-vfm` | `experiment/egotempo-benchmark` | `aab86b4` | uncommitted belief work by another agent |
| `/home/g1/projects/chronicle-packaging` | `feat/model-free-demo` | `1638606` | old merged packaging branch |

Do not reset or overwrite `/home/g1/projects/chronicle-vfm`: it contains
another agent's uncommitted `chronicle/belief/` prototype, `docs/BELIEF.md`,
tests, README and CLI edits. That work is independent of the perception work.

## 3. Main branch: what is already shipped

`main` is the public branch: `https://github.com/g1gowda123/chronicle`.

Current state:

- Package/distribution name: `chronicle-memory`; import package: `chronicle`.
- Model-free quickstart works:
  - `pip install "git+https://github.com/g1gowda123/chronicle.git@main"`
  - `chronicle demo`
  - `chronicle ask "where is my cup?" --db chronicle/demo_data/demo.db`
  - `chronicle doctor`
  - `chronicle eval`
- Product evaluation: **60/60** deterministic cases on the bundled 10.6 s demo.
  This is a self-authored smoke suite, not a general video benchmark.
- Research video QA (old `research/vfm-merged`): **15/18** (Cook 8/10,
  HoloAssist 7/8).
- Long-video HoloAssist diagnostic: **5/40** lenient on the low-res clip;
  **7/40** after transcript on high-res; strict evidence-aware score **3/40**.
- Main features include: lazy imports, optional UI, package data, CI matrix,
  demo database, doctor read-only check, model-free ask/eval, time-window
  queries, temporal location, counts, object source/destination, ordinals,
  confirmations, and evaluation-scope documentation.
- Latest CI for `1200f84` is green.
- README/docs distinguish product, GraphRAG, research video QA, and long-video
  diagnostics. Do not quote 60/60 as general video understanding.

## 4. Perception findings

### Low-res long-video baseline

- Worktree `/home/g1/projects/chronicle-longvideo`, branch
  `research/long-video-qa`, commit `990013e`.
- Data: 257 s HoloAssist `z210-sep-29-22-switch`, 454x256, 40 handcrafted
  questions in `data/qa/holoassist_z210_40q.json`.
- Result: **5/40 (12.5%)**, 50-54 VLM calls, 2,565 frames.
- Category scores: object identity 0/8, object location 2/7, counting 2/9,
  temporal order 1/6, activity 2/10, error correction 0/8.
- Only 1 of the 5 lenient passes is substantive; the rest are generic
  “last recorded event ... Work Table” strings matching a surface token.

### High-resolution branch

- Worktree `/home/g1/projects/chronicle-hires`, branch
  `research/high-res-longvideo`, commit `803eed7`.
- Downloaded a 896x504 pitch-shifted version of the same 257 s session
  (`data/real/holoassist_z210_hires_pitchshift.mp4`, untracked).
- High-res only: **5/40**; resolution alone does not fix the failure.
- 2x2 tiled YOLO-World restricted to fine labels: **5/40**, no usable fine
  objects. Tiling is off by default in `config_longvideo.yaml`.
- Whisper-medium transcript (21 segments) added speech evidence:
  - lenient **7/40**, strict **3/40**;
  - only +2 are genuine transcript wins (instructor quote and correction);
  - the other passes are still generic.
- MediaPipe hand crops, GroundingDINO-tiny, Florence caption/grounding, and
  Qwen2.5-VL-3B were probed; none reliably recovered wrist strap, game card,
  SD card, kickstand, power button, or slots.
- The high-res video has audio; the compressed 454x256 clip has none. This is
  why the transcript experiment only runs on the high-res branch.

### Root causes

- The graph has only 13-17 entities. Fine objects are absent.
- Action edges collapse to coarse `PICK_UP`; no attach/insert/open/press.
- There are no placement anchors for many objects.
- `person`/hand tracks absorb object questions.
- Counting is track-based; four joy-con tracks vs gold two.
- Some QA requires speech/instruction understanding; the low-res clip has no
  audio.
- Evaluation is lenient: empty/generic predictions can match. The empty
  prediction bug was fixed with `if not pred_norm: return False` and a test.
- The 5/40 and 7/40 numbers are diagnostic upper bounds, not headline claims.

## 5. SPORE: proposed general perception layer

Design doc: `docs/research/SPORE.md`.

Core thesis:

> **Perceive less. Name later.**

Chronicle should store **label-free, interaction-centric capsules** during
ingestion and bind semantics only when a query needs them. This is a
perception-layer change, not a new memory architecture.

Architecture:

1. **Cheap causal encoder**
   - low-res frames at low rate;
   - hand/motion/change gating;
   - contact proposals;
   - OCR triggers;
   - ASR/VAD.
2. **Capsule store**
   - `ObjectCapsule`: best views, CLIP embedding, contact state, support
     relation, optional label;
   - `ActionCapsule`: contact window and state delta;
   - `SceneCapsule`, `TextCapsule`, `SpeechCapsule`.
3. **Query router**
   - parse object/action/time/relation;
   - retrieve candidate capsules;
   - allocate a hard compute budget.
4. **Foveal late binding**
   - open-vocab detector/segmenter on best views;
   - small action model on contact windows;
   - bind labels only after visual verification.
5. **Evidence fusion**
   - write only grounded facts;
   - preserve provenance and confidence;
   - abstain with `unobserved` if unsupported.

Novelty: deferred semantic binding for a persistent egocentric memory. Closest
work: SmartRes (dynamic resolution routing), Q-Gate (query-modulated keyframe
selection), EgoP2C (plan-perceive-compose), WISH/EgoLoc (hand-object evidence).
None of them defer object semantics into a capsule store for a memory/query
system.

## 6. Next concrete work

On `/home/g1/projects/chronicle-spore`, branch `research/spore-perception`:

1. Implement `chronicle/perception/capsules.py`:
   - `ObjectCapsule`, `ActionCapsule`, `SceneCapsule`, `TextCapsule`,
     `SpeechCapsule`, `CapsuleStore`.
2. Implement a standalone `chronicle/perception/spore.py`:
   - low-res sampling;
   - hand/motion gating;
   - best-view selection;
   - query-time capsule retrieval;
   - open-vocabulary binding on crops.
3. Build a small multi-domain evaluation:
   - Cook 45 s (`data/qa/cook_45s.json`);
   - HoloAssist 34 s (`data/qa/holoassist_switch.json`);
   - z210 only as a secondary diagnostic.
4. Report:
   - object/action evidence recall/precision;
   - open-vocabulary binding accuracy;
   - QA;
   - expert calls/query, frames processed, VRAM, latency;
   - ablations: dense baseline, uniform sampling, no hand gating, no best-view,
     no late binding.
5. Only move stable, general improvements to `main` through a clean PR or
   cherry-pick. Never force-push `main`.

## 7. Environment

- Python: `/home/g1/projects/chronicle/.venv/bin/python` (3.12.13).
- CUDA: RTX 4060 Laptop, 8 GB.
- Models: `/home/g1/projects/chronicle-vfm/models` (YOLO-World-L, Florence,
  CLIP assets); symlink into experimental worktrees as needed.
- Full test command: `python -m pytest -q`.
- Product eval: `chronicle eval`.
- Known flaky test in some worktrees:
  `tests/test_query_frames.py::test_integrated_runtime_attaches_frame_to_answer`.
  It passes alone; do not assume a code regression from one CI failure.
- GitHub unauthenticated API cannot fetch failed job logs. Add JUnit artifacts
  or inspect locally instead.

## 8. Do not do

- Do not tune prompts/thresholds to the 40 z210 questions.
- Do not copy HoloAssist annotations into the runtime graph.
- Do not add a large VLM to every frame.
- Do not replace Chronicle’s graph/query engine with a generic VLM.
- Do not overwrite the uncommitted belief work in `chronicle-vfm`.
- Do not push unstable experiments to `main`.
- Do not quote 60/60, 7/40, or 5/40 as general video understanding.

## 9. Current immediate state

- `main`: `1200f84`, clean, CI green.
- `research/spore-perception`: based on `1200f84`; only
  `docs/research/SPORE.md` and this handoff are new.
- `research/high-res-longvideo`: `803eed7`; `config_longvideo_1280.yaml`
  untracked; no 1280 run executed.
- `research/long-video-qa`: `990013e`, clean.
- No code has been written for SPORE yet. The next agent should start with the
  capsule dataclasses and a standalone prototype, not another z210 threshold
  sweep.

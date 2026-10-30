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
| `/home/g1/projects/chronicle-spore` | `research/spore-perception` | pushed (see `git log`) | SPORE perceiver + benchmarks + split store + auto-eval + action consolidation + evidence-byte metric |
| `/home/g1/projects/chronicle-vfm` | `experiment/egotempo-benchmark` | `aab86b4` | uncommitted belief work by another agent |
| `/home/g1/projects/chronicle-packaging` | `feat/model-free-demo` | `1638606` | old merged packaging branch |

Do not reset or overwrite `/home/g1/projects/chronicle-vfm`: it contains
another agent's uncommitted `chronicle/belief/` prototype, `docs/BELIEF.md`,
tests, README and CLI edits. That work is independent of the perception work.

### SPORE branch status

`chronicle/perception/capsules.py` and `chronicle/perception/spore.py` now
implement a working label-free capsule perceiver and budgeted late-binding
query engine. `scripts/spore_real_bench.py` evaluates Cook 45 s and Switch 34 s
against dense YOLO-World pseudo-GT. Current headline: Cook loc@.30 1.00 /
mean IoU .692 / semantic .70 / 7.9 detector calls per query (dense: 150);
Switch loc@.30 .75 / mean IoU .611 / semantic .875 / 7.4 calls (dense: 128).
First-touch cup time is 4.6 s (annotated 4.5 s) via a fingertip-contact gate;
steady-state query latency is ~201 ms mean / 151 ms median at 7.9 calls after
a one-time model/predictor warmup (down from 617 / 337 ms at 14.4 calls).
Vocabulary switching is cached per label (~0.1 s vs ~2.1 s for `set_classes`).
Context-first binding is the default: retained keyframes answer first, and the
crop detector fallback runs only for context-less stores or `view_binding=True`.
Contact episodes are merged per object and short still fragments are folded
into their same-hand neighbour (24 actions for 12 annotated switch events;
recall 12/12 at IoU .1, 11/12 at .2-.3; precision .917 at .1, .792 at .2), and
counting uses a 32-frame uniform temporal grid so action-boundary changes do
not change counts.
z210 257 s is a scaling diagnostic: 298 s ingest and loc@.30 1.00, but semantic
.05 because the 454x256 capture cannot resolve the fine objects being counted
(temporal/activity questions also need verbs/speech not present). The
300-capsule hard budget cuts its store from 44.2 MB to 17.1 MB with no metric
loss; contexts, not capsules, carry most answer evidence. Play-domain QA now
scores 1.0 semantic / .667 loc@.30 / .475 IoU after fixing list-answer
stringification; its 407 capsules are bounded to 10.4 MB. Automatic
held-out questions generated from dense tracks score Cook .90 / Play .90 /
Switch .667 loc@.30 (any-instance Switch .889). The store now supports
`save(..., split=True)`: Cook's LLM-readable index is 1.02 MB instead of
8.22 MB, with evidence in sidecar blobs. View pixels are optional when
contexts are retained: Cook drops to 3.48 MB, Switch 2.37 MB, Play 4.55 MB,
with unchanged QA and fewer detector calls (contexts, not crop views, carry
localisation evidence). A new evidence-byte metric shows one answer touches
86.8 KB (Cook), 42.6 KB (Switch), 82.1 KB (Play), and 44.7 KB (z210) of JPEG
evidence, i.e. 0.2-1.0% of the store. See `SPORE.md` and `SPORE_PAPER.md` for
ablations, threats to validity, and open issues.


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

1. Keep the label-free capsule path stable: low-res sampling, hand/motion
   gating, best-view selection, context-frame retention, query-time routing,
   fused detection tracks, and temporal/action scans are implemented.
2. Next experiments, in priority order:
   - paper-grade evidence: add a real annotated subset or a second detector so
     the headline is not purely pseudo-GT; report variance over seeds;
   - multi-domain transfer beyond Cook/Switch/Play, without tuning to z210;
   - activity questions that genuinely synthesise a verb from hand motion and
     contact, not just an action-start timestamp;
   - a stronger fine-object detector path, since YOLO-World returns zero for
     game cards, SD cards, wrist straps, and comfort grips even on hand crops;
   - explicit evidence-byte budget for long video; the split index fixes
     LLM-readability, but context blobs still scale with duration.
3. Update `SPORE.md` and `SPORE_PAPER.md` with every experiment that changes a
   default or a headline number; keep negative results and threats to validity.
4. Only move stable, general improvements to `main` through a clean PR or
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
- `research/spore-perception`: based on `1200f84`; perceiver, query engine,
  benchmarks, context frames, fused tracks, counting and temporal scans are
  implemented. Newer count/support/docs edits are uncommitted on top of
  `606d5c8`.
- `research/high-res-longvideo`: `803eed7`; `config_longvideo_1280.yaml`
  untracked; no 1280 run executed.
- `research/long-video-qa`: `990013e`, clean.
- The next agent should improve count attribution/attribute answers or try a
  stronger fine-object detector, not another z210 prompt/threshold sweep.

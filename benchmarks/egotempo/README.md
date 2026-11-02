# EgoTempo benchmark

EgoTempo (CVPR 2025) is an open-ended egocentric video QA benchmark built on
Ego4D. It is the best fit for Chronicle because the questions require temporal
reasoning over a whole clip, not a single frame or a caption.

## Dataset facts

| Item | Value |
|---|---|
| Questions | 500 |
| Question types | 10 (50 each) |
| Clips | 367 unique clip IDs (paper reports 365 annotated clips) |
| Source videos | 222 Ego4D videos |
| Total clip time | ~272 minutes (~4.5 h) |
| Average clip length | ~45 s |
| QA license | CC-BY-4.0 |
| Video license | Ego4D license |

Question types:

- action sequence
- action-specific object
- counting actions
- counting objects
- future action prediction
- locating object
- object sequence
- object-specific action
- spatial relationship
- temporal event ordering

## What is already in this repo

- `data/benchmarks/egotempo/egotempo_openQA.json` — the 500 public QA pairs.
- `data/benchmarks/egotempo/manifest.json` — clips grouped by source video.
- `scripts/egotempo_manifest.py` — rebuilds the manifest.
- `scripts/prepare_egotempo.py` — downloads `video_540ss` and trims clips.
- `scripts/run_egotempo.py` — runs Chronicle on each clip and writes predictions.
- `scripts/evaluate_egotempo.py` — token diagnostic and official-style LLM judge.

## Data access (the blocking step)

EgoTempo videos are Ego4D sources and are **not redistributed**. You must:

1. Sign the Ego4D license: <https://ego4d-data.org/docs/start-here/>.
2. Wait for the AWS credentials email (~48 h).
3. Configure `~/.aws/credentials`.
4. Install the CLI:
   `pip install git+https://github.com/facebookresearch/Ego4d.git#egg=ego4d`

No Ego4D credentials are stored in this repository.

## Disk budget

`video_540ss` is the low-resolution short-side variant used by the official
EgoTempo notebook, not the 7 TB full dataset. A 222-video download is expected
to fit in roughly 30–60 GB; the trimmed 4.5 h clip set is roughly 5–10 GB.
Process in batches and delete source videos after trimming if space is tight.

## Commands

```bash
# 1. Build the manifest
python scripts/egotempo_manifest.py

# 2. Download + trim a pilot (first 5 source videos)
python scripts/prepare_egotempo.py --limit-videos 5

# 3. Run Chronicle on the pilot clips
python scripts/run_egotempo.py --limit-clips 5 --out runs/egotempo/pilot

# 4. Score with the local LLM judge
python scripts/evaluate_egotempo.py \
  --predictions runs/egotempo/pilot/predictions.json \
  --judge ollama --model qwen2.5:7b-instruct \
  --out runs/egotempo/pilot/eval.json

# Full benchmark
python scripts/prepare_egotempo.py
python scripts/run_egotempo.py --out runs/egotempo/full
python scripts/evaluate_egotempo.py \
  --predictions runs/egotempo/full/predictions.json \
  --judge ollama --model qwen2.5:7b-instruct \
  --out runs/egotempo/full/eval.json
```

Use `--dry-run` to inspect the download/trim commands without executing them.
Use `--skip-download` after the source videos are present, and
`--skip-existing` to resume a run.

## Evaluation protocol

The official EgoTempo evaluation is open-ended QA with an LLM judge. The judge
receives the question, the gold answer and the prediction, and returns a score
from 0 to 5. We treat **score >= 3 as correct** and also report the mean score
and a per-question-type breakdown.

`scripts/evaluate_egotempo.py --judge token` is a deterministic diagnostic
(exact/substring/token overlap). It is **not** the official metric and should
not be the headline result.

## Fixed vocabulary

The runner defaults to `--vocabulary-preset broad`, a fixed generic object
vocabulary. It is never derived from the questions. Report the vocabulary in
the paper; it is part of the method, not a hidden tuning knob.

## Status

- QA annotations + manifest: **ready**
- Download/trim/run/judge scripts: **ready**
- End-to-end numbers: **blocked on Ego4D credentials**

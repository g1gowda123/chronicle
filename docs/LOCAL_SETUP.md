# Local setup for offline evaluation (RTX 4060 / 8 GB)

This is the reproducible setup for `eval_pipeline.py`, `baseline_dense.py` and
`generate_paper_assets.py`. It assumes an NVIDIA RTX 4060 with a working CUDA
driver. All model paths resolve from the repository root and are listed in
`config.yaml`.

## 1. Python environment

From the repository root:

```bash
cd /home/g1/projects/chronicle-vfm

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
```

Install the CUDA build of PyTorch first so pip does not select a CPU wheel:

```bash
# CUDA 12.1 wheels work with current NVIDIA drivers; use cu124 if your driver
# requires it.
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121
```

Then install Chronicle and the evaluation stack:

```bash
pip install -e '.[perception,semantic-models,research]'
```

`requirements-semantic-models.txt` is an alternative for non-editable installs:

```bash
pip install -r requirements-semantic-models.txt
pip install -e '.[perception,research]'
```

Florence-2 is pinned to `transformers>=4.44,<5`. Transformers 5.x removes a
config attribute used by the official Florence-2 remote code and fails with
`'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'`.
The tested version is **4.46.3**.

Verify CUDA before downloading models:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("vram_mb", round(torch.cuda.get_device_properties(0).total_memory / 1024**2))
PY
```

Expected: `cuda True`, GPU name `NVIDIA GeForce RTX 4060`, ~8192 MB.

## 2. Model assets

`config.yaml` expects these exact paths:

```text
models/
├── yolov8l-worldv2.pt   # default detector
├── yolov8s-worldv2.pt   # efficiency ablation / optional S+L ensemble
├── hand_landmarker.task
└── Florence-2-base/
    ├── config.json
    ├── preprocessor_config.json
    ├── model.safetensors
    └── ...
```

### One-time download (online machine)

```bash
python scripts/fetch_models.py --download
```

This downloads:

- `yolov8l-worldv2.pt` (default) and `yolov8s-worldv2.pt` (ablation) from the
  Ultralytics releases;
- `hand_landmarker.task` from the MediaPipe model storage;
- `microsoft/Florence-2-base` into `models/Florence-2-base/`.

If Hugging Face CLI is preferred instead:

```bash
huggingface-cli download microsoft/Florence-2-base \
  --local-dir models/Florence-2-base
python scripts/fetch_assets.py --only all
```

### Offline verification (no network)

```bash
python scripts/fetch_models.py --check
```

This checks file presence and sizes only. It never downloads.

### OpenCLIP cache

The semantic engine uses OpenCLIP ViT-B-32. Cache it once while online:

```bash
python - <<'PY'
import open_clip
open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
print("OpenCLIP cached")
PY
```

For a strictly offline run:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

## 3. Config for the RTX 4060

`config.yaml` already points at the local Florence directory. For this GPU:

```yaml
perception:
  device: cuda
  detector_imgsz: 640
semantic:
  device: cuda
  require_florence: true      # fail fast instead of StubCaptioner
  florence_num_beams: 3
  florence_max_new_tokens: 128
```

`require_florence: true` means a missing model is an error, not a silent stub
fallback. `--offline` explicitly disables this for harness smoke tests.

### Qwen captioner and text backend

Two Qwen roles can be enabled independently:

| Role | Component | Backend | Config / flag |
|---|---|---|---|
| VLM captions | `chronicle/semantic/qwen_vl.py` | Ollama `qwen2.5vl:3b` | `--captioner qwen` |
| Text QA / GraphRAG | `chronicle/semantic/engine.py` | Ollama `qwen2.5:1.5b` | `--llm-backend ollama --llm-model qwen2.5:1.5b` |

Pull both models once while online:

```bash
ollama pull qwen2.5vl:3b
ollama pull qwen2.5:1.5b
```

The captioner is a drop-in replacement for Florence-2, so the state machine,
tracker and graph are unchanged:

```bash
python eval_pipeline.py \
  --video data/real/cook/aria_cook_45s.mp4 \
  --qa data/qa/cook_45s.json \
  --config config.yaml \
  --captioner qwen \
  --llm-backend ollama --llm-model qwen2.5:1.5b \
  --out runs/bench/aria_cook_45s_qq \
  --evaluate
```

`semantic.action_llm_enabled` controls whether the text LLM also verifies
per-clip actions during ingestion. It defaults to `false` in `config.yaml`:
ingestion stays deterministic and the LLM is reserved for QA, so per-question
LLM latency and call counts reflect answering rather than captioning. Set it to
`true` only when you explicitly want LLM-based action verification.

On an 8 GB RTX 4060 the measured device-wide peaks are:

| Configuration | Cook 45 s | HoloAssist 34 s |
|---|---:|---:|
| Chronicle + Florence-2 | 2.1 GB | 2.1 GB |
| Dense 1 FPS + Florence-2 | 0.8 GB | 0.8 GB |
| Chronicle + Qwen2.5-VL | 5.5 GB | 5.5 GB |
| Chronicle + Qwen2.5-VL + Qwen2.5 text | 6.8 GB | 6.8 GB |

All configurations fit in 8 GB, but the Qwen stack leaves little headroom.
Close other GPU processes before a run; `peak_gpu_used_mb` in `results.json`
reports the device-wide peak and includes the Ollama process.

Peak VRAM is device-wide, so an Ollama model left resident by an earlier run
inflates later Florence-only numbers. Unload idle models before measuring:

```bash
ollama stop qwen2.5vl:3b
ollama stop qwen2.5:1.5b
```

Florence + template runs need no Ollama daemon at all and should be measured
after those commands for a clean device peak.

## 4. Run the evaluation

One-command reproduction:

```bash
scripts/run_paper_eval.sh \
  data/real/cook/aria_cook_45s.mp4 \
  data/qa/cook_45s.json \
  runs/eval
```

The script also accepts environment variables for the model matrix:

```bash
CAPTIONER=qwen LLM_BACKEND=ollama LLM_MODEL=qwen2.5:1.5b SUFFIX=_qq \
  scripts/run_paper_eval.sh \
  data/real/cook/aria_cook_45s.mp4 data/qa/cook_45s.json runs/bench
```

- `CAPTIONER=florence|qwen`
- `LLM_BACKEND=template|ollama`
- `LLM_MODEL` (default `qwen2.5:1.5b`)
- `SUFFIX` names the outputs, e.g. `runs/bench/aria_cook_45s_qq`
  and `paper/table_comparison_qq.tex`.

QA suites used for the paper:

| File | Labels | Source |
|---|---|---|
| `data/qa/cook_45s.json` | silver | `data/vfm/events-cook/ledger.json` (Qwen2.5-VL) |
| `data/qa/holoassist_switch.json` | gold | HoloAssist human action annotations |
| `data/qa/ego4d_example.json` | illustrative | hand-written example |

Rows may carry `answers` (accepted alternatives) and `tolerance_s` for numeric
questions such as "when did X happen?".

Or run each stage separately.

Full Chronicle run (Task 3):

```bash
python eval_pipeline.py \
  --video data/real/cook/aria_cook_45s.mp4 \
  --qa data/qa/cook_45s.json \
  --config config.yaml \
  --captioner florence \
  --llm-backend template \
  --out runs/eval/cook \
  --evaluate
```

Outputs:

```text
runs/eval/cook/results.json
runs/eval/cook/summary_report.md
```

Dense 1 FPS baseline (Task 4):

```bash
python baseline_dense.py \
  --video data/real/cook/aria_cook_45s.mp4 \
  --qa data/qa/cook_45s.json \
  --config config.yaml \
  --captioner florence \
  --llm-backend template \
  --out runs/eval/baseline_cook
```

Outputs:

```text
runs/eval/baseline_cook/baseline_results.json
runs/eval/baseline_cook/baseline_summary.md
```

Paper table (Task 5):

```bash
python generate_paper_assets.py \
  --chronicle runs/eval/cook/results.json \
  --baseline runs/eval/baseline_cook/baseline_results.json \
  --out paper/table_comparison.tex
```

Outputs:

```text
paper/table_comparison.tex
paper/table_comparison.md
paper/table_comparison.json
```

Supplementary matrix across all model configurations:

```bash
python generate_matrix_table.py \
  --run-dir runs/bench \
  --run-dir runs/bench_holo \
  --out paper/table_matrix.tex
```

## 5. Offline smoke test

No weights, no CUDA, no Ollama:

```bash
python eval_pipeline.py --offline --max-frames 30 --evaluate \
  --out runs/eval/smoke
```

The smoke run is for plumbing only. It uses the colour detector, colour hands,
deterministic embeddings and the stub captioner.

## 6. Troubleshooting

### `vlm_backends` contains `StubCaptioner`

The Florence model was not loaded. Check:

```bash
python scripts/fetch_models.py --check
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

With `semantic.require_florence: true`, the engine raises before ingestion
rather than silently running the stub.

### MediaPipe falls back to colour hands

`models/hand_landmarker.task` is missing. Run:

```bash
python scripts/fetch_assets.py --only hands
```

### CUDA out of memory on 8 GB

Reduce only inference cost, not the state machine:

```yaml
semantic:
  florence_num_beams: 1
  florence_max_new_tokens: 64
perception:
  detector_imgsz: 512
  frame_stride: 2
```

### Florence generation issues

Florence-2 is a remote-code model. Keep `transformers>=4.44`, and make sure the
local directory contains the tokenizer and processor files from the snapshot.

### No Ollama

`semantic.llm_backend: template` is the default and uses deterministic GraphRAG
answers. Set `ollama` only if a local daemon is already running.

Check the daemon and the pulled models before a full run:

```bash
ollama list          # expect qwen2.5vl:3b and qwen2.5:1.5b
ollama ps            # confirms which model is resident
```

`eval_pipeline.py` performs a preflight check when `--captioner qwen` or
`--llm-backend ollama` is used and fails fast with the missing model name. If
Ollama is not running, omit those flags and stay on Florence + template.

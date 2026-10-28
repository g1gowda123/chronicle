#!/usr/bin/env bash
# One-command reproduction: Chronicle evaluation + dense baseline + LaTeX table.
#
# Usage:
#   scripts/run_paper_eval.sh [video] [qa] [output-dir]
#
# Optional environment:
#   CAPTIONER=florence|qwen   semantic captioner for both systems (default: florence)
#   LLM_BACKEND=template|ollama  GraphRAG/answer phrasing backend (default: template)
#   LLM_MODEL=qwen2.5:1.5b    Ollama text model when LLM_BACKEND=ollama
#   SUFFIX=_qwen              output-name suffix, useful for A/B runs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VIDEO="${1:-data/real/cook/aria_cook_45s.mp4}"
QA="${2:-data/qa/ego4d_example.json}"
OUT="${3:-runs/eval}"
NAME="$(basename "${VIDEO%.*}")"
CAPTIONER="${CAPTIONER:-florence}"
LLM_BACKEND="${LLM_BACKEND:-template}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:1.5b}"
SUFFIX="${SUFFIX:-}"
CHRON_NAME="${NAME}${SUFFIX}"
BASE_NAME="baseline_${NAME}${SUFFIX}"

if [[ ! -f "$VIDEO" ]]; then
  echo "video not found: $VIDEO" >&2
  exit 1
fi
if [[ ! -f "$QA" ]]; then
  echo "QA file not found: $QA" >&2
  exit 1
fi

if [[ "$CAPTIONER" == "florence" ]]; then
  echo "==> model check"
  python scripts/fetch_models.py --check
else
  echo "==> captioner: $CAPTIONER (server-side model, no download)"
fi

LLM_ARGS=(--llm-backend "$LLM_BACKEND")
if [[ "$LLM_BACKEND" == "ollama" || "$LLM_BACKEND" == "auto" ]]; then
  LLM_ARGS+=(--llm-model "$LLM_MODEL")
fi

echo "==> Chronicle evaluation (captioner=$CAPTIONER, llm=$LLM_BACKEND)"
python eval_pipeline.py \
  --video "$VIDEO" \
  --qa "$QA" \
  --config config.yaml \
  --captioner "$CAPTIONER" \
  "${LLM_ARGS[@]}" \
  --out "$OUT/$CHRON_NAME" \
  --evaluate

echo "==> Dense 1 FPS baseline (captioner=$CAPTIONER)"
python baseline_dense.py \
  --video "$VIDEO" \
  --qa "$QA" \
  --config config.yaml \
  --captioner "$CAPTIONER" \
  "${LLM_ARGS[@]}" \
  --out "$OUT/$BASE_NAME"

echo "==> paper assets"
python generate_paper_assets.py \
  --chronicle "$OUT/$CHRON_NAME/results.json" \
  --baseline "$OUT/$BASE_NAME/baseline_results.json" \
  --chronicle-label "Chronicle (${CAPTIONER}, ${LLM_BACKEND})" \
  --baseline-label "Dense 1 FPS (${CAPTIONER}, ${LLM_BACKEND})" \
  --out "paper/table_comparison${SUFFIX}.tex"

echo "done: $OUT/$CHRON_NAME/results.json and $OUT/$BASE_NAME/baseline_results.json"

#!/usr/bin/env bash
# Launch the Chronicle VFM Studio (upload -> compiled memory).
#
# Examples:
#   ./run_studio.sh --qwen
#   ./run_studio.sh --no-qwen --port 8093
#   ./run_studio.sh --qwen --qwen-model qwen2.5:7b-instruct --vlm-model qwen2.5vl:3b
#   ./run_studio.sh --help
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/home/g1/projects/chronicle/.venv/bin/python}"

if ! "$PY" -c "import nicegui" >/dev/null 2>&1; then
  echo "NiceGUI is missing from $PY." >&2
  echo "Install the UI extra in the main repo: cd /home/g1/projects/chronicle && .venv/bin/pip install 'nicegui>=3.0'" >&2
  exit 1
fi

exec "$PY" "$HERE/prototype/studio_app.py" "$@"

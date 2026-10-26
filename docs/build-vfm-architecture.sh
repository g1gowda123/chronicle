#!/usr/bin/env bash
# Rebuild the VFM architecture note (LaTeX -> PDF).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -n "${PDFLATEX:-}" ]; then
    ENGINE="$PDFLATEX"
elif command -v pdflatex >/dev/null 2>&1; then
    ENGINE="$(command -v pdflatex)"
elif [ -x "${HOME}/.TinyTeX/bin/x86_64-linux/pdflatex" ]; then
    ENGINE="${HOME}/.TinyTeX/bin/x86_64-linux/pdflatex"
else
    echo "pdflatex not found. Install TeX Live/TinyTeX or set PDFLATEX." >&2
    exit 1
fi

"${ENGINE}" -interaction=nonstopmode -halt-on-error VFM_ARCHITECTURE.tex
"${ENGINE}" -interaction=nonstopmode -halt-on-error VFM_ARCHITECTURE.tex
rm -f VFM_ARCHITECTURE.aux VFM_ARCHITECTURE.log VFM_ARCHITECTURE.out
echo "wrote $(pwd)/VFM_ARCHITECTURE.pdf"

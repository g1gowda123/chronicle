#!/usr/bin/env bash
# Rebuild the short (surface-level) VFM guide (LaTeX -> PDF).
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

"${ENGINE}" -interaction=nonstopmode -halt-on-error CHRONICLE_SHORT.tex
"${ENGINE}" -interaction=nonstopmode -halt-on-error CHRONICLE_SHORT.tex
rm -f CHRONICLE_SHORT.aux CHRONICLE_SHORT.log CHRONICLE_SHORT.out
echo "wrote $(pwd)/CHRONICLE_SHORT.pdf"

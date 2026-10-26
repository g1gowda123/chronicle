#!/usr/bin/env bash
# Rebuild the condensed specification PDF.
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

"${ENGINE}" -interaction=nonstopmode -halt-on-error chronicle-spec-condensed.tex
"${ENGINE}" -interaction=nonstopmode -halt-on-error chronicle-spec-condensed.tex
rm -f chronicle-spec-condensed.aux chronicle-spec-condensed.log chronicle-spec-condensed.out

echo "wrote $(pwd)/chronicle-spec-condensed.pdf"

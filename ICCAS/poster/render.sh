#!/bin/bash
# Build PPTX -> PDF -> PNG preview
set -e
cd "$(dirname "$0")"
python3 build_poster.py
soffice --headless --convert-to pdf ICCAS2026_Poster_Quadruped.pptx --outdir . >/dev/null 2>&1
pdftoppm -png -r 40 -singlefile ICCAS2026_Poster_Quadruped.pdf preview_small
pdftoppm -png -r 100 -singlefile ICCAS2026_Poster_Quadruped.pdf preview_full
echo "done: $(ls -la ICCAS2026_Poster_Quadruped.pdf | awk '{print $5}') bytes PDF"

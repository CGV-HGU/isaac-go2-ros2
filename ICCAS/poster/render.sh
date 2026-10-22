#!/bin/bash
# Build PPTX -> PDF -> PNG preview
set -e
cd "$(dirname "$0")"
python3 make_poster.py
soffice --headless --convert-to pdf ICCAS2026_Poster_Quadruped.pptx --outdir . >/dev/null 2>&1
pdftoppm -png -r 75 -singlefile ICCAS2026_Poster_Quadruped.pdf poster_preview_75dpi
pdftoppm -png -r 150 -singlefile ICCAS2026_Poster_Quadruped.pdf poster_preview_150dpi
echo "Done: Generated $(ls -lh ICCAS2026_Poster_Quadruped.pdf | awk '{print $5}') PDF and previews."

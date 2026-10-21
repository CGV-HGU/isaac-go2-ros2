# ICCAS 2026 Academic Poster Project

This folder contains the official A0 poster generation pipeline and source assets for the ICCAS 2026 conference paper:
**"Bridging the Sim-to-Real Gap in Quadrupedal Visual Navigation via 3D Gaussian Splatting"**

## Files
- `make_poster.py`: Python automation script that builds the complete A0 poster using `python-pptx`.
- `ICCAS2026_Poster_Quadruped.pptx`: The master editable PowerPoint presentation formatted to the official ICCAS A0 portrait template (84.1 cm x 118.0 cm).
- `ICCAS2026_Poster_Quadruped.pdf`: Print-ready high-resolution vector PDF generated from the presentation.
- `poster_preview_75dpi.png`: High-resolution visual preview.
- `assets/`: Scaled and cropped high-resolution figures, QR codes, and sub-assets.

## Poster Preview
![ICCAS 2026 Poster Preview](poster_preview_75dpi.png)

## How to Re-generate
```bash
python3 make_poster.py
soffice --headless --convert-to pdf ICCAS2026_Poster_Quadruped.pptx --outdir .
```

## How to Edit Session Code
Open `make_poster.py`, change `SESSION_CODE = "Session Code [TuP...]"`, and re-run the script or edit the top badge directly in PowerPoint.

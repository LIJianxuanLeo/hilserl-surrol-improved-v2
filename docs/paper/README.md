# HIL-SERL Reproducibility Paper (LaTeX)

Academic paper reporting the HIL-SERL reproducibility study with V1 sparse and V2 dense reward designs.

## Format

| Property | Value |
|---|---|
| Paper size | A4 |
| Margins | 1 inch all sides |
| Font | Times New Roman (via `mathptmx`) |
| Body size | 12pt |
| Line spacing | 1.5× (`\onehalfspacing`) |
| Citations | APA style — natbib `\citep`/`\citet` with `apalike`-compatible bibliography |
| Color | All-black text (no colored links) |
| Language | English only |

## Compile

Requires a standard TeX Live distribution (e.g., MacTeX, TeX Live 2023+).

```bash
make            # compiles paper.pdf via latexmk
make view       # opens the PDF (macOS)
make clean      # removes intermediate files
```

Or manually:
```bash
latexmk -pdf paper.tex
```

## Files

```
paper.tex       Single-file source (self-contained bibliography)
Makefile        Convenience compile commands
figures/
├── fig_v1_curves.pdf       V1 6-panel training curves
├── fig_v2_curves.pdf       V2 6-panel training curves
└── fig_compare_v1v2.pdf    V1 vs V2 4-panel overlay
```

## Citation

If you use this work, please cite:

> ZHU, J. (2026). HIL-SERL for Robotic Pick-and-Lift: A Reproducibility Study with
> Sparse and Dense Reward Designs. MSC Project, April 2026.

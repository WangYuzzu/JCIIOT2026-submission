# English LaTeX technical report

This directory contains the editable conference-style English report. It uses
a neutral two-column academic layout and deliberately contains no ICLR/EMNLP
branding, anonymous-review text, or proceedings footer.

Regenerate the vector trajectory figure directly from the canonical evidence:

```bash
python plot_trajectories.py
```

For Overleaf, upload this directory, select `main.tex`, and use XeLaTeX. The
pre-generated trajectory vector figure and two image-generated architecture
figures are included, so Overleaf does not need Python or an image API. Their
generation prompts and semantic review are recorded in
[`IMAGEGEN_FIGURES.md`](IMAGEGEN_FIGURES.md). Alternatively, compile locally
with Tectonic:

```bash
tectonic main.tex
```

The checked four-page PDF is published at `../TECHNICAL_REPORT_EN.pdf`. Build
artifacts are ignored; the LaTeX source, BibTeX database, figure generator, and
vector figure remain versioned and editable.

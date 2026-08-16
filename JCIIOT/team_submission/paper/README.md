# English LaTeX technical report

This directory contains the editable conference-style English report. It uses
a neutral two-column academic layout and deliberately contains no ICLR/EMNLP
branding, anonymous-review text, or proceedings footer.

Regenerate the vector trajectory figure directly from the canonical evidence:

```bash
python plot_trajectories.py
```

For Overleaf, upload this directory, select `main.tex`, and use XeLaTeX. The
pre-generated vector figure is included, so Overleaf does not need Python.
Alternatively, compile locally with Tectonic:

```bash
tectonic main.tex
```

The checked four-page PDF is published at `../TECHNICAL_REPORT_EN.pdf`. Build
artifacts are ignored; the LaTeX source, BibTeX database, figure generator, and
vector figure remain versioned and editable.

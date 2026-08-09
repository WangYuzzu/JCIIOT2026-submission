#!/usr/bin/env python3
"""Build the submission PDF from TECHNICAL_REPORT.md."""

from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import HTML


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "TECHNICAL_REPORT.md"
OUTPUT = HERE / "TECHNICAL_REPORT.pdf"

CSS = r"""
@font-face {
  font-family: "Noto CJK";
  src: url("file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf");
}
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
  @bottom-center { content: "JCIIOT 2026 · " counter(page) " / " counter(pages); color: #64748b; font-size: 8pt; }
}
body { font-family: "Noto CJK", sans-serif; color: #18212f; font-size: 9.3pt; line-height: 1.55; }
h1 { color: #0f3d5e; font-size: 23pt; border-bottom: 3px solid #2a9d8f; padding-bottom: 8px; }
h2 { color: #0f3d5e; font-size: 15pt; margin-top: 18px; border-bottom: 1px solid #b8c7d1; padding-bottom: 3px; }
h3 { color: #24566f; font-size: 11.5pt; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 8.3pt; }
th { background: #dbeef0; color: #12384a; }
th, td { border: 0.6px solid #91a4b0; padding: 4px 5px; vertical-align: top; }
tr:nth-child(even) td { background: #f6f8fa; }
pre { background: #102a3a; color: #e9f4f3; padding: 8px; border-radius: 4px; font-size: 7.8pt; white-space: pre-wrap; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 0.9em; color: #8a2c2c; }
pre code { color: inherit; }
blockquote { border-left: 4px solid #e9c46a; background: #fff9e9; margin: 8px 0; padding: 5px 10px; }
a { color: #176b87; text-decoration: none; }
"""


def main() -> int:
    html_body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<style>{CSS}</style></head><body>{html_body}</body></html>"""
    HTML(string=document, base_url=str(HERE)).write_pdf(OUTPUT)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

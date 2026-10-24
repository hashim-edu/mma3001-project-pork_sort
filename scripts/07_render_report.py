"""Render the written report to a print-ready A4 PDF.

The report is edited as a Claude Docs document and exported as Markdown. This
script turns that Markdown into an A4 PDF (11 pt Times, 20 mm margins) with the figures from
``results/figures`` placed at sensible widths, and reports the page on which
the References start (the page limit applies to the body only).

    python scripts/07_render_report.py report/report.md report/report.pdf

Needs the ``markdown`` and ``playwright`` packages (Chromium).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"

#: Figure label -> (file, width as a percentage of the text width)
FIGURES = {
    "Figure 1": ("pipeline.png", 92),
    "Figure 2": ("roc_blocked_test.png", 48),
    "Figure 3": ("examples_blocked_test.jpg", 78),
    "Figure 4": ("roofline.png", 58),
    "Figure C1": ("audit_test_train_gap.png", 55),
    "Figure C2": ("audit_similarity_vs_gap.png", 65),
    "Figure C3": ("audit_class_presence.png", 70),
    "Figure C4": ("classical_resolution.png", 80),
    "Figure C5": ("yolo_resolution.png", 72),
    "Figure C6": ("cost_ratio_sensitivity.png", 65),
}

CSS = """
@page { size: A4; margin: 20mm; }
body { font-family: 'Times New Roman', 'Liberation Serif', serif; font-size: 11pt; line-height: 1.3;
       color: #111; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; }
h2 { font-size: 12pt; margin: 9pt 0 3pt 0; }
h3 { font-size: 11pt; margin: 6pt 0 2pt 0; }
p { margin: 0 0 4pt 0; text-align: justify; }
ul, ol { margin: 0 0 4pt 0; padding-left: 16pt; }
li { margin: 0 0 1.5pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 2pt 0 5pt 0;
        page-break-inside: avoid; }
th, td { border: 0.5pt solid #999; padding: 1.5pt 4pt; vertical-align: top; }
th { background: #eee; }
figure { margin: 3pt 0 5pt 0; text-align: center; page-break-inside: avoid; }
figure img { display: block; margin: 0 auto; }
p.cap, figcaption { font-size: 9pt; font-style: italic; text-align: left; margin-bottom: 5pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; }
pre { font-size: 9pt; margin: 2pt 0 4pt 0; text-align: center; }
blockquote { margin: 2pt 14pt 4pt 14pt; font-style: italic; }
h1, h2, h3, p.cap, figure { break-after: avoid; page-break-after: avoid; }
p, li { orphans: 2; widows: 2; }
.back { break-before: page; page-break-before: always; font-size: 10pt; }
.back table { font-size: 8.6pt; break-inside: auto; page-break-inside: auto; }
tr { break-inside: avoid; page-break-inside: avoid; }
td code { font-size: 7.8pt; overflow-wrap: anywhere; }
.back td:first-child { width: 30%; }
.back p { text-align: left; }
"""


def to_html(md_text: str) -> str:
    # Image placeholders from the Docs export -> <figure> with a local image
    def fig(m):
        label = m.group(1).strip()
        if label not in FIGURES:
            return m.group(0)
        f, w = FIGURES[label]
        return f'<figure><img src="file://{FIG / f}" style="width:{w}%"></figure>'

    md_text = re.sub(r"&#91;image: (Figure [A-Z]?\d+)\\\]", fig, md_text)
    # Display formula: the latex code block becomes a centred plain-text formula
    md_text = re.sub(
        r"```latex\n(.*?)\n```", lambda m: "<pre>C(τ) = [ r · FN(τ) + FP(τ) ] / N</pre>", md_text, flags=re.S
    )
    html = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    # Italic paragraphs that start with Figure/Table are captions
    html = re.sub(r"<p><em>((?:Figure|Table) [^<]*)</em></p>", r'<p class="cap">\1</p>', html)
    head = f"<head><meta charset='utf-8'><style>{CSS}</style></head>"
    return f"<!doctype html><html>{head}<body>{keep_together(html)}</body></html>"


def keep_together(html: str) -> str:
    """Start the References and appendices on a new page.

    The body's page count can then be read off directly. Headings, table
    captions and figures are kept with the block that follows them by the
    ``break-after: avoid`` rules in the stylesheet.
    """
    return html.replace("<h2>References</h2>", '<div class="back"><h2>References</h2>', 1) + (
        "</div>" if "<h2>References</h2>" in html else ""
    )


def main(src: str, out: str) -> None:
    md_text = Path(src).read_text()
    html_path = Path(out).with_suffix(".html")
    html_path.write_text(to_html(md_text))
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page()
        page.goto(f"file://{html_path.resolve()}")
        page.pdf(
            path=out,
            format="A4",
            print_background=True,
            margin={"top": "20mm", "bottom": "20mm", "left": "20mm", "right": "20mm"},
            display_header_footer=True,
            header_template="<span></span>",
            footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#666'>"
            "<span class='pageNumber'></span></div>",
        )
        b.close()
    n = int(
        re.search(
            r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", out], capture_output=True, text=True).stdout
        ).group(1)
    )
    ref_page = None
    for i in range(1, n + 1):
        t = subprocess.run(
            ["pdftotext", "-f", str(i), "-l", str(i), out, "-"], capture_output=True, text=True
        ).stdout
        if re.search(r"^\s*References\s*$", t, re.M):
            ref_page = i
            break
    print(f"{out}: {n} pages; References start on page {ref_page}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

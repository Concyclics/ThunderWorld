# ThunderWorld visual research report

English report, evidence through 2026-09-11. The main edition now uses one visual
question per page: **14 landscape pages, 14 figures, approximately 74% less prose**
than the detailed English edition. All experimental values come from archived
measurements; this revision does not rerun inference.

- [Visual PDF](REPORT.pdf) and [HTML](REPORT.html).
- [Editable visual narrative](REPORT.md).
- [Detailed English appendix](REPORT_DETAILED.md), [PDF](REPORT_DETAILED.pdf), and
  [HTML](REPORT_DETAILED.html): retained equations, full explanation, and references.
- [Figure index](FIGURES.md) and [standalone figure package](FIGURES.zip).
- [Original Chinese source](REPORT.zh.md): retained for translation verification.

The visual edition includes five new vector flow diagrams, eight new quantitative
figures, and the unchanged AI-generated architecture overview. New figures have
editable SVG, standalone PDF, and PNG editions in `assets/figures/`.
`data/visuals/` contains plotted values; `data/visual_manifest.json` records source
hashes and aggregation definitions. Heatmaps and step curves are recomputed from
all 28 archived observation trajectories and reconcile with the prior summaries.

The overview's [prompt](assets/architecture_prompt.txt),
[correction prompt](assets/architecture_edit_prompt.txt), and
[provenance](assets/image_provenance.json) are preserved. It is a conceptual image,
not measured data. The new flow diagrams are also explicitly schematic.

To rebuild from the repository root:

```bash
MPLCONFIGDIR=/tmp/thunderworld-report-mpl data_local/e2e_venv/bin/python reports/thunderworld_paper_20260911/build_figures.py
python3 reports/thunderworld_paper_20260911/build_report.py
```

The first command uses the existing scientific Python environment and local raw
sampling archives. The second uses installed Markdown-it and Playwright/Chromium.
HTML embeds all figure bytes. Repository evidence links and links to standalone
figures require the bundle/repository alongside it. Narrow screens can scroll
individual figures or open them at full size.

`qa/` records numerical reconciliation, browser screenshots, mobile layout, and
PDF structure/rendered pages. `SHA256SUMS` covers bundle integrity. Regenerate it
and the figure archive after editing any figure or source.

Current scientific boundaries remain unchanged: the first quantizer is RTN;
automatic budgets often fall back to uniform repair; attention repair,
repair-aware quantization, energy, and cross-device results are not established.

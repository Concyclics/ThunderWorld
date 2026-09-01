# Technical report

[`thunderworld-current-evidence.html`](thunderworld-current-evidence.html) is a
self-contained, source-backed report generated from the checked-in evidence
snapshot. It includes the current quality–compute frontier, relation asymmetry,
head taxonomy, timestep patterns, kernel scaling, negative results, and the
retained implementation plan.

The checked-in report passed schema, payload-equality, and structural checks.
Browser-enhanced interaction was not certified because the local Chromium
installation entered the report reader's fallback state; the semantic HTML
fallback and chart data tables remain available without JavaScript or network
access.

Rebuild the canonical report payload with:

```bash
python3 scripts/build_report_artifact.py \
  --data-dir data \
  --output reports/artifact.json
```

See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for source and QA
details.

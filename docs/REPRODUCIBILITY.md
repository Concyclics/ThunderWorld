# Reproducibility and evidence audit

## Public artifact design

The original workspace contains model checkpoints, downloaded datasets,
generated videos, raw attention traces, and multi-gigabyte Parquet files. They
are intentionally excluded from this Git repository.

The public snapshot retains:

- reviewed aggregate rows;
- exact experiment-family denominators and step counts;
- source logical paths;
- byte sizes and SHA-256 hashes for every source aggregate;
- scripts that rebuild and validate the public data bundle;
- one self-contained HTML technical report.

No machine-local absolute path is stored in the public provenance manifest.

## Rebuild from the private workspace

```bash
python3 scripts/build_evidence_snapshot.py \
  --source-root /path/to/WorldModel \
  --output-dir data

python3 scripts/validate_evidence.py \
  --data-dir data \
  --output data/validation_report.json

python3 scripts/build_report_artifact.py \
  --data-dir data \
  --output reports/artifact.json
```

The report is packaged with the Data Analytics portable report builder. The
checked-in HTML is self-contained and does not fetch remote data.

## Validation checks

`scripts/validate_evidence.py` currently verifies:

1. the source manifest is unique and every source has a SHA-256 digest;
2. Generator taxonomy counts sum to 2,464 and shares sum to one;
3. all four directional ratios and confidence bounds match the curated rows;
4. the q12 50% policy has three 20-step seeds and passes SSIM≥0.9 plus
   PSNR≥24 dB;
5. the q12 75% policy is correctly marked provisional with only two seeds;
6. every comparable 20-step all-head sparse result fails SSIM 0.8;
7. the apparent q16 SSIM 0.827 result is excluded because it used four steps;
8. all Amdahl estimates recompute from the declared shares;
9. all-sparse H200 speedup increases across the tested 4K/8K/16K lengths;
10. timestep safe-share growth recomputes from the 20-step rows;
11. cross→self transfer remains negative on every held-out summary measure;
12. the eight-task equal-budget campaign has complete expected row counts and
    no missing output.

The current validation report passes all implemented checks but remains rated
`share_with_caveats`, because missing task-level and independent-prompt evidence
is a scope limitation rather than an arithmetic failure.

## Source snapshot files

| File | Purpose |
| --- | --- |
| `data/evidence_snapshot.json` | canonical nested aggregate |
| `data/quality_compute_frontier.csv` | comparable 20-step quality/compute operating points |
| `data/allhead_quality_20step.csv` | blanket all-head failure frontier |
| `data/directed_relations.csv` | paired relation asymmetry |
| `data/timestep_profile.csv` | 20-step local safe-share profile |
| `data/kernel_scaling.csv` | H200 all-sparse bucket scaling |
| `data/finding_ledger.csv` | retained/reworked/dropped contribution mapping |
| `data/source_manifest.json` | logical source identities and SHA-256 hashes |
| `data/validation_report.json` | validation receipt and required caveats |

## Statistical unit

When publication-scale confidence intervals are added, the independent unit
must be the prompt or robot episode. Three seeds are averaged within each prompt
before paired bootstrap. Seeds must not be treated as independent samples.

The current relation-atlas confidence intervals resample head-layer units and
are explicitly labeled structural rather than population-level intervals.

## Report QA

The portable artifact passed canonical schema validation, payload equality, and
structural verification. The installed Chromium entered the portable reader's
fallback state, so enhanced-reader responsive layout and source-dialog
interaction were not browser-certified in this build. The semantic HTML
fallback, source metadata, charts-as-tables, and exact artifact payload are
present in the self-contained file.

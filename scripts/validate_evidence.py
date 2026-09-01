#!/usr/bin/env python3
"""Validate the public ThunderWorld evidence snapshot and headline claims."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def amdahl(attention_speedup: float, generator_share: float = 0.9, attention_share: float = 0.387) -> float:
    fraction = generator_share * attention_share
    return 1.0 / (1.0 - fraction + fraction / attention_speedup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/validation_report.json"))
    args = parser.parse_args()

    snapshot = load_json(args.data_dir / "evidence_snapshot.json")
    manifest = load_json(args.data_dir / "source_manifest.json")
    frontier = load_csv(args.data_dir / "quality_compute_frontier.csv")
    allhead = load_csv(args.data_dir / "allhead_quality_20step.csv")
    directed = load_csv(args.data_dir / "directed_relations.csv")
    steps = load_csv(args.data_dir / "timestep_profile.csv")

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str, severity: str = "blocker") -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "severity": severity, "evidence": evidence}
        )

    source_ids = [row["id"] for row in manifest["sources"]]
    check(
        "source manifest is complete and unique",
        len(source_ids) == snapshot["data_quality"]["source_files"]
        and len(source_ids) == len(set(source_ids))
        and all(len(row["sha256"]) == 64 for row in manifest["sources"]),
        f"{len(source_ids)} reviewed source artifacts with SHA-256 hashes",
    )

    taxonomy = snapshot["characterization"]["taxonomy"]
    taxonomy_share = sum(row["share"] for row in taxonomy.values())
    taxonomy_count = sum(row["count"] for row in taxonomy.values())
    check(
        "Generator taxonomy has a valid denominator",
        close(taxonomy_share, 1.0) and taxonomy_count == 2464,
        f"shares sum to {taxonomy_share:.12f}; counts sum to {taxonomy_count}",
    )

    ratios = [float(row["asymmetry_ratio"]) for row in directed]
    ci_lows = [float(row["ci95_low"]) for row in directed]
    check(
        "directed relation headline matches the source rows",
        len(directed) == 4 and close(min(ratios), 4.094989217986449) and max(ratios) > 13.5,
        f"four relation pairs; asymmetry range {min(ratios):.2f}–{max(ratios):.2f}×",
    )
    check(
        "all directional structural intervals stay above 3.5×",
        min(ci_lows) > 3.5,
        f"minimum head-layer bootstrap lower bound {min(ci_lows):.2f}×",
        severity="caveat",
    )

    q50, q75, full = snapshot["runtime"]["quality_frontier"]
    check(
        "0.90 development point has three seeds",
        q50["seeds"] == 3
        and q50["steps"] == 20
        and q50["ssim_min"] >= 0.9
        and q50["psnr_min_db"] >= 24.0,
        (
            f"min SSIM {q50['ssim_min']:.4f}, min PSNR {q50['psnr_min_db']:.2f} dB, "
            f"QK reduction {q50['joint_qk_reduction']:.1%}"
        ),
    )
    check(
        "0.80 aggressive point is correctly marked provisional",
        q75["seeds"] == 2
        and q75["steps"] == 20
        and q75["ssim_min"] >= 0.8
        and "provisional" in q75["evidence_status"],
        (
            f"two seeds; min SSIM {q75['ssim_min']:.4f}; "
            f"QK reduction {q75['joint_qk_reduction']:.1%}"
        ),
        severity="caveat",
    )
    check(
        "blanket all-head policy fails the 0.80 gate",
        max(float(row["global_ssim"]) for row in allhead) < 0.8
        and full["ssim_min"] < 0.8,
        f"best comparable 20-step SSIM {full['ssim_min']:.4f}",
    )

    excluded = snapshot["runtime"]["excluded_noncomparable_result"]
    check(
        "the apparent all-head 0.827 result is excluded for step mismatch",
        excluded["steps"] == 4
        and excluded["global_ssim"] > 0.8
        and excluded["comparable"] is False,
        f"SSIM {excluded['global_ssim']:.4f} came from {excluded['steps']} steps",
    )

    for row in (q50, q75, full):
        expected = amdahl(float(row["attention_speedup"]))
        check(
            f"Amdahl estimate recomputes for {row['policy']}",
            close(expected, float(row["estimated_e2e_speedup"])),
            (
                f"attention {row['attention_speedup']:.3f}× -> "
                f"estimated E2E {row['estimated_e2e_speedup']:.3f}×"
            ),
        )

    kernel_rows = snapshot["runtime"]["kernel_scaling"]
    ar61 = sorted(
        (row for row in kernel_rows if row["ar_tokens"] == 61), key=lambda row: row["dm_tokens"]
    )
    check(
        "all-sparse kernel speedup scales with tested DM length",
        [row["dm_tokens"] for row in ar61] == [4096, 8192, 16384]
        and all(left["attention_speedup"] < right["attention_speedup"] for left, right in zip(ar61, ar61[1:])),
        "AR=61 speedups: "
        + ", ".join(f"{row['dm_tokens']}: {row['attention_speedup']:.2f}×" for row in ar61),
        severity="caveat",
    )

    step0 = next(row for row in steps if int(row["step"]) == 0)
    step19 = next(row for row in steps if int(row["step"]) == 19)
    growth = float(step19["safe_share_0p05"]) / float(step0["safe_share_0p05"])
    check(
        "late-step local sparsifiability growth recomputes",
        close(growth, snapshot["trajectory"]["timestep_headline"]["safe_share_growth_x"]),
        f"safe share grows {growth:.3f}× from step 0 to step 19",
    )

    cross_self = snapshot["trajectory"]["cross_to_self_negative"]["headline_metrics"]
    check(
        "cross-to-self transfer remains a negative result",
        cross_self["position_controlled_residual_mean_correlation"] < 0
        and cross_self["heldout_cross_anchor_omitted_reduction"] < 0
        and cross_self["heldout_cross_query_budget_omitted_reduction"] < 0,
        (
            "residual correlation "
            f"{cross_self['position_controlled_residual_mean_correlation']:.3f}; "
            "both held-out omitted-mass changes are negative"
        ),
    )

    candidate = snapshot["data_quality"]["candidate_campaign"]
    check(
        "full equal-budget candidate campaign passed completeness checks",
        candidate["passed"] is True
        and candidate["missing_outputs"] == []
        and candidate["head_metric_rows"] == candidate["expected_head_metric_rows"]
        and candidate["quality_rows"] == candidate["expected_quality_rows"],
        (
            f"{candidate['head_metric_rows']:,} head rows and "
            f"{candidate['quality_rows']:,} quality rows"
        ),
    )

    blockers = [row for row in checks if row["severity"] == "blocker" and not row["passed"]]
    caveat_failures = [row for row in checks if row["severity"] == "caveat" and not row["passed"]]
    # The arithmetic is internally consistent, but the strongest quality point is
    # still one prompt and the kernel evidence is a microbenchmark. Therefore the
    # correct stakeholder confidence remains "share with caveats" even when every
    # implemented consistency check passes.
    assessment = "share_with_caveats" if not blockers else "needs_revision"

    report = {
        "schema_version": 1,
        # Tie the receipt to the immutable snapshot rather than wall-clock time so
        # independent validation runs are byte-for-byte reproducible.
        "generated_at": snapshot["compiled_at"],
        "assessment": assessment,
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "blockers": blockers,
        "caveat_failures": caveat_failures,
        "checks": checks,
        "required_caveats": snapshot["claim_boundaries"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    if blockers:
        raise SystemExit(f"Evidence validation failed with {len(blockers)} blocker(s)")


if __name__ == "__main__":
    main()

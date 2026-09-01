#!/usr/bin/env python3
"""Build a compact, public evidence snapshot from the private experiment workspace.

The source workspace contains checkpoints, videos, traces, and large Parquet files that
must not be committed. This script extracts only reviewed aggregate metrics, preserves a
SHA-256 manifest for provenance, and removes machine-local absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


LOGICAL_SOURCES = {
    "candidate_validation": "reports/data/candidate_probe_v1/validation.json",
    "modality_summary": "reports/data/modality_relations/analysis_summary.json",
    "modality_relations": "reports/data/modality_relations/relation_summary.csv",
    "contribution_summary": "reports/data/contribution_evidence/summary.json",
    "cross_self_negative": "reports/data/cross_self_transfer/decision_summary.json",
    "trajectory_validation": (
        "reports/data/kernels/trajectory_head_search/paper_tables/validation_summary.json"
    ),
    "timestep_summary": "reports/timestep_patterns_v1/summary.json",
    "timestep_profile": "reports/timestep_patterns_v1/step_profile.csv",
    "engineering_summary": "reports/engineering_adoption_v1/summary.json",
    "engineering_validation": "reports/engineering_adoption_v1/validation.json",
    "engineering_quality": "reports/engineering_adoption_v1/quality_seeds.json",
    "gqa50_kernel": "reports/data/engineering_adoption_v1/mixed_head_q12_gqa50.json",
    "gqa75_kernel": "reports/data/engineering_adoption_v1/mixed_head_q12_gqa75.json",
    "all_sparse_kernel": "reports/data/engineering_adoption_v1/joint_full_moment_v2.json",
    "drsa_analysis": "reports/drsa_experiment_v1/analysis.json",
    "q12_gqa50_seed17": "reports/data/kernels/q12_multiseed_gqa50_t2v_seed17_quality.json",
    "q12_gqa50_seed29": "reports/data/kernels/q12_multiseed_gqa50_t2v_seed29_quality.json",
    "q12_gqa50_seed53": "reports/data/kernels/q12_multiseed_gqa50_t2v_seed53_quality.json",
    "q12_gqa75_seed29": "reports/data/kernels/q12_multiseed_gqa75_t2v_seed29_quality.json",
    "q12_gqa75_seed53": "reports/data/kernels/q12_multiseed_gqa75_t2v_seed53_quality.json",
    "allhead_q8": "reports/data/kernels/q8_keep375_allhead_20step_seed29_quality.json",
    "allhead_q12": "reports/data/kernels/q12_allhead_20step_seed29_quality.json",
    "allhead_q16": "reports/data/kernels/q16_keep3125_allhead_20step_seed29_quality.json",
    "allhead_q24": "reports/data/kernels/q24_keep25_allhead_20step_seed29_quality.json",
    "allhead_q32": "reports/data/kernels/q32_keep1875_allhead_20step_seed29_quality.json",
    "allhead_q40": "reports/data/kernels/q40_keep125_allhead_20step_seed29_quality.json",
    "allhead_q48": "reports/data/kernels/q48_keep0_allhead_20step_seed29_quality.json",
    "allhead_cache2": "reports/data/kernels/cache2_gqa100_20step_seed29_quality.json",
    "excluded_q16_quality": "reports/data/kernels/q16_allhead_exact_taylor_t2v_quality.json",
    "excluded_q16_args": "outputs/runtime_qinterp16_taylor_t2v/t2v_61f/sample_args.json",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def amdahl_e2e(attention_speedup: float, generator_share: float = 0.9, attention_share: float = 0.387) -> float:
    accelerated = generator_share * attention_share
    return 1.0 / (1.0 - accelerated + accelerated / attention_speedup)


def quality_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    ssims = [float(row["global_ssim"]) for row in rows]
    psnrs = [float(row["psnr_db"]) for row in rows]
    return {
        "ssim_min": min(ssims),
        "ssim_mean": mean(ssims),
        "ssim_max": max(ssims),
        "psnr_min_db": min(psnrs),
        "psnr_mean_db": mean(psnrs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    paths = {key: source_root / logical for key, logical in LOGICAL_SOURCES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required evidence sources:\n" + "\n".join(missing))

    source_manifest = []
    for key, logical in LOGICAL_SOURCES.items():
        path = paths[key]
        source_manifest.append(
            {
                "id": key,
                "logical_path": logical,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    candidate_validation = read_json(paths["candidate_validation"])
    modality_summary = read_json(paths["modality_summary"])
    contribution = read_json(paths["contribution_summary"])
    cross_self = read_json(paths["cross_self_negative"])
    trajectory = read_json(paths["trajectory_validation"])
    timestep = read_json(paths["timestep_summary"])
    timestep_profile = read_csv(paths["timestep_profile"])
    engineering = read_json(paths["engineering_summary"])
    engineering_validation = read_json(paths["engineering_validation"])
    drsa = read_json(paths["drsa_analysis"])
    gqa50_kernel = read_json(paths["gqa50_kernel"])["rows"]
    gqa75_kernel = read_json(paths["gqa75_kernel"])["rows"]
    all_sparse_kernel = read_json(paths["all_sparse_kernel"])["rows"]

    relation_rows = read_csv(paths["modality_relations"])
    anchor_relations = []
    anchor_keys = {
        ("droid_forward", "generator:action.clean<-DM.action.clean"),
        ("robot_forward", "generator:action.clean<-DM.action.clean"),
        ("t2vs", "generator:sound.noisy<-DM.sound.noisy"),
    }
    for row in relation_rows:
        if (row["task"], row["relation"]) in anchor_keys:
            anchor_relations.append(
                {
                    "task": row["task"],
                    "relation": row["relation"],
                    "attention_mass": f(row, "mean_relation_mass"),
                    "score_fraction": f(row, "mean_score_fraction"),
                    "mass_per_score_share": f(row, "mass_per_score_share"),
                }
            )

    taxonomy_counts = modality_summary["taxonomy_counts"]
    # The relation atlas includes 448 Reasoner units. Remove them to recover the
    # published Generator-only 2,464-unit taxonomy.
    generator_counts = {
        "specialist": int(taxonomy_counts["specialist"] - 320),
        "hybrid": int(taxonomy_counts["hybrid"] - 21),
        "mixer": int(taxonomy_counts["mixer"] - 107),
    }
    generator_total = sum(generator_counts.values())
    taxonomy = {
        key: {"count": value, "share": value / generator_total}
        for key, value in generator_counts.items()
    }

    q50_quality = [
        read_json(paths["q12_gqa50_seed17"]),
        read_json(paths["q12_gqa50_seed29"]),
        read_json(paths["q12_gqa50_seed53"]),
    ]
    q75_quality = [
        read_json(paths["q12_gqa75_seed29"]),
        read_json(paths["q12_gqa75_seed53"]),
    ]

    allhead_specs = [
        ("q8_keep37.5", "allhead_q8", 0.375),
        ("q12_keep18.75", "allhead_q12", 0.1875),
        ("q16_keep31.25", "allhead_q16", 0.3125),
        ("q24_keep25", "allhead_q24", 0.25),
        ("q32_keep18.75", "allhead_q32", 0.1875),
        ("q40_keep12.5", "allhead_q40", 0.125),
        ("q48_keep0", "allhead_q48", 0.0),
        ("cache2_gqa100", "allhead_cache2", math.nan),
    ]
    allhead_quality = []
    for policy, source_key, keep_ratio in allhead_specs:
        quality = read_json(paths[source_key])
        allhead_quality.append(
            {
                "policy": policy,
                "steps": 20,
                "seed": 29,
                "keep_ratio": None if math.isnan(keep_ratio) else keep_ratio,
                "global_ssim": float(quality["global_ssim"]),
                "psnr_db": float(quality["psnr_db"]),
                "comparable": True,
            }
        )

    excluded_quality = read_json(paths["excluded_q16_quality"])
    excluded_args = read_json(paths["excluded_q16_args"])
    excluded_result = {
        "policy": "q16_allhead_exact_taylor",
        "global_ssim": float(excluded_quality["global_ssim"]),
        "psnr_db": float(excluded_quality["psnr_db"]),
        "steps": int(excluded_args["num_steps"]),
        "seed": int(excluded_args["seed"]),
        "comparable": False,
        "reason": "4-step rollout; excluded from 20-step quality frontier",
    }

    kernel_scaling = []
    for row in all_sparse_kernel:
        kernel_scaling.append(
            {
                "execution": "all sparse",
                "dm_tokens": int(row["dm_tokens"]),
                "ar_tokens": int(row["ar_tokens"]),
                "attention_speedup": float(row["amortized_speedup"]),
                "joint_qk_reduction": float(row["joint_qk_reduction"]),
                "output_relative_l2_vs_joint_repair": float(
                    row["relation_vs_joint_output_relative_l2"]
                ),
                "device": row["device"],
            }
        )

    gqa50_16k = next(
        row for row in gqa50_kernel if row["dm_tokens"] == 16384 and row["ar_tokens"] == 61
    )
    gqa75_16k = next(
        row for row in gqa75_kernel if row["dm_tokens"] == 16384 and row["ar_tokens"] == 61
    )
    all_sparse_16k = next(
        row for row in all_sparse_kernel if row["dm_tokens"] == 16384 and row["ar_tokens"] == 61
    )

    q50_stats = quality_stats(q50_quality)
    q75_stats = quality_stats(q75_quality)
    allhead_best = max(row["global_ssim"] for row in allhead_quality)
    allhead_best_psnr = max(
        row["psnr_db"] for row in allhead_quality if row["global_ssim"] == allhead_best
    )

    quality_frontier = [
        {
            "policy": "q12 · 50% GQA sparse",
            "sparse_group_fraction": 0.50,
            "joint_qk_reduction": float(gqa50_16k["joint_qk_reduction"]),
            "attention_speedup": float(gqa50_16k["attention_speedup"]),
            "estimated_e2e_speedup": amdahl_e2e(float(gqa50_16k["attention_speedup"])),
            "ssim_min": q50_stats["ssim_min"],
            "ssim_mean": q50_stats["ssim_mean"],
            "psnr_min_db": q50_stats["psnr_min_db"],
            "prompts": 1,
            "seeds": 3,
            "steps": 20,
            "evidence_status": "validated development screen",
            "quality_gate": "0.90 development",
        },
        {
            "policy": "q12 · 75% GQA sparse",
            "sparse_group_fraction": 0.75,
            "joint_qk_reduction": float(gqa75_16k["joint_qk_reduction"]),
            "attention_speedup": float(gqa75_16k["attention_speedup"]),
            "estimated_e2e_speedup": amdahl_e2e(float(gqa75_16k["attention_speedup"])),
            "ssim_min": q75_stats["ssim_min"],
            "ssim_mean": q75_stats["ssim_mean"],
            "psnr_min_db": q75_stats["psnr_min_db"],
            "prompts": 1,
            "seeds": 2,
            "steps": 20,
            "evidence_status": "provisional aggressive screen",
            "quality_gate": "0.80 aggressive",
        },
        {
            "policy": "100% all-head sparse · best comparable",
            "sparse_group_fraction": 1.0,
            "joint_qk_reduction": float(all_sparse_16k["joint_qk_reduction"]),
            "attention_speedup": float(all_sparse_16k["amortized_speedup"]),
            "estimated_e2e_speedup": amdahl_e2e(float(all_sparse_16k["amortized_speedup"])),
            "ssim_min": allhead_best,
            "ssim_mean": allhead_best,
            "psnr_min_db": allhead_best_psnr,
            "prompts": 1,
            "seeds": 1,
            "steps": 20,
            "evidence_status": "fails 0.80 quality screen",
            "quality_gate": "none",
        },
    ]

    directional = []
    for row in contribution["directional_pairs"]:
        directional.append(
            {
                "task": row["task"],
                "relation_pair": row["pair"],
                "head_layers": int(row["head_layers"]),
                "forward_mass": float(row["left_mass"]),
                "reverse_mass": float(row["right_mass"]),
                "asymmetry_ratio": float(row["asymmetry_ratio"]),
                "ci95_low": float(row["ratio_ci95_low"]),
                "ci95_high": float(row["ratio_ci95_high"]),
                "direction_consistency": float(row["paired_direction_consistency"]),
            }
        )

    step_rows = []
    for row in timestep_profile:
        step_rows.append(
            {
                "step": int(row["step"]),
                "stage": int(row["stage"]),
                "safe_share_0p05": f(row, "safe_share_0p05"),
                "safe_jaccard_0p05": None
                if not row["safe_jaccard_0p05"]
                else f(row, "safe_jaccard_0p05"),
                "mean_robust_output_l2": f(row, "mean_robust_output_l2"),
                "p95_robust_output_l2": f(row, "p95_robust_output_l2"),
            }
        )

    findings = [
        {
            "id": "F1",
            "finding": "有向跨模态关系存在 4.09–13.53× 注意力质量不对称",
            "implementation": "GQA group × directed relation budget/skip，保留 hard anchors",
            "decision": "KEEP · primary novelty",
            "evidence_level": "structural pilot + 20-step equal-QK screen",
            "caveat": "结构性置信区间基于 head-layer 单元；仍缺 24 个独立 prompt 的确认实验",
        },
        {
            "id": "F2",
            "finding": "关系/层/head 异质且 archetype 跨 step 稳定",
            "implementation": "离线 head archetype，在线只调 budget/refresh/fallback；按 GQA bucket 执行",
            "decision": "KEEP · core method",
            "evidence_level": "2,464 Generator units; 94.5% pass stability",
            "caveat": "固定 hand depth relocation 在闭环中失败，layer 只能作为 compiler feature",
        },
        {
            "id": "F3",
            "finding": "action/sound 小关系是高信息密度 dense anchors",
            "implementation": "action/audio/control/special/physical-time neighbors 强制保留",
            "decision": "KEEP · merge into semantic method",
            "evidence_level": "134–302× mass per score share",
            "caveat": "仍需 equal-score anchor removal 的 task-level counterfactual",
        },
        {
            "id": "F4",
            "finding": "token oracle 显著强于 natural block oracle",
            "implementation": "local geometry + query/head/timestep-conditioned semantic block list",
            "decision": "KEEP · reworked primitive",
            "evidence_level": "35.4–43.3% token saving vs 11.6–19.4% natural-block saving",
            "caveat": "naive K-only reorder 失败；semantic reorder 不作为默认运行时",
        },
        {
            "id": "F5",
            "finding": "局部误差不能单独预测完整 diffusion trajectory 风险",
            "implementation": "joint-projected error Gram + full-rollout/UCB trajectory compiler",
            "decision": "KEEP · policy contribution",
            "evidence_level": "median GQA cancellation ratio 0.412; depth schedule equal-QK quality regression",
            "caveat": "当前 Gram/UCB 仍缺多任务 confirmatory rollout",
        },
        {
            "id": "F6",
            "finding": "重要 block/layout 跨 diffusion step 高度稳定，但 output 不可直接复用",
            "implementation": "stage-wise budget + descriptor/block-list reuse；保留 QK/PV 重算",
            "decision": "KEEP · system allocator",
            "evidence_level": "safe-set Jaccard 0.930; anchor Jaccard 0.919",
            "caveat": "previous-output cache SSIM 0.77–0.80，默认禁用",
        },
        {
            "id": "N1",
            "finding": "浅层 DM←AR 热块不能可靠预测深层 DM←DM 热块",
            "implementation": "self/cross 独立选块，仅在 trajectory compiler 中联合校准",
            "decision": "DROP · primary algorithm",
            "evidence_level": "position-controlled residual correlation -0.065",
            "caveat": "保留为负结果与消融",
        },
        {
            "id": "N2",
            "finding": "全 layer/head/step 统一 sparse 在当前 primitive 下不可行",
            "implementation": "只把安全 layer-stage cells 编译到 all-sparse kernel bucket",
            "decision": "DROP · blanket policy",
            "evidence_level": "可比 20-step all-head 最佳 SSIM 0.770",
            "caveat": "all-sparse 16K 1.93× 仅是 kernel 上限，不是质量验证后的端到端结果",
        },
    ]

    protocols = [
        {
            "name": "full equal-budget candidate campaign",
            "tasks": 8,
            "prompts_per_task": 24,
            "seeds": 3,
            "steps": "20 and 40",
            "layers": 36,
            "heads": 32,
            "purpose": "geometry / semantic / reorder / hybrid counterfactual characterization",
        },
        {
            "name": "relation atlas",
            "tasks": 6,
            "prompts_per_task": 1,
            "seeds": 1,
            "steps": 4,
            "layers": 7,
            "heads": 32,
            "purpose": "directed modality and anchor discovery",
        },
        {
            "name": "timestep trace",
            "tasks": 1,
            "prompts_per_task": 24,
            "seeds": 3,
            "steps": 20,
            "layers": 36,
            "heads": 32,
            "purpose": "local sparsifiability and step/layout stability",
        },
        {
            "name": "q12 50% GQA quality screen",
            "tasks": 1,
            "prompts_per_task": 1,
            "seeds": 3,
            "steps": 20,
            "layers": 36,
            "heads": 32,
            "purpose": "latest 0.90 development fidelity point",
        },
        {
            "name": "q12 75% GQA aggressive screen",
            "tasks": 1,
            "prompts_per_task": 1,
            "seeds": 2,
            "steps": 20,
            "layers": 36,
            "heads": 32,
            "purpose": "provisional 0.80 aggressive mode",
        },
        {
            "name": "H200 kernel microbench",
            "tasks": 0,
            "prompts_per_task": 0,
            "seeds": 0,
            "steps": "random-tensor benchmark",
            "layers": "single attention layer",
            "heads": 32,
            "purpose": "kernel scaling upper bound; not full Cosmos rollout",
        },
    ]

    snapshot = {
        "schema_version": 1,
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "evidence_as_of": engineering["as_of"],
        "assessment": "share_with_caveats",
        "headline": {
            "paper_thesis": (
                "World-model sparsity is relation-local, while approximation error is trajectory-global."
            ),
            "current_0p9_policy": quality_frontier[0],
            "aggressive_0p8_policy": quality_frontier[1],
            "all_sparse_upper_bound": quality_frontier[2],
            "full_sparse_policy_supported": False,
            "recommended_runtime": (
                "GQA-group head routing with directed relation budgets, dense anchors, semantic/local "
                "block lists, timestep reuse, and one joint online-softmax state"
            ),
        },
        "protocols": protocols,
        "characterization": {
            "directional_relations": directional,
            "taxonomy": taxonomy,
            "taxonomy_total_generator_units": generator_total,
            "dominant_relation_pass_stability": modality_summary[
                "generator_dominant_pass_stability"
            ],
            "cross_task_agreement": modality_summary["cross_task_agreement"],
            "anchors": anchor_relations,
            "layer_schedule": contribution["layer_schedule_checks"],
            "score_saving_potential": modality_summary["score_saving_potential"],
        },
        "trajectory": {
            "gqa_cancellation_ratio_median": trajectory["headline"][
                "cancellation_ratio_median"
            ],
            "local_oracle_coverage": trajectory["headline"]["local_oracle_coverage"],
            "coverage_for_1p2x_attention": trajectory["headline"][
                "coverage_for_1_2x_attention"
            ],
            "cross_to_self_negative": cross_self,
            "timestep_headline": timestep["headline"],
        },
        "runtime": {
            "device": engineering["device"],
            "definitions": engineering["definitions"],
            "engineering_validation": engineering_validation,
            "quality_frontier": quality_frontier,
            "allhead_quality_20step": allhead_quality,
            "excluded_noncomparable_result": excluded_result,
            "kernel_scaling": kernel_scaling,
            "drsa_screen": drsa,
        },
        "findings": findings,
        "data_quality": {
            "candidate_campaign": candidate_validation,
            "timestep": timestep["data_quality"],
            "trajectory_integrity": trajectory["integrity"],
            "source_files": len(source_manifest),
        },
        "claim_boundaries": [
            "Paired SSIM/PSNR measure same-seed dense-reference fidelity, not semantic quality or task success.",
            "H200 microbenchmarks use random tensors and measure attention kernels, not complete Cosmos3 rollouts.",
            "The 0.90 point covers one T2V prompt across three seeds; the 0.80 point covers two seeds.",
            "The 4-step all-head SSIM 0.827 result is excluded from the comparable 20-step frontier.",
            "Final VBench, control, action, and task-success non-inferiority remains unmeasured.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evidence_snapshot.json", snapshot)
    write_json(
        output_dir / "source_manifest.json",
        {
            "schema_version": 1,
            "generated_at": snapshot["compiled_at"],
            "source_root_redacted": True,
            "sources": source_manifest,
        },
    )
    write_json(
        output_dir / "quality_seed_records.json",
        {
            "q12_gqa50": [
                {
                    "seed": seed,
                    "global_ssim": row["global_ssim"],
                    "psnr_db": row["psnr_db"],
                    "temporal_delta_psnr_db": row["temporal_delta_psnr_db"],
                }
                for seed, row in zip((17, 29, 53), q50_quality, strict=True)
            ],
            "q12_gqa75": [
                {
                    "seed": seed,
                    "global_ssim": row["global_ssim"],
                    "psnr_db": row["psnr_db"],
                    "temporal_delta_psnr_db": row["temporal_delta_psnr_db"],
                }
                for seed, row in zip((29, 53), q75_quality, strict=True)
            ],
        },
    )
    write_csv(
        output_dir / "quality_compute_frontier.csv",
        quality_frontier,
        list(quality_frontier[0].keys()),
    )
    write_csv(
        output_dir / "allhead_quality_20step.csv",
        allhead_quality,
        list(allhead_quality[0].keys()),
    )
    write_csv(
        output_dir / "kernel_scaling.csv",
        kernel_scaling,
        list(kernel_scaling[0].keys()),
    )
    write_csv(
        output_dir / "directed_relations.csv",
        directional,
        list(directional[0].keys()),
    )
    write_csv(
        output_dir / "timestep_profile.csv",
        step_rows,
        list(step_rows[0].keys()),
    )
    write_csv(
        output_dir / "finding_ledger.csv",
        findings,
        list(findings[0].keys()),
    )


if __name__ == "__main__":
    main()

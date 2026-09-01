#!/usr/bin/env python3
"""Build the canonical portable-report artifact from the audited snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TITLE = "ThunderWorld 当前实验信息与结论（2026-09）"


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source(source_id: str, path: str, description: str, executed_at: str) -> dict[str, Any]:
    if path.endswith(".csv"):
        code = f"SELECT * FROM read_csv_auto('{path}', header = true)"
    else:
        code = f"SELECT * FROM read_json_auto('{path}', format = 'auto')"
    return {
        "id": source_id,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": code,
            "description": description,
            "tables_used": [path],
            "executed_at": executed_at,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/artifact.json"))
    args = parser.parse_args()

    evidence = load(args.data_dir / "evidence_snapshot.json")
    validation = load(args.data_dir / "validation_report.json")
    generated_at = evidence["compiled_at"]

    directional = []
    for row in evidence["characterization"]["directional_relations"]:
        directional.append(
            {
                **row,
                "display_relation": row["relation_pair"].replace("↤", "←"),
            }
        )

    taxonomy = [
        {
            "archetype": name,
            "share": row["share"],
            "head_layer_query_units": row["count"],
        }
        for name, row in evidence["characterization"]["taxonomy"].items()
    ]

    anchors = []
    anchor_names = {
        "droid_forward": "DROID action←action",
        "robot_forward": "Bridge action←action",
        "t2vs": "T2VS sound←sound",
    }
    for row in evidence["characterization"]["anchors"]:
        anchors.append({**row, "display_relation": anchor_names[row["task"]]})

    frontier = evidence["runtime"]["quality_frontier"]
    allhead = evidence["runtime"]["allhead_quality_20step"]
    kernel = [
        {**row, "ar_context": f"AR {row['ar_tokens']}"}
        for row in evidence["runtime"]["kernel_scaling"]
    ]
    timestep_rows = []
    # Reuse the compact curated CSV values embedded in evidence by reconstructing
    # the 20-point series from the checked public CSV.
    import csv

    with (args.data_dir / "timestep_profile.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            timestep_rows.append(
                {
                    "step": int(row["step"]),
                    "stage": int(row["stage"]),
                    "safe_share_0p05": float(row["safe_share_0p05"]),
                    "mean_robust_output_l2": float(row["mean_robust_output_l2"]),
                    "p95_robust_output_l2": float(row["p95_robust_output_l2"]),
                }
            )

    overview = [
        {
            "min_directional_ratio": min(row["asymmetry_ratio"] for row in directional),
            "current_qk_reduction": frontier[0]["joint_qk_reduction"],
            "current_min_ssim": frontier[0]["ssim_min"],
            "aggressive_qk_reduction": frontier[1]["joint_qk_reduction"],
            "aggressive_min_ssim": frontier[1]["ssim_min"],
            "all_sparse_16k_attention_speedup": frontier[2]["attention_speedup"],
            "all_sparse_best_20step_ssim": frontier[2]["ssim_min"],
        }
    ]

    decisions = evidence["findings"]
    protocols = evidence["protocols"]

    source_specs = [
        source(
            "snapshot",
            "data/evidence_snapshot.json",
            "Audited aggregate evidence snapshot produced from hashed experiment artifacts.",
            generated_at,
        ),
        source(
            "validation",
            "data/validation_report.json",
            "Independent arithmetic, comparability, denominator, and completeness checks.",
            validation["generated_at"],
        ),
        source(
            "quality_frontier",
            "data/quality_compute_frontier.csv",
            "Comparable 20-step quality/compute frontier and H200 attention microbench link.",
            generated_at,
        ),
        source(
            "directional",
            "data/directed_relations.csv",
            "Four paired directed modality relations measured at head-layer grain.",
            generated_at,
        ),
        source(
            "kernel",
            "data/kernel_scaling.csv",
            "H200 NVL random-tensor attention microbenchmark for the all-sparse kernel bucket.",
            generated_at,
        ),
        source(
            "timestep",
            "data/timestep_profile.csv",
            "Twenty-step T2V local sparsifiability profile over 24 prompts and three seeds.",
            generated_at,
        ),
        source(
            "findings",
            "data/finding_ledger.csv",
            "Frozen finding-to-implementation-to-contribution decision ledger.",
            generated_at,
        ),
        source(
            "allhead",
            "data/allhead_quality_20step.csv",
            "Comparable seed-29, 20-step all-head sparse quality checks.",
            generated_at,
        ),
    ]

    manifest_sources = [
        {
            **item,
            "label": item["query"]["description"],
            "path": item["query"]["tables_used"][0],
        }
        for item in source_specs
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": (
                "Audited synthesis of ThunderWorld relation-local sparsity findings, trajectory-global "
                "failure modes, quality/compute frontier, and sparse-kernel scaling."
            ),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "direction_card",
                    "description": "四类有向模态关系中最弱的不对称倍数。",
                    "dataset": "overview",
                    "sourceId": "snapshot",
                    "metrics": [
                        {"label": "最低方向不对称", "field": "min_directional_ratio", "format": "number"}
                    ],
                },
                {
                    "id": "current_card",
                    "description": "当前 0.90 development 档；一条 T2V prompt、三个 seeds。",
                    "dataset": "overview",
                    "sourceId": "quality_frontier",
                    "metrics": [
                        {"label": "当前 joint-QK reduction", "field": "current_qk_reduction", "format": "percent"},
                        {"label": "最差 SSIM", "field": "current_min_ssim", "format": "number"},
                    ],
                },
                {
                    "id": "aggressive_card",
                    "description": "0.80 aggressive 档；一条 T2V prompt、两个 seeds，仍属 provisional。",
                    "dataset": "overview",
                    "sourceId": "quality_frontier",
                    "metrics": [
                        {"label": "Aggressive joint-QK reduction", "field": "aggressive_qk_reduction", "format": "percent"},
                        {"label": "最差 SSIM", "field": "aggressive_min_ssim", "format": "number"},
                    ],
                },
                {
                    "id": "kernel_card",
                    "description": "16K DM、AR=61 的 all-sparse attention microbenchmark；不是端到端结果。",
                    "dataset": "overview",
                    "sourceId": "kernel",
                    "metrics": [
                        {"label": "All-sparse attention speedup", "field": "all_sparse_16k_attention_speedup", "format": "number"},
                        {"label": "20-step 最佳 all-head SSIM", "field": "all_sparse_best_20step_ssim", "format": "number"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "direction_chart",
                    "title": "有向模态关系注意力质量不对称",
                    "subtitle": "四类关系的 head-layer bootstrap 95% 下界均高于 3.5×。",
                    "type": "bar",
                    "dataset": "directional",
                    "sourceId": "directional",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "display_relation", "type": "nominal", "label": "有向关系对"},
                        "y": {"field": "asymmetry_ratio", "type": "quantitative", "label": "前向 / 反向 attention mass"},
                        "tooltip": [
                            {"field": "ci95_low", "type": "quantitative", "label": "95% CI lower"},
                            {"field": "ci95_high", "type": "quantitative", "label": "95% CI upper"},
                            {"field": "direction_consistency", "type": "quantitative", "label": "方向一致率", "format": "percent"},
                        ],
                    },
                },
                {
                    "id": "taxonomy_chart",
                    "title": "Generator head-relation archetype 构成",
                    "subtitle": "共 2,464 个 layer × head × query-role 单元；只有 16.6% 属于 mixer。",
                    "type": "bar",
                    "dataset": "taxonomy",
                    "sourceId": "snapshot",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "archetype", "type": "nominal", "label": "Archetype"},
                        "y": {"field": "share", "type": "quantitative", "label": "单元占比", "format": "percent"},
                        "tooltip": [
                            {"field": "head_layer_query_units", "type": "quantitative", "label": "单元数"}
                        ],
                    },
                },
                {
                    "id": "timestep_chart",
                    "title": "严格局部可稀疏 head share 随 denoising step 上升",
                    "subtitle": "p95 output-L2≤0.05 的 share 从 3.82% 增至 12.93%，但不代表可复用 attention output。",
                    "type": "line",
                    "dataset": "timestep",
                    "sourceId": "timestep",
                    "encodings": {
                        "x": {"field": "step", "type": "quantitative", "label": "Diffusion step"},
                        "y": {"field": "safe_share_0p05", "type": "quantitative", "label": "严格可稀疏 share", "format": "percent"},
                    },
                },
                {
                    "id": "frontier_chart",
                    "title": "当前 20-step 质量—QK Pareto",
                    "subtitle": "50% sparse groups 通过 0.90 development screen；75% 仅通过 provisional 0.80；100% all-head 未过 0.80。",
                    "type": "scatter",
                    "dataset": "frontier",
                    "sourceId": "quality_frontier",
                    "encodings": {
                        "x": {"field": "joint_qk_reduction", "type": "quantitative", "label": "Generator joint-QK reduction", "format": "percent"},
                        "y": {"field": "ssim_min", "type": "quantitative", "label": "最差 paired SSIM"},
                        "color": {"field": "evidence_status", "type": "nominal", "label": "证据状态"},
                        "tooltip": [
                            {"field": "policy", "type": "nominal", "label": "策略"},
                            {"field": "attention_speedup", "type": "quantitative", "label": "16K attention speedup"},
                            {"field": "estimated_e2e_speedup", "type": "quantitative", "label": "Amdahl E2E estimate"},
                            {"field": "seeds", "type": "quantitative", "label": "Seeds"},
                        ],
                    },
                },
                {
                    "id": "kernel_chart",
                    "title": "All-sparse kernel 的收益随 DM 长度增长",
                    "subtitle": "H200 NVL random-tensor attention microbenchmark；4K 几乎不赚，16K 接近 2×。",
                    "type": "line",
                    "dataset": "kernel",
                    "sourceId": "kernel",
                    "encodings": {
                        "x": {"field": "dm_tokens", "type": "quantitative", "label": "DM tokens"},
                        "y": {"field": "attention_speedup", "type": "quantitative", "label": "Attention speedup"},
                        "color": {"field": "ar_context", "type": "nominal", "label": "AR context"},
                        "tooltip": [
                            {"field": "joint_qk_reduction", "type": "quantitative", "label": "Joint-QK reduction", "format": "percent"},
                            {"field": "output_relative_l2_vs_joint_repair", "type": "quantitative", "label": "Kernel check rel-L2"},
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "protocol_table",
                    "title": "实验族与证据粒度",
                    "subtitle": "不同实验族不可直接池化；特别是 4-step、20-step 与 random-tensor microbench。",
                    "dataset": "protocols",
                    "sourceId": "snapshot",
                    "columns": [
                        {"field": "name", "label": "实验族", "type": "text"},
                        {"field": "tasks", "label": "任务数", "format": "number"},
                        {"field": "prompts_per_task", "label": "Prompts/task", "format": "number"},
                        {"field": "seeds", "label": "Seeds", "format": "number"},
                        {"field": "steps", "label": "Steps", "type": "text"},
                        {"field": "purpose", "label": "用途", "type": "text"},
                    ],
                },
                {
                    "id": "anchor_table",
                    "title": "高信息密度 dense anchors",
                    "subtitle": "小关系计算占比低，但 attention mass / score share 高两个数量级。",
                    "dataset": "anchors",
                    "sourceId": "snapshot",
                    "columns": [
                        {"field": "display_relation", "label": "关系", "type": "text"},
                        {"field": "attention_mass", "label": "Attention mass", "format": "percent"},
                        {"field": "score_fraction", "label": "Score share", "format": "percent"},
                        {"field": "mass_per_score_share", "label": "Mass / score share", "format": "number"},
                    ],
                },
                {
                    "id": "frontier_table",
                    "title": "质量—计算精确值",
                    "subtitle": "E2E 为 Amdahl 估计：Generator share=0.9、attention share=0.387；非实测完整 rollout latency。",
                    "dataset": "frontier",
                    "sourceId": "quality_frontier",
                    "columns": [
                        {"field": "policy", "label": "策略", "type": "text"},
                        {"field": "joint_qk_reduction", "label": "Joint-QK cut", "format": "percent"},
                        {"field": "ssim_min", "label": "Min SSIM", "format": "number"},
                        {"field": "psnr_min_db", "label": "Min PSNR (dB)", "format": "number"},
                        {"field": "attention_speedup", "label": "Attention speedup", "format": "number"},
                        {"field": "estimated_e2e_speedup", "label": "Estimated E2E", "format": "number"},
                        {"field": "seeds", "label": "Seeds", "format": "number"},
                    ],
                },
                {
                    "id": "allhead_table",
                    "title": "可比 20-step all-head sparse 失败边界",
                    "subtitle": "所有记录均为同一 seed=29 的 20-step T2V；最高 SSIM 仍低于 0.8。",
                    "dataset": "allhead",
                    "sourceId": "allhead",
                    "columns": [
                        {"field": "policy", "label": "策略", "type": "text"},
                        {"field": "keep_ratio", "label": "Keep ratio", "format": "percent"},
                        {"field": "global_ssim", "label": "Global SSIM", "format": "number"},
                        {"field": "psnr_db", "label": "PSNR (dB)", "format": "number"},
                        {"field": "steps", "label": "Steps", "format": "number"},
                    ],
                },
                {
                    "id": "decision_table",
                    "title": "Finding → implementation → contribution",
                    "subtitle": "KEEP、REWORK 与 DROP 决策均绑定证据等级和失败条件。",
                    "dataset": "decisions",
                    "sourceId": "findings",
                    "columns": [
                        {"field": "id", "label": "ID", "type": "text"},
                        {"field": "finding", "label": "Finding", "type": "text"},
                        {"field": "implementation", "label": "Implementation", "type": "text"},
                        {"field": "decision", "label": "Decision", "type": "text"},
                        {"field": "evidence_level", "label": "Evidence", "type": "text"},
                        {"field": "caveat", "label": "Caveat", "type": "text"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## Technical summary\n\n"
                        "**主线可以固定为：世界模型中的稀疏性是 relation-local 的，但近似误差是 trajectory-global 的。** "
                        "当前最可信的方法形态是 `GQA group × directed relation × layer/stage` 编译：保留 action/audio/control 等 dense anchors，"
                        "在 vision residual 上执行 semantic/local block list，并通过同一个 online-softmax 状态保持 AR/DM joint normalization。\n\n"
                        "最新 0.90 development 点在一条 T2V prompt 的 3 seeds 上实现 36.68% joint-QK reduction，最差 SSIM 0.923；"
                        "0.80 aggressive 点将 reduction 提高到 55.02%，但只有 2 seeds。全程 all-head sparse 的可比 20-step 最佳 SSIM 仅 0.770，"
                        "所以 all-sparse kernel 只能作为安全 layer/stage bucket，而不能覆盖完整 trajectory。"
                    ),
                },
                {"id": "metric_strip", "type": "metric-strip", "cardIds": ["direction_card", "current_card", "aggressive_card", "kernel_card"]},
                {
                    "id": "evidence_boundary",
                    "type": "markdown",
                    "sourceId": "validation",
                    "body": (
                        "## 证据可共享，但仍需保留范围限定\n\n"
                        f"一致性审计通过 {validation['checks_passed']}/{validation['checks_total']} 项检查，未发现计算或可比性 blocker。"
                        "不过 strongest quality point 仍只有一条独立 prompt，kernel 数据来自 random tensors，最终 VBench/control/action/task-success 尚未运行。"
                        "因此整体评级是 **share with caveats**，不是 publication-ready acceleration claim。"
                    ),
                },
                {"id": "protocols", "type": "table", "tableId": "protocol_table"},
                {
                    "id": "relation_finding",
                    "type": "markdown",
                    "sourceId": "directional",
                    "body": (
                        "## 方向不是 metadata，而是可执行预算维度\n\n"
                        "sound↔vision、action↔vision 和 clean↔noisy 的正反方向相差 4.09–13.53×。"
                        "这要求 runtime 保留 `(query role, key role)`，低质量方向可降预算或 skip，高质量反向及 hard anchors 维持 dense。"
                        "后续 20-step equal-QK counterfactual 也支持方向选择：I2V 删除低质量方向 SSIM 0.975，删除反向高质量方向只有 0.776。"
                    ),
                },
                {"id": "direction_visual", "type": "chart", "chartId": "direction_chart"},
                {
                    "id": "head_finding",
                    "type": "markdown",
                    "sourceId": "snapshot",
                    "body": (
                        "## Head-wise 分类有稳定信号，但执行单位应回到 GQA group\n\n"
                        "Generator 单元中 37.7% 是 specialist、45.7% 是 hybrid、16.6% 是 mixer；主导关系跨 denoising passes 稳定率为 94.5%。"
                        "这支持 MInference-like 离线 archetype，但 production policy 应先逐 Q-head 测量，再编译为 GQA-aligned buckets；"
                        "只有成本模型证明值得时才在一个 KV group 内拆成两个 Q-head buckets。"
                    ),
                },
                {"id": "taxonomy_visual", "type": "chart", "chartId": "taxonomy_chart"},
                {
                    "id": "anchor_finding",
                    "type": "markdown",
                    "sourceId": "snapshot",
                    "body": (
                        "## 小模态关系应作为 dense anchors，而不是优先稀疏对象\n\n"
                        "action self 与 sound self 只占 0.12%–0.37% scores，却承载 33.5%–48.9% attention mass，"
                        "信息密度达到 134–302×。它们计算便宜、任务风险高，应从 vision residual 的稀疏预算中剥离并强制保留。"
                    ),
                },
                {"id": "anchor_detail", "type": "table", "tableId": "anchor_table"},
                {
                    "id": "trajectory_finding",
                    "type": "markdown",
                    "sourceId": "snapshot",
                    "body": (
                        "## Layer/head 局部安全不能替代 trajectory 校准\n\n"
                        "Token oracle 可删 35.4%–43.3%，natural block oracle 只能删 11.6%–19.4%，证明 semantic blockification 的必要性；"
                        "但 naive K-only reorder 与手写深度调度都失败。GQA 组合误差的 median cancellation ratio 为 0.412，"
                        "而 equal-QK depth relocation 将 SSIM 从 0.906 降至 0.861。因而 compiler 必须使用 joint-projected error Gram、"
                        "完整 rollout 和 UCB，而不能按局部 output drift 或层先验直接加和。"
                    ),
                },
                {
                    "id": "timestep_finding",
                    "type": "markdown",
                    "sourceId": "timestep",
                    "body": (
                        "## Diffusion 后期更可稀疏，但应复用 layout 而非 output\n\n"
                        "严格局部可稀疏 share 从 step 0 的 3.82% 增长到 step 19 的 12.93%，相邻 safe-set Jaccard 均值约 0.93。"
                        "这支持 stage-wise budget 与 descriptor/block-list reuse。相反，previous-output cache 的 paired SSIM 只有约 0.77–0.80，"
                        "因此保留 QK/softmax/PV 重算，默认禁用 attention-output reuse。"
                    ),
                },
                {"id": "timestep_visual", "type": "chart", "chartId": "timestep_chart"},
                {
                    "id": "negative_result",
                    "type": "markdown",
                    "sourceId": "snapshot",
                    "body": (
                        "## Cross→self 热块迁移是明确负结果\n\n"
                        "去除位置偏置后的 DM←AR→DM←DM residual correlation 为 -0.065；"
                        "held-out transferred anchors 与 query budgets 都使 omitted mass 变差。两分支应独立选块，"
                        "只在 trajectory compiler 中建模组合误差。"
                    ),
                },
                {
                    "id": "frontier_finding",
                    "type": "markdown",
                    "sourceId": "quality_frontier",
                    "body": (
                        "## 当前可工程化 Pareto 是 50%/75% sparse groups，不是 blanket all-sparse\n\n"
                        "0.90 档的 measured 16K attention speedup 为 1.21×；0.80 aggressive 档为 1.41×。"
                        "按 Generator share=0.9、attention share=0.387 的 Amdahl 模型，分别只对应约 1.065× 与 1.112× 端到端。"
                        "全 sparse bucket 在 16K 达到 1.93× attention，但完整 20-step all-head 质量仍失败。"
                    ),
                },
                {"id": "frontier_visual", "type": "chart", "chartId": "frontier_chart"},
                {"id": "frontier_detail", "type": "table", "tableId": "frontier_table"},
                {
                    "id": "allhead_failure",
                    "type": "markdown",
                    "sourceId": "allhead",
                    "body": (
                        "## 0.827 的 apparent pass 被排除：它只有 4 steps\n\n"
                        "严格核对 `sample_args.json` 后，该 q16 exact-Taylor 结果只运行 4 个 diffusion steps。"
                        "在可比的 20-step all-head 记录中，最佳 SSIM 是 0.770，所有策略均未达到 0.8。"
                    ),
                },
                {"id": "allhead_detail", "type": "table", "tableId": "allhead_table"},
                {
                    "id": "kernel_finding",
                    "type": "markdown",
                    "sourceId": "kernel",
                    "body": (
                        "## Sparse kernel 的收益会随输入长度 scaling，但会被 Amdahl 上限截断\n\n"
                        "AR=61 时，all-sparse attention speedup 从 DM 4K 的 1.03×、8K 的 1.32× 增长到 16K 的 1.93×。"
                        "因此 production runtime 应依赖实测 lookup table：短序列或高密度策略回退 FA3，长视频和同质 all-sparse buckets 才走 fused sparse path。"
                    ),
                },
                {"id": "kernel_visual", "type": "chart", "chartId": "kernel_chart"},
                {
                    "id": "decision_heading",
                    "type": "markdown",
                    "body": (
                        "## 固化后的 finding → implementation → contribution\n\n"
                        "最终只保留三项论文贡献：relation/head/modality characterization；interaction-aware trajectory compiler；"
                        "保持 joint-softmax 语义的 sparse runtime。其余发现要么并入这三项，要么作为负结果。"
                    ),
                },
                {"id": "decision_detail", "type": "table", "tableId": "decision_table"},
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "snapshot",
                    "body": (
                        "## 指标、基线与实验解释\n\n"
                        "`joint-QK reduction` 是相对 DM←DM + DM←AR score count 的削减，不包含 FFN、QKV/O projection、VAE、routing 或通信；"
                        "`attention speedup` 是 H200 dense three-way latency / sparse amortized attention latency；"
                        "`estimated E2E` 是显式 Amdahl 估算。paired SSIM/PSNR 比较同 seed dense reference，"
                        "用于 trajectory fidelity screen，不能代替 VBench、control adherence、action error 或 task success。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "validation",
                    "body": (
                        "## 限制与稳健性\n\n"
                        "- 关系 atlas 的 bootstrap 单位是 head-layer，不是独立 prompt。\n"
                        "- 0.90 点只有一条 prompt × 3 seeds；0.80 点只有一条 prompt × 2 seeds。\n"
                        "- 16K speedup 来自 random-tensor attention microbench，不是完整 Cosmos3 rollout。\n"
                        "- 目前没有 final-test VBench、PAI-C judge、AV-sync 或 DROID task-success 结果。\n"
                        "- Context-parallel 动态稀疏尚未实现，多卡路径必须回退 dense。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 推荐的最小下一步\n\n"
                        "1. 为 q12 75% GQA policy 补 seed 17，完成 3-seed aggressive screen。\n"
                        "2. 只测试一个 `前 15 steps=50/75%，最后 5 steps=all-sparse` schedule，验证 selective full-sparse bucket。\n"
                        "3. 在一个 I2V 与一个 DROID case 上执行相同 20-step policy，检查 directed anchors 与 control/action failure。\n"
                        "4. 把 semantic block list 与 directed relation skip 融合进同一 joint-online-softmax kernel；分别报告 routing、packing、kernel、merge。\n"
                        "5. 若多任务 0.90 档无法同时达到 ≥30% QK cut 与任务非劣，停止扩大 kernel，转为 characterization/negative-result 论文。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        "- 方向性不对称能否在每任务 24 个独立 prompts/episodes 上维持？\n"
                        "- 25%/37.5%/50% equal-active-tile 下，query-conditioned semantic block list 能否缩小 oracle gap？\n"
                        "- normalized sigma 的 stage policy 能否从 20 steps 迁移到 40 steps、从 Nano 迁移到 Edge？\n"
                        "- 在真实 14,720/74,000 DM token 配置中，attention share 与端到端 speedup 是否符合 Amdahl 假设？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "overview": overview,
                "directional": directional,
                "taxonomy": taxonomy,
                "anchors": anchors,
                "timestep": timestep_rows,
                "frontier": frontier,
                "allhead": allhead,
                "kernel": kernel,
                "decisions": decisions,
                "protocols": protocols,
            },
        },
        "sources": source_specs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

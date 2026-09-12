# 共享 GPU 下的计算约束校准

这是无训练、无梯度的 Edge I2V 工作流。所有 `shared-calibration` 结果禁止进入正式延迟排行榜。MMA 预算不是延迟预算。

## 可执行入口

使用 Cosmos 环境并设置包路径：

```bash
export PYTHONPATH=packages/fp4-repair/src:/tmp/thunderworld-joint-deps
PY=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
EDGE=/home/yufeng/thunderworld/ThunderWorld/artifacts/huggingface/models/Cosmos3-Edge-a9d944e
EXP=/home/yufeng/thunderworld/compute_quality_edge_v1
COHORT=tools/joint_inference/cohorts/edge_i2v_compute_v1.json
```

`resources.shared_worker` 使用跨 GPU0/3 的进程锁，每次一个 GPU 实验 worker；未指定设备时选择空闲显存最多的一张。启动前要求至少 12 GiB 空闲；失败保留进度，无精度或输入回退。

```bash
$PY tools/joint_inference/capture_budget.py --model "$EDGE" \
  --case tools/joint_inference/cases/robot_observe.json --cohort "$COHORT" \
  --output "$EXP/capture"
$PY tools/joint_inference/calibrate_budget.py --capture "$EXP/capture" \
  --output "$EXP/base_calibration"
$PY tools/joint_inference/search_budget.py --capture "$EXP/capture" \
  --assets "$EXP/base_calibration" --output "$EXP/repair_search"
$PY tools/joint_inference/export_budget.py "$EXP/repair_search/search.json" \
  --output "$EXP/candidates"
```

采集、基础校准、修复搜索、资产细化和请求生成支持 `--resume`，只跳过已保存的完整处理单元。导出目录使用新目录，保留历史文件。

每个候选的 `plan.json` 使用 `branch|phase|module` 与 `branch|phase|layer` 键。A 配额是每个 M128 块的 K128 整数数目；W 支持是展平的 N128/K128 索引。QK/PV 配额分别为每个 query head、每个 M128 块的 K-token128 块数。所有 attention base blocks 保留。

```bash
$PY tools/joint_inference/refine_budget_assets.py --capture "$EXP/capture" \
  --candidate "$EXP/candidates/b1.25_grow_mixed" \
  --output "$EXP/refined_b1.25"
```

细化时只在所有阶段和 CFG 已支付 W1 的共同支持内尝试 outlier 拆分。每 K16 组最多移出一个贡献 outlier；离线编码 W1 使用原始缩放权重减去实际解码 W0。资产保存 `mandatory_w_blocks`，运行时拒绝关闭这些块。一个请求只使用一份固定 W0。

## 运行、预算与调试

```bash
$PY -m thunderworld_fp4.cosmos.cli infer --model "$EDGE" \
  --config "$EXP/candidates/b1.25_grow_mixed/config.json" \
  --assets "$EXP/refined_b1.25" --resource-mode shared-calibration \
  --compute-budget 1.25 --budget-plan "$EXP/candidates/b1.25_grow_mixed/plan.json" \
  --case tools/joint_inference/cases/robot_observe.json \
  --cohort "$COHORT" --cohort-role trajectory --seeds 7103 7104 \
  --output "$EXP/development_b1.25"
```

准备时以实际输入遍历所有策略，收集形状和合法固定 launch；在捕获任何 Transformer Graph 前编译完整 35 步 MMA 清单并检查上限。未准备形状、缺失策略、跨任务误用和超预算均报错。BF16 selector MMA 分账并计入总预算。动态 A/W 交叉项使用每 K 块 W 支持频数之和的 top-A 上界。M 尾块按实际 BM 发射量计费。

A 选择读取当前量化残差，结合离线权重通道敏感度；base packing 只写出残差摘要，后续只编码被选中的残差块。K/V 残差按当前 QK/PV masks 的 GQA 并集打包。残差地址空间仍预留完整 plane，不能把写流量减少称为 scratch 分配等比例减少。共享 W1 bank 在不同阶段使用索引映射，未重复常驻保存整个阶段 bank。

`debug off` 不增加诊断事件和逐步回读。`timing/trace` 保留原接口。shared 模式禁止 `benchmark` 与 `--tune`；本轮使用固定安全 launch，不将旧设备的调优结论移植过来。

## 队列与种子隔离

`run_shared.py --jobs jobs.json --role trajectory|validation|final` 串行独立加载配置，写入原始 NPZ、每请求 JSON、完整预算清单与输入图片哈希。作业格式：

```json
{"jobs":[{"name":"candidate_b1.25","config":"/absolute/config.json","assets":"/absolute/assets","budget":1.25,"candidate":true}]}
```

校准 7101–7102；开发轨迹 7103–7104；验证 92101–92108；最终 101001–101016。不得用最终测试选参。学生轨迹采集使用 `capture_budget.py --student-config ... --student-assets ... --reference-capture ...`，存下当前输入，并以原始 BF16 权重求同输入参考；允许最多一轮追加校准。

`evaluate.py --cohort ...` 保留历史 91001–91008 默认规则，新 cohort 需显式提供。`select_budget.py` 只接受完整验证结果，先检查相对该档最佳未校准对照的 SSIM/L2，再按 LPIPS 选取。最终作业绑定验证结果、配置和资产哈希。两枚开发种子不满足置信区间验收。

目前的全局分配使用测得的局部 A/W、QK/PV **组合**误差作为轨迹候选提案。跨层非线性误差不能由局部误差相加证明，完整生成与独立验证仍是必要步骤。只可称为已搜索配置内的计算量约束前沿。

## 完整复测与对照

未经校准资产可用 `prepare --config tools/joint_inference/configs/dual.json --resource-mode shared-calibration` 离线生成，随后由 `controls_budget.py --capture ... --assets ... --output ...` 按四档预算导出 A 优先、W 优先及均衡投影对照。加 `--joint-attention` 额外导出均匀联合 ACT2/dual 对照；余数按 head 索引分配，不使用校准 head finding。当前 attention 接口无 W-only 补项，因此 W 优先对照保留 base attention。所有对照均保留实际整数配额对应的预算使用率，不把未用预算算成已经执行。

开发集选定每档最多三个候选后，将其与所有对照、BF16、原生 FP8、普通 FP4、全量 dual 写入 `jobs.json`，运行：

```bash
$PY tools/joint_inference/run_compute_cohort.py --model "$EDGE" \
  --case tools/joint_inference/cases/robot_table.json --cohort "$COHORT" \
  --jobs "$EXP/validation_complete_jobs.json" --output "$EXP/retest" \
  --lpips-cache /tmp/thunderworld-lpips-cache --metric-workers 4
```

该入口依次执行验证、CPU 指标评估、冻结选参、最终测试及报告；失败立即停止，可用 `--resume` 恢复。不会自动改变输入、精度或已冻结配置。CPU 指标线程不增加 GPU 实验进程，4 线程结果已与串行结果逐项一致性校验。

`final_compute_report.py` 分别报告同预算对照门槛与 FP8 门槛、配对置信区间、最差种子退化，以及 LPIPS/MMA/常驻量化权重三维非支配集合。只有同预算质量门槛通过后才能扩展下一任务；任何扩展都必须创建独立 cohort、校准资产与预算清单。

内存报告区分量化投影常驻权重、完整模型常驻状态、支持索引、warmup 后非模型活跃分配及请求 allocated/reserved 峰值。非模型活跃分配包含 activation、scratch 和 Graph 输入输出，不能当作独立 scratch 峰值。

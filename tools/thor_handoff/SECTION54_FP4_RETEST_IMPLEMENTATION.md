# Thor §5.4 七方法补测实现

本轮只做速度、内存及原生算子正确性。冻结 CUTLASS mainloop v6；不移植 v8、调优 baseline、运行 BF16/FP8 速度或视频质量/judge。

## 入口

`standard_comparison.py` 已扩展为按模型冻结七方法、显式资产路径、真实 adapter 资格、eager/Graph 分别检查，以及两个场景的独立 profiling。Edge/Nano 的 runtime_options 原样保留。

`section54_campaign.py` 依次执行：等待 Nano 资产到齐 → CPU 哈希/模型/投影验证与配置冻结 → 原生算子资格 → 新 ARC 实例 memcheck/racecheck → 两模型统一 420 次计时 → Edge 10 次独立 profile。数值资格失败保留 unavailable；资源不足、OOM 或出现其他计算任务保留该次尝试并暂停等待。不会结束其他用户进程。

```bash
cd /home/chenhan/thunderworld/src/ThunderWorld
source tools/thor_handoff/env.sh
python -u tools/thor_handoff/section54_campaign.py \
  --output /home/chenhan/thunderworld/comparisons/section54_fp4_retest_20260923
```

同一 campaign 有 driver 锁；推理/资格进程继续使用 gpu-test.lock。复用时配置与 worker 哈希必须完全一致。失败尝试进入 retained_attempts，不覆盖历史数据。

## 代码与检查

- `section54_protocol.py`：只读 safetensors 头获得真实目标集合/维度；检查资产模型身份和全部 payload 哈希；新帧 FPS、分位数和 CSV。
- `baseline_adapter.py`：按模型传入 ARC/AWQ 资产；检查目标名称、维度、实际类和计算 dtype，不把 BF16 装载配置当作实际投影 dtype。
- `extend_arc_shapes.py`：新目录中仅增加 K12288 的 reorder 调用与模板实例；不改量化器/旧库。新库为 external/arcquant-sm110-section54。
- `probe_baselines.py`：从真实模型维度覆盖 M129/M416/M960、零输入和当前输入；ARC 两个覆盖率、SharQ 原 BF16 中间舍入、AWQ eager/Graph 对齐独立解码参考。新增 ARC 库做 memcheck/racecheck。
- `section54_resources.py`：外部进程/tegrastats 采样。若整次 run 内观察到其他计算进程，排除整次计时并保留日志。采样不等于连续 GPU 活动证明。
- `section54_profiles.py`：每 kernel 只归一类，融合 kernel 不拆分；不把累计 kernel 时间当请求 latency，也不从 CSV 推断 CPU 阻塞或时间线间隙。
- `test_section54_retest.py`：7 项 CPU 协议测试，覆盖真实 eager adapter、拒绝 BF16 冒充、维度/探针拒绝、profile 输入选择、FPS/P90 与融合分类。

冻结后的代码不可直接修改再续跑；应另开修订 cohort。结果不会自动获得 performance_eligible=true。

## 数据交付

每模型目录包含 manifest/fixed-model、assets-verified、qualification、runs、完整日志及主机遥测，和 requests.csv、summary.json/csv、round_summary.csv、report.md。原始 result 保留计算账本、预热后 inventory、实际配置/后端与源代码/库 hash。Graph/scratch 与权重分开，Marlin workspace 不作为常驻权重。

profile 在两个模型全部计时完成后启动；每个方法×场景有独立文件名。大 .nsys-rep/.sqlite 可留 Thor，仅回传路径/hash、CSV、JSON、日志。

顶层 campaign_status.json 记录队列状态；本机工作目录为 /home/yufeng/thunderworld/thor_section54_fp4_retest_20260923。

# §5.4：Thor FP4 速度补测交接

2026-09-23。目标是补齐论文 Table 3 的 Thor 两模型速度和 Figure 5 的代表性执行分解。速度只比较 FP4 方案，不增加 BF16／FP8 计时，不重新调优 baseline，不跑全量视频质量／judge。本说明已只读核对 Thor；本轮没有启动 GPU 测试。

机器可读需求见同目录 section54_fp4_retest_spec.json。它是 coding agent 的实现规格，不是现有 runner 可以直接读取的 manifest。

## 1. 需要补齐什么

| 方法 | 报告的 method_id | Edge 当前批次 | Nano 当前批次 |
|---|---|---|---|
| 普通 W4A4＋FP4 attention | fp4_base | v6 已有五轮 | v6 只有一轮 |
| NVFP4-AWQ，Marlin W4A16 | awq_marlin | 只有旧版批次 | 缺接入／计时 |
| ARCQuant 50% | arcquant | 只有旧版批次 | 缺接入／计时 |
| ARCQuant 100% | arcquant100 | 只有旧版批次 | 缺接入／计时 |
| SharQ | sharq_perf | 只有旧版批次 | 缺接入／计时 |
| ThunderWorld 1.5× | b1.5_protected_n | v6 已有五轮 | v6 只有一轮 |
| ThunderWorld 2× | b2_joint | v6 已有五轮 | v6 只有一轮 |

优先一次冻结这七种方法，在 Edge、Nano 各跑五轮。每轮两个固定场景、每场景三个种子，每个模型—方法 30 请求，总计 420 个无探针计时请求、70 次独立加载／预热。不把旧版四个外部基线直接接到最新 v6 结果后面算加速。若某路径不能通过资格检查，保留失败原因和空项，其余方法继续完成。

Figure 5 最小补测：仅 Edge，ARC50、ARC100、SharQ、1.5×、2×，两个场景各一个种子，共 10 次独立 profiling 请求。先结束所有无探针计时，再采 profile。Nano 全方法分解、AWQ 分解、2.5×、大分辨率、batch／并发不是本轮必需项。

Thor 与 Figure 1 对齐的是方法定义、模型 revision 和生成设置。Thor 现有两个场景不同于 PRO 的六场景面板，不能据此计算同输入的跨设备速度比；图注继续分别说明面板。

## 2. 已有环境与冻结输入

- SSH：chenhan@thor。
- 工作区：/home/chenhan/thunderworld。
- 代码：/home/chenhan/thunderworld/src/ThunderWorld。
- 环境：source tools/thor_handoff/env.sh；沿用现有 ARM Python／Torch／Triton／CUDA 环境。
- 当前修复后端：/home/chenhan/thunderworld/variants/cutlass-mainloop-v6。
- variant.json SHA256：319232a793a0955a3afcd5eb669b50a249686d052489c4ebf241c7e11a08bd5f。
- 设备为 SM110；沿用真实 sm_110a 构建，不复用桌面 SM120 二进制。
- Edge revision：a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba。
- Nano revision：411f42a8fdfb8c5b2583cb8786e0938f49796eaa。
- 两模型和现有修复资产都已在 Thor。读取 data/{edge,nano}/package.json、study.json，models/ 中的 thunderworld_verified.json，以及 assets/{edge,nano}/restored.json 确認身份与文件。
- 本机冻结部署资产的生成投影数为 Edge 168、Nano 252；每个 baseline 对齐目标名称集合，不只检查数量。

沿用下面两个 scene，prompt、negative prompt、图像和 first-frame 处理全部从各模型的 study.json 读取：

| case_index | scene_id | 图像 SHA256 |
|---|---|---|
| 0 | paibench__av_660aa819-4a75-45d0-b79a-e6c2f2c321dc | a7c1d54b6146e00e2905599327af3f861a27bee409157c28a381a4cb3e114a79 |
| 1 | rbench__common_manipulation_0041 | 9145f8c54b0289b31aada8a6072e3f58cdfe41e5c06d60dc10f145d91501383f |

固定 320×192、61 帧（60 个新帧）、35 steps、CFG 6、flow shift 3；播放 fps=24 与生成吞吐无关。batch=1、concurrency=1。两个场景都使用种子 301017、301018、301031，五轮顺序使用固定 RNG seed 230923 预先生成并写入 manifest。相同模型内所有方法执行相同的 scene×seed×round。

保留每方法已经冻结的执行配置：

- 普通 W4A4、我方两档使用 v6 原生 FP4 GEMM／FP4 attention 和 Transformer Graph。
- ARCQuant、SharQ 保留已接入的 BF16 attention 和 eager 限制，不强行改成 Graph 或 FP4 attention。
- AWQ 使用打包 NVFP4 权重、Marlin W4A16、BF16 attention，采用已验证的 Graph 路径。
- Edge 我方冻结的 calibrated／pointwise／rotary／norm／codec Graph 选项不删改；Nano 沿用自己的空 runtime_options／codec Graph off。
- 不把 v8 内存控制的全关开关／新加载器／紧凑 scratch 无声混入 v6。每行记录实际 runtime_options、attention_backend、execution 和库哈希；这是集成方案比较，不是仅改变算术精度的消融。

## 3. 必需资产：已有和待同步

Thor 已有且旧 Edge 运行通过：

- data/arc_orders：Edge 的 ARC 通道顺序。
- assets/awq_marlin：Edge 的 NVFP4-AWQ 打包权重。
- external/arcquant-sm110、external/sharq-sm110：本机编译的原生库和 build.json／补丁。
- comparisons/edge_20260922/{arcquant,sharq,awq_marlin}-probe.json：三个 complete=true 的旧 Edge 资格凭据。

这些全局 asset 路径是 Edge 的，不能拿去跑 Nano。本机已有 Nano 正确资产，不需要重新校准：

| 资产 | PRO 机器源目录 | 建议 Thor 目标目录 | 大小 |
|---|---|---|---|
| Nano ARC 排序 | /home/yufeng/thunderworld/full_quality_ready_20260919/nano/arc_orders | /home/chenhan/thunderworld/data/nano/arc_orders | 10,714,348 bytes，约 10.2 MiB |
| Nano packed AWQ | /home/yufeng/thunderworld/full_quality_ready_20260919/nano/awq_native | /home/chenhan/thunderworld/assets/nano/awq_marlin | 3,910,489,226 bytes，约 3.64 GiB |

两个 manifest 均 complete=true；本次已逐文件核验 ARC orders.pt 和全部 252 份 AWQ 模块文件。manifest SHA256：

- ARC：8a1897912bb56027c620ee1d352258e0251eb289a5a223cf6864b8afcd8ce029。
- AWQ：8306a287a834f478110d1894891be2b6861b683a5dee7f57575f7cce034daf19。

资产传输命令在 PRO 机器运行，尚未在本轮执行：

~~~bash
rsync -a --ignore-existing --info=progress2 \
  /home/yufeng/thunderworld/full_quality_ready_20260919/nano/arc_orders/ \
  chenhan@thor:/home/chenhan/thunderworld/data/nano/arc_orders/
rsync -a --ignore-existing --info=progress2 \
  /home/yufeng/thunderworld/full_quality_ready_20260919/nano/awq_native/ \
  chenhan@thor:/home/chenhan/thunderworld/assets/nano/awq_marlin/
~~~

接收端按 manifest 重新核对文件哈希、model_identity、模块集合及维度。已有同名资产不一致时停止复用，另开版本目录。无需传模型、生成视频、13 GiB 的 decoded AWQ 权重或完整教师面板。Nano AWQ 资产内部定义是既有 ModelOpt NVFP4_AWQ_LITE_W4A16，不能改称 INT4 AWQ 或换校准定义；论文展示名保持 NVFP4-AWQ。

## 4. coding agent 先补的接入点

现有入口不能直接完成七方法复测，需要以下有限扩展。复用 CosmosRunner，不另建推理框架。

1. **standard_comparison.py 的方法与模型绑定。** 当前只支持 bf16／fp8／fp4／repair15／repair20／fp4_bf16_attention；加入 awq_marlin、arcquant50、arcquant100、sharq。prepare 的 Edge 分支仍引用旧 manifest，Nano 分支要求方法已存在于 study.json。改为冻结两个模型各自的原方法规格和外部 adapter 规格；禁止把 external method 当成 study 中现成键直接索引。
2. **tune_speed.py 的测量／profile worker。** 当前 choices 只有五项，且没有安装 baseline_adapter。加入上述外部方法，在 prepare 后、warmup／Graph capture 前安装对应算子；不调用 tune。固定资产权重离线准备，在线激活仍消费当前输入。
3. **baseline_adapter.py 的模型相关资产。** ARC 的 root/data/arc_orders、AWQ 的 root/assets/awq_marlin 目前硬编码为 Edge；从冻结 manifest 显式传入每模型路径。SharQ 保留 extra_fusion=True、cosmos_rmsnorm_fusion=False；模块加载冲突以独立进程解决。
4. **资格与检查不能只加 choices。** standard_comparison.checked_run 当前硬编码全部 graph、并按方法名推定 precision／attention，外部包装器可能保留 BF16 config 来接管投影。以 adapter receipt 的真实 GEMM dtype、模块覆盖和每方法 execution 检查，不把 BF16 占位 config 冒充真正的 BF16 计算或将它改标签冒充 FP4。Graph 方法确认计时期间不新增 capture，eager baseline 按其实际路径验证。
5. **Nano 的 baseline 形状。** probe_baselines.py 当前只有 Edge 的三组随机形状，AWQ 资产路径也固定为 Edge。扩展为从每模型实际模块／预热 shape 清单检查 M129／M416／M960、尾块、零输入和当前输入更新；Nano 特别覆盖 K4096／K12288、N1024／N4096／N12288 和 gate/up/down。检查 ARC 现有编译实例是否覆盖这些 K，必要时只补形状实例，在新 build 目录保存来源、补丁和 sm_110a 库哈希。SharQ 稀疏首项的 BF16 舍入保持原定义。新改内核运行 memcheck／racecheck；未改变且覆盖匹配的已锁定资格可以复用。
6. **统计与 profile 接口。** 当前 tune_speed.py、standard_comparison.py、compare_speed.py 仍按 61/T 输出 FPS。本轮必须改为 new_frame_fps=(frames−1)/T，汇总为 60/median(T)，并删除自动生成 BF16／FP8 速度比的报告列。profile 当前遍历全部方法、只用 case 0、seed 第一项；增加 --profile-methods／--profile-cases／--profile-seed，在独立目录记录每个方法×场景，不能覆盖早先 trace。

不要直接运行 compare_speed.py 的默认 prepare/run：它固定 Edge、旧后端和九方法（包含 BF16／FP8），不能补齐本轮矩阵。不要直接运行 standard_comparison.py 的默认方法集，同样会多跑高精度和 attention 归因组。

代码基线已核对本机与 Thor 四个文件哈希一致，见配套 JSON。ARCQuant 官方 commit 为 dbf841e123204ee0584006541dea3b4a539b2e1c，SharQ 为 3cc584fe48e32e59c8b4e37323470dcdd606bb11；沿用 data/baseline_sources.json 锁定版本，不另做 baseline 参数／性能搜索。

## 5. 运行协议与预期命令

以下七方法命令和 profile 参数是完成第 4 节扩展后的目标接口；目前原 CLI 尚不接受新增方法，不能直接复制启动。

~~~bash
cd /home/chenhan/thunderworld/src/ThunderWorld
source tools/thor_handoff/env.sh
thor_tools="$TW_THOR_ROOT/src/ThunderWorld/tools/thor_handoff"
thor_campaign="$TW_THOR_ROOT/comparisons/section54_fp4_retest_20260923"

for thor_model in edge nano; do
  python "$thor_tools/standard_comparison.py" prepare \
    --model "$thor_model" --rounds 5 \
    --variant "$TW_THOR_ROOT/variants/cutlass-mainloop-v6" \
    --methods fp4 awq_marlin arcquant50 arcquant100 sharq repair15 repair20 \
    --output "$thor_campaign/$thor_model"
  python "$thor_tools/standard_comparison.py" run \
    --output "$thor_campaign/$thor_model"
  python "$thor_tools/standard_comparison.py" report \
    --output "$thor_campaign/$thor_model"
done

# All uninstrumented measurements must finish before this step.
python "$thor_tools/standard_comparison.py" profile \
  --output "$thor_campaign/edge" \
  --profile-methods arcquant50 arcquant100 sharq repair15 repair20 \
  --profile-cases 0 1 --profile-seed 301017
~~~

- 每个方法×轮次独立进程，两个 case 均完成准备、capture 和完整请求预热。请求内没有临时编译、调参或建图。
- debug=off，不开逐层／逐步 hook 或 Event；完整请求两端同步计时。计时包含 CPU 输入准备、fresh AR prefill、35 步、解码和 CPU 输出转换；文件保存、输出 hash、准备和预热另记。
- AR cache 在请求内复用；每次新请求重新 prefill，不能命中上一请求的输出／AR cache 来制造加速。
- 复用 gpu-test.lock，单进程串行；请求边界检查其他计算任务，记录期间系统负载和温度／频率／功耗模式。沿用 MAXN／DVFS 并如实记录；不擅自 sudo 更改功耗、风扇或锁频。
- 可用统一内存不足 12 GiB、OOM 或其他计算任务出现时保存进度并暂停该实验；不减少帧数、分辨率、steps 或改变数值路径。
- 不评估视频 fidelity／judge，也不要求不同量化方法输出一致；仍检查输出完整、有限、形状正确，以及原生算子数值、Graph 当前输入和预算可行性。不得隐式回退 BF16 GEMM／attention。
- 每请求保留完整计算账单，我方 1.5／2 档必须通过含 padding 的预算检查。
- 顺便记录相同速度配置的 peak allocated/reserved；预热后以去重 tensor storage 记录常驻资产（权重／codes／scales／W1／索引／额外布局），与 Graph/scratch 分列。库存扫描在计时外进行，不移植 v8 数字。
- median、线性插值 P90 从原始请求计算。报告每轮、每场景和 pooled 结果；若给区间，按轮成组重采样，不把同场景重复请求当独立质量样本。

## 6. 需要交回的文件与字段

建议新目录 comparisons/section54_fp4_retest_20260923/{edge,nano}，不修改任何旧 campaign。

1. manifest.json：model revision／模型凭据、输入图像和 prompt hash、方法身份、投影范围、精度／attention／eager/Graph、运行开关、资产与库 SHA256、环境／功耗、种子及每轮方法顺序。
2. runs/round-*/METHOD/result.json 和完整 stdout/stderr：每请求 wall_seconds、scene_id、seed、round、steps／帧数、debug、profiled、new_frame_fps、peak allocated/reserved、输出 shape、计算账单、prepare/warmup/capture 分时。cold start 以进程开始至首个有效输出等明确边界另记，不能混入暖请求。
3. requests.csv（每请求一行）和 summary.json／summary.csv：cohort、device、model、method_id、rounds、requests、median_seconds、p90_seconds、new_frame_fps_at_median、memory bytes、source/manifest hash；另列 relative_to_plain_fp4、对应 ARC 档位和 SharQ 的比较。所有缺失／失败方法有明确状态，不能按零时延处理。
4. qualification/：每模型外部基线数值／原生指令／当前输入校验，新 kernel 的 sanitizer 日志，实际 shape 清单、调优命中或固定安全 launch 说明。
5. profiles/：10 次 profile 各自的 result.json、kernel 明细 CSV、分类汇总 JSON/CSV、nsys-rep／sqlite 的路径及 SHA256。分类为 preparation_selection、projection_ffn、attention、combine、other；每 kernel 只归一次，融合 kernel 整体保留。累计 kernel 时间不充当 wall time，不将 CPU API 累计时间加到 GPU 堆叠。
6. report.md：七方法×两模型完成状态，runtime-only 范围、额外修复成本、外部基线加速、内存以及版本差异。所有新数据先明确为本次实测批次；五轮完成本身不自动将 performance_eligible 改为 true，更不能写成吞吐上限或全量质量验收。

JSON／CSV／日志／源码补丁／输入哈希清单同步回 PRO 即可。大 trace 可保留在 Thor，给可定位路径和 hash；本轮无需保存或回传全部 MP4／NPZ。

## 7. 已有结果用于回归核对

以下是旧观测值，只用于发现路径接错，不作为新数据必须达到的目标：

| 现有 v6 median 秒 | W4A4 | 我方 1.5× | 我方 2× |
|---|---:|---:|---:|
| Edge，五轮 | 6.615 | 7.167 | 7.846 |
| Nano，一轮 | 15.080 | 21.170 | 24.432 |

旧 Edge campaign 的 AWQ／ARC50／ARC100／SharQ 为 10.778／13.903／14.686／13.719 秒，但同批我方为 13.304／13.970 秒；它们不与上表 v6 拼算。新批次应完整报告快慢变化，不筛选只保留更快的轮次。

论文取数端：docs/paper/manuscript/scripts/import_efficiency.py 与 evidence/section54_measured_20260923。补测结束后以新 cohort 导入，不覆盖 Figure 1 原始 PRO 6000 点，也不把 Thor 时延与桌面全量质量拼成同配置前沿。

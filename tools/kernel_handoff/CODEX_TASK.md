# 接收端 Codex：RTX 5090 FP4 kernel 性能调试

用户目标：在自己的 RTX 5090 服务器上优化 ThunderWorld 主要 FFN/attention 修复配置的性能。先读本目录 README，检查本机获准使用的 GPU，然后执行工作；不要从源服务器的 GPU 编号推断目标机器的分配。

## 已完成的工作

- 已合并内核基线 `9711abb09619a6e5e40be9ba0fb71b1fd63a145b`。
- `packages/fp4-repair`：独立 FFN/core/projection 配置，head quotas/masks，原生 FP4 attention。
- `experiments/ffn_block_repair_20260907/native.py`：dense W4 + shared K-channel A repair + offline compact W-block repair，支持在同 K 循环复用 W0。
- 旧实验中有共享 GPU 性能数据，不能作为本机 5090 的性能结论。
- 交接数据只含真实输入、校准统计、prescale 和期望权重哈希。模型从 HF 固定 revision 下载，目标机离线准备 W0/W1。

## 不改变的算法语义

1. W 的变换、量化、残差、选块、打包均离线。在线 overlap 只针对当前激活量化和计算。
2. FFN、attention core、attention projection 配置独立；head finding 必须真正影响各 head 配额或块支持。
3. Attention 所有连接保留 dense FP4 base，仅按块加修复，不以删连接来取得速度。
4. A0W0 + A1W0 + A0W1；交叉项若开启，单独计入计算预算。不要双重计数。
5. 不通过降低修复配额、改变输入、复用旧 step 激活或扩大误差容忍度来掩盖性能回退。
6. 本轮 W0 是 local BF16 + AWQ prescale 的 RTN W4；不得称作完整 AWQ checkpoint 精度结果。

## 执行顺序

1. 新建优化分支，保存起始 commit、环境信息、GPU 功率/时钟/占用情况；不擅自更改系统级功耗或时钟配置。
2. 核对输入包 SHA256，下载 manifest 指定 HF 分片，运行 prepare_weights；哈希不一致先排查环境/权重源。
3. 包测试与 quick matrix 通过后，完整运行 main matrix 并冻结 baseline。主表保留实际小 M/N；stress 只测吞吐上限。
4. 对每类形状分开判断 compute、memory、occupancy、launch/producer、softmax/P 量化瓶颈；不要强求每个小矩阵都达到营销峰值。
5. FFN 优先：W0 复用、M32/M64、K128/K256、shared-memory layout、寄存器压力、流水预取、激活 producer；gate/up 同输入的 producer 共享是待实现优化，不是已有功能。down 的长 K 需要单独调。
6. Attention 优先：在线 P 量化与 softmax 依赖、寄存器/occupancy、split 数、K/V 数据搬运、producer overlap。保持 lazy residual quantization 和当前 head/block mask 语义。
7. 单次改变一个主要实现因素，重复交错测量并保存所有样本。profile-only 或预量化 GEMM 结果单独列，不能当作整算子加速。
8. 质量/数值检查通过后才保留性能候选；运行真实三 step、tail、GQA、per-head support、动态输入图重放和必要 sanitizer。仅吞吐改善而数值失败的 case 必须保留失败记录，不纳入推荐。

## 需要交付的结果

- 每个主要配置的 BF16、原始 FP4、优化 FP4：整算子延迟中位数和样本范围、真实 shape、实际修复比例、源码 SHA256。
- FFN 全流程与预打包 GEMM 分开；attention 若增加分阶段 probe，注明数据依赖和缓存差异。
- Nsight 证据：原生 E2M1 MMA、Tensor Core 利用率、SM 活跃度、DRAM 吞吐、寄存器/溢出、occupancy 和主要 stall。
- 正确性误差、BF16 一致性、输出 ownership/图重放验证，额外权重/metadata/scratch 分开计量。
- 建议配置按 gate/up/down 和 M、attention M/N/head profile 分开，不假设一个 tile 适合全部。
- 无法“打满”的配置说明可测的限制，并报告相对本机可达 roofline 的差距。不得把 native FP4 指令存在等同于已打满算力。

## 参考资料

- `docs/FFN_BLOCK_REPAIR_20260907.md`
- `packages/fp4-repair/README.md` 与 `VALIDATION.md`
- `experiments/ffn_block_repair_20260907/plans.json`
- `results/f4/ffn_block_repair_20260907/summary.json`

可选 RePair4 源码：`https://github.com/Concyclics/RePair4.git`，基线 `a3b39e5d04fc0b60ab05ded30f65989786281671`。关注 `kernels/sm100/online_nvfp4_v2.cuh`、`dual_blockscaled_mainloop.cuh`、`blockscaled_layout.cuh`。目录名不是目标架构声明；5090 仍按 SM120 编译和验证。旧 ThunderWorld CUTLASS prescale 实验未随 main 发布，且 build 脚本含源机路径；当前可复现默认路线为本交接的独立 Triton kernels。

# 同卡多进程生成队列

`judge_parallel_queue.py` 为冻结的 judge 队列提供每卡一到四个独立生成进程。
每张卡的主进程可以执行 Edge/Nano 的全部五种方法；其他进程领取明确准入的 Edge 方法。
两个 GPU 仍共用原 `queue.json`，领取任务使用文件锁，每个任务始终只有一个 owner。

沿用原来的 320×192、61 帧、35 步、CFG、种子、离线资产和算子；不改变任何已冻结的
数值源文件。`judge_concurrent_infer.py` 仅适配资源锁，然后调用原 `dataset_infer.run`。
不同进程隔离 RNG、模型、CUDA Graph、临时缓冲区和默认 stream。没有修改 GPU 的 SM 配额、
时钟、功耗或 MPS 配置。

## 锁与恢复

调度器持有原来的全局共享锁，以及每张物理 GPU 的独占设备锁。经准入的子进程继承这些
文件描述符；外部旧版校准进程仍不能抢占本池的设备锁。进程组按 worker 隔离，停止某一路
会终止其当前生成子进程。异常或 OOM 暂停整个池并保留断点，不降低精度或改变输入。

历史 GPU0 资格记录中，Edge 最大 reserved 约 7.62 GiB，Nano 最大约 25.43 GiB。
启动前每卡按主进程 28 GiB、附加 Edge 进程各 12 GiB、空闲余量 12 GiB 作检查；
四进程要求启动前至少 76 GiB 空闲。运行期间每卡最多一个进程领取 Nano，防止四份 Nano 超出显存。
原始逐任务显存检查继续生效。共享环境中的其他进程仍可能改变资源状况，这些吞吐记录
不能作为独占硬件的性能排行榜。

主进程复核原 GPU 资格记录。新增进程先生成参考场景，逐位核验四份输出和 A→B→A，
再领取正式任务。进程内部始终串行执行完整请求；各任务的 NPZ、视频、日志和 metadata
仍使用原目录。未完成任务重新入队，已保存种子通过原 SHA/身份检查后直接复用。

## 并发准入试验

先腾出用于试验的一张卡，其余卡可以继续正式生成：

```bash
export TW_PYTHON=/home/yufeng/thunderworld/.venvs/cosmos3/bin/python
export PYTHONPATH=/tmp/thunderworld-joint-deps
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_concurrent_infer.py probe \
  --root /home/yufeng/thunderworld/judge_full_i2v_20260919 --gpu 1 \
  --output /home/yufeng/thunderworld/judge_concurrency_probe_20260921
```

五种方法分别执行单路、两路、单路三阶段。每阶段是相同的两个独立任务块：每块保存四个
参考视频并执行 A→B→A。统计整个阶段墙钟时间，包含加载、预热、输出转换、校验和文件写入。
这是短任务块的调度诊断，不等同于完整队列的稳态吞吐或纯推理 latency。
两路吞吐至少高于两次单路中较快者 10%，且全部输出哈希匹配，才标为可进入实际队列试运行。
GPU0/1 有其他负载时还需结合实际队列观察，不能把变化都归因于 Tensor Core 利用率。

## 启动与状态

先停止旧调度器、worker 和 exporter，并保留已保存的种子。首次启用使用
`launch --root ROOT --gpus 0 1 --slots-per-gpu 2 --secondary-methods METHOD... --probe-report REPORT`，
其中默认 `--admission measured` 要求方法通过报告中的两路准入检查，最多启用两路。
用户明确指定四路时使用 `--admission requested`，不把尚未完成的四路吞吐比较伪装成已通过；
新增进程的逐位参考输出校验始终执行。原始 job、试验 cohort、控制器和资源适配器均以 SHA 绑定。
配置完成后，续跑使用以下命令沿用已保存策略：

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_parallel_queue.py launch \
  --root /home/yufeng/thunderworld/judge_full_i2v_20260919
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_parallel_queue.py status \
  --root /home/yufeng/thunderworld/judge_full_i2v_20260919
```

启用后不要用旧 `judge_generation.py` 或 `judge_device_control.py` 恢复同一目录。
`parallel_policy.json` 保存卡数、每卡进程数、方法准入和证据 SHA；`pool_status.json` 保存
总体进度及各卡各路状态。每路日志位于 `parallel_workers/gpuN.slotM/`，任务日志仍在
`tasks/<task>/generation.log`。每次实际生成的资源适配记录追加在输出目录的
`concurrency_invocations.jsonl`，与原数值 receipt 分开保存。

## 2026-09-21 当前部署

用户明确要求 GPU0、GPU1 每卡四进程，已采用以下策略，额外进程可以领取全部五种 Edge 方法：

```bash
bash tools/fp4_handoff/python.sh tools/contribution_study/judge_parallel_queue.py launch \
  --root /home/yufeng/thunderworld/judge_full_i2v_20260919 --gpus 0 1 --slots-per-gpu 4 \
  --secondary-methods arcquant arcquant100 sharq_perf b1.5_protected_n b2_joint \
  --probe-report /home/yufeng/thunderworld/judge_concurrency_probe_20260921/progress.json \
  --admission requested
```

两路对照已完成 ARCQuant 100%、Ours 1.5×、Ours 2×；两路相对较快的单路分别约为
1.43×、1.68×、1.70×，且各阶段输出均匹配参考。用户随后要求直接启用每卡四路，因此
ARCQuant 50% 和 SharQ 的成对吞吐试验中断并保留原始记录。上述两路收益不代表四路收益。
六个新增进程均已完成全部五种 Edge 方法的资格校验：120 份参考输出哈希一致，30 次
A→B→A 校验通过。2026-09-20 20:03:28 UTC 起，八路全部进入正式队列。

此后第一个完整五分钟窗口保存 92 个视频，合计 1,104 视频/小时、18.71 生成 FPS。
切换前 19:00–19:30 UTC 的双卡每卡单路窗口为 508 视频/小时；两窗口均为 Edge
ARCQuant 100%，观测比值为 2.17。两次共享负载可能不同，且新窗口较短，不能把它解释为
独占设备的因果加速比，也不能代表其他方法的四路吞吐。详见
[`并发观测记录`](../../results/judge_full_i2v_20260919/concurrency_20260921/README.md)。

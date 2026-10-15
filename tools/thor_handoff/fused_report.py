"""Report matched Thor v5/v6 repair-mainloop fusion, including unchanged outputs."""
import argparse
import json
from pathlib import Path
from build_baselines import sha,write
from cutlass_report import read_result,compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    qualified=json.loads(a.qualification.read_text())
    if not qualified['complete'] or not qualified['fused']:raise ValueError('Incomplete fused qualification')
    if len(qualified['rows'])<42 or not all(r['previous_bitwise_equal'] for r in qualified['rows']):
        raise ValueError('Fusion must preserve the previous CUTLASS arithmetic')
    for tool in ('memcheck','racecheck'):
        path=a.qualification.parent/(tool+'.json');q=json.loads(path.read_text())
        marker='ERROR SUMMARY: 0 errors' if tool=='memcheck' else 'RACECHECK SUMMARY: 0 hazards'
        if not q['complete'] or q['library_sha256']!=qualified['library_sha256'] or q['sources']!=qualified['sources']:
            raise ValueError('Sanitizer source/binary mismatch')
        if marker not in path.with_suffix('.log').read_text():raise ValueError('Sanitizer failed')
    report=dict(complete=False,quality_evaluation=False,performance_eligible=False,methods={},
        qualification=dict(path=str(a.qualification),sha256=sha(a.qualification),
            checks=len(qualified['rows']),sparse_packing=len(qualified['sparse_packing']),
            graph_checks=sum(r.get('current_inputs_graph',False) for r in qualified['rows']),
            all_operator_outputs_bitwise_equal=True,memcheck='passed',racecheck='passed'),
        limitations=['Thor SM110 only; MAXN without locked clocks; sequential diagnostic, not formal leaderboard.',
            'Same two inputs x three seeds, 320x192, 61 frames, 35 steps, CFG 6, Graph, debug off.',
            'Complete request includes encode/decode/CPU output, excludes preparation and file serialization.',
            'No new generation-quality scoring; hash parity is relative to the prior v5.',
            'Global-memory byte counts are logical traffic from the cost ledger, not measured DRAM traffic.'])
    lines=['# Thor CUTLASS 稀疏修复主循环融合','',
        'Edge，320×192、61 帧、35 步、CFG=6，两个输入各三个固定种子，Graph、debug off。',
        'v5 和 v6 均在本轮独立加载、充分预热后测量。计时包含编码、生成、解码和 CPU 输出。',
        'MAXN 未锁频；顺序诊断，不是五轮随机交错的正式速度排行榜。','',
        '| 档位 | v5 两个 GEMM kernel 秒 | v6 融合秒 | 延迟下降 | v6 FPS | 输出哈希一致 |',
        '|---|---:|---:|---:|---:|---:|']
    for method,label in [('b1.5_protected_n','1.5×'),('b2_joint','2×')]:
        before,brow=read_result(a.measurements/'v5'/method/'result.json')
        after,arow=read_result(a.measurements/'v6'/method/'result.json')
        change=compare(before,after)
        if change['hash_matches']!=6:raise ValueError('Full-request output changed')
        if arow['max_intermediate_buffer_bytes'] or arow['max_extra_intermediate_traffic_per_request']:
            raise ValueError('Global FP32 intermediate still present')
        if arow['max_gemm_launches_per_request']!=brow['max_gemm_launches_per_request']//2:
            raise ValueError('Unexpected GEMM launch count')
        report['methods'][method]=dict(before=brow,after=arow,comparison=change)
        lines.append(f"| {label} | {brow['median_seconds']:.3f} | {arow['median_seconds']:.3f} | {change['latency_reduction']:.1%} | {arow['fps']:.3f} | {change['hash_matches']}/6 |")
    lines+=['','| 档位 | v5 P90 秒 | v6 P90 秒 | 请求 MMA/C0 范围 | W1 支持并集 MiB |',
        '|---|---:|---:|---:|---:|']
    for method,label in [('b1.5_protected_n','1.5×'),('b2_joint','2×')]:
        row=report['methods'][method];b=row['before'];n=row['after']
        if b['w1_union_bytes']!=n['w1_union_bytes'] or b['prepared_weight_bytes']!=n['prepared_weight_bytes']:
            raise ValueError('Weight storage changed')
        lo,hi=min(n['compute_ratios']),max(n['compute_ratios'])
        lines.append(f"| {label} | {b['p90_seconds']:.3f} | {n['p90_seconds']:.3f} | {lo:.6f}–{hi:.6f} | {n['w1_union_bytes']/2**20:.3f} |")
    lines+=['','| 档位 | allocated GiB 前→后 | reserved GiB 前→后 | GEMM kernel/请求 前→后 | 去除的逻辑读写 GiB/请求 |',
        '|---|---:|---:|---:|---:|']
    for method,label in [('b1.5_protected_n','1.5×'),('b2_joint','2×')]:
        row=report['methods'][method];b=row['before'];n=row['after']
        lines.append(f"| {label} | {b['allocated_gib']:.3f} → {n['allocated_gib']:.3f} | {b['reserved_gib']:.3f} → {n['reserved_gib']:.3f} | {b['max_gemm_launches_per_request']} → {n['max_gemm_launches_per_request']} | {b['max_extra_intermediate_traffic_per_request']/2**30:.3f} |")
    lines+=['',
        'v6 将基础项和选中的 A/W/交叉项装入同一 CUTLASS 主循环。代码数据由 TMA 搬运，基础尺度继续使用 TMA，残差 K16 尺度由加载 warp 的 cp.async 搬入同一流水槽。',
        'MMA warp 消费这些槽并累加到同一个 FP32 Tensor Memory accumulator；仅最终 epilogue 输出 BF16。',
        '保留 v5 的基础项→A→W/交叉项顺序。动态 mask 的实际选中数量决定生产者和消费者的阶段数，包括少于配额及空修复情况。',
        '权重、W1 支持并集、阶段/head 计划、padding、MMA 预算和在线量化定义均未改变；attention 保持 v5。',
        '最大 33.75 MiB 的单算子 FP32 中间缓冲已去除。上述流量是 global 地址空间的逻辑字节数，不代表全部访问都到达 DRAM。',
        'GEMM kernel 次数来自实际预热执行清单及阶段调用次数。allocated 峰值基本不变；Graph 存储复用与其他阶段决定了请求峰值，不能将去除的单算子缓冲直接当作峰值显存节省。',
        f"{len(qualified['rows'])} 组算子与 v5 逐位一致，{len(qualified['sparse_packing'])} 组稀疏 packing 校验通过；6 组 Graph 当前输入/动态 mask、尾块与全零检查通过，memcheck/racecheck 无报错。",
        '完整请求哈希一致证明这组输入未改变输出；不增加对 FP8 的质量结论，也不说明跨模型或跨设备已经通过验收。','']
    report['complete']=True;a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'summary.json',report);(a.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:v['comparison'] for k,v in report['methods'].items()},indent=2))


if __name__=='__main__':main()

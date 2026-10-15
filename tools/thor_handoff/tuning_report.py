"""Rebuild the speed-only Thor optimization readout from completed JSON files."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def percentile(values,q):
    values=sorted(values);pos=(len(values)-1)*q;i=int(pos)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(pos-i)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sources={}
    def read(rel):
        path=a.measurements/rel
        if not path.exists():return None
        sources[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    report=dict(scope='Thor Edge speed-only diagnostic; two inputs, three seeds per input, debug off, 35 steps',
        resolution=[320,192],frames=61,quality_evaluated=False,formal_leaderboard=False,
        power_mode='MAXN, clocks not locked',methods=[],sources=sources)
    for tag,budget in [('15',1.5),('20',2.)]:
        before=read(f'e2e-anchor{tag}/result.json');after=read(f'e2e-transposed{tag}/result.json')
        if not before or not after or not before['complete'] or not after['complete']:
            raise RuntimeError(f'Incomplete end-to-end comparison {budget}')
        key=lambda row:(row['case_index'],row['seed'])
        assert sorted(map(key,before['rows']))==sorted(map(key,after['rows']))
        assert before['fixed_manifest_sha256']==after['fixed_manifest_sha256']
        row=dict(budget=budget,requests=len(after['rows']),variant=after['variant'],by_case=[])
        for label,data in [('before',before),('after',after)]:
            values=[r['wall_seconds'] for r in data['rows']]
            assert all(r['debug']=='off' and r['execution']=='graph' and r['steps']==35 for r in data['rows'])
            assert all(r['compute_ledger']['feasible'] for r in data['rows'])
            row[label]=dict(median_seconds=statistics.median(values),p90_seconds=percentile(values,.9),
                generated_fps=61/statistics.median(values),
                peak_allocated_gib=max(r['peak_allocated_bytes'] for r in data['rows'])/2**30,
                peak_reserved_gib=max(r['peak_reserved_bytes'] for r in data['rows'])/2**30,
                compute_ratio_max=max(r['compute_ledger']['ratio'] for r in data['rows']))
        row['speedup']=row['before']['median_seconds']/row['after']['median_seconds']
        row['latency_reduction']=1-1/row['speedup']
        for case in sorted({r['case_index'] for r in after['rows']}):
            bs=[r['wall_seconds'] for r in before['rows'] if r['case_index']==case]
            ns=[r['wall_seconds'] for r in after['rows'] if r['case_index']==case]
            row['by_case'].append(dict(case_index=case,before_median=statistics.median(bs),after_median=statistics.median(ns)))
        report['methods'].append(row)
    qualified=read('transposed-attention-qualified.json')
    assert qualified and qualified['complete']
    report['correctness']=dict(tests=len(qualified['tests']),
        maximum_relative_l2_to_frozen_operator=max(r['relative_l2'] for r in qualified['tests']),
        every_repair_mask_identical=all(r['identical_masks'] for r in qualified['tests']),
        graph_tests=sum(r.get('current_input_graph',False) for r in qualified['tests']),
        kernel_sha256=qualified['kernel_sha256'])
    for tool in ('memcheck','racecheck'):
        path=a.measurements/f'transposed-attention-{tool}.log'
        contents=path.read_text()
        assert ('ERROR SUMMARY: 0 errors' in contents if tool=='memcheck' else '0 errors, 0 warnings' in contents)
        report['correctness'][tool]='passed'
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    concurrent=read('streams2-15.json')
    if concurrent and concurrent['complete']:
        single=report['methods'][0]['by_case'][0]['after_median']
        report['concurrency']=dict(slots=2,requests=len(concurrent['rows']),generated_fps=concurrent['generated_fps'],
            single_case_generated_fps=61/single,throughput_ratio=concurrent['generated_fps']/(61/single),
            peak_allocated_gib=concurrent['peak_allocated_bytes']/2**30,
            mean_request_seconds=statistics.mean(r['seconds'] for r in concurrent['rows']))
    tma=read('tma-gemm-qualified.json')
    tma_memcheck=a.measurements/'tma-gemm-memcheck.log'
    tma_memcheck_passed=tma_memcheck.exists() and 'ERROR SUMMARY: 0 errors' in tma_memcheck.read_text()
    report['tma_status']=dict(promoted=False,complete=bool(tma and tma['complete']),
        locally_received_tests=len(tma.get('rows',[])) if tma else 0,
        memcheck_passed=tma_memcheck_passed,
        reason='Numerical qualification and sanitizer acceptance are separate gates. TMA has not been deployed; no end-to-end gain is claimed.')
    report['pending']=['Resolve TMA sanitizer API errors and qualify any changed implementation before deployment.',
        'See the separate fp4_fp8_attribution_20260922 campaign for subsequent V-layout results.',
        'See resume_state.json for the historical benchmark controller state.']
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# Thor Edge FP4 性能调优实测','',
        '已完成并可用的是 `transposed-attention-v2`。本轮只测速度；没有开展生成质量评分。',
        'NVIDIA Thor SM110 / 20 SM，MAXN 未锁频；320×192、61 帧、35 步、CFG 6、flow shift 3、Graph、debug off。',
        'PAI-Bench 与 RBench 各一个固定输入，每个输入 3 个种子。每版分别充分预热，计入编码、生成、解码和 NumPy 输出转换；不含视频文件写入。',
        '这是 6 请求的调优诊断，尚非五轮随机交错的正式性能排行榜。','',
        '| 修复预算 | 优化前中位秒 | 优化后中位秒 | 延迟下降 | 加速比 | FPS 前→后 | 峰值 allocated GiB 前→后 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in report['methods']:
        b=row['before'];n=row['after']
        lines.append(f"| {row['budget']:g}× | {b['median_seconds']:.3f} | {n['median_seconds']:.3f} | {row['latency_reduction']:.1%} | {row['speedup']:.3f}× | {b['generated_fps']:.3f} → {n['generated_fps']:.3f} | {b['peak_allocated_gib']:.3f} → {n['peak_allocated_gib']:.3f} |")
    lines+=['','## 实现与证据','',
        '旧 Q32 attention 的 PTX 包含 384 个静态 BF16 MMA 指令位置，没有 NVFP4 MMA；旧 Q64 则使用 tcgen05 的 f16 路径。GEMM doctor 的原生 FP4 证据不能外推到 attention。',
        '新内核用 K×Qᵀ、V×Pᵀ 的转置计算，让原生 MMA 的 M 保持 128、N 使用 query tile 64。保留全部 base blocks、当前输入选择器、原有 QK/PV 修复项、softmax 与 P 量化；W0/W1 资产不变。',
        '实际发射 tile 参与预算编译。1.5× 最大请求比率为 '+f"{report['methods'][0]['after']['compute_ratio_max']:.9f}"+'，2× 为 '+f"{report['methods'][1]['after']['compute_ratio_max']:.9f}"+'，均通过硬约束。',
        '首个直接 Q128 原型在 RBench 上达到 1.504764766×，被预算检查拒绝，没有通过减少修复掩盖。',
        '旧版本的一次 Nsight GPU kernel 时间分解：attention 33.1%（4.45 秒），GEMM 30.7%（4.13 秒）；该结果带 profiler，不能作为正式请求耗时。',
        f"新 attention 通过 {report['correctness']['tests']} 项测试，涵盖尾块、GQA、空/全修复、QK/PV 独立配置、四项、显式 mask、全零及 Graph 当前输入更新。与冻结算子的最大相对 L2 为 {report['correctness']['maximum_relative_l2_to_frozen_operator']:.3g}，修复 mask 全部相同；memcheck/racecheck 零错误。",
        '更换底层累加路径不保证逐位一致；算子数值检查不等于生成质量验收。','',
        '## 并发与未启用候选','']
    if 'concurrency' in report:
        c=report['concurrency'];lines += [f"相同 PAI-Bench 输入，单路 {c['single_case_generated_fps']:.3f} FPS，双 CUDA stream {c['generated_fps']:.3f} FPS（+{(c['throughput_ratio']-1)*100:.1f}%）。双路 allocated 峰值 {c['peak_allocated_gib']:.3f} GiB；吞吐收益有限，单请求延迟反而增加。"]
    lines += ['寄存器预取 GEMM 版本变慢，未启用。TMA 双缓冲在部分 A 修复配置上有内核收益，但纯 W 修复等配置也会变慢，不能全局替换。未集成到上述端到端数字。',
        'row-major packed-V 的后续资格检查与请求测量另存于 fp4_fp8_attribution_20260922，不混入本次 v2 对照。','',
        '## 后续验证状态','',
        '追加 TMA 验证时曾发生 SSH / Tailscale 失联，之后已恢复连接。设备未重启，原因未确认，不能断言由内核造成。',
        f"当前已收回 TMA 数值测试 {report['tma_status']['locally_received_tests']} 项；数值完整性为 {report['tma_status']['complete']}，memcheck 通过状态为 {tma_memcheck_passed}。",
        '现有 TMA memcheck 日志包含 cuFuncSetAttribute 属性 17 的 API 错误。不能将完整数值结果等同于 sanitizer 通过；该版本保持未启用。历史速度队列状态见独立恢复记录。',
        '已验证的 attention 版本、完整请求 JSON、安全检查及代码均保留。没有将未完成实验计入收益，也没有宣称已达到 Thor 的硬件吞吐上限。','',
        '原始文件位于 `'+str(a.measurements)+'`；逐文件 SHA256、P90、逐场景数据和显存分项见同目录生成的 `summary.json`。']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(methods=report['methods'],report=str(a.output/'report.md')),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

"""Measured Thor speed explanation with explicit operator/profile boundaries."""
import argparse
import collections
import csv
import json
from pathlib import Path
import random
import statistics
from build_baselines import sha,write
from attribution_report import percentile

# Both symbols disappear in the quantized traces and together account for
# exactly the 11,760 replaced generation projections. Other nvjet symbols
# remain present in all six arms and belong to the unquantized GEMM scope.
BF16_GENERATION_KERNELS={
    'nvjet_sm110_tst_256x192_64x4_2x1_2cta_v_bz_TNT',
    'nvjet_sm110_tst_128x192_64x7_2x1_2cta_v_bz_TNT',
}

def category(name,method,instances):
    if method=='bf16' and name in BF16_GENERATION_KERNELS and instances==5880:
        return 'generation_projection_gemm'
    if method!='bf16' and 'cutlass::device_kernel<' in name:
        return 'generation_projection_gemm'
    if name in ('scaled_amax','finish_amax','pack_activation','pointwise_amax','pack_kernel',
                'base_summary','choose','residual_only','calibrated_relu_amax') or 'dynamic_per_token_scaled_fp8_quant_kernel' in name:
        return 'activation_pack_and_select'
    if name=='select_tiles':return 'ambiguous_selector'
    if name in ('streaming_attention','pack_qk','pack_v','combine','summarize','select_blocks',
                'pack_qk_sparse','pack_v_sparse','support_union','select_tiles_head','residual_qk','residual_v'):
        return 'fp4_attention_including_pack'
    if 'pytorch_flash::flash_fwd_kernel<' in name and instances==1960:
        return 'generation_bf16_attention'
    lower=name.lower()
    if ('gemm' in lower and 'implicit_gemm' not in lower) or name.startswith('nvjet_'):
        return 'other_explicit_gemm'
    if any(s in lower for s in ('cudnn::','implicit_gemm','convolve','conv_', 'winograd')):
        return 'convolution_and_layout'
    if any(s in lower for s in ('copy','catarray','transpose','permut','nhwc')):
        return 'other_copy_layout'
    if any(s in lower for s in ('elementwise','reduce_kernel','normaliz','layer_norm','rms_norm','rotary','rope','relu','norm_')):
        return 'other_pointwise_norm_reduce'
    return 'other_unclassified'


def profiles(root):
    result={};key=None
    for path in sorted(root.glob('*-kernels_cuda_gpu_kern_sum.csv')):
        name=path.name.removesuffix('-kernels_cuda_gpu_kern_sum.csv')
        receipt=json.loads((root/(name+'.complete.json')).read_text())
        if not receipt['complete'] or receipt['csv_sha256']!=sha(path):raise ValueError('Unverified profile')
        if receipt['source_manifest_sha256']!=sha(root.parent/'manifest.json'):raise ValueError('Unmatched profile manifest')
        p=root/(name+'-request/result.json');data=json.loads(p.read_text())
        if sha(p)!=receipt['request_sha256'] or not data['complete'] or len(data['rows'])!=1 or not data['rows'][0]['profiled']:
            raise ValueError('Unverified profiled request')
        row=data['rows'][0];current=[row[k] for k in ('case','seed','steps')]
        if key is None:key=current
        if current!=key:raise ValueError('Unpaired profiling input')
        groups=collections.defaultdict(lambda:dict(seconds=0.,calls=0));kernels=[]
        for kernel in csv.DictReader(path.open()):
            n=kernel['Name'];seconds=int(kernel['Total Time (ns)'])/1.e9;calls=int(kernel['Instances'])
            cat=category(n,name,calls);groups[cat]['seconds']+=seconds;groups[cat]['calls']+=calls
            kernels.append(dict(name=n,seconds=seconds,calls=calls,category=cat))
        if groups['generation_projection_gemm']['calls']!=11760:
            raise ValueError('Unexpected generated projection count: '+name)
        total=sum(v['seconds'] for v in groups.values())
        result[name]=dict(source=str(path),sha256=sha(path),request=current,
            kernel_sum_seconds=total,profiled_wall_seconds=row['wall_seconds'],groups=dict(groups),
            top_kernels=sorted(kernels,key=lambda r:r['seconds'],reverse=True),
            limitation='Kernel sum with profiling; not uninstrumented latency, SM utilization or measured DRAM traffic.')
    return result


def paired_rounds(root,method,reference):
    diffs=[];ratios=[];keys=None
    for i in range(5):
        folder=root/'runs'/f'round-{i:02d}'
        a=json.loads((folder/method/'result.json').read_text())['rows']
        b=json.loads((folder/reference/'result.json').read_text())['rows']
        index=lambda rows:{(r['case'],r['seed'],r['steps']):r['wall_seconds'] for r in rows}
        x,y=index(a),index(b)
        if x.keys()!=y.keys() or (keys is not None and keys!=x.keys()):raise ValueError('Unpaired round')
        keys=x.keys();diffs.append(statistics.mean(x[k]-y[k] for k in keys))
        ratios.append(statistics.mean(y.values())/statistics.mean(x.values()))
    rng=random.Random(230925)
    boot=[statistics.mean(rng.choices(diffs,k=5)) for _ in range(10000)]
    return dict(mean_latency_difference_seconds=statistics.mean(diffs),
        round_differences_seconds=diffs,round_speedups=ratios,
        paired_round_bootstrap95_seconds=[percentile(boot,.025),percentile(boot,.975)],
        caveat='Only five round blocks and two fixed scenes; descriptive conditional interval, not broad workload generalization.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime-control',type=Path,help='Completed auxiliary-runtime attribution summary')
    a=p.parse_args();summary=json.loads((a.campaign/'summary.json').read_text())
    if not summary['complete']:raise ValueError('Five-round benchmark incomplete')
    bench={r['name']:r for r in summary['rows']}
    gpu=profiles(a.campaign/'profiles')
    if set(gpu)!=set(bench):raise ValueError('Incomplete profile collection')
    probe=json.loads(a.probe.read_text())
    if not probe['complete'] or len(probe['rows'])!=16:raise ValueError('Operator comparison incomplete')
    control=None
    if a.runtime_control:
        control=json.loads(a.runtime_control.read_text())
        if not control['complete']:raise ValueError('Runtime attribution incomplete')
    sec=lambda name,key:gpu[name]['groups'].get(key,{}).get('seconds',0.)
    gemm_share=sec('bf16','generation_projection_gemm')/gpu['bf16']['kernel_sum_seconds']
    findings=dict(bf16_generation_gemm_kernel_sum_share=gemm_share,
        ideal_kernel_sum_speedup_if_only_generation_gemm_becomes_4x=1/(1-gemm_share+gemm_share/4),
        plain_fp4_projection_gemm_speedup_vs_bf16=sec('bf16','generation_projection_gemm')/sec('fp4','generation_projection_gemm'),
        plain_fp4_projection_gemm_speedup_vs_fp8=sec('fp8','generation_projection_gemm')/sec('fp4','generation_projection_gemm'),
        plain_fp4_attention_extra_kernel_seconds=sec('fp4','fp4_attention_including_pack')-sec('fp8','generation_bf16_attention'))
    paired={name:{ref:paired_rounds(a.campaign,name,ref) for ref in ('bf16','fp8')}
        for name in bench if name not in ('bf16','fp8')}
    report=dict(complete=True,quality_evaluation=False,performance_eligible=False,
        benchmark_source=dict(path=str(a.campaign/'summary.json'),sha256=sha(a.campaign/'summary.json')),
        probe_source=dict(path=str(a.probe),sha256=sha(a.probe)),paired=paired,profiles=gpu,
        findings=findings,runtime_control=control,
        runtime_control_source=None if a.runtime_control is None else dict(path=str(a.runtime_control),sha256=sha(a.runtime_control)),
        limitations=['Nsight kernel sums are independently instrumented; never substitute them for request timing.',
            'FP4 + BF16 attention changes arithmetic, and is only an attribution ablation.',
            'BF16 generation attribution is inferred from two cuBLAS symbols (5,880 calls each) replaced by the 11,760 quantized projection kernels; other GEMMs remain separate.',
            'Frozen repair recipes enable auxiliary norm/rotary/calibrated/codec fusions; baseline recipes retain original flags. Full-request table compares existing implementations, not only arithmetic types.',
            'Convolution/layout bucket also includes im2col/layout kernels; it is not a fully attributed VAE stage.',
            'No DRAM bandwidth/SM occupancy hardware counters, no claim that Tensor Cores are saturated.'])
    lines=['# Thor：FP4 速度差距的同条件解释','',
        '本报告将五轮无探针完整请求、单独 Nsight 请求、算子微基准分别列账；不做新的视频质量评分。','',
        '## 1. 同输入与生成设置的完整请求','',
        'Edge，320×192、61 帧、35 步、CFG 6；Transformer Graph、debug off、batch=1。每方法五轮、每轮两个输入各三个种子，共 30 次。',
        'MAXN，因无免密 root 未修改频率；各轮方法顺序固定随机，逐请求检查其他计算进程。所有基线保持原实现，没有另行调优。','',
        '| 方法 | 中位秒 | P90 秒 | FPS | 相对 BF16 | 相对 FP8 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary['rows']:
        lines.append(f"| {r['label']} | {r['median_seconds']:.3f} | {r['p90_seconds']:.3f} | {r['fps']:.3f} | {r['speedup_vs_bf16']:.3f}× | {r['speedup_vs_fp8']:.3f}× |")
    lines+=['','普通 FP4、FP8、BF16 的外围执行开关一致。修复版沿用已部署的 norm、RoPE、calibrated pointwise 和 codec Graph 优化；',
        '因此修复版这一表是现有实现的对比，不是仅改变精度格式的消融。不能将这些外围收益归因于 FP4 GEMM。',
        'FPS = 61 / 请求秒数，包含条件帧；文件序列化不计时。仅比较速度，本轮不重测视频质量。']
    if control:
        lines+=['','独立补测将上述外围优化一同关闭，保留相同 Transformer Graph、量化资产、修复计划和在线输入。',
            '以下为四个配置随机顺序、每配置六请求的一轮归因实验，不能替代上面的五轮统计。','',
            '| 修复预算 | 完整实现秒 | 外围开关与基线一致秒 | 外围优化延迟下降 | 六请求输出哈希一致 |',
            '|---|---:|---:|---:|---|']
        for c in control['pairs']:
            lines.append(f"| {c['budget']} | {c['production_median_seconds']:.3f} | {c['common_median_seconds']:.3f} | {c['auxiliary_fusion_latency_reduction']:.1%} | {c['all_output_hashes_equal']} |")
    else:
        lines+=['','未载入外围开关消融结果；修复表中的实现收益不能视为严格配平的纯精度收益。']
    lines+=['','相对延迟差按同轮、同输入、同种子配对，再以轮为 block 做 bootstrap。负值代表更快。样本仅五轮，区间只作本机这两个场景的描述。','',
        '| 方法 | 对照 | 平均延迟差秒 | 配对轮 bootstrap 95% 区间 |',
        '|---|---|---:|---|']
    for name,refs in paired.items():
        for ref,d in refs.items():
            lo,hi=d['paired_round_bootstrap95_seconds']
            lines.append(f"| {bench[name]['label']} | {ref} | {d['mean_latency_difference_seconds']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines+=['','## 2. 实际 GPU 工作构成','',
        '以下为一个相同 PAI-Bench 请求的 Nsight kernel 累计秒数。它含 profiler 影响，不能与上表相减得出 CPU 时间，也不等于 Tensor Core 利用率。',
        'FP4 attention 桶包含 QKV packing、per-head 选择、流式 QK—softmax—P 量化—PV 及 combine；选择 kernel 中的 MMA 不单列为 CUTLASS 投影。',
        'BF16 生成投影由两个 cuBLAS `nvjet_sm110_tst_*` 符号推断，各 5,880 次；量化组恰好替换为 11,760 次 CUTLASS GEMM。',
        '其他共享 nvjet/GEMM 符号仍列入其他显式 GEMM；不把 VAE 的 implicit GEMM 计为可量化生成投影。','',
        '| 部分 | BF16 | FP8 | 普通 FP4 | 1.5× | 2× | FP4＋BF16 attn |',
        '|---|---:|---:|---:|---:|---:|---:|']
    groups=[('generation_projection_gemm','生成投影 GEMM'),
        ('activation_pack_and_select','投影 packing／选择／融合 pointwise'),
        ('fp4_attention_including_pack','FP4 attention 全路径'),
        ('generation_bf16_attention','BF16 FlashAttention'),
        ('other_explicit_gemm','其他显式 GEMM'),
        ('convolution_and_layout','卷积及相关布局'),('other_copy_layout','其他拷贝／布局'),
        ('other_pointwise_norm_reduce','其他逐元素／归一化／归约'),
        ('ambiguous_selector','同名 selector，未强行归属'),('other_unclassified','其余未分类')]
    order=['bf16','fp8','fp4','repair15','repair20','fp4_bf16_attention']
    for group,label in groups:
        values=[]
        for name in order:
            values.append(gpu[name]['groups'].get(group,{}).get('seconds',0))
        lines.append('| '+label+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['',f"生成投影 GEMM 在 BF16 的 kernel 累计时间中占 **{gemm_share:.1%}**。若只把这部分变成 4×、其他部分保持不变，",
        f"累计 kernel 时间的理想加速仅 **{findings['ideal_kernel_sum_speedup_if_only_generation_gemm_becomes_4x']:.3f}×**。这是本次 trace 的 Amdahl 示意，不是无探针请求的硬上限。",
        f"普通 FP4 生成投影 GEMM 已比 BF16 快 {findings['plain_fp4_projection_gemm_speedup_vs_bf16']:.2f}×、比 FP8 快 {findings['plain_fp4_projection_gemm_speedup_vs_fp8']:.2f}×；",
        f"但 FP4 attention 全路径多用约 {findings['plain_fp4_attention_extra_kernel_seconds']:.3f} 秒 kernel 时间。卷积/布局/归一化等仍不在当前 FP4 生成投影范围内。",
        '投影 packing 桶含融合激活函数；FP4 桶的绝对时间没有超过 FP8，不能声称此次差距主要来自 FP4 量化更慢。',
        '修复版的 packing、残差和 attention 确实增加了工作；同时外围融合减少了 copy/pointwise。各桶变化不等于单独优化的因果收益。']
    lines+=['','## 3. 预算与峰值的正确口径','',
        'NVIDIA 技术文档表 2 给出 T5000：稠密 FP4 1035、稠密 FP8 517、稠密 FP16 258 TFLOPS。2070 是稀疏 FP4，不能作为当前稠密主项＋软件 tile 修复的吞吐分母。',
        '来源：[NVIDIA Jetson Thor Technical Brief，表 2](https://dam-cdn.nvd.orangelogic.com/AssetLink/v686u306w26ftuk863xjeimilimtx55e.pdf)。',
        '即使某段 GEMM 有 4× 理论吞吐，若修复执行 r 倍工作，该段理想计算加速也只有 4/r；相对 FP8 的相同 GEMM 范围则约为 2/r。完整请求还受量化、访存、attention 非线性和未量化部分限制。',
        '这里的 r 应按组件分别计算，不能把整请求预算直接套给每个算子。','',
        '| 预算 | 场景 | 投影 issued FLOPs/C0 | attention issued FLOPs/C0 |',
        '|---|---|---:|---:|']
    for name in ('repair15','repair20'):
        for case,cost in bench[name]['cost_by_case'].items():
            c=cost['components']
            lines.append(f"| {bench[name]['label']} | {case} | {c['linear']['ratio']:.3f} | {c['attention']['ratio']:.3f} |")
    lines+=['','计数包含 padding、补项及 selector MMA；它不包括标量指令、搬运、同步等成本，也不是硬件计数器。',
        'Thor 标称带宽 273 GB/s，相对稠密 FP4 峰值对应约 3791 FLOP/byte 的计算／带宽比。以 M960/N9216/K2048 为例，',
        '若两个 FP4 操作数各读取一次（含 K16 scale）、写出一次 BF16 结果，简化算术强度约为 1232 FLOP/byte；再加在线 packing 会增加流量。',
        '因此实际形状的访存和片上复用必须纳入解释。该简化模型未计缓存、压缩和流水，不是已测 DRAM 饱和证据；hot/rotating 对照用于检验缓存敏感性。',
        '## 4. 同形状算子：GEMM 与在线 packing 分开','',
        '使用原 cuBLAS BF16、原 vLLM CUTLASS FP8、当前 CUTLASS FP4 和已安装 vLLM 的官方 NVFP4 API。权重量化离线，输出均 BF16；不调参。',
        'hot 为同一组 buffer 重放，rotating 轮换独立输入/权重；足够大形状覆盖至少两倍 L2，JSON 记录未达到该条件的小形状。微基准不能代替完整模型结果。','',
        '| M/N/K | 缓存 | BF16 ms | FP8 GEMM ms | FP4 v6 GEMM ms | 库 FP4 GEMM ms | FP8 含打包 ms | v6 FP4 含打包 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in probe['rows']:
        v=r['measurements'];keys=['bf16_cublas','fp8_cutlass','fp4_v6','fp4_library','fp8_with_pack','fp4_v6_with_pack']
        lines.append('| '+ '/'.join(map(str,r['shape']))+' | '+r['cache']+' | '+
            ' | '.join(f"{v[k]['median_ms']:.4f}" for k in keys)+' |')
    maximum_errors={k:max(r['checks'][k] for r in probe['rows'])
        for k in ('fp4_decoded_relative_l2','fp8_decoded_relative_l2','official_cutlass_relative_l2')}
    hot={tuple(r['shape']):r['measurements'] for r in probe['rows'] if r['cache']=='hot'}
    down=hot[(960,2048,9216)]
    micros=lambda name:down[name]['median_ms']*1000
    actual_shapes=[(960,1024,2048),(960,2048,2048),(960,9216,2048),(960,2048,9216)]
    library_reductions=[1-hot[shape]['fp4_library']['median_ms']/hot[shape]['fp4_v6']['median_ms'] for shape in actual_shapes]
    report['operator_findings']=dict(maximum_decoded_errors=maximum_errors,
        all_official_bitwise_equal=all(r['checks']['official_bitwise_equal'] for r in probe['rows']),
        library_raw_gemm_latency_reduction_for_m960_hot=library_reductions,
        bf16_reduced_precision_reduction=probe['bf16_reduced_precision_reduction'])
    lines+=['',f"所有形状当前 FP4 与库 NVFP4 输出逐位一致：{report['operator_findings']['all_official_bitwise_equal']}。",
        f"独立解码参考（每形状前 16 行）的最大相对 L2：FP4 {maximum_errors['fp4_decoded_relative_l2']:.6f}，FP8 {maximum_errors['fp8_decoded_relative_l2']:.6f}。",
        f"保留 PyTorch BF16 reduced-precision-reduction 原设置：{probe['bf16_reduced_precision_reduction']}；没有将其改成统一的严格 FP32 accumulation 实验。",
        '这是一组相同尺寸、相同输出类型的现有实现对照，量化定义和库默认累加策略分别记录，不能说所有算术语义相同。',
        f"FFN 下投影 M960/N2048/K9216：FP4 纯 GEMM {micros('fp4_v6'):.1f} µs，比 FP8 {micros('fp8_cutlass'):.1f} µs 快；",
        f"包含在线 packing 后，FP4 为 {micros('fp4_v6_with_pack'):.1f} µs，FP8 为 {micros('fp8_with_pack'):.1f} µs，排序反转。",
        '这是实际测得的形状依赖开销，不是说所有 FP4 packing 都比 FP8 慢。',
        f"同 codes/scales 下，已安装库的 NVFP4 在四个主要 M960 形状的 hot GEMM 延迟比当前固定 launch 低 {min(library_reductions):.1%}–{max(library_reductions):.1%}。",
        '这证明形状调度仍有优化空间，但结果只覆盖无修复 GEMM，不能直接外推到稀疏修复或完整请求。8192 方阵也未达到标称峰值；增大矩阵不保证打满算力。',
        '实际 M960 的投影只有四种主要形状；M129/M416 是尾块/其他工作负载探测，4096/8192 是大矩阵参考，不将后两者当作 Cosmos 请求性能。',
        'hot 仅指相同 buffer 重放，不保证工作集驻留 L2；例如 8192 方阵单份工作集已超过 L2。运行前后未锁频，小形状部分重复有 DVFS 波动，原始三次值完整保留。',
        '标准化口径参考：[CUTLASS Profiler](https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/profiler.html)。本算子表使用已安装库接口和 Graph，并非宣称执行了官方完整 profiler 搜索或 MLPerf。','',
        '## 5. 结论边界与下一步','',
        '优化优先级由本轮已测差距决定：先定位 attention 全路径和在线 packing 的可消除工作，再考虑 GEMM 的形状特化。不能由 nvidia-smi GPU-Util 或使用原生 FP4 指令推断算力已经打满。',
        '优先验证 attention 的 P 量化、V 读取和加载流水，目标是在保留全部 base block 与现有修复支持下消除相对 FlashAttention 的差距。',
        '随后验证 FFN 激活/amax 与 GEMM epilogue 融合，保持原 BF16 舍入；QKV 仅在 prescale、量化与支持语义兼容时共享 packing。',
        '修复主循环下一步可验证残差尺度直接采用 TMA 布局，减少 producer warp 的 cp.async/wait；权重尺度离线重排，当前激活残差在选择后直接写目标布局。这项尚未实现或计入本轮收益。',
        '无修复形状可借鉴库 dispatch，而修复形状需独立资格检查。不能将标准 dense GEMM 的收益直接套给含 CSR/动态 mask 的修复 kernel。',
        '目前代码和产物均限定 sm_110a；原生 FP4 GEMM 实测胜过 FP8，现有证据不支持“主要因为编错架构”的解释。未采集硬件 SM/DRAM 计数器，不能宣布计算已饱和。',
        '修改修复预算或将 FP4 attention 换回 BF16 会改变算法，必须另过质量与成本约束，不作为本轮无损实现优化。','']
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'analysis.json',report);(a.output/'analysis.md').write_text('\n'.join(lines))


if __name__=='__main__':main()

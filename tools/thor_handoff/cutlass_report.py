"""Build matched Thor CUTLASS speed/cost evidence; never infer quality or a ceiling."""
import argparse
import json
from pathlib import Path
import statistics
from build_baselines import sha,write
from attribution_report import percentile


METHODS={'fp4_base':'普通 W4A4','b1.5_protected_n':'修复 1.5×','b2_joint':'修复 2×'}
PREVIOUS={'fp4_base':'fp4-row-major','b1.5_protected_n':'repair15-row-major','b2_joint':'repair20-row-major'}


def read_result(path):
    data=json.loads(path.read_text())
    if not data['complete'] or data['operation']!='measure' or len(data['rows'])!=6:
        raise ValueError(f'Incomplete measurement: {path}')
    keys=[(r['case'],r['seed'],r['steps']) for r in data['rows']]
    if len(set(keys))!=6:raise ValueError('Duplicate measurement')
    for r in data['rows']:
        if r['steps']!=35 or r['calls']!=70 or r['debug']!='off' or r['execution']!='graph' or r.get('profiled'):
            raise ValueError(f'Changed request/timing contract: {path}')
        if r['compute_ledger'] and not r['compute_ledger']['feasible']:raise ValueError('Infeasible budget')
    elapsed=[r['wall_seconds'] for r in data['rows']]
    row=dict(source=str(path),sha256=sha(path),median_seconds=statistics.median(elapsed),
        p90_seconds=percentile(elapsed,.9),fps=61/statistics.median(elapsed),
        allocated_gib=max(r['peak_allocated_bytes'] for r in data['rows'])/2**30,
        reserved_gib=max(r['peak_reserved_bytes'] for r in data['rows'])/2**30,
        prepare_seconds=data['prepare_seconds'],warmup_timings=data['timings'],
        paired_keys=sorted(keys),model_identity=data['runner']['model_identity'],
        asset_identity=data['runner']['asset_identity'],
        variant=data.get('variant'),backend_override=data.get('effective_backend_override'),
        resident_model_state_bytes=data['runner']['resident_model_state_bytes'],
        prepared_weight_bytes=data['runner']['prepared_weight_bytes'],
        per_case={str(i):statistics.median(r['wall_seconds'] for r in data['rows'] if r['case_index']==i) for i in (0,1)})
    ledgers=[r['compute_ledger'] for r in data['rows'] if r['compute_ledger']]
    if ledgers:
        row['compute_ratios']=sorted(set(r['ratio'] for r in ledgers))
        row['max_intermediate_buffer_bytes']=max(r.get('cutlass_fp32_gmem_scratch_bytes',0) for l in ledgers for r in l['rows'])
        row['max_extra_intermediate_traffic_per_request']=max(sum(r.get('cutlass_fp32_intermediate_read_write_bytes',0)*r['calls'] for r in l['rows']) for l in ledgers)
        row['max_gemm_launches_per_request']=max(sum(r.get('gemm_kernel_launches',1)*r['calls'] for r in l['rows'] if r['kind']=='linear') for l in ledgers)
        row['w1_union_bytes']=max(l['w1_support_union_bytes'] for l in ledgers)
    return data,row


def compare(before,after):
    for key in ('fixed_manifest_sha256','effective_config','runtime_options'):
        if before[key]!=after[key]:raise ValueError(f'Unmatched {key}')
    for key in ('model_identity','asset_identity','targets'):
        if before['runner'][key]!=after['runner'][key]:raise ValueError(f'Changed {key}')
    index=lambda d:{(r['case'],r['seed'],r['steps']):r for r in d['rows']}
    b=index(before);a=index(after)
    if b.keys()!=a.keys():raise ValueError('Unpaired requests')
    for key in b:
        bl,al=b[key]['compute_ledger'],a[key]['compute_ledger']
        if bool(bl)!=bool(al):raise ValueError('Changed ledger scope')
        if bl and any(bl[k]!=al[k] for k in ('base_mma','total_mma','ratio','plan_sha256')):
            raise ValueError('Changed repair computation/plan')
    x=statistics.median(r['wall_seconds'] for r in b.values())
    y=statistics.median(r['wall_seconds'] for r in a.values())
    return dict(speedup=x/y,latency_reduction=1-y/x,
        hash_matches=sum(b[k]['sha256']==a[k]['sha256'] for k in b),requests=6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurements',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=dict(complete=False,performance_eligible=False,quality_evaluation=False,methods={},
        limitations=['Two inputs x three seeds; diagnostic sequential measurements, MAXN without locked clocks.',
            'Thor only. No generation-quality scoring. FP32 base/repair accumulation order changes.',
            'FPS counts generated frames including the conditioning frame: 61 / full request seconds.',
            'Full request includes decode and CPU output; model preparation, warmup and file serialization excluded.',
            'Torch allocated/reserved memory is not total unified-memory physical usage.'])
    lines=['# Thor CUTLASS / FP4 attention 加速复测','',
        'Edge，320×192、61 帧、35 步、CFG=6，两个输入各三个固定种子。Graph、debug off。',
        '完整请求包含编码、生成、解码和 CPU 输出；不含加载、建图和视频文件保存。Thor 只评估速度。',
        '所有输入、模型、修复资产、预算和生成投影范围均配对检查。顺序诊断、未锁频，不作为正式速度排行榜。','',
        '| 方法 | 原 v3 秒 | CUTLASS v4 秒 | 联合 v5 秒 | 联合延迟下降 | 联合 FPS |',
        '|---|---:|---:|---:|---:|---:|']
    for method,label in METHODS.items():
        paths=[a.previous/PREVIOUS[method]/'result.json']+[a.measurements/ver/method/'result.json' for ver in ('cutlass-base-v4','cutlass-attention-v5')]
        (old,oldrow),(base,baserow),(joint,jointrow)=[read_result(p) for p in paths]
        gains=dict(cutlass=compare(old,base),attention=compare(base,joint),joint=compare(old,joint))
        report['methods'][method]=dict(old=oldrow,cutlass=baserow,joint=jointrow,comparisons=gains)
        lines.append(f"| {label} | {oldrow['median_seconds']:.3f} | {baserow['median_seconds']:.3f} | {jointrow['median_seconds']:.3f} | {gains['joint']['latency_reduction']:.1%} | {jointrow['fps']:.3f} |")
    lines += ['', 'v4 使用 CUTLASS NVFP4 TMA 基础 GEMM，FP32 中间结果送入原生稀疏 FP4 补项，末尾仅做一次 BF16 转换。',
        'v5 在此基础上将 QK/PV 各组补项连续发射，复用 base/residual 操作数，以一次 commit/wait 完成一组；保留在线 softmax、P 量化、GQA 和全部 base blocks。',
        'v5 的 attention 为本项目的原生 FP4 流式实现。官方 FlashAttention/FlashInfer 的适用范围见下文。',
        '', '| 方法 | v3 allocated GiB | v5 allocated GiB | v5 reserved GiB | v4→v5 哈希一致 |',
        '|---|---:|---:|---:|---:|']
    for method,label in METHODS.items():
        r=report['methods'][method]
        lines.append(f"| {label} | {r['old']['allocated_gib']:.3f} | {r['joint']['allocated_gib']:.3f} | {r['joint']['reserved_gib']:.3f} | {r['comparisons']['attention']['hash_matches']}/6 |")
    fp8path=a.previous/'fp8'/'result.json'
    if fp8path.exists():
        _,fp8=read_result(fp8path);report['historical_fp8_same_inputs']=fp8
        lines+=['',f"历史同条件 FP8 为 {fp8['median_seconds']:.3f} 秒 / {fp8['fps']:.3f} FPS；FP8 使用原生 W8A8 投影及 BF16 attention。不是本轮重新交错测量，不据微小差异宣称稳定胜出。"]
    qualified=json.loads((a.measurements/'qualified.json').read_text())
    attn=json.loads((a.measurements/'attention.json').read_text())
    if not qualified['complete'] or not attn['complete']:raise ValueError('Qualification incomplete')
    report['qualification']=dict(gemm_checks=len(qualified['rows']),sparse_pack_checks=len(qualified['sparse_packing']),
        gemm_max_relative_l2=max(r['relative_l2'] for r in qualified['rows']),
        attention_checks=len(attn['tests']),attention_bitwise=all(r['native_v3_bitwise_equal'] for r in attn['tests']))
    for name in ('memcheck','racecheck','attention_memcheck','attention_racecheck'):
        log=a.measurements/(name+'.log')
        marker='RACECHECK SUMMARY: 0 hazards' if 'racecheck' in name else 'ERROR SUMMARY: 0 errors'
        if marker not in log.read_text():raise ValueError(f'Sanitizer failed: {name}')
    report['attention_operator_diagnostics']={}
    for name in ('attention_act2','attention_bench'):
        path=a.measurements/(name+'.json')
        if not path.exists():continue
        data=json.loads(path.read_text())
        if not data['complete'] or not all(r['native_v3_bitwise_equal'] for r in data['tests']):
            raise ValueError(f'Incomplete additional attention check: {name}')
        report['attention_operator_diagnostics'][name]=dict(source=str(path),sha256=sha(path),
            terms=data['terms'],tests=data['tests'])
    if report['attention_operator_diagnostics']:
        lines+=['','代表性 attention 算子包含 packing、在线选择、softmax、P 量化和 combine；Graph 下诊断计时，单位 ms。',
            '形状 M960/N1504，Q heads=16、KV heads=8。与前述完整请求计时独立。','',
            '| 补项方案 | 配额 | v3 ms | 分组异步 ms | 降幅 |','|---|---|---:|---:|---:|']
        for name,d in report['attention_operator_diagnostics'].items():
            for r in d['tests']:
                if r['shape']==[960,1504,16,8] and r['mode'] in ('partial','full'):
                    x,y=r['native_full_attention_ms'],r['tested_full_attention_ms']
                    lines.append(f"| {d['terms']} 项 | {r['mode']} | {x:.4f} | {y:.4f} | {1-y/x:.1%} |")
    for name in ('act2_memcheck','act2_racecheck'):
        path=a.measurements/(name+'.json')
        if path.exists():
            d=json.loads(path.read_text());marker='RACECHECK SUMMARY: 0 hazards' if 'racecheck' in name else 'ERROR SUMMARY: 0 errors'
            if not d['complete'] or marker not in path.with_suffix('.log').read_text():raise ValueError('ACT2 sanitizer incomplete')
            report['qualification'][name]='passed'
    lines+=['',f"GEMM {len(qualified['rows'])} 项及稀疏 packing {len(qualified['sparse_packing'])} 项通过，最大算子相对 L2 为 {report['qualification']['gemm_max_relative_l2']:.3g}。",
        f"Attention {len(attn['tests'])} 项与 v3 逐位一致；GEMM、attention 均通过 memcheck/racecheck。覆盖 Graph 当前输入/修复 mask、GQA、尾块、空/全修复。",
        'CUTLASS 改变部分 FP32 累加顺序，完整生成质量未重测；数值检查不等于质量门槛通过。',
        '补项仍使用原生 FP4，未修改支持或预算。中间 FP32 写回/读取、常驻尺度副本和实际显存均列入 JSON；尚未实现单内核 CUTLASS 稀疏修复融合。', '',
        '官方支持核对（2026-09-22）：FlashInfer NVFP4 attention 标注 SM120/121；FlashAttention CuTe 的 dtype 入口尚无 E2M1。Thor SM110 不能直接接入这些官方 FP4 attention 接口。',
        '来源：[FlashInfer](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/nvfp4_attention_sm120.py)、[FlashAttention](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py)。','']
    report['complete']=True;a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'summary.json',report);(a.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:v['comparisons'] for k,v in report['methods'].items()},indent=2))


if __name__=='__main__':main()

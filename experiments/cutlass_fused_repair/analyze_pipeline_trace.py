"""Classify CUPTI kernels without summing CPU and GPU clocks or splitting fused work.

Stage ownership uses GPU annotation intervals. Kernel durations are instrumented
active-time samples, not uninstrumented wall latency. Ambiguous ATen elementwise
kernels remain grouped rather than falsely attributed to a particular module.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def category(name):
    if 'cudnn' in name.lower():
        if any(s in name for s in ('nchwToNhwc', 'nhwcToNchw')): return 'Layout / casts / copies'
        return 'VAE convolution'
    if 'GemmUniversal' in name or name == 'gemm': return 'FP4 GEMM incl. fused repair'
    if any(s in name for s in ('flash_fwd', 'fmha_', 'streaming_attention')) or name == 'combine':
        return 'Attention incl. softmax / repair'
    if 'gemm' in name.lower() or 'splitKreduce' in name: return 'BF16 / FP32 GEMM + reduction'
    if name in ('scaled_amax', 'finish_amax', 'pack_activation', 'pack_kernel', 'pack_qk', 'pack_v',
                'residual_only', 'residual_qk', 'residual_v', '_swizzle'):
        return 'Activation quantization / residual packing'
    if name in ('choose', 'select_tiles_head', 'support_union', 'base_summary'):
        return 'Repair selection / support'
    if any(s in name for s in ('pointwise_amax', 'silu_kernel', 'relu', 'launch_clamp',
                               'pow_tensor_scalar_kernel_impl<c10::BFloat16')):
        return 'Nonlinear activation / clipping'
    if any(s in name for s in ('MeanOps', 'NormTwoOps', 'rsqrt_kernel',
                               'pow_tensor_scalar_kernel_impl<float', 'norm_square', 'norm_finish')):
        return 'Normalization core'
    if any(s in name.lower() for s in ('copy', 'memcpy', 'catarray', 'fillfunctor', 'upsample', 'transpose')):
        return 'Layout / casts / copies'
    if 'elementwise' in name or 'reduce_kernel' in name or name == 'rope_kernel':
        return 'Other elementwise / reductions (incl. RoPE / residual)'
    return 'Other kernels'


def union_us(events):
    end = None; total = 0.
    for e in sorted(events, key=lambda e:e['ts']):
        lo, hi = e['ts'], e['ts'] + e['dur']
        total += hi - lo if end is None else max(0., hi - max(lo, end))
        end = hi if end is None else max(end, hi)
    return total


def analyze(path):
    events = json.loads(path.read_text())['traceEvents']
    spans = sorted((e for e in events if e.get('cat') == 'gpu_user_annotation'
                    and e['name'] != 'TW/request'), key=lambda e:(e['ts'], -e['dur']))
    starts = [e['ts'] for e in spans]
    kernels = [e for e in events if e.get('cat') == 'kernel']
    device_events = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    by_name = defaultdict(lambda: [0, 0.])
    grouped = defaultdict(lambda: [0, 0.]); stages = defaultdict(list)
    for e in kernels:
        i = bisect.bisect_right(starts, e['ts']) - 1
        # Walk out of nested annotations, e.g. AR prefill inside Transformer.
        while i >= 0 and e['ts'] + e['dur'] > spans[i]['ts'] + spans[i]['dur'] + 1:
            i -= 1
        span = spans[i] if i >= 0 else None
        stage = span['name'].removeprefix('TW/') if span else 'request_other'
        cat = category(e['name']); key = (stage, cat)
        grouped[key][0] += 1; grouped[key][1] += e['dur'] / 1000
        by_name[stage, cat, e['name']][0] += 1; by_name[stage, cat, e['name']][1] += e['dur'] / 1000
        stages[stage].append(e)
    span_by_stage = Counter()
    for e in spans: span_by_stage[e['name'].removeprefix('TW/')] += e['dur'] / 1000
    runtime = Counter(e['name'] for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver'))
    result = dict(complete=True, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='One instrumented full request. Category times sum kernel durations, not CPU time or wall latency. GPU interval unions include overlaps once. No tensor-core utilization or saturation claim.',
        request_cpu_annotation_ms=next(e['dur']/1000 for e in events if e.get('cat')=='user_annotation' and e['name']=='TW/request'),
        kernel_count=len(kernels), kernel_sum_ms=sum(e['dur'] for e in kernels)/1000,
        kernel_union_ms=union_us(kernels)/1000, device_activity_union_ms=union_us(device_events)/1000,
        gpu_span_ms=(max(e['ts']+e['dur'] for e in device_events)-min(e['ts'] for e in device_events))/1000,
        stages=[dict(stage=s,kernels=len(es),kernel_ms=sum(e['dur'] for e in es)/1000,
                     kernel_union_ms=union_us(es)/1000,gpu_annotation_span_ms=span_by_stage.get(s)) for s,es in stages.items()],
        categories=[dict(stage=s,category=c,count=v[0],kernel_ms=v[1]) for (s,c),v in grouped.items()],
        kernels=[dict(stage=s,category=c,name=n,count=v[0],kernel_ms=v[1]) for (s,c,n),v in sorted(by_name.items(),key=lambda kv:-kv[1][1])],
        cuda_api_counts=dict(runtime))
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(15,6),layout='constrained')
    stages = ['transformer','video_encode','video_decode','request_other']
    if any(r['stage']=='ar_prefill' for r in result['stages']):stages.insert(0,'ar_prefill')
    cats = list(dict.fromkeys(r['category'] for r in result['categories']))
    totals = Counter()
    for r in result['categories']: totals[r['category']] += r['kernel_ms']
    cats.sort(key=lambda c:-totals[c]); cmap=plt.get_cmap('tab20')
    for idx,cat in enumerate(cats):
        left=[sum(r['kernel_ms'] for r in result['categories'] if r['stage']==s and r['category'] in cats[:idx]) for s in stages]
        vals=[sum(r['kernel_ms'] for r in result['categories'] if r['stage']==s and r['category']==cat) for s in stages]
        axes[0].barh(stages,vals,left=left,color=cmap(idx),label=cat)
    axes[0].invert_yaxis();axes[0].set_xlabel('Summed GPU kernel time (ms, instrumented)')
    axes[0].set_title('Where GPU work runs')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=8,ncol=1)
    axes[1].barh(cats,[totals[c] for c in cats],color=[cmap(i) for i in range(len(cats))])
    for i,c in enumerate(cats):axes[1].text(totals[c]+3,i,f'{totals[c]:.1f} ms',va='center',fontsize=9)
    axes[1].set_xlim(0,max(totals.values())*1.25);axes[1].invert_yaxis()
    axes[1].set_title('What the kernels do');axes[1].set_xlabel('Summed GPU kernel time (ms)')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('RTX 5090 · Cosmos3-Edge · Joint FP4 budget 1.25\n192×320 · 61 frames · 35 steps · seed 7103',fontsize=13)
    for ext in ('png','svg'):fig.savefig(output/f'pipeline_breakdown.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);result=analyze(a.trace)
    (a.output/'breakdown.json').write_text(json.dumps(result,indent=2)+'\n');plot(result,a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('kernels','cuda_api_counts')},indent=2))


if __name__=='__main__':main()

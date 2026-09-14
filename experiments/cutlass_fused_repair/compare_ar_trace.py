"""Compare instrumented full-request work before and after request-local AR cache."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);old=json.loads(a.before.read_text());new=json.loads(a.after.read_text())
    def totals(result):
        t=Counter();c=Counter()
        for r in result['categories']:t[r['category']]+=r['kernel_ms'];c[r['category']]+=r['count']
        return t,c
    assert any(r['stage']=='ar_prefill' and r['kernels']>0 for r in new['stages'])
    assert new['cuda_api_counts']['cudaGraphLaunch']==72
    ot,oc=totals(old);nt,nc=totals(new)
    rows=[dict(category=k,before_ms=ot[k],after_ms=nt[k],before_count=oc[k],after_count=nc[k]) for k in ot]
    assert oc['FP4 GEMM incl. fused repair']==nc['FP4 GEMM incl. fused repair']==11760
    result=dict(complete=True,scope='Separate CUPTI diagnostic samples; kernel counts and instrumented durations, not formal end-to-end timing or a utilization measurement.',
        rows=rows,before_stages=old['stages'],after_stages=new['stages'],
        before_kernel_count=old['kernel_count'],after_kernel_count=new['kernel_count'],
        before_graph_launches=old['cuda_api_counts']['cudaGraphLaunch'],after_graph_launches=new['cuda_api_counts']['cudaGraphLaunch'])
    (a.output/'trace_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'FP4 GEMM incl. fused repair':'FP4 GEMM + repair','BF16 / FP32 GEMM + reduction':'BF16 / FP32 GEMM + reduction',
        'VAE convolution':'VAE convolution','Layout / casts / copies':'Layout / casts / copies',
        'Other elementwise / reductions (incl. RoPE / residual)':'Other elementwise / reductions',
        'Activation quantization / residual packing':'Activation quantization / packing',
        'Attention incl. softmax / repair':'Attention / softmax / repair','Normalization core':'Normalization core',
        'Nonlinear activation / clipping':'Nonlinear activation','Repair selection / support':'Repair selection','Other kernels':'Other'}
    ordered=sorted(rows,key=lambda r:-r['before_ms']);fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    for i,r in enumerate(ordered):
        ax.barh(i-.18,r['before_ms'],height=.35,color='#94a3b8',label='Original' if i==0 else None)
        ax.barh(i+.18,r['after_ms'],height=.35,color='#2563eb',label='AR cache (prefill included)' if i==0 else None)
    ax.set_yticks(range(len(ordered)),[labels[r['category']] for r in ordered]);ax.invert_yaxis()
    ax.set_xlabel('Summed GPU kernel time, ms (instrumented)');ax.legend();ax.spines[['top','right']].set_visible(False)
    ax.set_title('RTX 5090 · Edge FP4 budget 1.25 · Where AR caching removes work')
    for ext in ('png','svg'):fig.savefig(a.output/f'ar_kernel_breakdown.{ext}',dpi=180,bbox_inches='tight')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

"""Reviewable steady-state launch tuning and independent hardware counters."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmarks',type=Path,nargs='+',required=True)
    p.add_argument('--hardware',type=Path,required=True);p.add_argument('--shared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[];verified=0
    for source in a.benchmarks:
        r=json.loads(source.read_text());assert r['complete']
        for x in r['rows']:
            assert all(c['relative_l2']==0 for c in x['candidates']);verified+=len(x['candidates'])
            rows.append(dict(model=r['model'],shape=x['shape'],tier=x['tier'],
                original_us=x['original_ms']*1000,optimized_us=x['optimized_ms']*1000,
                tuning_speedup=x['original_ms']/x['optimized_ms'],bf16_speedup=x['bf16_ms']/x['optimized_ms'],
                operator_tflops=x['achieved_tflops_operator'],packed_gemm_tflops=x['achieved_tflops_gemm'],
                bm=x['winner']['bm'],stages=x['winner']['num_stages'],select_us=x['select_ms']*1000,pack_us=x['pack_ms']*1000))
    with (a.output/'kernels.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hardware=json.loads(a.hardware.read_text());assert hardware['complete'];profiles=[]
    for p in hardware['profiles']:
        ks=p['kernels'];gemm=[k for k in ks if 'gemm' in k['kernel']];assert len(gemm)==1;g=gemm[0]
        profiles.append(dict(tag=p['tag'],tensor_activity=g['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'],
            occupancy=g['sm__warps_active.avg.pct_of_peak_sustained_active'],
            non_gemm_fraction=1-g['duration_ns']/sum(k['duration_ns'] for k in ks)))
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained');ys=np.arange(len(profiles))
    labels=[x['tag'].replace('_',' / ') for x in profiles]
    for ax,key,title in zip(axs,['tensor_activity','occupancy'],['Tensor pipe elapsed-cycle activity','Active warp occupancy']):
        vals=[x[key] for x in profiles];ax.barh(ys,vals,color='#2563eb');ax.set(yticks=ys,yticklabels=labels,xlim=(0,100),xlabel='Percent',title=title)
        for y,v in zip(ys,vals):ax.text(v+1,y,f'{v:.1f}%',va='center',fontsize=9)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('RTX 5090: representative dynamic/full-repair down projections\nNsight counters; tensor activity is not percentage of theoretical FP4 FLOP/s')
    fig.savefig(a.output/'tensor_core_activity.png',dpi=160);plt.close(fig)
    shared=json.loads(a.shared.read_text());assert shared['complete'] and all(x['bitwise_equal'] for x in shared['rows'])
    result=dict(verified_launch_candidates=verified,kernel_configurations=len(rows),profiles=profiles,
                shared_pack_speedup_range=[min(x['speedup'] for x in shared['rows']),max(x['speedup'] for x in shared['rows'])],
                max_tuning_speedup=max(x['tuning_speedup'] for x in rows),
                operator_tflops_range=[min(x['operator_tflops'] for x in rows),max(x['operator_tflops'] for x in rows)])
    (a.output/'kernel_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

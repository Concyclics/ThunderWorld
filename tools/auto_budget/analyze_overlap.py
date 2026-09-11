"""Inspect CUPTI kernel intervals, separately from unprofiled speed measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();rows=[];shown=[]
    for tier in ['a50','act2']:
        path=a.directory/f'torch_trace_cases.{tier}.chrome.json';trace=json.loads(path.read_text())['traceEvents']
        for mode in ['serial','parallel']:
            ranges=[e for e in trace if e.get('cat')=='user_annotation' and e['name']==f'overlap_probe/{tier}/{mode}'];assert len(ranges)==1
            span=ranges[0];kernels=[e for e in trace if e.get('cat')=='kernel' and span['ts']<=e['ts'] and e['ts']+e['dur']<=span['ts']+span['dur']]
            groups={}
            for k in kernels:groups.setdefault(k['args']['correlation'],[]).append(k)
            assert len(groups)==5;details=[]
            for group in groups.values():
                gs=[e for e in group if e['name']=='gemm'];assert len(gs)==1;g=gs[0]
                producers=[e for e in group if e['name']!='gemm'];assert len(producers)==(4 if tier=='a50' else 3)
                overlap=sum(max(0,min(g['ts']+g['dur'],e['ts']+e['dur'])-max(g['ts'],e['ts'])) for e in producers)
                start=min(e['ts'] for e in group);end=max(e['ts']+e['dur'] for e in group)
                details.append(dict(gpu_span_us=end-start,gemm_us=g['dur'],producer_us=sum(e['dur'] for e in producers),overlap_us=overlap,
                    kernels=[dict(name=e['name'],start_us=e['ts']-start,duration_us=e['dur'],stream=e['args']['stream']) for e in group]))
            if mode=='serial':assert all(e['overlap_us']==0 for e in details)
            rows.append(dict(tier=tier,mode=mode,graphs=5,overlap_us_total=sum(e['overlap_us'] for e in details),
                producer_us_total=sum(e['producer_us'] for e in details),median_gemm_us=statistics.median(e['gemm_us'] for e in details),details=details))
            shown.append((tier,mode,sorted(details,key=lambda d:d['gpu_span_us'])[2]))
    fig,axs=plt.subplots(2,2,figsize=(12,5),layout='constrained');colors={'gemm':'#2563eb','select_tiles':'#a855f7','scaled_amax':'#f59e0b','finish_amax':'#dc2626','pack_activation':'#16a34a'}
    for ax,(tier,mode,row) in zip(axs.flat,shown):
        for e in row['kernels']:
            ax.broken_barh([(e['start_us'],e['duration_us'])],((0 if e['name']=='gemm' else 1)-.25,.5),facecolors=colors[e['name']])
        ax.set(title=f'Edge M316 {tier} / {mode}; overlap {row["overlap_us"]:.2f} us',yticks=[0,1],yticklabels=['TC GEMM','SIMT producer'],xlabel='GPU time from first kernel (us)',xlim=(0,100),ylim=(-.6,1.6));ax.grid(axis='x',alpha=.2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=v,label=k) for k,v in colors.items()],loc='outside lower center',ncol=5,fontsize=8)
    fig.suptitle('Independent-input concurrency: CUPTI timelines\nMedian GPU-span replay of five per configuration; profiler timings are not the normal speed benchmark')
    fig.savefig(a.directory/'overlap_timeline.png',dpi=160);plt.close(fig)
    out=dict(passed=True,scope='Kernel-time overlap of independent tasks, not per-SM co-residency or production speed. CUDA Graph fork/join, five traced replays per case.',rows=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.directory/'trace_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in rows:print(r['tier'],r['mode'],'overlap us',round(r['overlap_us_total'],3),'producer us',round(r['producer_us_total'],3))


if __name__=='__main__':main()

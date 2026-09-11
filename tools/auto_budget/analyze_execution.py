"""Validate graph/fusion trajectories and keep framework/kernel gains separate."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def geo(values):return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[2]
    paired=[];summaries=[];local=[];pipe=[];tuned=[]
    for model in ['edge','nano']:
        reports={v:json.loads((root/f'runs5090/execution_{model}_{v}/result.json').read_text()) for v in ['eager','graph','fused']}
        assert all(r['complete'] for r in reports.values())
        assert all(r['model_revision']==reports['eager']['model_revision'] and r['budget_policy']==reports['eager']['budget_policy'] and r['cases']==reports['eager']['cases'] for r in reports.values())
        for source in ['tools/finding_resample/run.py','tools/auto_budget/dynamic.py','tools/auto_budget/fusion.py','tools/auto_budget/phase_graph.py','tools/cosmos_e2e/graph.py','experiments/ffn_block_repair_20260907/native.py']:
            assert len({r['source_sha256'][source] for r in reports.values()})==1,source
        def key(x):return x['mode'],x['case'],x['seed'],x['repeat']
        banks={v:{key(x):x for x in r['rows']} for v,r in reports.items()};assert all(set(b)==set(banks['eager']) for b in banks.values())
        tasks={c['id']:c['task'] for c in reports['eager']['cases']}
        for k,e in banks['eager'].items():
            g,f=banks['graph'][k],banks['fused'][k]
            assert e['sha256']==g['sha256']==f['sha256'],(model,k,'changed output')
            if k[0]!='bf16':assert e['budget_actual']==g['budget_actual']==f['budget_actual'],(model,k,'changed budget')
            branches=1 if tasks[k[1]]=='policy' else 2
            for row in [g,f]:
                expected={f'{b}:{ph}':n for b in range(branches) for ph,n in enumerate([7,11,10,7])} if k[0].startswith('ffn_auto') else {f'{b}:0':35 for b in range(branches)}
                assert row['phase_graph']['phase_hits']==expected and row['phase_graph']['replays']==35*branches
                assert row['phase_graph']['captures']==branches*(4 if k[0].startswith('ffn_auto') else 1)
            paired.append(dict(model=model,mode=k[0],case=k[1],seed=k[2],repeat=k[3],bitwise_equal=True,
                eager_seconds=e['wall_seconds'],graph_seconds=g['wall_seconds'],fused_seconds=f['wall_seconds'],
                eager_transformer_seconds=e['transformer_seconds'],graph_transformer_seconds=g['transformer_seconds'],fused_transformer_seconds=f['transformer_seconds'],
                graph_speedup=e['wall_seconds']/g['wall_seconds'],fusion_speedup=g['wall_seconds']/f['wall_seconds'],
                combined_speedup=e['wall_seconds']/f['wall_seconds'],
                eager_peak_gib=e['peak_allocated_bytes']/2**30,graph_peak_gib=g['peak_allocated_bytes']/2**30,fused_peak_gib=f['peak_allocated_bytes']/2**30,
                fused_speedup_bf16=banks['fused']['bf16',*k[1:]]['wall_seconds']/f['wall_seconds'],
                ssim_bf16=f.get('vs_bf16',{}).get('video',{}).get('ssim_mean',1.)))
        for case in tasks:
            for mode in reports['eager']['args']['modes']:
                xs=[x for x in paired if x['model']==model and x['case']==case and x['mode']==mode]
                summaries.append(dict(model=model,case=case,mode=mode,n=len(xs),graph_speedup=geo(x['graph_speedup'] for x in xs),
                    fusion_speedup=geo(x['fusion_speedup'] for x in xs),combined_speedup=geo(x['combined_speedup'] for x in xs),
                    fused_speedup_bf16=geo(x['fused_speedup_bf16'] for x in xs),ssim_mean=statistics.mean(x['ssim_bf16'] for x in xs),
                    **{v+'_seconds':statistics.mean(x[v+'_seconds'] for x in xs) for v in ['eager','graph','fused']},
                    **{v+'_transformer_seconds':statistics.mean(x[v+'_transformer_seconds'] for x in xs) for v in ['eager','graph','fused']},
                    **{v+'_peak_gib':max(x[v+'_peak_gib'] for x in xs) for v in ['eager','graph','fused']}))
        fig,axs=plt.subplots(1,len(tasks),figsize=(5*len(tasks),4),squeeze=False,layout='constrained')
        modes=reports['eager']['args']['modes']
        for ax,case in zip(axs.flat,tasks):
            for i,v in enumerate(['eager','graph','fused']):
                ys=[];lo=[];hi=[]
                for mode in modes:
                    values=[x['wall_seconds'] for x in reports[v]['rows'] if x['case']==case and x['mode']==mode]
                    mean=statistics.mean(values);ys.append(mean);lo.append(mean-min(values));hi.append(max(values)-mean)
                ax.bar([j+(i-1)*.24 for j in range(len(modes))],ys,width=.23,label=v,yerr=[lo,hi],capsize=2)
            ax.set(title=tasks[case],xticks=range(len(modes)),xticklabels=[m.removeprefix('ffn_') for m in modes],ylabel='Generator seconds',ylim=(0,None));ax.grid(axis='y',alpha=.2)
        axs[0,-1].legend(fontsize=8);fig.suptitle(model+': same outputs and budgets, 35 steps\nTwo seeds; bars are means with observed ranges, not confidence intervals')
        fig.savefig(a.output/f'{model}_execution.png',dpi=160);plt.close(fig)
        bench=json.loads((root/f'runs5090/graph_fusion_{model}_bench/result.json').read_text());assert bench['complete']
        for x in bench['ffn']:
            assert x['bitwise_equal'];local.append(dict(model=model,shape=x['shape'],tier=x['tier'],speedup=x['speedup'],
                original_us=x['times']['original']['ms']*1000,fused_us=x['times']['fused']['ms']*1000))
        for x in bench['pipelines']:
            assert x['bitwise_equal']
            for c in ['128','256','512']:pipe.append(dict(model=model,shape=x['shape'],tier=x['tier'],chunk=int(c),num_chunks=math.ceil(int(x['shape'].split(':')[0])/int(c)),speedup=x['times']['original']['ms']/x['times'][c]['ms']))
        retuned=json.loads((root/f'runs5090/graph_fusion_{model}_chunk_tuning/result.json').read_text());assert retuned['complete']
        for x in retuned['rows']:
            assert x['bitwise_equal'] and len(x['trials'])==6 and all(t['bitwise_equal'] for t in x['trials'])
            tuned.append(dict(model=model,shape=x['shape'],tier=x['tier'],chunk=x['chunk'],num_chunks=math.ceil(int(x['shape'].split(':')[0])/x['chunk']),bm=x['best']['bm'],num_stages=x['best']['num_stages'],
                speedup=x['speedup'],original_us=x['times']['original']['ms']*1000,tuned_us=x['times']['tuned']['ms']*1000))
    for name,rows in [('paired.csv',paired),('summary.csv',summaries),('fusion_ffn.csv',local),('double_buffer.csv',pipe),('tuned_double_buffer.csv',tuned)]:
        with (a.output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(passed=True,paired_trajectories=len(paired),total_trajectories=len(paired)*3,
                local_ffn_cases=len(local),double_buffer_cases=len(pipe),
                retuned_double_buffer_cases=len(tuned),retuned_launch_candidates=len(tuned)*6,
                retuned_speedup_range=[min(x['speedup'] for x in tuned),max(x['speedup'] for x in tuned)],
                retuned_multichunk_speedup_range=[min(x['speedup'] for x in tuned if x['num_chunks']>1),max(x['speedup'] for x in tuned if x['num_chunks']>1)],
                retuned_multichunk_winners=sum(x['speedup']>1 for x in tuned if x['num_chunks']>1),
                same_outputs_and_budgets=True,phase_boundaries_verified=True,
                graph_speedup_range=[min(x['graph_speedup'] for x in summaries),max(x['graph_speedup'] for x in summaries)],
                fp4_graph_speedup_range=[min(x['graph_speedup'] for x in summaries if x['mode']!='bf16'),max(x['graph_speedup'] for x in summaries if x['mode']!='bf16')],
                fp4_fusion_speedup_range=[min(x['fusion_speedup'] for x in summaries if x['mode']!='bf16'),max(x['fusion_speedup'] for x in summaries if x['mode']!='bf16')],
                fusion_speedup_range=[min(x['fusion_speedup'] for x in summaries),max(x['fusion_speedup'] for x in summaries)])
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

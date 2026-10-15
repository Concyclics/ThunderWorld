"""Audit and summarize the joint_e2e experiment without reusing panel timings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def read(path):return json.loads(path.read_text())


def trajectory(root, cache):
    import torch
    import lpips
    from thunderworld_fp4.cosmos.quality import compare
    torch.set_num_threads(8); torch.hub.set_dir(str(cache))
    metric = lpips.LPIPS(net='alex', version='0.1').eval()
    rows = []; equivalence = []
    for seed in (7103,7104):
        def signals(name):
            with np.load(root/f'trajectory_00_{name}/seed{seed}.npz') as data:return dict(data)
        reference = signals('bf16')
        for budget in ('125','150'):
            old, new = signals('cutlass'+budget), signals('joint'+budget)
            equal = old.keys()==new.keys() and all(old[k].dtype==new[k].dtype and
                old[k].shape==new[k].shape and old[k].tobytes()==new[k].tobytes() for k in old)
            equivalence.append(dict(budget=budget,seed=seed,bitwise_equal=equal))
            for name, output in [('cutlass'+budget,old),('joint'+budget,new)]:
                rows.append(dict(config=name,seed=seed,metrics=compare(output,reference,metric)))
    return dict(complete=True, scope='Two development seeds; BF16 consistency, not a held-out quality gate',
        equivalence=equivalence,rows=rows,
        lpips_state_sha256={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in metric.state_dict().items()})


def diagnostic(root):
    matrix=read(root/'diagnostic_matrix.json')
    if not matrix['complete']:raise ValueError('Incomplete diagnostics')
    rows=[]
    for job in matrix['jobs']:
        result=read(Path(job['output'])/'result.json')
        row=next(r for r in result['rows'] if r['seed']==7103)
        reference=next(r for r in read(root/f"trajectory_00_{job['config']}/result.json")['rows'] if r['seed']==7103)
        if row['performance_eligible'] or row['debug']!='timing':raise ValueError('Expected diagnostic timing')
        if row['sha256']!=reference['sha256']:raise ValueError('Diagnostics changed the output')
        stages=Counter()
        for stage in row['diagnostics']['stages']:stages[stage['name']]+=stage['milliseconds']
        transformer=sum(row['diagnostics']['transformer_step_ms'])
        rows.append(dict(config=job['config'],seed=7103,wall_seconds=row['wall_seconds'],
            transformer_interval_ms=transformer,stage_interval_ms=dict(stages),
            event_sum_minus_host_wall_ms=transformer+sum(stages.values())-row['wall_seconds']*1000,
            additive_decomposition=False,
            bitwise_equal_to_uninstrumented=True))
    return dict(complete=True,rows=rows,scope='Separate diagnostic: seed 7103 after a full seed 6099 warm request. CUDA event intervals can include dispatch gaps and need not sum to the host timer; do not infer a CPU residual, additive stage percentages, or pure kernel active time. Not leaderboard timings.')


def performance(root):
    matrix=read(root/'performance_matrix.json')
    if not matrix['complete']:raise ValueError('Unfinished matrix')
    values=defaultdict(list); rounds=defaultdict(set); hashes={}; by_pair={}; env=None; source=None; common=None
    shapes={}; startup={}; memory={}; budgets={}; configurations={}
    for job in matrix['jobs']:
        result=read(Path(job['output'])/'result.json'); name=job['config']
        if not result['complete'] or result['tuning_missing']:raise ValueError('Incomplete or missing launch profiles')
        if env is not None and result['environment']!=env:raise ValueError('Mixed environments')
        env=result['environment']
        if source is not None and result['source_sha256']!=source:raise ValueError('Mixed source versions')
        source=result['source_sha256']
        request={k:result[k] for k in ('model_identity','case','image_sha256')}
        if common is not None and common!=request:raise ValueError('Unmatched model or request')
        common=request
        if not result['config']['strict_tuning'] or result['config']['execution']!='graph':raise ValueError('Unfrozen or non-Graph execution')
        signature={k:result[k] for k in ('config','asset_identity','cutlass_artifact','observed_shapes')}
        if name in configurations and configurations[name]!=signature:raise ValueError('Configuration, assets, or launches changed across rounds')
        configurations[name]=signature
        if name=='bf16' and (result['config']['precision']!='bf16' or result['config']['attention_backend']!='bf16'):
            raise ValueError('Invalid BF16 control')
        rounds[name].add(job['round']); shapes[name]=result['observed_shapes']
        startup.setdefault(name,[]).append(result['timings'])
        memory[name]={k:result[k] for k in ('prepared_weight_bytes','resident_model_state_bytes','support_buffer_bytes')}
        for row in result['rows']:
            if row['debug']!='off' or not row['performance_eligible'] or row['steps']!=35:
                raise ValueError('Invalid timing scope')
            if row['timing_scope']!='complete_request_v2_including_host_setup':raise ValueError('Incomplete request timer')
            key=(name,row['seed'])
            if key in hashes and hashes[key]!=row['sha256']:raise ValueError('Nondeterministic outputs')
            hashes[key]=row['sha256']; values[name].append(row)
            by_pair[name,job['round'],row['seed']]=row['wall_seconds']
            budgets[name]=row['compute_ledger']
    for name, rows in values.items():
        if len(rows)!=15 or rounds[name]!=set(range(5)):raise ValueError('Expected five rounds and three seeds')
        if len({json.dumps(hashes[name,s],sort_keys=True) for s in (93001,93002,93003)})!=3:
            raise ValueError('Distinct request seeds produced stale or identical outputs')
    for budget in ('125','150'):
        if configurations['joint'+budget]['asset_identity']!=configurations['cutlass'+budget]['asset_identity']:
            raise ValueError('Same-budget asset mismatch')
    summary=[]; rng=np.random.default_rng(120)
    for name,rows in values.items():
        samples=[r['wall_seconds'] for r in rows]
        paired=np.array([statistics.median(by_pair['bf16',r,s]/by_pair[name,r,s] for s in (93001,93002,93003)) for r in range(5)])
        boot=np.median(paired[rng.integers(0,5,(10000,5))],axis=1)
        summary.append(dict(config=name,n=15,median_seconds=statistics.median(samples),
            p90_seconds=float(np.quantile(samples,.9)),
            bf16_speedup=statistics.median(r['wall_seconds'] for r in values['bf16'])/statistics.median(samples),
            paired_round_median_speedup=float(np.median(paired)),paired_round_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),**memory[name],
            dispatch_profiles=dict(Counter(json.dumps([s['backend'],s['launch'],s.get('direct_scale_pack',False)],sort_keys=True) for s in shapes[name]))))
        if name.startswith('joint'):
            anchor='cutlass'+name[-3:]
            paired=np.array([statistics.median(by_pair[anchor,r,s]/by_pair[name,r,s] for s in (93001,93002,93003)) for r in range(5)])
            boot=np.median(paired[rng.integers(0,5,(10000,5))],axis=1)
            summary[-1].update(same_budget_cutlass_speedup=statistics.median(r['wall_seconds'] for r in values[anchor])/statistics.median(samples),
                same_budget_paired_round_speedup=float(np.median(paired)),same_budget_paired_round_ci95=np.quantile(boot,[.025,.975]).tolist())
    equal=[dict(budget=b,seed=s,bitwise_equal=hashes['joint'+b,s]==hashes['cutlass'+b,s])
        for b in ('125','150') for s in (93001,93002,93003)]
    return dict(complete=True,scope='Edge I2V warm complete requests; one input; 35 steps; five isolated rounds x three seeds',
        environment=env,source_sha256=source,request=common,rows=summary,performance_output_equivalence=equal,
        startup_timings=startup,compute=budgets,
        timing_samples=[dict(config=n,round=r,seed=s,wall_seconds=v) for (n,r,s),v in by_pair.items()])


def plot(root,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['bf16','cutlass125','joint125','cutlass150','joint150']
    labels=['BF16','CUTLASS\n1.25','Joint\n1.25','CUTLASS\n1.50','Joint\n1.50']
    rows={r['config']:r for r in result['rows']}
    colors=['#64748b','#93c5fd','#2563eb','#5eead4','#0f766e']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    x=np.arange(5)
    for ax,field,title,unit in [(axes[0],'median_seconds','Complete request latency','seconds'),
                               (axes[1],'peak_allocated_bytes','Peak allocated GPU memory','GiB')]:
        values=[rows[n][field]/(2**30 if field.endswith('bytes') else 1) for n in names]
        ax.bar(x,values,color=colors)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(x,labels);ax.set_title(title);ax.set_ylabel(unit);ax.set_ylim(0,max(values)*1.18)
        ax.spines[['top','right']].set_visible(False)
    quality=read(root/'trajectory_summary.json')
    for name,label,color in zip(names,labels,colors):
        lpips=0 if name=='bf16' else statistics.mean(r['metrics']['video']['lpips'] for r in quality['rows'] if r['config']==name)
        axes[2].scatter(rows[name]['median_seconds'],lpips,c=color,s=90,label=label.replace('\n',' '))
    axes[2].set_xlabel('Complete request latency (seconds)');axes[2].set_ylabel('LPIPS relative to BF16 (lower is closer)')
    axes[2].set_title('Speed and BF16 consistency');axes[2].legend(fontsize=8)
    axes[2].spines[['top','right']].set_visible(False)
    fig.suptitle('RTX 5090 · Cosmos3-Edge I2V · 192×320 · 61 frames · 35 steps · CUDA Graph',fontsize=13)
    fig.text(.5,-.04,'Timing: 5 isolated rounds × 3 seeds. Quality: 2 development seeds; not a held-out quality certification.',ha='center',fontsize=9)
    for ext in ('png','svg'):fig.savefig(root/f'e2e_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    times={(r['config'],r['round'],r['seed']):r['wall_seconds'] for r in result['timing_samples']}
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for name,label,color in zip(names[1:],labels[1:],colors[1:]):
        ratios=[statistics.median(times['bf16',r,s]/times[name,r,s] for s in (93001,93002,93003)) for r in range(5)]
        axes[0].plot(range(1,6),ratios,'o-',label=label.replace('\n',' '),color=color)
    for name,color in [('joint125',colors[2]),('joint150',colors[4])]:
        ratios=[statistics.median(times['cutlass'+name[-3:],r,s]/times[name,r,s] for s in (93001,93002,93003)) for r in range(5)]
        axes[1].plot(range(1,6),ratios,'o-',label='Budget '+('1.25' if name.endswith('125') else '1.50'),color=color)
    for ax,title in zip(axes,['Complete-request speedup vs BF16','Joint vs same-budget tuned CUTLASS']):
        ax.axhline(1,color='#64748b',linestyle='--',linewidth=1)
        ax.set_xticks(range(1,6));ax.set_xlabel('Isolated process round');ax.set_ylabel('Paired median speedup (×)')
        ax.set_title(title);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    for ext in ('png','svg'):fig.savefig(root/f'paired_rounds.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--stage',choices=['trajectory','performance','diagnostic'],required=True)
    p.add_argument('--lpips-cache',type=Path,default=Path('data_local/perceptual_cache'))
    a=p.parse_args()
    result=trajectory(a.root,a.lpips_cache) if a.stage=='trajectory' else (performance(a.root) if a.stage=='performance' else diagnostic(a.root))
    path=a.root/f'{a.stage}_summary.json';path.write_text(json.dumps(result,indent=2)+'\n')
    if a.stage=='performance':plot(a.root,result)
    print(path)


if __name__=='__main__':main()

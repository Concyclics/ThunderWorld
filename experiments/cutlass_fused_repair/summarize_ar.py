"""Verify exact-output AR benchmark controls; report paired timing and memory."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    manifest=json.loads((a.root/'performance_matrix.json').read_text());assert manifest['complete']
    fixed={};common=None;hashes={};samples={};values=defaultdict(list);meta={};prefills=[]
    for job in manifest['jobs']:
        r=json.loads((Path(job['output'])/'result.json').read_text());base=job['base'];name=job['name']
        assert r['complete'] and not r['tuning_missing']
        assert r['source_sha256']==manifest['pilot_source_sha256']
        identity={k:r[k] for k in ('environment','source_sha256','model_identity','case','image_sha256')}
        if common is not None:assert common==identity
        common=identity
        config={k:r[k] for k in ('config','asset_identity','cutlass_artifact','observed_shapes')}
        if base in fixed:assert fixed[base]==config
        fixed[base]=config
        assert r['ar_cache']['enabled']==job['cached']
        assert not r['codec_graph']['enabled'] and not r['norm_fusion']['enabled']
        assert r['config']['strict_tuning'] and r['config']['execution']=='graph'
        if job['cached']:assert r['ar_cache']['last_request_prefills']==2 and r['ar_cache']['prefill_graphs']==2
        assert len(r['rows'])==3 and {x['seed'] for x in r['rows']}=={93001,93002,93003}
        meta[name]=dict(ar_cache=r['ar_cache'],model_bytes=r['resident_model_state_bytes'])
        for row in r['rows']:
            assert row['performance_eligible'] and row['debug']=='off' and row['steps']==35
            assert row['timing_scope']=='complete_request_v2_including_host_setup'
            hkey=(base,row['seed'])
            if hkey in hashes:assert hashes[hkey]==row['sha256'],'Changed output or stale cache'
            hashes[hkey]=row['sha256'];samples[name,job['round'],row['seed']]=row['wall_seconds'];values[name].append(row)
    assert len(samples)==90
    for name in values:
        assert {(r,s) for n,r,s in samples if n==name}=={(r,s) for r in range(5) for s in (93001,93002,93003)}
    medians={n:statistics.median(r['wall_seconds'] for r in rows) for n,rows in values.items()}
    result=[];rng=np.random.default_rng(914)
    for name,rows in values.items():
        assert len(rows)==15
        base=name.rsplit('_',1)[0];anchor=base+'_original';fair='bf16_ar' if name.endswith('_ar') else 'bf16_original'
        assert len({json.dumps(hashes[base,s],sort_keys=True) for s in (93001,93002,93003)})==3
        def paired(control):
            ratios=np.array([statistics.median(samples[control,r,s]/samples[name,r,s] for s in (93001,93002,93003)) for r in range(5)])
            boot=np.median(ratios[rng.integers(0,5,(10000,5))],axis=1)
            return dict(round_speedups=ratios.tolist(),median=float(np.median(ratios)),ci95=np.quantile(boot,[.025,.975]).tolist())
        result.append(dict(name=name,n=15,median_seconds=medians[name],p90_seconds=float(np.quantile([r['wall_seconds'] for r in rows],.9)),
            same_precision_speedup=medians[anchor]/medians[name],same_precision_paired=paired(anchor),
            fair_bf16_speedup=medians[fair]/medians[name],fair_bf16_paired=paired(fair),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),**meta[name]))
    summary=dict(complete=True,scope='Edge I2V; 5 isolated randomized rounds x 3 seeds; full request including fresh AR prefill; exact output equality across each base/candidate and all repeats',
        common=common,rows=result,samples=[dict(name=n,round=r,seed=s,seconds=t) for (n,r,s),t in samples.items()])
    (a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['bf16_original','bf16_ar','joint125_original','joint125_ar','cutlass150_original','cutlass150_ar']
    labels=['BF16','BF16 + AR','FP4 1.25','FP4 1.25 + AR','FP4 1.50','FP4 1.50 + AR'];by={r['name']:r for r in result}
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');colors=['#94a3b8','#475569','#93c5fd','#2563eb','#5eead4','#0f766e']
    for ax,field,title in [(axes[0],'median_seconds','Complete request latency (s)'),(axes[1],'peak_allocated_bytes','Peak allocated memory (GiB)')]:
        v=[by[n][field]/(2**30 if field.endswith('bytes') else 1) for n in names]
        ax.bar(range(6),v,color=colors);ax.set_xticks(range(6),labels,rotation=25,ha='right');ax.set_title(title)
        for i,x in enumerate(v):ax.text(i,x,f'{x:.3f}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(v)*1.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('RTX 5090 · Cosmos3-Edge · AR prefill included · Same-budget outputs bitwise equal')
    fig.savefig(a.root/'ar_comparison.png',dpi=180,bbox_inches='tight');fig.savefig(a.root/'ar_comparison.svg',bbox_inches='tight')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

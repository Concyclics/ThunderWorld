"""Separate the frozen repair recipe's auxiliary fusions from FP4 arithmetic.

This is a one-round paired attribution experiment, not a new speed leaderboard.
It reuses tune_speed and leaves BF16/FP8 implementations and frozen assets alone.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

from build_baselines import sha,write
from standard_comparison import snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',type=Path,default=Path('/home/chenhan/thunderworld/variants/cutlass-mainloop-v6'))
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args();worker=Path(__file__).with_name('tune_speed.py')
    plans=[dict(name=f'{budget}_{mode}',method=method,common=mode=='common')
        for budget,method in [('repair15','b1.5_protected_n'),('repair20','b2_joint')]
        for mode in ('production','common')]
    if not a.report_only:
        a.output.mkdir(parents=True,exist_ok=False)
        order=plans.copy();random.Random(230926).shuffle(order)
        write(a.output/'manifest.json',dict(order=order,worker_sha256=sha(worker),
            variant=str(a.variant),variant_sha256=sha(a.variant/'variant.json'),
            source_sha256=sha(Path(__file__)),seeds=[301017,301018,301031],cases=[0,1],
            quality_evaluation=False,performance_eligible=False,
            scope='One randomized round: identical repair assets/support and Transformer Graph; method-specific runtime fusions toggled together.'))
        for row in order:
            cmd=[sys.executable,'-u',str(worker),'measure','--method',row['method'],
                '--variant',str(a.variant),'--cases','0','1','--repeats','3',
                '--seeds','301017','301018','301031','--output',str(a.output/row['name'])]
            if row['common']:cmd+=['--common-runtime']
            env=dict(before=snapshot(),command=cmd)
            with (a.output/(row['name']+'.log')).open('x') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            env.update(after=snapshot(),returncode=result.returncode)
            write(a.output/(row['name']+'.environment.json'),env)
            if result.returncode:raise RuntimeError('Failed control retained: '+row['name'])
    records={};sources={}
    for row in plans:
        path=a.output/row['name']/'result.json';d=json.loads(path.read_text())
        if not d['complete'] or len(d['rows'])!=6:raise ValueError('Incomplete control')
        if d['method']!=row['method']:raise ValueError('Method mismatch')
        if row['common'] and (d['runtime_options'] or not d['runtime_attribution']['common_runtime']):
            raise ValueError('Common runtime not applied')
        if d['runner']['codec_graph']['enabled']==row['common']:raise ValueError('Wrong codec Graph mode')
        for r in d['rows']:
            if r['profiled'] or r['debug']!='off' or r['execution']!='graph' or r['steps']!=35:
                raise ValueError('Changed timing scope')
            if not r['compute_ledger']['feasible']:raise ValueError('Budget violation')
        records[row['name']]=d;sources[str(path)]=sha(path)
    pairs=[]
    for budget in ('repair15','repair20'):
        prod,common=[records[budget+'_'+mode] for mode in ('production','common')]
        if prod['effective_config']!=common['effective_config']:raise ValueError('Config changed')
        if prod['runner']['asset_identity']!=common['runner']['asset_identity']:raise ValueError('Assets changed')
        index=lambda d:{(r['case'],r['seed']):r for r in d['rows']}
        x,y=index(prod),index(common)
        if len(x)!=6 or x.keys()!=y.keys():raise ValueError('Unpaired requests')
        if any(x[k]['compute_ledger']!=y[k]['compute_ledger'] for k in x):raise ValueError('Compute plan changed')
        med=lambda d:statistics.median(r['wall_seconds'] for r in d['rows'])
        pairs.append(dict(budget=budget,production_median_seconds=med(prod),common_median_seconds=med(common),
            auxiliary_fusion_latency_reduction=1-med(prod)/med(common),
            all_output_hashes_equal=all(x[k]['sha256']==y[k]['sha256'] for k in x),
            production_options=prod['runtime_options'],common_options=common['runtime_options'],
            paired_latency_delta_seconds=[x[k]['wall_seconds']-y[k]['wall_seconds'] for k in sorted(x)]))
    write(a.output/'summary.json',dict(complete=True,quality_evaluation=False,performance_eligible=False,
        pairs=pairs,sources=sources,limitation='One paired six-request round; distinguishes auxiliary runtime optimizations, not a five-round matched-arithmetic comparison.'))
    print(json.dumps(pairs,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

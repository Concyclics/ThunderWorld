"""Freeze and run a small paired structure-vs-random/window repair experiment."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from structure_repair import COVERAGES, digest, fit_prior, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JOBS = {
    'edge_i2v': dict(model='edge', task='i2v', development='robot_observe', cases=['robot_observe', 'robot_table'], layers=[0,7,14,20,27]),
    'edge_policy': dict(model='edge', task='policy', development='policy_observe', cases=['policy_observe', 'policy_table'], layers=[0,7,14,20,27]),
    'nano_av': dict(model='nano', task='av', development='av_waterfall', cases=['av_waterfall', 'av_pottery'], layers=[0,9,18,26,35]),
}


def prepare(a):
    if a.root.exists():
        raise FileExistsError(a.root)
    a.root.mkdir(parents=True)
    jobs = {}
    for name, entry in JOBS.items():
        entry = dict(entry)
        paths = [a.observations / entry['model'] / f'bf16_{entry["development"]}_seed{s}_r0_heads.npz' for s in (6101,6102)]
        density = fit_prior(paths)
        prior = a.root / 'priors' / (name + '.npz')
        prior.parent.mkdir(exist_ok=True)
        np.savez_compressed(prior, density=density)
        model = a.models / ('Cosmos3-Edge-a9d944e' if entry['model']=='edge' else 'Cosmos3-Nano-411f42a')
        cases = a.cases / (entry['model'] + '_cases.json')
        if not model.is_dir() or not cases.is_file():
            raise FileNotFoundError('Model or case manifest missing')
        available = {v['id']:v for v in json.loads(cases.read_text())['cases']}
        if available[entry['development']]['split'] != 'development':
            raise ValueError('Prior fit must use an explicitly development scene')
        entry.update(prior=str(prior.resolve()), prior_sha256=digest(prior),
            source_observations={str(p.resolve()):digest(p) for p in paths},
            model_path=str(model.resolve()), case_manifest=str(cases.resolve()), case_manifest_sha256=digest(cases))
        jobs[name] = entry
    source = [HERE / n for n in ('structure_repair.py','structure_repair_run.py','structure_repair_study.py')]
    source += list((ROOT/'packages/fp4-repair/src/thunderworld_fp4').glob('*.py'))
    protocol = dict(schema='structure-repair-paired-v1', jobs=jobs, frozen=True,
        calibration_seeds=[6101,6102], evaluation_seeds=[241101,241102,241103,241104],
        steps=35, probe_steps=[0,7,18,28], branch=0, terms=3, coverages=list(COVERAGES),
        random_repeats=8, physical_gpu=a.gpu, performance_eligible=False,
        selection='Frozen development role/head/layer/phase mean attention density; no test error, QK probability or BF16 output used in selection',
        budget='Exact same H*K repair count within each M128 row; verify issued M32/N128 FP4 QK+PV MMA including padding',
        masks='Explicit QK/PV masks, all dense base blocks retained; current full Q/K/V input; no output reuse',
        statistics='Average random replicates within panel; equal sampled layers/steps within request; paired seed bootstrap within each of two fixed scenes; no head/panel pseudoreplication',
        claims='Exploratory within-attention consistency, not full-generation quality, broad scene generalization, or timing',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source},
        expected_requests=24, expected_attention_panels=480)
    write(a.root/'protocol.json',protocol)
    print(json.dumps(dict(frozen=True, requests=24, panels=480, calibration_trajectories=6)),flush=True)


def run(a):
    protocol=json.loads((a.root/'protocol.json').read_text())
    for name, checksum in protocol['source_sha256'].items():
        if digest(ROOT/name)!=checksum:
            raise ValueError('Source changed after protocol freeze: '+name)
    state=dict(complete=False,performance_eligible=False,completed=[])
    for name, job in protocol['jobs'].items():
        if a.jobs and name not in a.jobs:
            continue
        out=a.root/name
        previous=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
        if previous and previous['complete']:
            state['completed'].append(name);continue
        for key, path in [('prior_sha256',job['prior']),('case_manifest_sha256',job['case_manifest'])]:
            if digest(path)!=job[key]:raise ValueError('Frozen input changed')
        command=[sys.executable,str(HERE/'structure_repair_run.py'),'--model',job['model'],
            '--model-path',job['model_path'],'--case-manifest',job['case_manifest'],'--cases',*job['cases'],
            '--output',str(out),'--seeds',*map(str,protocol['evaluation_seeds']),'--steps','35','--sample',
            '--modes','bf16','--structure-prior',job['prior'],'--probe-layers',*map(str,job['layers']),
            '--random-repeats',str(protocol['random_repeats'])]
        if previous:command.append('--resume')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(protocol['physical_gpu']),TW_SHARED_CALIBRATION_GPUS=str(protocol['physical_gpu']),
                 TW_SHARED_CALIBRATION_LOCK_SCOPE='device',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        log=a.root/(name+'.log')
        state.update(active=name,command=command,log=str(log));write(a.root/'progress.json',state)
        print(json.dumps(dict(starting=name,log=str(log))),flush=True)
        with log.open('a') as f:
            completed=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if completed.returncode:
            state.update(paused=True,exit_code=completed.returncode);write(a.root/'progress.json',state)
            raise RuntimeError('Experiment stopped; inspect '+str(log))
        state['completed'].append(name);write(a.root/'progress.json',state)
    state.update(complete=len(state['completed'])==len(protocol['jobs']),active=None)
    write(a.root/'progress.json',state)


def paired(values, baseline, seeds, replicates=10000):
    """Seed-resampled intervals conditional on the fixed scenes (not scene CI)."""
    values=np.asarray(values,np.float64);baseline=np.asarray(baseline,np.float64)
    if values.shape!=baseline.shape or values.ndim!=2 or values.shape[1]!=len(seeds):
        raise ValueError('Paired scene/seed shape mismatch')
    rng=np.random.default_rng(924071)
    idx=rng.integers(0,len(seeds),size=(replicates,values.shape[0],len(seeds)))
    scene=np.arange(values.shape[0])[None,:,None]
    aa=values[scene,idx].mean((1,2));bb=baseline[scene,idx].mean((1,2))
    difference=aa-bb;reduction=100*(bb-aa)/bb
    # Six planned primary comparisons: three coverages x two baselines.
    alpha=.05/6
    return dict(mean_nmse=float(values.mean()),baseline_mean_nmse=float(baseline.mean()),
        relative_error_reduction_pct=float(100*(baseline.mean()-values.mean())/baseline.mean()),
        difference_ci95=np.quantile(difference,[.025,.975]).tolist(),
        difference_ci_familywise95=np.quantile(difference,[alpha/2,1-alpha/2]).tolist(),
        reduction_ci95_pct=np.quantile(reduction,[.025,.975]).tolist(),
        request_wins=int((values<baseline).sum()),requests=int(values.size),
        worst_request_relative_change_pct=float((100*(values-baseline)/baseline).max()),
        interval_scope='Paired seed bootstrap conditional on these fixed scenes; exploratory small sample')


def report(a):
    protocol=json.loads((a.root/'protocol.json').read_text());cohorts={};all_complete=True
    text=['# Same-budget structure-aware attention repair','','Local complete attention output NMSE relative to the current-input native BF16 operator. All base blocks remain. This is not generated-video quality or timing.','']
    for name, job in protocol['jobs'].items():
        path=a.root/name/'result.json'
        if not path.exists():all_complete=False;continue
        run=json.loads(path.read_text());all_complete &= run['complete']
        groups=defaultdict(list);rows=[];ratios=defaultdict(set);unions=defaultdict(list)
        completed={}
        for request in run['rows']:
            case,seed=request['case'],request['seed']
            file=a.root/name/f'bf16_{case}_seed{seed}_r0_heads.repair.json'
            panel=json.loads(file.read_text())
            if not panel['complete']:raise ValueError('Incomplete panel set')
            completed[case,seed]=True
            for r in panel['rows']:
                r=dict(r,case=case,seed=seed);rows.append(r)
                method='random' if r['method'].startswith('random_') else r['method']
                groups[case,seed,r['coverage'],method,r['step'],r['layer']].append(r['nmse'])
                if method not in ('base','full_three'):ratios[r['coverage']].add(r['mma_ratio'])
                unions[r['coverage'],method].append(r['gqa_kv_union_density'])
        complete=len(completed)==len(job['cases'])*len(protocol['evaluation_seeds'])
        if not complete:
            cohorts[name]=dict(complete=False,requests=len(completed));continue
        averages=defaultdict(list)
        for key, errors in groups.items():averages[key[:4]].append(float(np.mean(errors)))
        def table(cov,method):
            return [[float(np.mean(averages[c,s,cov,method])) for s in protocol['evaluation_seeds']] for c in job['cases']]
        result=dict(complete=True,requests=len(completed),panels=len(completed)*len(job['layers'])*4,
            base_nmse=float(np.mean(table(0.,'base'))),full_three_nmse=float(np.mean(table(1.,'full_three'))),budgets={})
        text += [f'## {name}', '',f'Base NMSE {result["base_nmse"]:.6g}; full three-term NMSE {result["full_three_nmse"]:.6g}.', '',
            '| Attention MMA | Random | Window | Structure | vs random | vs window |','|---|---:|---:|---:|---:|---:|']
        for cov in COVERAGES:
            methods={method:float(np.mean(table(cov,method))) for method in ('random','window','structure','structure_no_head','structure_no_step')}
            comparisons={base:paired(table(cov,'structure'),table(cov,base),protocol['evaluation_seeds']) for base in ('random','window')}
            result['budgets'][str(cov)]=dict(mma_ratios=sorted(ratios[cov]),mean_nmse=methods,comparisons=comparisons,
                gqa_kv_union_density={m:float(np.mean(v)) for (c,m),v in unions.items() if c==cov})
            r=','.join(f'{x:.3f}x' for x in sorted(ratios[cov]))
            text += [f'| {r} | {methods["random"]:.6g} | {methods["window"]:.6g} | {methods["structure"]:.6g} | {comparisons["random"]["relative_error_reduction_pct"]:+.2f}% | {comparisons["window"]["relative_error_reduction_pct"]:+.2f}% |']
        cohorts[name]=result
        write(a.root/'report'/f'{name}_raw.json',dict(rows=rows,performance_eligible=False))
        text += ['', 'Positive percentages denote lower NMSE; negative percentages denote a loss. Intervals and per-role errors are retained in JSON.', '']
    summary=dict(complete=all_complete and len(cohorts)==len(protocol['jobs']),protocol_sha256=digest(a.root/'protocol.json'),
        performance_eligible=False,cohorts=cohorts,scope=protocol['claims'])
    write(a.root/'report/results.json',summary)
    (a.root/'report/REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['prepare','run','report'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--observations',type=Path)
    p.add_argument('--models',type=Path)
    p.add_argument('--cases',type=Path)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--jobs',nargs='+',choices=list(JOBS))
    a=p.parse_args();a.root=a.root.resolve()
    if a.operation=='prepare' and any(getattr(a,k) is None for k in ('observations','models','cases')):
        p.error('prepare requires --observations, --models and --cases')
    globals()[a.operation](a)


if __name__=='__main__':main()

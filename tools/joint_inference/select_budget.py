"""Freeze one candidate and matched uncalibrated control per budget on validation only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from thunderworld_fp4.cosmos.cohorts import Cohort
from capture_budget import atomic_json


def freeze(job):
    result={**job,'frozen':True,'config_sha256':hashlib.sha256(Path(job['config']).read_bytes()).hexdigest()}
    config=json.loads(Path(job['config']).read_text())
    if config.get('budget_plan'):
        result['budget_plan_sha256']=hashlib.sha256(Path(config['budget_plan']).read_bytes()).hexdigest()
    if job.get('assets'):
        result['assets_manifest_sha256']=hashlib.sha256((Path(job['assets'])/'manifest.json').read_bytes()).hexdigest()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('quality','jobs','cohort','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();quality=json.loads(a.quality.read_text());jobs=json.loads(a.jobs.read_text())['jobs'];cohort=Cohort.load(a.cohort)
    if not quality['complete'] or quality['suite']!='validation':raise ValueError('Complete independent validation required')
    if quality.get('cohort_sha256')!=cohort.fingerprint:raise ValueError('Validation/cohort provenance mismatch')
    if a.output.exists():raise FileExistsError(a.output)
    matrix_path=a.quality.parent/'matrix.json';runtime_ledgers={}
    if matrix_path.exists():
        if quality['matrix_sha256']!=hashlib.sha256(matrix_path.read_bytes()).hexdigest():raise ValueError('Validation matrix changed')
        for job in json.loads(matrix_path.read_text())['jobs']:
            run=json.loads((Path(job['output'])/'result.json').read_text())
            if run.get('compute_ledger'):runtime_ledgers[job['config']]=run['compute_ledger']
    def metrics(name):
        rows=[r for r in quality['rows'] if r['config']==name]
        cohort.validate('validation',[r['seed'] for r in rows],complete=True)
        result={key:float(np.mean([r['metrics']['video'][key] for r in rows])) for key in ('lpips','ssim_mean','relative_l2')}
        extra={'policy':{'action':('relative_l2','rmse')},'av':{'waveform':('log_spectral_l1','relative_l2'),'sound_latent':('relative_l2',)}}
        for modality,keys in extra.get(cohort.value['task'],{}).items():
            for key in keys:result[modality+'.'+key]=float(np.mean([r['metrics'][modality][key] for r in rows]))
        if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite validation metrics')
        return result
    decisions=[];frozen=[]
    for budget in cohort.value['budgets']:
        controls=[j for j in jobs if j.get('control',False) and j.get('budget')==budget]
        candidates=[j for j in jobs if j.get('candidate',False) and j.get('budget')==budget]
        if not controls or not candidates or len(candidates)>3:raise ValueError('Missing controls/candidates or too many candidates')
        control=min(controls,key=lambda j:metrics(j['name'])['lpips']);baseline=metrics(control['name'])
        def guarded(job):
            m=metrics(job['name'])
            return m['ssim_mean']>=baseline['ssim_mean'] and all(m[key]<=baseline[key] for key in baseline if key not in ('lpips','ssim_mean'))
        eligible=[j for j in candidates if guarded(j)]
        if not eligible:
            frozen.append(freeze(control))
            decisions.append(dict(budget=budget,selected=control['name'],optimized_candidate=False,
                reason='No optimized candidate satisfies SSIM/L2 guardrails; retain best feasible uncalibrated FP4 control',control=control['name']))
            continue
        def key(job):
            m=metrics(job['name'])
            # The actual request inventory takes precedence over a pre-export
            # estimate, which may use older metadata or different padded shapes.
            ledger=runtime_ledgers.get(job['name'])
            if ledger is None:ledger=json.loads((Path(job['config']).parent/'compute.json').read_text())
            # Inputs for tie breakers must be supplied, not inferred from nominal coverage.
            traffic=ledger.get('packing_traffic',{})
            # These are disjoint linear and attention writes; do not sum aliases.
            packing=traffic.get('base_write_bytes',0)+traffic.get('base_code_scale_write_bytes',0)+traffic.get('base_factor_write_bytes',0)+traffic.get('residual_write_bytes',traffic.get('selected_residual_write_bytes',0))+traffic.get('selected_residual_write_bytes_upper',0)
            return (m['lpips'],ledger['total_mma'],packing,ledger['w1_support_union_bytes'])
        selected=min([control,*eligible],key=key)
        if selected['name']==control['name']:
            frozen.append(freeze(control))
            decisions.append(dict(budget=budget,selected=control['name'],optimized_candidate=False,
                reason='Uncalibrated control is the lowest-error feasible validation configuration',
                control=control['name'],metrics=baseline))
            continue
        for job in (selected,control):
            frozen.append(freeze(job))
        decisions.append(dict(budget=budget,selected=selected['name'],control=control['name'],metrics=metrics(selected['name'])))
    for job in jobs:
        if job['name'] in ('bf16','fp8','fp4','full_dual'):frozen.append(freeze(job))
    atomic_json(a.output,dict(complete=True,cohort_sha256=cohort.fingerprint,
                validation_sha256=hashlib.sha256(a.quality.read_bytes()).hexdigest(),decisions=decisions,jobs=frozen))


if __name__=='__main__':main()

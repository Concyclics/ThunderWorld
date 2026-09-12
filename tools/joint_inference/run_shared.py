"""Serialized cohort request execution, atomic progress, no shared timing ranking."""
import argparse,json,hashlib,gc
from pathlib import Path
from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
from thunderworld_fp4.cosmos.cohorts import Cohort
from capture_budget import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','case','cohort','jobs','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--role',choices=['trajectory','validation','final'],required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();cohort=Cohort.load(a.cohort);job_manifest=json.loads(a.jobs.read_text());jobs=job_manifest['jobs']
    if a.role=='final':
        if job_manifest.get('cohort_sha256')!=cohort.fingerprint or not job_manifest.get('validation_sha256'):
            raise ValueError('Final job manifest requires recorded validation selection')
        for job in jobs:
            if not job.get('frozen',False) or hashlib.sha256(Path(job['config']).read_bytes()).hexdigest()!=job.get('config_sha256'):
                raise ValueError('Frozen configuration changed or missing for candidate/control/baseline')
            frozen_config=json.loads(Path(job['config']).read_text())
            if frozen_config.get('budget_plan'):
                if hashlib.sha256(Path(frozen_config['budget_plan']).read_bytes()).hexdigest()!=job.get('budget_plan_sha256'):
                    raise ValueError('Frozen repair plan changed')
            if job.get('assets'):
                if hashlib.sha256((Path(job['assets'])/'manifest.json').read_bytes()).hexdigest()!=job.get('assets_manifest_sha256'):
                    raise ValueError('Frozen assets changed')
    if a.role=='validation':
        from collections import Counter
        if any(n>3 for n in Counter(j['budget'] for j in jobs if j.get('candidate',False)).values()):
            raise ValueError('At most 3 validation candidates per budget')
    if a.role=='final':
        from collections import Counter
        if any(n>1 for n in Counter(j['budget'] for j in jobs if j.get('candidate',False)).values()):
            raise ValueError('Only one frozen candidate per budget')
    if a.role=='final' and any(not j.get('frozen',False) for j in jobs if j.get('candidate',False)):
        raise ValueError('Final candidates must be frozen before seeing final seeds')
    case=json.loads(a.case.read_text())
    if case['task']!=cohort.value['task']:raise ValueError('Case/cohort task mismatch')
    if case.get('image'):case['image']=str((a.case.parent/case['image']).resolve())
    if a.role=='trajectory' and case.get('split')!='development':raise ValueError('Trajectory selection requires development case')
    a.output.mkdir(parents=True,exist_ok=a.resume)
    matrix=dict(complete=False,model=cohort.value['model'],suite='quality' if a.role=='final' else a.role,
                cohort_sha256=cohort.fingerprint,resource_mode='shared-calibration',performance_eligible=False,jobs=[])
    atomic_json(a.output/'matrix.json',matrix)
    with shared_worker():
        try:select_shared_device()
        except Exception as error:
            matrix.update(error=repr(error),paused=True);atomic_json(a.output/'matrix.json',matrix);raise
        import torch,numpy as np
        from dataclasses import replace
        from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
        from thunderworld_fp4.cosmos.runner import model_identity
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        try:
            for job in jobs:
                config=replace(RunConfig.from_dict(json.loads(Path(job['config']).read_text())),resource_mode='shared-calibration')
                if job.get('reference_output'):
                    folder=Path(job['reference_output']);reference=json.loads((folder/'result.json').read_text())
                    if not reference['complete'] or reference.get('role')!=a.role or reference.get('cohort_sha256')!=cohort.fingerprint:
                        raise ValueError('Reference run cohort/role mismatch')
                    if RunConfig.from_dict(reference['config'])!=config or reference['case']!=case or reference['model_identity']!=model_identity(a.model):
                        raise ValueError('Reference model/config/input mismatch')
                    cohort.validate(a.role,[r['seed'] for r in reference['rows']],complete=True)
                    if any(r['steps']!=35 for r in reference['rows']):raise ValueError('Reference step mismatch')
                    if case.get('image') and reference.get('image_sha256')!=hashlib.sha256(Path(case['image']).read_bytes()).hexdigest():
                        raise ValueError('Reference image content mismatch')
                    matrix['jobs'].append(dict(case=case['id'],config=job['name'],output=str(folder.resolve()),reused=True))
                    atomic_json(a.output/'matrix.json',matrix);continue
                folder=a.output/job['name'];folder.mkdir(exist_ok=a.resume)
                path=folder/'result.json'
                result=dict(complete=False,config=config.to_dict(),case=case,rows=[],cohort_sha256=cohort.fingerprint,
                    role=a.role,performance_eligible=False,job=job,
                    image_sha256=hashlib.sha256(Path(case['image']).read_bytes()).hexdigest() if case.get('image') else None)
                if a.resume and path.exists():
                    previous=json.loads(path.read_text())
                    if RunConfig.from_dict(previous['config'])!=config or previous['job']!=job or previous['cohort_sha256']!=cohort.fingerprint:
                        raise ValueError('Resume job/cohort mismatch')
                    prior_identity=previous.get('model_identity')
                    if previous['case']!=case or previous.get('image_sha256')!=result['image_sha256'] or (prior_identity is not None and prior_identity!=model_identity(a.model)) or (previous['rows'] and prior_identity is None):
                        raise ValueError('Resume model/input mismatch')
                    result=previous
                matrix['jobs'].append(dict(case=case['id'],config=job['name'],output=str(folder.resolve())))
                if result['complete']:continue
                select_shared_device()  # Check free memory again before loading; no auto precision/input changes.
                runner=CosmosRunner(a.model,config,assets=job.get('assets'),audio=case['task']=='av')
                try:
                    runner.prepare();runner.warmup(case,35)
                    result.update(runner.receipt());atomic_json(path,result)
                    for seed in cohort.seeds[a.role]:
                        if any(r['seed']==seed for r in result['rows']):continue
                        signals,row=runner.infer(case,seed,35)
                        output=folder/f'seed{seed}.npz';np.savez_compressed(output,**signals)
                        row['outputs']=output.name;row['outputs_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
                        # Store the full request ledger once, with a per-row digest.
                        ledger=row.pop('compute_ledger',None)
                        if ledger:
                            result['compute_ledger']=ledger
                            row['compute_ledger_sha256']=hashlib.sha256(json.dumps(ledger,sort_keys=True).encode()).hexdigest()
                        result['rows'].append(row);atomic_json(path,result)
                        print(json.dumps(dict(job=job['name'],seed=seed,complete=len(result['rows']))),flush=True)
                    result['complete']=True;result.pop('error',None);result.pop('paused',None);atomic_json(path,result)
                except Exception as error:
                    result.update(error=repr(error),paused=True);atomic_json(path,result);raise
                finally:
                    if runner.engine:runner.engine.clear()
                    del runner;gc.collect();torch.cuda.empty_cache()
                atomic_json(a.output/'matrix.json',matrix)
            matrix['complete']=True;atomic_json(a.output/'matrix.json',matrix)
        except Exception as error:
            matrix.update(error=repr(error),paused=True);atomic_json(a.output/'matrix.json',matrix);raise


if __name__=='__main__':main()

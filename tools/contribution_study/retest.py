"""Exclusive-device retest dispatcher: five shuffled rounds, three fixed seeds.

Use only native, quality-frozen jobs. The installed CLI enforces idle GPU,
debug-off, exact device tuning lookup and the original request budget.
"""
import argparse,json,os,random,subprocess,sys
from pathlib import Path
from baselines import sha,write


def verify_outputs(result,expected):
    observed={str(r['seed']):r['sha256'] for r in result['rows']}
    if not result['complete'] or len(observed)!=len(result['rows']) or observed!=expected:
        raise RuntimeError('Device/tuning changed complete outputs; full quality requalification required before timing')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('0','3'),default='0');p.add_argument('--execute',action='store_true')
    a=p.parse_args();manifest=json.loads(a.manifest.read_text())
    if not manifest.get('frozen') or not manifest.get('quality_selection_sha256'):raise ValueError('Freeze quality selection before timing')
    a.output.mkdir(parents=True,exist_ok=False)
    jobs=manifest['jobs'];commands=[]
    for job in jobs:
        if job.get('execution_type')!='native':raise ValueError('Only qualified native jobs can enter this timing suite')
        if sha(job['config'])!=job['config_sha256']:raise ValueError('Configuration changed')
        cfg=json.loads(Path(job['config']).read_text())
        if cfg.get('budget_plan') and sha(cfg['budget_plan'])!=job.get('budget_plan_sha256'):raise ValueError('Plan changed')
        if job.get('assets') and sha(Path(job['assets'])/'manifest.json')!=job.get('assets_sha256'):raise ValueError('Assets changed')
        common=['--model',manifest['model'],'--case',manifest['case'],'--config',job['config'],
            '--resource-mode','exclusive','--debug','off','--execution','graph','--steps','35','--ar-cache','--first-frame']
        if job.get('assets'):common+=['--assets',job['assets']]
        job['_common']=common
        commands.append(dict(kind='offline_device_tuning',job=job['name'],command=[sys.executable,'-m','thunderworld_fp4.cosmos','warmup',
            *common,'--tune','--output',str(a.output/(job['name']+'_prepare'))]))
        expected=job.get('quality_signal_hashes')
        if not expected:raise ValueError('Frozen complete-output hashes required before device timing')
        verify=a.output/(job['name']+'_verify')
        commands.append(dict(kind='complete_output_equivalence',job=job['name'],expected=expected,
            result=str(verify/'result.json'),command=[sys.executable,'-m','thunderworld_fp4.cosmos','infer',
            *common,'--tuning',str(a.output/(job['name']+'_prepare')/'tuning.json'),
            '--seeds',*sorted(expected,key=int),'--output',str(verify)]))
    rng=random.Random(170917)
    for round_id in range(5):
        order=list(jobs);rng.shuffle(order)
        for job in order:
            commands.append(dict(kind='complete_requests',round=round_id,job=job['name'],command=[sys.executable,'-m','thunderworld_fp4.cosmos','benchmark',
                *job['_common'],'--tuning',str(a.output/(job['name']+'_prepare')/'tuning.json'),
                '--seeds','91001','91002','91003','--output',str(a.output/f'round{round_id}_{job["name"]}')]))
    write(a.output/'commands.json',dict(manifest_sha256=sha(a.manifest),device=a.device,commands=commands,
        scope='Fresh per-configuration process and warmup each round. New device tuning, no 5090/PRO6000 profile sharing. Timing seeds never select quality.'))
    if a.execute:
        for index,row in enumerate(commands):
            with (a.output/f'{index:03d}.log').open('w') as log:
                subprocess.run(row['command'],env={**os.environ,'CUDA_VISIBLE_DEVICES':a.device},stdout=log,stderr=subprocess.STDOUT,check=True)
            if row['kind']=='complete_output_equivalence':
                result=json.loads(Path(row['result']).read_text())
                verify_outputs(result,row['expected'])


if __name__=='__main__':main()

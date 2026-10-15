"""Sequential isolated-process matrix. Quality seeds and performance rounds are distinct."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-name',choices=['edge','nano'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',choices=['smoke','validation','quality','performance','tuning','cold'],required=True)
    p.add_argument('--configs',nargs='+',default=['bf16','fp8','four'])
    p.add_argument('--case-configs',type=Path,help='Frozen case -> config names mapping')
    p.add_argument('--cases',nargs='+')
    p.add_argument('--tuning-root',type=Path)
    p.add_argument('--steps',type=int,default=35)
    p.add_argument('--reference-matrix',type=Path,help='Reuse complete same-suite BF16/FP8 output pairs')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cases=a.cases or (['robot_table','policy_table']+(['av_pottery'] if a.model_name=='nano' else []))
    seeds={'smoke':[90991],'validation':list(range(92001,92009)),
           'quality':list(range(91001,91009)),'performance':[93001,93002,93003], 'tuning':[6099], 'cold':[6099]}[a.suite]
    rounds=5 if a.suite=='performance' else 1
    rng=random.Random(120);jobs=[]
    case_configs=json.loads(a.case_configs.read_text()) if a.case_configs else None
    if a.reference_matrix:
        previous=json.loads(a.reference_matrix.read_text())
        if not previous['complete'] or previous['suite']!=a.suite or previous['model']!=a.model_name or a.suite=='performance':
            raise ValueError('Reference matrix must be complete and match quality cohort/model')
        jobs=[dict(j,reused=True) for j in previous['jobs'] if j['config'] in ('bf16','fp8') and j['case'] in cases]
        if any(c in ('bf16','fp8') for c in a.configs):p.error('Reused controls must not also be requested')
    for repeat in range(rounds):
        configs=list(a.configs)
        if a.suite=='performance':rng.shuffle(configs)
        for case in cases:
            selected=list(case_configs[case]) if case_configs else configs
            if case_configs and a.suite=='performance':rng.shuffle(selected)
            audio_outputs=['latent','waveform'] if case=='av_pottery' and a.suite=='performance' else ['waveform']
            for config in selected:
                for audio in audio_outputs:
                    dest=a.output/f'{repeat:02d}_{case}_{config}_{audio}'
                    cmd=[sys.executable,'-m','thunderworld_fp4.cosmos',
                         {'performance':'benchmark','tuning':'warmup','cold':'warmup'}.get(a.suite,'infer'),
                         '--model',str(a.model),'--config',str(ROOT/'tools/joint_inference/configs'/f'{config}.json'),
                         '--case',str(ROOT/'tools/joint_inference/cases'/f'{case}.json'),
                         '--output',str(dest),'--steps',str(2 if a.suite=='smoke' else a.steps),
                         '--seeds',*map(str,seeds),'--audio-output',audio]
                    if a.suite in ('performance','cold'):
                        if a.tuning_root is None:p.error('Performance requires --tuning-root with frozen profiles')
                        cmd+=['--tuning',str(a.tuning_root/f'{case}_{config}.json')]
                    if a.suite=='tuning':cmd+=['--tune']
                    jobs.append(dict(command=cmd,output=str(dest),config=config,case=case,round=repeat,audio_output=audio))
    manifest=dict(complete=False,suite=a.suite,model=a.model_name,device=os.environ.get('CUDA_VISIBLE_DEVICES'),jobs=jobs)
    path=a.output/'matrix.json'
    def save():path.write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    for job in jobs:
        if job.get('reused'):continue
        if a.suite=='cold':
            from thunderworld_fp4.cosmos.cli import idle_guard
            idle_guard()
        print('Running',job['output'],flush=True)
        process_start=time.perf_counter()
        environment=dict(os.environ)
        if a.suite=='cold':environment['THUNDERWORLD_COLD_START']=str(process_start)
        with open(job['output']+'.log','w') as log:
            job['exit_code']=subprocess.run(job['command'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
        job['process_seconds']=time.perf_counter()-process_start
        if a.suite=='cold':idle_guard()
        save()
        if job['exit_code']:raise RuntimeError('Failed job: '+job['output'])
        if a.suite=='tuning':
            import shutil
            profiles=a.output/'profiles';profiles.mkdir(exist_ok=True)
            shutil.copyfile(Path(job['output'])/'tuning.json',profiles/f"{job['case']}_{job['config']}.json")
    manifest['complete']=True;save()


if __name__=='__main__':main()

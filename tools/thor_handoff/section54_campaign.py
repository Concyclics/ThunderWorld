"""Resumable serial qualification -> 420 requests -> ten profiles on idle Thor.

Resource contention preserves the failed attempt and waits. It never kills other
users' work or converts shared-load timings into results. Numerical qualification
failures mark that external method unavailable; unexpected runtime failures stop.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from build_baselines import sha,write
from compare_speed import compute_pids

METHODS=['fp4','awq_marlin','arcquant50','arcquant100','sharq','repair15','repair20']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tools=Path(__file__).resolve().parent
    state=dict(complete=False,started=time.time(),scope='420 FP4-only requests, ten Edge profiles; no quality evaluation',pid=os.getpid())
    def status(**values):
        state.update(values,updated=time.time());write(args.output/'campaign_status.json',state)
    def idle():
        stable=0
        while stable<3:
            import psutil
            pids=compute_pids();available=psutil.virtual_memory().available
            if pids or available<12*2**30:
                status(state='waiting_for_resources',other_compute_pids=pids,available_bytes=available);stable=0
            else:stable+=1
            if stable<3:time.sleep(20)
        status(state='resources_ready',other_compute_pids=[],available_bytes=available)
    def launch(command,log,gpu=True):
        if gpu:idle()
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        return result.returncode,log.read_text(errors='replace')
    def paused(text):
        return any(s in text for s in ('PAUSED:','out of memory','gpu-test.lock','BlockingIOError'))
    def preserve(path,tag):
        if not path.exists():return
        dest=args.output/'retained_attempts'/f'{tag}-{time.time_ns()}'
        dest.parent.mkdir(exist_ok=True);shutil.move(str(path),str(dest))
        return str(dest)
    try:
        # rsync installs files by rename; wait for the complete manifest/payload
        # list before preparing, then freeze_assets verifies every checksum.
        requirements=json.loads((tools/'section54_fp4_retest_spec.json').read_text())
        while True:
            missing=[]
            for asset in requirements['missing_assets']:
                folder=Path(asset['destination']);manifest=folder/'manifest.json'
                if not manifest.exists():missing.append(str(manifest));continue
                if sha(manifest)!=asset['manifest_sha256']:raise ValueError('Transferred asset manifest mismatch')
                data=json.loads(manifest.read_text());entries=data.get('modules',{}).values() if 'modules' in data else [data]
                missing.extend(str(folder/e['file']) for e in entries if not (folder/e['file']).exists())
            if not missing:break
            status(state='waiting_for_assets',missing_files=len(missing),examples=missing[:3]);time.sleep(20)
        # CPU-only prepare may run while another user occupies the GPU.
        for model in ('edge','nano'):
            dest=args.output/model
            if not (dest/'manifest.json').exists():
                if dest.exists():raise ValueError('Incomplete prepare retained: '+str(dest))
                status(state='preparing',model=model)
                code,text=launch([sys.executable,str(tools/'standard_comparison.py'),'prepare','--model',model,
                    '--root',str(args.root),'--output',str(dest),'--rounds','5','--methods',*METHODS],
                    args.output/f'prepare-{model}.log',gpu=False)
                if code:raise RuntimeError('CPU preparation failed: '+text[-2000:])
        # Operator checks are independent of generated video fidelity.
        for model in ('edge','nano'):
            dest=args.output/model;qdir=dest/'qualification';qdir.mkdir(exist_ok=True)
            fixed=dest/'fixed-model.json'
            for method,aliases in [('arcquant',['arcquant50','arcquant100']),('sharq',['sharq']),('awq_marlin',['awq_marlin'])]:
                if all((qdir/(n+'.qualified.json')).exists() or (qdir/(n+'.unavailable.json')).exists() for n in aliases):continue
                probe=qdir/(method+'-probe.json');log=qdir/(method+'-probe.log')
                while True:
                    status(state='qualifying',model=model,method=method)
                    code,text=launch([sys.executable,str(tools/'probe_baselines.py'),'--root',str(args.root),
                        '--method',method,'--manifest',str(fixed),'--output',str(probe)],log)
                    if code and paused(text):
                        preserve(probe,model+'-'+method+'-probe');preserve(log,model+'-'+method+'-probe-log');continue
                    break
                okay=code==0 and json.loads(probe.read_text()).get('complete',False)
                sanitation=[]
                if okay and method=='arcquant':
                    for checker in ('memcheck','racecheck'):
                        output=qdir/(checker+'-arcquant.json');slog=qdir/(checker+'-arcquant.log')
                        while True:
                            cmd=['compute-sanitizer','--tool',checker,'--error-exitcode','86',sys.executable,
                                str(tools/'probe_baselines.py'),'--root',str(args.root),'--method','arcquant',
                                '--manifest',str(fixed),'--output',str(output),'--quick']
                            if model=='nano':cmd+=['--only-k','12288']
                            rc,stext=launch(cmd,slog)
                            if rc and paused(stext):
                                preserve(output,model+'-'+checker);preserve(slog,model+'-'+checker+'-log');continue
                            break
                        passed=rc==0 and json.loads(output.read_text()).get('complete',False)
                        if not passed:text=stext
                        sanitation.append(dict(tool=checker,passed=passed,path=str(slog),sha256=sha(slog)))
                        okay=okay and passed
                receipt=dict(complete=bool(okay),probe=str(probe),probe_sha256=sha(probe) if probe.exists() else None,
                    manifest_sha256=sha(fixed),probe_source_sha256=sha(tools/'probe_baselines.py'),
                    sanitizer=sanitation,reason=None if okay else text[-3000:],quality_evaluation=False)
                for name in aliases:write(qdir/(name+('.qualified.json' if okay else '.unavailable.json')),receipt)
            # v6 kernels unchanged; retain model-specific prior numerical checks.
            prior=args.root/'comparisons'/('nano_standard_20260923/qualification' if model=='nano' else 'cutlass_fused_20260922')
            if not prior.exists():raise ValueError('Prior v6 qualification missing')
            version=json.loads((args.root/'variants/cutlass-mainloop-v6/variant.json').read_text())
            proof=version['qualification']
            if sha(proof['path'])!=proof['sha256']:raise ValueError('v6 numerical qualification changed')
            write(qdir/'v6-reuse.json',dict(complete=True,variant=str(args.root/'variants/cutlass-mainloop-v6'),
                receipt_sha256=sha(args.root/'variants/cutlass-mainloop-v6/variant.json'),
                prior_qualification=str(prior),kernel_changed=False,parent_proof=proof,
                model_specific_checks={str(f):sha(f) for f in prior.glob('*.json')},
                scope='Reuse frozen v6 and prior Edge/Nano checks; each fresh worker verifies source/library hashes, current request outputs and budgets'))
        for model in ('edge','nano'):
            dest=args.output/model
            while not (dest/'summary.json').exists() or not json.loads((dest/'summary.json').read_text())['complete']:
                status(state='timing',model=model)
                code,text=launch([sys.executable,'-u',str(tools/'standard_comparison.py'),'run','--root',str(args.root),'--output',str(dest)],args.output/f'run-{model}.log')
                if not code:break
                # Resource failures are retryable; keep every invalid attempt.
                run_status=json.loads((dest/'status.json').read_text())
                workerlog=dest/'logs'/f"round-{run_status.get('round',0):02d}-{run_status.get('method','unknown')}.log"
                detailed=text+'\n'+(workerlog.read_text(errors='replace') if workerlog.exists() else '')
                if not paused(detailed):raise RuntimeError('Measurement failed: '+detailed[-3000:])
                target=dest/'runs'/f"round-{run_status.get('round',0):02d}"/run_status.get('method','unknown')
                preserve(target,model+'-invalid-run')
                for f in workerlog.parent.glob(workerlog.stem+'*'):preserve(f,model+'-invalid-log')
        status(state='profiling',model='edge')
        while True:
            code,text=launch([sys.executable,'-u',str(tools/'standard_comparison.py'),'profile','--root',str(args.root),
                '--output',str(args.output/'edge'),'--profile-methods','arcquant50','arcquant100','sharq','repair15','repair20',
                '--profile-cases','0','1','--profile-seed','301017'],args.output/'profile-edge.log')
            if not code:break
            if not paused(text):raise RuntimeError('Profile failed: '+text[-3000:])
            for f in (args.output/'edge/profiles').glob('*-request'):
                tag=f.name.removesuffix('-request')
                if (f.parent/(tag+'.complete.json')).exists():continue
                for item in f.parent.glob(tag+'*'):preserve(item,'invalid-profile')
        totals={model:json.loads((args.output/model/'summary.json').read_text())['successful_requests'] for model in ('edge','nano')}
        status(complete=True,state='complete',requests=totals,planned_requests=420,finished=time.time())
    except Exception as error:
        status(state='stopped_requires_review',error=repr(error));raise


if __name__=='__main__':main()

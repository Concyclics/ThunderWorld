"""Qualified two-worker continuation of the frozen Super quality experiment."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from baselines import sha,write
from super_quality_campaign import check_campaign, METHODS

HERE=Path(__file__).resolve().parent
CODE=('super_cached_runtime.py','super_prefix_prepare.py','super_cached_worker.py',
      'super_cached_campaign.py','super_cached_bootstrap.py','test_super_cached_runtime.py')


def now():return datetime.now(timezone.utc).isoformat()


def source_hashes():return {n:sha(HERE/n) for n in CODE}


def env(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TW_SHARED_CALIBRATION_GPUS='0,2',
        TW_SHARED_CALIBRATION_LOCK_SCOPE='device',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',VLLM_PLUGINS='')


def reconcile(root):
    tasks=json.loads((root/'generation_tasks.json').read_text())['tasks']
    queue=json.loads((root/'generation_queue.json').read_text())
    for task in tasks:
        receipt=root/'tasks'/task['id']/'generation/progress.json'
        if task['id'] in queue['completed']:
            if sha(receipt)!=queue['completed'][task['id']]:raise ValueError('Changed completed task')
        elif receipt.exists() and json.loads(receipt.read_text()).get('complete'):
            data=json.loads(receipt.read_text())
            if data['identity']['cases']!=task['cases'] or data['identity']['method']!=task['method']:
                raise ValueError('Wrong drained task identity')
            queue['completed'][task['id']]=sha(receipt)
    write(root/'generation_queue.json',queue)
    return tasks,queue


def prepare(root,deployment):
    check_campaign(root)
    deployment.mkdir(parents=True,exist_ok=True)
    tasks,queue=reconcile(root)
    remaining=[t for t in tasks if t['id'] not in queue['completed']]
    cases=sorted({c for t in remaining for c in t['cases']})
    write(deployment/'prefix_cases.json',cases)
    value=dict(created_utc=now(),phase='deployment_preparation',source_sha256=source_hashes(),
        campaign_sha256=sha(root/'campaign.json'),full_manifest_sha256=sha(root/'super/final/manifest.json'),
        original_completed_chunks=len(queue['completed']),original_completed_videos=sum(2*len(t['cases']) for t in tasks if t['id'] in queue['completed']),
        remaining_chunks=len(remaining),remaining_scenes=len(cases),gpus=[0,2],workers=2,
        scope='Only execution placement and deterministic AR prefix reuse change; all generation and quantization definitions stay frozen')
    write(deployment/'preparation.json',value)
    print(json.dumps(value),flush=True)


def qualification(root,deployment,bank):
    check_campaign(root)
    if not json.loads((bank/'manifest.json').read_text())['complete']:raise ValueError('Complete prefix bank required')
    code=source_hashes();bank_sha=sha(bank/'manifest.json')
    state=dict(complete=False,source_sha256=code,bank_sha256=bank_sha,checks={},started_utc=now(),performance_eligible=False)
    write(deployment/'qualification.json',state)
    smoke=root/'super/smoke/manifest.json';study=json.loads(smoke.read_text())
    def worker(gpu):
        checks={}
        for method in METHODS:
            out=deployment/'qualification'/f'gpu{gpu}'/method
            log=deployment/f'qualify_gpu{gpu}_{method}.log'
            cmd=[sys.executable,str(HERE/'super_cached_runtime.py'),'--bank',str(bank),
                str(HERE/'dataset_infer.py'),str(smoke),'--method',method,'--output',str(out),'--smoke']
            with log.open('ab') as f:
                subprocess.run(cmd,env=env(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)
            progress=json.loads((out/'progress.json').read_text())
            if not progress['complete'] or not progress.get('roundtrip_bitwise_equal'):raise ValueError('Failed A-B-A qualification')
            hashes=[]
            for entry in study['cases']:
                cid=entry['case']['id']
                old_path=root/'super/smoke/generation'/method/cid/'result.json'
                new_path=out/cid/'result.json'
                old=json.loads(old_path.read_text());new=json.loads(new_path.read_text())
                a={r['seed']:r['sha256'] for r in old['rows']};b={r['seed']:r['sha256'] for r in new['rows']}
                if a!=b:raise ValueError(f'Full-output mismatch: GPU{gpu}/{method}/{cid}')
                for r in new['rows']:
                    if r['actual_mma_ratio']!=next(x['actual_mma_ratio'] for x in old['rows'] if x['seed']==r['seed']):
                        raise ValueError('Compute budget changed')
                hashes.append(dict(reference=str(old_path),reference_sha256=sha(old_path),candidate=str(new_path),
                    candidate_sha256=sha(new_path),signals_bitwise_equal=True))
            checks[method]=dict(complete=True,outputs_bitwise_equal=True,ABA_bitwise_equal=True,receipts=hashes,
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for e in study['cases']
                    for r in json.loads((out/e['case']['id']/'result.json').read_text())['rows']))
            write(deployment/f'qualification_gpu{gpu}.json',dict(complete=False,checks=checks))
            print(json.dumps(dict(qualified_gpu=gpu,method=method,bitwise_equal=True)),flush=True)
        write(deployment/f'qualification_gpu{gpu}.json',dict(complete=True,checks=checks))
        return gpu,checks
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,g) for g in (0,2)]
        for f in as_completed(futures):
            gpu,checks=f.result();state['checks'][str(gpu)]=checks;write(deployment/'qualification.json',state)
    if source_hashes()!=code or sha(bank/'manifest.json')!=bank_sha:raise ValueError('Qualification inputs changed')
    state.update(complete=True,finished_utc=now());write(deployment/'qualification.json',state)


def seal(root,deployment,bank):
    check_campaign(root)
    q=json.loads((deployment/'qualification.json').read_text())
    if not q['complete'] or q['source_sha256']!=source_hashes() or q['bank_sha256']!=sha(bank/'manifest.json'):
        raise ValueError('Matching complete two-GPU qualification required')
    value=dict(schema=1,created_utc=now(),complete=True,source_sha256=source_hashes(),
        campaign_sha256=sha(root/'campaign.json'),generation_manifest_sha256=sha(root/'super/final/manifest.json'),
        bank=str(bank.resolve()),bank_sha256=sha(bank/'manifest.json'),qualification_sha256=sha(deployment/'qualification.json'),
        gpus=[0,2],workers=2,chunks_per_resident_session=4,performance_eligible=False,
        scope='Independent single-GPU DM workers, exact BF16 AR prefixes; offline prefix work/storage separately recorded. No numerical or sampling policy changes.')
    file=deployment/'manifest.json'
    if file.exists():
        old=json.loads(file.read_text());value['created_utc']=old['created_utc']
        if value!=old:raise ValueError('Changed sealed deployment')
    write(file,value)


def verify_deployment(root,deployment):
    check_campaign(root)
    m=json.loads((deployment/'manifest.json').read_text())
    if not m['complete'] or m['source_sha256']!=source_hashes():raise ValueError('Changed cached deployment code')
    for file,key in [(root/'campaign.json','campaign_sha256'),(root/'super/final/manifest.json','generation_manifest_sha256'),
                     (Path(m['bank'])/'manifest.json','bank_sha256'),(deployment/'qualification.json','qualification_sha256')]:
        if sha(file)!=m[key]:raise ValueError('Changed cached deployment asset: '+str(file))
    return m


def make_aliases(root,task):
    for cid in task['cases']:
        folder=root/'tasks'/task['id']/'generation'/cid;folder.mkdir(parents=True,exist_ok=True)
        canonical=root/'super/final/generation'/task['method']/cid
        canonical.parent.mkdir(parents=True,exist_ok=True)
        if not canonical.exists():canonical.symlink_to(folder,target_is_directory=True)
        if canonical.resolve()!=folder.resolve():raise ValueError('Changed output alias')
        log=folder/'generation.log'
        if not log.exists() and not log.is_symlink():log.symlink_to(root/('generate_'+task['id']+'.log'))
        if log.resolve()!=root/('generate_'+task['id']+'.log'):raise ValueError('Changed generation log alias')


def run(root,deployment):
    spec=verify_deployment(root,deployment)
    lock=(root/'campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks,queue=reconcile(root)
    state=json.loads((root/'progress.json').read_text())
    state.pop('paused',None);state.pop('error',None)
    state.update(phase='parallel_cached_AR_generation',deployment=dict(path=str(deployment/'manifest.json'),sha256=sha(deployment/'manifest.json')),
        workers=2,updated_utc=now())
    write(root/'progress.json',state);write(root/'super/workflow.json',dict(paused=False,generation_complete=False))
    exporter_log=(deployment/'video_export.log').open('ab')
    exporter=subprocess.Popen([sys.executable,str(HERE/'judge_video_export.py'),'--root',str(root/'super'),'--follow'],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=exporter_log,stderr=subprocess.STDOUT)
    active={};reserved=set();job_number=0
    try:
        while len(queue['completed'])<len(tasks):
            verify_deployment(root,deployment)
            if exporter.poll() is not None and exporter.returncode!=0:raise RuntimeError('Video exporter failed')
            for gpu,job in list(active.items()):
                process=job['process']
                for task in job['tasks']:
                    file=root/'tasks'/task['id']/'generation/progress.json'
                    if file.exists() and json.loads(file.read_text()).get('complete') and task['id'] not in queue['completed']:
                        queue['completed'][task['id']]=sha(file)
                        state['stages']['generate_'+task['id']]=sha(file)
                if process.poll() is not None:
                    job['log'].close()
                    if process.returncode:raise RuntimeError('Cached worker failed: '+str(job['file']))
                    if any(t['id'] not in queue['completed'] for t in job['tasks']):raise RuntimeError('Worker omitted a chunk')
                    reserved.difference_update(t['id'] for t in job['tasks']);del active[gpu]
            write(root/'generation_queue.json',queue)
            pending=[t for t in tasks if t['id'] not in queue['completed'] and t['id'] not in reserved]
            if shutil.disk_usage(root).free<150*2**30:raise RuntimeError('Less than 150 GiB free disk')
            from thunderworld_fp4.cosmos.resources import free_memory
            free=free_memory()
            for gpu in (0,2):
                if gpu in active or not pending or free.get(gpu,0)<88*1024:continue
                method=pending[0]['method'];selected=[t for t in pending if t['method']==method][:spec['chunks_per_resident_session']]
                for task in selected:make_aliases(root,task)
                job_number+=1;file=deployment/'jobs'/f'{time.time_ns()}_{gpu}_{method}.json'
                write(file,dict(method=method,tasks=selected,gpu=gpu,started_utc=now()))
                logfile=file.with_suffix('.log').open('ab')
                command=[sys.executable,str(HERE/'super_cached_worker.py'),'--root',str(root),'--deployment',str(deployment),'--job',str(file)]
                process=subprocess.Popen(command,env=env(gpu),stdout=logfile,stderr=subprocess.STDOUT)
                active[gpu]=dict(process=process,tasks=selected,file=file,log=logfile,method=method)
                reserved.update(t['id'] for t in selected);pending=[t for t in pending if t['id'] not in reserved]
            count=sum(2*len(t['cases']) for t in tasks if t['id'] in queue['completed'])
            partial=0
            for job in active.values():
                for task in job['tasks']:
                    if task['id'] in queue['completed']:continue
                    for cid in task['cases']:
                        file=root/'tasks'/task['id']/'generation'/cid/'result.json'
                        if file.exists():partial+=len(json.loads(file.read_text()).get('rows',[]))
            state.update(generated_videos=count,generated_videos_including_active=count+partial,updated_utc=now(),
                phase='parallel_cached_AR_generation' if active else 'waiting_resources',
                active_jobs={str(g):dict(pid=j['process'].pid,method=j['method'],job=str(j['file']),tasks=[t['id'] for t in j['tasks']]) for g,j in active.items()})
            write(root/'progress.json',state)
            if len(queue['completed'])<len(tasks):time.sleep(15)
        for job in active.values():
            if job['process'].wait()!=0:raise RuntimeError('Last worker failed')
        queue['complete']=True;write(root/'generation_queue.json',queue)
        state.update(generated_videos=30492,generated_videos_including_active=30492,active_jobs={})
        write(root/'super/workflow.json',dict(paused=False,generation_complete=True))
        if exporter.wait()!=0:raise RuntimeError('Export failed')
        if not json.loads((root/'super/export_progress.json').read_text())['complete']:raise RuntimeError('Incomplete exports')
        for phase,script,args,receipt in [
            ('full_pixel_metrics','super_quality_metrics.py',['--root',str(root)],root/'metrics/progress.json'),
            ('full_task_judge','super_task_judge.py',['run','--campaign',str(root)],root/'judge/report/complete.json')]:
            state.update(phase=phase,updated_utc=now());write(root/'progress.json',state)
            with (root/(phase+'.log')).open('ab') as log:
                subprocess.run([sys.executable,str(HERE/script),*args],env=dict(env('0,2'),TW_SHARED_CALIBRATION_LOCK_SCOPE='global'),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            if not json.loads(receipt.read_text())['complete']:raise RuntimeError('Incomplete '+phase)
            state['stages'][phase]=sha(receipt)
        state.update(complete=True,phase='generation_metrics_and_task_judge_complete',completed_utc=now(),task_judge_complete=True)
        write(root/'progress.json',state)
    except BaseException as error:
        state.update(complete=False,paused=True,error=repr(error),updated_utc=now());write(root/'progress.json',state)
        write(root/'super/workflow.json',dict(paused=True,error=repr(error)))
        raise
    finally:
        for job in active.values():
            if job['process'].poll() is None:
                job['process'].terminate();job['process'].wait()
        if exporter.poll() is None:exporter.terminate();exporter.wait()
        exporter_log.close();lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','qualify','seal','run'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--deployment',type=Path)
    p.add_argument('--bank',type=Path)
    a=p.parse_args();root=a.root.resolve();deployment=(a.deployment or root/'cached_ar_deployment_20261003').resolve()
    bank=(a.bank or deployment/'prefix_bank').resolve()
    if a.operation=='prepare':prepare(root,deployment)
    elif a.operation=='qualify':qualification(root,deployment,bank)
    elif a.operation=='seal':seal(root,deployment,bank)
    else:run(root,deployment)


if __name__=='__main__':main()

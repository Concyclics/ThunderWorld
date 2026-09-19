"""Frozen judge-video generation with a shared, cost-ordered two-GPU task queue."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

from baselines import sha, write
from quality_benchmark import copy_checked, immutable, quality_sources

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
METHODS=('arcquant','arcquant100','sharq_perf','b1.5_protected_n','b2_joint')
RESOURCE_SOURCE='packages/fp4-repair/src/thunderworld_fp4/cosmos/resources.py'


def now():return datetime.now(timezone.utc).isoformat()


def sources():
    return dict(quality_sources(),**{str(p.relative_to(REPO)):sha(p) for p in
        (HERE/'judge_generation.py',HERE/'judge_video_export.py',HERE/'full_i2v_dataset.py')})


def attach_reuse(root, model, base, cohort):
    """Reference immutable original receipts; never rewrite their request identity."""
    previous=Path(cohort['reuse_root'])/model
    manifest=previous/'final/manifest.json'
    if sha(manifest)!=cohort['reuse_manifest_sha256'][model]:raise ValueError('Changed reuse manifest')
    old=json.loads(manifest.read_text());old_job=json.loads((previous.parent/'manifest.json').read_text())
    if any(old_job['source_sha256'].get(k)!=v for k,v in quality_sources().items()):
        raise ValueError('Changed numerical sources since reusable generation')
    if old['methods']!=base['methods'] or old['adapters']!=base['adapters'] or old['steps']!=base['steps']:
        raise ValueError('Changed reusable inference configuration')
    if not json.loads((previous/'export_progress.json').read_text())['complete']:
        raise ValueError('Reusable videos incomplete')
    by_id={e['case']['id']:e for e in old['cases']}
    ids=set(cohort['reused_case_ids']);entries={e['case']['id']:e for e in base['cases']}
    if ids!=set(by_id) or not ids<=set(entries):raise ValueError('Changed reuse cohort')
    receipts={};metadata={}
    for cid in sorted(ids):
        entry=entries[cid]
        if any(entry[k]!=by_id[cid][k] for k in ('case','seeds','image_sha256')):
            raise ValueError('Changed reusable case: '+cid)
        for method in base['methods']:
            folder=previous/'final/generation'/method/cid
            receipt=folder/'result.json';value=json.loads(receipt.read_text())
            if not value['complete'] or {r['seed'] for r in value['rows']}!=set(entry['seeds']):
                raise ValueError('Incomplete reusable request')
            receipts[method+'/'+cid]=sha(receipt)
            for seed in entry['seeds']:
                p=previous/'videos'/entry['benchmark']/method/cid/f'seed{seed}.json'
                sample=json.loads(p.read_text())
                if sample['method_spec']!=base['methods'][method]:raise ValueError('Changed reusable method')
                metadata[f'{method}/{cid}/{seed}']=sha(p)
            link=root/model/'final/generation'/method/cid;link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():link.symlink_to(folder,target_is_directory=True)
            if link.resolve()!=folder.resolve():raise ValueError('Changed reuse alias')
    return dict(root=str(previous),manifest_sha256=sha(manifest),case_ids=sorted(ids),
                receipt_sha256=receipts,metadata_sha256=metadata,
                videos=sum(len(entries[cid]['seeds']) for cid in ids)*len(base['methods']))


def prepare(a):
    if len(a.gpus)!=2 or len(set(a.gpus))!=2 or any(i<0 for i in a.gpus) or not 1<=a.chunk_scenes<=12:
        raise ValueError('Two distinct physical GPUs and 1-12 scenes per task required')
    a.root.mkdir(parents=True,exist_ok=True)
    cohort=json.loads(a.cohort.read_text()) if a.cohort else None
    if cohort:
        from collections import Counter
        if (not cohort['frozen'] or not cohort['complete'] or cohort['coverage']!='all_I2V_cases' or
            dict(Counter(e['benchmark'] for e in cohort['cases']))!={'rbench':650,'paibench':1044} or
            len({e['case']['id'] for e in cohort['cases']})!=1694 or
            any(len(e['seeds'])!=2 for e in cohort['cases']) or
            any(cohort['protocol'].get(k)!=v for k,v in dict(width=320,height=192,frames=61,
                fps=24,steps=35,guidance_scale=6.,flow_shift=3.).items())):
            raise ValueError('Unsupported or incomplete full-corpus manifest')
    encoding=dict(fps=24,ffmpeg='/usr/bin/ffmpeg',ffprobe='/usr/bin/ffprobe',
        mp4=dict(codec='libx264',pixel_format='yuv420p',crf=15,preset='medium'),
        lossless=dict(codec='ffv1',pixel_format='bgr0',level=3,verify_rgb_roundtrip=True),
        float_to_rgb='round(clip(video,0,1)*255) to uint8; original float32 NPZ retained')
    codec_tools={key:dict(path=encoding[key],sha256=sha(encoding[key]),version=subprocess.check_output(
        [encoding[key],'-version'],text=True).splitlines()[0]) for key in ('ffmpeg','ffprobe')}
    job=dict(schema=1,models={},methods=list(METHODS),gpus=a.gpus,steps=35,video_export=encoding,
        codec_tools=codec_tools,source_sha256=sources(),performance_eligible=False,tasks=[],
        scheduling=dict(kind='Shared longest-estimated-task-first queue; any model on either GPU',
            chunk_scenes=a.chunk_scenes,estimates='Historical shared-environment smoke request times for scheduling only, not performance evidence'),
        scope=cohort['scope'] if cohort else 'User-selected five methods on the frozen final 72-scene I2V subset; judge inputs only, no new quality-based selection or performance ranking.')
    if cohort:job.update(cohort=str(a.cohort.resolve()),cohort_sha256=sha(a.cohort))
    for model in ('edge','nano'):
        parent=a.source/model;q=json.loads((parent/'qualification.json').read_text())
        if (not q['complete'] or q['final_manifest_sha256']!=sha(parent/'final/manifest.json') or
            q['shape_preflight_sha256']!=sha(parent/'shape_preflight.json')):
            raise ValueError('Unqualified original model matrix')
        if cohort:
            smoke_link=a.root/model/'smoke';smoke_link.parent.mkdir(parents=True,exist_ok=True)
            if not smoke_link.exists():smoke_link.symlink_to(parent/'smoke',target_is_directory=True)
            if smoke_link.resolve()!=(parent/'smoke').resolve():raise ValueError('Changed shape reference')
        current=quality_sources()
        changes={k:dict(previous=q['quality_sources'].get(k),current=current.get(k))
                 for k in set(current)|set(q['quality_sources']) if current.get(k)!=q['quality_sources'].get(k)}
        if set(changes)-{RESOURCE_SOURCE}:raise ValueError('Numerical or worker sources changed: '+str(changes))
        study=json.loads((parent/'final/manifest.json').read_text())
        if len(study['cases'])!=72 or sum(len(e['seeds']) for e in study['cases'])!=144:
            raise ValueError('Changed frozen final cohort')
        if cohort:study['cases']=copy.deepcopy(cohort['cases'])
        copies={}
        for e in study['cases']:
            src=Path(e['case']['image']);target=a.root/'inputs'/e['benchmark']/(e['case']['id']+src.suffix)
            copy_checked(src,target,e['image_sha256']);copies[e['case']['id']]=str(target)
        base=copy.deepcopy(study)
        base['methods']={n:base['methods'][n] for n in METHODS}
        base['adapters']={n:v for n,v in base['adapters'].items() if n in METHODS}
        base.update(primary_comparisons=[],source_qualification=str(parent/'qualification.json'),
            source_qualification_sha256=sha(parent/'qualification.json'),video_export=encoding,input_copies=copies,
            gpu_policy=dict(allowed=a.gpus,lock_scope='device',assignment='dynamic'),scope=job['scope'],source_output_hashes=None)
        if cohort:
            base.update(role='full_I2V_corpus',cohort_sha256=sha(a.cohort),protocol=cohort['protocol'],
                prior_scene_role_counts=cohort['prior_scene_role_counts'])
            base['reuse']=attach_reuse(a.root,model,base,cohort)
        immutable(a.root/model/'final/manifest.json',base)
        old=json.loads((parent/'smoke/manifest.json').read_text());hashes={};estimates={}
        for name in METHODS:
            if old['methods'][name]!=study['methods'][name]:raise ValueError('Changed method after smoke qualification')
            hashes[name]={};estimates[name]={}
            for e in old['cases']:
                folder=parent/'smoke/generation'/name/e['case']['id']
                receipt=json.loads((folder/'result.json').read_text())
                if not receipt['complete']:raise ValueError('Incomplete source smoke')
                for row in receipt['rows']:
                    if sha(folder/row['outputs'])!=row['outputs_sha256']:raise ValueError('Corrupt source smoke')
                hashes[name][e['case']['id']]={str(row['seed']):row['sha256'] for row in receipt['rows']}
                estimates[name][e['benchmark']]=statistics.mean(row['wall_seconds'] for row in receipt['rows'])
        hp=a.root/model/'qualification/source_output_hashes.json';immutable(hp,hashes)
        smoke=copy.deepcopy(base);smoke.update(role='development_gpu_qualification',cases=old['cases'],
            source_output_hashes=str(hp),source_output_hashes_sha256=sha(hp))
        smoke.pop('reuse',None)
        immutable(a.root/model/'qualification/manifest.json',smoke)
        job['models'][model]=dict(source_model_root=str(parent),source_manifest_sha256=sha(parent/'final/manifest.json'),
            final_manifest_sha256=sha(a.root/model/'final/manifest.json'),
            smoke_manifest_sha256=sha(a.root/model/'qualification/manifest.json'),source_admission_changes=changes,
            expected_videos=sum(len(e['seeds']) for e in base['cases'])*len(METHODS),
            expected_per_benchmark={b:sum(len(e['seeds']) for e in base['cases'] if e['benchmark']==b)*len(METHODS)
                                    for b in ('rbench','paibench')},
            reused_videos=base.get('reuse',{}).get('videos',0),
            qualification='Each GPU reproduces all five methods on four previous outputs plus A-B-A before claiming tasks')
        for name in METHODS:
            for benchmark in ('rbench','paibench'):
                entries=[e for e in study['cases'] if e['benchmark']==benchmark]
                if not cohort and len(entries)!=36:raise ValueError('Changed benchmark scene count')
                reuse=set(base.get('reuse',{}).get('case_ids',[]))
                entries=[e for e in entries if e['case']['id'] not in reuse]
                for start in range(0,len(entries),a.chunk_scenes):
                    part=entries[start:start+a.chunk_scenes];tid=f'{model}__{name}__{benchmark}__{start:03d}'
                    task=dict(id=tid,model=model,method=name,benchmark=benchmark,cases=[e['case']['id'] for e in part],
                        videos=sum(len(e['seeds']) for e in part),
                        estimated_seconds=len(part)*(2+4/35)*estimates[name][benchmark])
                    job['tasks'].append(task)
                    for e in part:
                        folder=a.root/'tasks'/tid/'generation'/e['case']['id'];folder.mkdir(parents=True,exist_ok=True)
                        canonical=a.root/model/'final/generation'/name/e['case']['id'];canonical.parent.mkdir(parents=True,exist_ok=True)
                        if not canonical.exists():canonical.symlink_to(folder,target_is_directory=True)
                        if canonical.resolve()!=folder.resolve():raise ValueError('Changed generation alias')
                        log=folder/'generation.log';target=a.root/'tasks'/tid/'generation.log'
                        if not log.is_symlink():log.symlink_to(target)
                        if log.resolve()!=target.resolve():raise ValueError('Changed log alias')
    job['tasks'].sort(key=lambda t:(-t['estimated_seconds'],t['id']))
    immutable(a.root/'manifest.json',job)
    if not (a.root/'queue.json').exists():
        write(a.root/'queue.json',dict(job_sha256=sha(a.root/'manifest.json'),tasks={t['id']:dict(status='pending',attempts=[]) for t in job['tasks']}))
    print(json.dumps(dict(prepared=True,root=str(a.root),
        videos=sum(m['expected_videos'] for m in job['models'].values()),
        reused_videos=sum(m['reused_videos'] for m in job['models'].values()),
        tasks=len(job['tasks']),gpus=a.gpus)),flush=True)


def verify_job(root):
    job=json.loads((root/'manifest.json').read_text())
    if job['source_sha256']!=sources():raise ValueError('Changed implementation after judge job was frozen')
    if job.get('cohort') and sha(job['cohort'])!=job['cohort_sha256']:raise ValueError('Changed corpus')
    for entry in job['codec_tools'].values():
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Changed video encoder')
    for model,spec in job['models'].items():
        for role,key in [('final','final_manifest_sha256'),('qualification','smoke_manifest_sha256')]:
            if sha(root/model/role/'manifest.json')!=spec[key]:raise ValueError('Changed model manifest')
        if job.get('cohort'):
            preflight=json.loads((root/model/'shape_preflight.json').read_text())
            if (not preflight['complete'] or preflight['manifest_sha256']!=spec['final_manifest_sha256'] or
                preflight['source_sha256']!=sha(HERE/'quality_shape_preflight.py') or
                set(preflight['checks'])!={'b1.5_protected_n','b2_joint'} or
                any(len(c['cases'])!=1694 or not all(r['feasible'] for r in c['cases'])
                    for c in preflight['checks'].values())):
                raise ValueError('Full-corpus shape/budget preflight incomplete')
    return job


def queue_update(root, callback):
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((root/'queue.json').read_text())
        if state['job_sha256']!=sha(root/'manifest.json'):raise ValueError('Changed queue identity')
        result=callback(state)
        write(root/'queue.json',state)
        return result


def claim(root, job, gpu):
    def take(state):
        for task in job['tasks']:
            item=state['tasks'][task['id']]
            if item['status']=='pending':
                item.update(status='running',gpu=gpu,pid=os.getpid(),started_utc=now())
                item['attempts'].append(dict(gpu=gpu,pid=os.getpid(),started_utc=item['started_utc']))
                return task
    return queue_update(root,take)


def worker(a):
    job=verify_job(a.root)
    if a.gpu not in job['gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('Wrong worker GPU')
    r=a.root/f'gpu{a.gpu}';r.mkdir(exist_ok=True)
    state=dict(complete=False,pid=os.getpid(),gpu=a.gpu,phase='qualification',qualified={},completed_tasks=[],started_utc=now())
    def event(kind,**fields):
        line=json.dumps(dict(utc=now(),event=kind,gpu=a.gpu,**fields))
        with (r/'events.jsonl').open('a') as stream:stream.write(line+'\n')
        print(line,flush=True)
    def command(log,args):
        log.parent.mkdir(parents=True,exist_ok=True)
        event('command_start',command=list(map(str,args)),log=str(log))
        with log.open('a') as stream:
            stream.write('\n'+json.dumps(dict(utc=now(),command=list(map(str,args))))+'\n');stream.flush()
            subprocess.run(list(map(str,args)),stdout=stream,stderr=subprocess.STDOUT,check=True)
    current=None
    def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    with (r/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for model in ('edge','nano'):
                for name in METHODS:
                    dest=r/'qualification'/model/name
                    command(dest/'generation.log',[sys.executable,HERE/'dataset_infer.py',a.root/model/'qualification/manifest.json',
                        '--method',name,'--output',dest,'--smoke','--require-source-hash'])
                    result=json.loads((dest/'progress.json').read_text())
                    if not result['complete'] or not result['roundtrip_bitwise_equal']:raise ValueError('GPU qualification incomplete')
                    state['qualified'][model+'/'+name]=sha(dest/'progress.json');write(r/'status.json',state)
                    event('qualified',model=model,method=name)
            state['phase']='generation';write(r/'status.json',state)
            while True:
                verify_job(a.root)
                current=claim(a.root,job,a.gpu)
                if current is None:break
                taskroot=a.root/'tasks'/current['id'];event('task_start',task=current['id'],videos=current['videos'])
                command(taskroot/'generation.log',[sys.executable,HERE/'dataset_infer.py',a.root/current['model']/'final/manifest.json',
                    '--method',current['method'],'--output',taskroot/'generation','--cases',*current['cases']])
                progress=taskroot/'generation/progress.json';result=json.loads(progress.read_text())
                if not result['complete'] or set(result['completed'])!=set(current['cases']):raise ValueError('Incomplete task')
                def finish(q):q['tasks'][current['id']].update(status='complete',completed_utc=now(),progress_sha256=sha(progress))
                queue_update(a.root,finish)
                state['completed_tasks'].append(current['id']);write(r/'status.json',state)
                event('task_complete',task=current['id']);current=None
            state.update(complete=True,phase='complete',completed_utc=now())
        except BaseException as error:
            state.update(paused=True,error=repr(error),traceback=traceback.format_exc(),stopped_utc=now())
            if current:
                def fail(q):q['tasks'][current['id']].update(status='failed',error=repr(error),stopped_utc=now())
                queue_update(a.root,fail)
            event('paused',error=repr(error));raise
        finally:write(r/'status.json',state)


def active(pid, root):
    p=Path(f'/proc/{pid}/cmdline')
    if not p.exists():return False
    cmd=p.read_bytes()
    return b'judge_generation.py' in cmd and str(root).encode() in cmd


def snapshot(root, job):
    queue=json.loads((root/'queue.json').read_text());models={}
    for model in job['models']:
        tasks=[queue['tasks'][t['id']] for t in job['tasks'] if t['model']==model]
        generated=sum(len(json.loads(p.read_text())['rows']) for p in (root/model/'final/generation').glob('*/*/result.json'))
        ep=root/model/'export_progress.json';exports=json.loads(ep.read_text()) if ep.exists() else {}
        models[model]=dict(generation_complete=all(t['status']=='complete' for t in tasks),
            generated=generated,expected=job['models'][model]['expected_videos'],
            reused=job['models'][model].get('reused_videos',0),exported=exports.get('completed',0),
            complete=all(t['status']=='complete' for t in tasks) and exports.get('complete',False),performance_eligible=False)
    workers={}
    for gpu in job['gpus']:
        p=root/f'gpu{gpu}/status.json'
        workers[str(gpu)]=json.loads(p.read_text()) if p.exists() else dict(phase='starting')
    return dict(complete=all(x['complete'] for x in models.values()),models=models,workers=workers,updated_utc=now(),performance_eligible=False)


def supervise(a):
    job=verify_job(a.root);children=[];state={}
    with (a.root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def recover(queue):
            for row in queue['tasks'].values():
                if row['status'] in ('running','failed'):
                    if active(row.get('pid',0),a.root):raise RuntimeError('Previous worker is still active')
                    row['status']='pending'
        queue_update(a.root,recover)
        try:
            for model in job['models']:
                write(a.root/model/'workflow.json',dict(complete=False,generation_complete=False,phase='qualification'))
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
                with (a.root/model/'exporter.log').open('ab') as log:
                    proc=subprocess.Popen([sys.executable,str(HERE/'judge_video_export.py'),'--root',str(a.root/model),'--follow'],
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(('exporter_'+model,proc))
            for gpu in job['gpus']:
                r=a.root/f'gpu{gpu}';r.mkdir(exist_ok=True)
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    TW_SHARED_CALIBRATION_GPUS=','.join(map(str,job['gpus'])),TW_SHARED_CALIBRATION_LOCK_SCOPE='device',
                    PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
                with (r/'supervisor.log').open('ab') as log:
                    proc=subprocess.Popen([sys.executable,str(HERE/'judge_generation.py'),'worker','--root',str(a.root),'--gpu',str(gpu)],
                        cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(('gpu'+str(gpu),proc))
            write(a.root/'children.json',dict(processes=[dict(name=n,pid=p.pid) for n,p in children]))
            while True:
                state=snapshot(a.root,job)
                for model,info in state['models'].items():write(a.root/model/'workflow.json',info)
                write(a.root/'pool_status.json',state)
                failed=[name for name,p in children if p.poll() is not None and p.returncode!=0]
                if failed:raise RuntimeError('Worker failed: '+str(failed))
                if state['complete']:
                    from judge_video_export import jsonlines
                    rows=[];blind=[]
                    for model in job['models']:
                        rows += [json.loads(s) for s in (a.root/model/'judge_index.jsonl').read_text().splitlines()]
                        blind += [json.loads(s) for s in (a.root/model/'judge_blind/inputs.jsonl').read_text().splitlines()]
                    jsonlines(a.root/'judge_index.jsonl',rows)
                    jsonlines(a.root/'judge_inputs.jsonl',sorted(blind,key=lambda r:r['sample_id']))
                    break
                if all(p.poll() is not None for _,p in children):raise RuntimeError('Workers exited with incomplete outputs')
                time.sleep(5)
        except BaseException as error:
            state.update(complete=False,paused=True,error=repr(error),traceback=traceback.format_exc(),updated_utc=now())
            for model in job['models']:write(a.root/model/'workflow.json',dict(state.get('models',{}).get(model,{}),paused=True,error=repr(error)))
            raise
        finally:
            if state.get('complete'):
                for name,p in children:
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
            for name,p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for name,p in children:
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            write(a.root/'pool_status.json',state)


def launch(a):
    verify_job(a.root);path=a.root/'launch.json'
    if path.exists() and active(json.loads(path.read_text())['pid'],a.root):
        print(path.read_text());return
    state=a.root/'pool_status.json'
    if state.exists() and json.loads(state.read_text()).get('complete'):
        print(state.read_text());return
    command=[sys.executable,str(HERE/'judge_generation.py'),'supervise','--root',str(a.root)]
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
    with (a.root/'supervisor.log').open('ab') as log:
        process=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(pid=process.pid,started_utc=now(),command=command,log=str(a.root/'supervisor.log'))
    write(path,value);print(json.dumps(value),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','launch','supervise','worker','status'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('/home/yufeng/thunderworld/full_quality_matrix_20260919'))
    p.add_argument('--gpus',nargs=2,type=int,default=[0,1]);p.add_argument('--gpu',type=int)
    p.add_argument('--chunk-scenes',type=int,default=6)
    p.add_argument('--cohort',type=Path,help='Full I2V corpus from full_i2v_dataset.py; default preserves the old subset')
    a=p.parse_args();a.root=a.root.resolve()
    if a.operation=='worker' and a.gpu is None:p.error('--gpu required for worker')
    if a.operation=='status':print(json.dumps(snapshot(a.root,verify_job(a.root)),indent=2))
    else:{'prepare':prepare,'launch':launch,'supervise':supervise,'worker':worker}[a.operation](a)


if __name__=='__main__':main()

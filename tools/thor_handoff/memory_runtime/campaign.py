"""Resumable serial Thor memory scaling campaign, one fresh process per point."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_baselines import write,sha
from compare_speed import idle_guard
from standard_comparison import freeze_model,snapshot

METHODS=['bf16','fp8','fp4_base','b1.5_protected_n','b2_joint']
SHAPES=[(320,192,61),(320,192,125),(320,192,189),(480,288,61),(640,384,61)]


def host_sample(pid,folder):
    sample={'time':time.time()}
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            k,_,v=line.partition(':')
            if k in ('VmRSS','VmHWM'):sample[k+'_bytes']=int(v.split()[0])*1024
        for line in Path('/proc/meminfo').read_text().splitlines():
            k,_,v=line.partition(':')
            if k in ('MemTotal','MemAvailable','SwapFree','SwapTotal'):sample[k+'_bytes']=int(v.split()[0])*1024
        if (folder/'phase.json').exists():sample['phase']=json.loads((folder/'phase.json').read_text())['stage']
    except FileNotFoundError:pass
    return sample


def prepare(a):
    a.output.mkdir(parents=True,exist_ok=False)
    prior=None
    if a.reuse_controls_from:
        prior=json.loads((a.reuse_controls_from/'manifest.json').read_text())
        if prior['worker_sha256']!=sha(Path(__file__).with_name('worker.py')):
            raise ValueError('Cannot reuse controls with a different worker')
        if prior['seeds']!=[301017,301018,301031] or prior['runtime_options']!={}:
            raise ValueError('Cannot reuse controls with different requests/runtime')
    fixed={}
    for model in ('edge','nano'):
        folder=a.output/'models'/model;folder.mkdir(parents=True)
        if prior:
            path=Path(prior['fixed'][model])
            if sha(path)!=prior['fixed_sha256'][model]:raise ValueError('Prior frozen manifest changed')
            copied=folder/path.name;shutil.copy2(path,copied);fixed[model]=str(copied)
        else:
            fixed[model]=str(freeze_model(SimpleNamespace(root=a.root,model=model,output=folder,
                           methods=['bf16','fp8','fp4','repair15','repair20'])))
    jobs=[];rng=random.Random(230929)
    # Matched original-shape controls isolate storage/lifetime from numerics.
    for model in ('edge','nano'):
        for mode,methods in [('legacy',['fp4_base','b1.5_protected_n','b2_joint']),
                             ('release',['b1.5_protected_n','b2_joint']),('compact',METHODS)]:
            order=methods.copy();rng.shuffle(order)
            for method in order:jobs.append(dict(model=model,mode=mode,method=method,shape=list(SHAPES[0])))
    for shape in SHAPES[1:]:
        for model in ('edge','nano'):
            order=METHODS.copy();rng.shuffle(order)
            for method in order:jobs.append(dict(model=model,mode='compact',method=method,shape=list(shape)))
    for i,job in enumerate(jobs):job['id']=f"{i:03d}-{job['model']}-{job['mode']}-{job['method']}-"+'x'.join(map(str,job['shape']))
    variants={mode:str(a.root/'variants'/('cutlass-mainloop-v6' if mode=='legacy' else a.variant)) for mode in ('legacy','release','compact')}
    reused=[];references={}
    if prior:
        previous_jobs={j['id']:j for j in prior['jobs']}
        for job in jobs:
            source=a.reuse_controls_from/'runs'/job['id']/'result.json'
            if not source.exists():continue
            data=json.loads(source.read_text())
            if not data['complete']:continue
            if previous_jobs.get(job['id'])!=job:raise ValueError('Prior job differs')
            if data['manifest_sha256']!=sha(fixed[job['model']]):raise ValueError('Prior model differs')
            if [r['seed'] for r in data['rows']]!=[301017,301018,301031]:raise ValueError('Prior seeds differ')
            references[job['id']]=dict(path=str(source),sha256=sha(source))
            if job['mode']!='legacy':continue
            if data['variant_receipt_sha256']!=sha(Path(variants['legacy'])/'variant.json'):
                raise ValueError('Prior legacy variant differs')
            destination=a.output/'runs'/job['id'];destination.parent.mkdir(exist_ok=True)
            shutil.copytree(source.parent,destination)
            logs=a.output/'logs';logs.mkdir(exist_ok=True)
            for path in (a.reuse_controls_from/'logs').glob(job['id']+'.*'):
                shutil.copy2(path,logs/path.name)
            reused.append(dict(job=job['id'],source=str(source),sha256=sha(source)))
    write(a.output/'manifest.json',dict(created=time.time(),jobs=jobs,variants=variants,
        reused_controls=reused,output_references=references,
        variant_sha256={k:sha(Path(v)/'variant.json') for k,v in variants.items()},fixed=fixed,
        fixed_sha256={k:sha(v) for k,v in fixed.items()},seeds=[301017,301018,301031],steps=35,
        worker_sha256=sha(Path(__file__).with_name('worker.py')),environment=snapshot(),
        runtime_options={},execution='graph',debug='off',concurrency=1,batch=1,
        memory_sample_interval_seconds=.25,quality_evaluation=False,performance_eligible=False,
        scope='One fixed RBench scene, three seeds per point, five input sizes, fresh process per point; memory diagnostic, not latency leaderboard',
        policy='Frozen repair supports/weights; recompile request budget for each shape; infeasible shapes are retained, never silently reduce repair',
        memory_semantics='Torch allocated/reserved, process RSS/HWM, and system available RAM are separate overlapping measures; do not add'))
    print(json.dumps(dict(jobs=len(jobs),measured_requests=len(jobs)*3)),flush=True)


def run(a):
    lock=(a.output/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=json.loads((a.output/'manifest.json').read_text())
    if sha(Path(__file__).with_name('worker.py'))!=m['worker_sha256']:raise ValueError('Frozen worker changed')
    for model,path in m['fixed'].items():
        if sha(path)!=m['fixed_sha256'][model]:raise ValueError('Frozen model manifest changed')
    for mode,path in m['variants'].items():
        if sha(Path(path)/'variant.json')!=m['variant_sha256'][mode]:raise ValueError('Frozen variant changed')
    state=dict(complete=False,total=len(m['jobs']),completed=0,infeasible=0,started=time.time())
    write(a.output/'status.json',state)
    for job in m['jobs']:
        folder=a.output/'runs'/job['id'];result=folder/'result.json'
        if result.exists():
            old=json.loads(result.read_text())
            if old['complete']:state['completed']+=1;continue
            if old['status']=='infeasible_budget':state['infeasible']+=1;continue
            raise RuntimeError('Incomplete evidence retained, inspect before resuming: '+str(folder))
        if a.max_jobs is not None and state['completed']+state['infeasible']>=a.max_jobs:break
        idle_guard();state.update(current=job,updated=time.time());write(a.output/'status.json',state)
        logs=a.output/'logs';logs.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('worker.py')),
             '--root',str(a.root),'--manifest',m['fixed'][job['model']],
             '--variant',m['variants'][job['mode']],'--method',job['method'],
             '--mode',job['mode'],'--shape',*map(str,job['shape']),'--output',str(folder),
             '--seeds',*map(str,m['seeds'])]
        if job['mode']!='legacy' and job['shape']==list(SHAPES[0]):cmd+=['--verify-release']
        before=snapshot()
        samples=[]
        with (logs/(job['id']+'.log')).open('w') as log, (logs/(job['id']+'.memory.jsonl')).open('w') as stream:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            state['worker_pid']=child.pid;write(a.output/'status.json',state)
            while child.poll() is None:
                sample=host_sample(child.pid,folder);samples.append(sample)
                stream.write(json.dumps(sample)+'\n');stream.flush()
                time.sleep(m['memory_sample_interval_seconds'])
            code=child.wait()
        by_phase={}
        for phase in sorted({s.get('phase','startup') for s in samples}):
            rows=[s for s in samples if s.get('phase','startup')==phase and 'VmRSS_bytes' in s and 'MemAvailable_bytes' in s]
            if rows:by_phase[phase]=dict(samples=len(rows),rss_max_bytes=max(s['VmRSS_bytes'] for s in rows),
                process_hwm_bytes=max(s['VmHWM_bytes'] for s in rows),system_available_min_bytes=min(s['MemAvailable_bytes'] for s in rows))
        write(logs/(job['id']+'.environment.json'),dict(before=before,after=snapshot(),command=cmd,exit_code=code,host_samples=by_phase,
             initial_sample=samples[0] if samples else None,scope='250 ms external host sampling; sampled extrema, not an exact system peak'))
        if not result.exists():raise RuntimeError('Worker failed before writing receipt: '+job['id'])
        row=json.loads(result.read_text())
        if code:
            if row['status']=='infeasible_budget':state['infeasible']+=1
            else:
                state.update(error=row.get('error'),status=row['status'],updated=time.time())
                write(a.output/'status.json',state);report(a);raise RuntimeError('Worker paused/failed: '+job['id'])
        elif row['complete']:state['completed']+=1
        else:raise RuntimeError('Worker returned incomplete result')
        reference=m.get('output_references',{}).get(job['id'])
        if row['complete'] and reference:
            if sha(reference['path'])!=reference['sha256']:raise ValueError('Output reference changed')
            old=json.loads(Path(reference['path']).read_text())
            equal=[v['sha256'] for v in old['rows']]==[v['sha256'] for v in row['rows']]
            write(folder/'loader-equivalence.json',dict(bitwise_equal=equal,reference=reference,
                current_sha256=sha(result),scope='Three matched complete 35-step requests'))
            if not equal:raise AssertionError('Loader changed complete-request output: '+job['id'])
        state.update(updated=time.time(),worker_pid=None);write(a.output/'status.json',state);report(a)
        print(json.dumps(dict(completed=state['completed'],infeasible=state['infeasible'],total=state['total'],job=job['id'])),flush=True)
    state.update(complete=state['completed']+state['infeasible']==state['total'],finished=time.time(),worker_pid=None)
    write(a.output/'status.json',state);report(a)


def report(a):
    m=json.loads((a.output/'manifest.json').read_text());rows=[];failures=[];raw={}
    for job in m['jobs']:
        path=a.output/'runs'/job['id']/'result.json'
        if not path.exists():continue
        d=json.loads(path.read_text())
        if not d['complete']:
            failures.append(dict(job=job,status=d['status'],error=d.get('error')));continue
        times=[r['wall_seconds'] for r in d['rows']]
        envpath=a.output/'logs'/(job['id']+'.environment.json')
        env=json.loads(envpath.read_text()) if envpath.exists() else {}
        host=env.get('host_samples',{}).get('measure',{})
        prep=d['memory']['after_prepare'];warm=d['memory']['after_graph_warmup']
        row=dict(**job,requests=len(times),seconds=statistics.median(times),fps=job['shape'][2]/statistics.median(times),
            peak_allocated_gib=max(r['peak_allocated_bytes'] for r in d['rows'])/2**30,
            peak_reserved_gib=max(r['peak_reserved_bytes'] for r in d['rows'])/2**30,
            prepare_peak_allocated_gib=prep['peak_allocated_bytes']/2**30,
            graph_warmup_peak_allocated_gib=warm['peak_allocated_bytes']/2**30,
            ready_process_rss_gib=d['memory']['ready']['process_rss_bytes']/2**30,
            ready_system_available_gib=d['memory']['ready']['system_available_bytes']/2**30,
            sampled_request_rss_max_gib=host.get('rss_max_bytes',0)/2**30,
            sampled_request_system_available_min_gib=host.get('system_available_min_bytes',0)/2**30,
            released_host_tensor_gib=d['host_release']['released_host_tensor_bytes']/2**30,
            budget_ratio=d['rows'][0].get('compute_ledger',{}).get('ratio') if d['rows'][0].get('compute_ledger') else None,
            host_release_bitwise_equal=d.get('host_release_bitwise_equal'),source_sha256=sha(path))
        rows.append(row);raw[job['id']]=d
    controls=[]
    for model in ('edge','nano'):
        for method in ['fp4_base','b1.5_protected_n','b2_joint']:
            matched={r['mode']:r for r in rows if r['model']==model and r['method']==method and r['shape']==list(SHAPES[0])}
            if not {'legacy','compact'}<=set(matched):continue
            old,new=matched['legacy'],matched['compact']
            oldhash=[r['sha256'] for r in raw[old['id']]['rows']];newhash=[r['sha256'] for r in raw[new['id']]['rows']]
            controls.append(dict(model=model,method=method,bitwise_equal=oldhash==newhash,
                peak_allocated_saved_gib=old['peak_allocated_gib']-new['peak_allocated_gib'],
                ready_process_rss_saved_gib=old['ready_process_rss_gib']-new['ready_process_rss_gib'],
                speedup=old['seconds']/new['seconds']))
    complete=len(rows)+sum(f['status']=='infeasible_budget' for f in failures)==len(m['jobs'])
    write(a.output/'summary.json',dict(complete=complete,rows=rows,failures=failures,controls=controls,
        manifest_sha256=sha(a.output/'manifest.json'),quality_evaluation=False,performance_eligible=False,
        scope=m['scope'],memory_semantics=m['memory_semantics']))
    lines=['# Thor Edge / Nano 内存优化与尺寸扩展复测','',
           '相同 RBench 输入，每点三个种子，35 步、CFG 6、Graph、debug off。每个点独立加载和预热；所有方法外围融合开关相同。',
           '250 ms 外部主机采样；延迟为内存诊断附带结果，不是正式延迟排行榜。未重测视频质量。','',
           '| 模型 | 模式 | 方法 | 宽×高×帧 | 请求秒 | FPS | 请求 allocated GiB | 加载 allocated 峰值 GiB | 建图 allocated 峰值 GiB | 就绪 RSS GiB | 请求期可用 RAM 最小采样 GiB |',
           '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['mode']} | {r['method']} | {'×'.join(map(str,r['shape']))} | {r['seconds']:.3f} | {r['fps']:.3f} | {r['peak_allocated_gib']:.3f} | {r['prepare_peak_allocated_gib']:.3f} | {r['graph_warmup_peak_allocated_gib']:.3f} | {r['ready_process_rss_gib']:.3f} | {r['sampled_request_system_available_min_gib']:.3f} |")
    lines+=['','legacy：原 v6；release：逐层加载＋CPU 资产释放、仍用完整 A1 缓冲；compact：再加入紧凑 A1 tile 缓冲。',
            'Torch、RSS 与系统 RAM 指标存在重叠，不能相加。加载、建图峰值与请求峰值分别报告，不用后者代表部署全过程峰值。','',
            '原形状匹配控制（相同输出 hash；仅验证实现一致性，不是质量评分）：','']
    for c in controls:lines.append(f"- {c['model']} {c['method']}: bitwise={c['bitwise_equal']}；请求 allocated 少 {c['peak_allocated_saved_gib']:.3f} GiB；就绪 RSS 少 {c['ready_process_rss_saved_gib']:.3f} GiB；速度比 {c['speedup']:.3f}×。")
    if failures:
        lines+=['','未完成/预算不可行点：','']
        for f in failures:lines.append(f"- {f['job']['id']}: {f['status']}，{f['error']}")
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','run','report'))
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-jobs',type=int)
    p.add_argument('--variant',default='cutlass-memory-v7')
    p.add_argument('--reuse-controls-from',type=Path)
    a=p.parse_args();globals()[a.operation](a)


if __name__=='__main__':main()

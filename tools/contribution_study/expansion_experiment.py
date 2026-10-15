"""Frozen ten-scene Edge experiment for nested native FP4 residual expansions.

Run with tools/fp4_handoff/python.sh. No performance or monotonicity claims are
made by this quality experiment; all outputs, including negative results, stay.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from baselines import sha, write

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT = WORKSPACE / 'fp4_expansion_demo_20260921'
VARIANTS = [
    ('fp4_base', 1, 1, False), ('fp4_three', 2, 2, True),
    ('fp4_four', 2, 2, False), ('fp4_six_a', 2, 3, False),
    ('fp4_six_w', 3, 2, False), ('fp4_nine', 3, 3, False),
    ('fp4_twelve_a', 3, 4, False), ('fp4_twelve_w', 4, 3, False),
    ('fp4_sixteen', 4, 4, False),
]


def prepare(root):
    manifest = root / 'edge/final/manifest.json'
    if manifest.exists():
        raise ValueError('Experiment already frozen; reuse the existing manifest')
    sources = {
        'methods': WORKSPACE / 'full_quality_matrix_20260919/edge/final/manifest.json',
        'scenes': WORKSPACE / 'judge_full_i2v_20260919/edge/final/manifest.json',
    }
    base = json.loads(sources['methods'].read_text())
    inventory = json.loads(sources['scenes'].read_text())
    groups = {'rbench': ['dual_arm', 'humanoid', 'quad', 'single_arm', 'spatial_relationship'],
              'paibench': ['av', 'human', 'industry', 'physics', 'robot']}
    cases, inputs = [], {}
    salt = 'nested-fp4-edge-ten-demo-v1'
    # Selection depends only on IDs, never generated outputs or baseline errors.
    for benchmark, categories in groups.items():
        for category in categories:
            pool = [e for e in inventory['cases'] if (e['benchmark'], e['category']) == (benchmark, category)]
            entry = copy.deepcopy(min(pool, key=lambda e: hashlib.sha256(
                (salt + '/' + e['case']['id']).encode()).hexdigest()))
            image = Path(entry['case']['image'])
            if sha(image) != entry['image_sha256']:
                raise ValueError('Input inventory image changed')
            target = root / 'inputs' / (entry['case']['id'] + image.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, target)
            entry['case']['image'] = str(target)
            entry['case'].update(height=192, width=320, frames=61)
            entry['seeds'] = [92121001 + len(cases)]
            cases.append(entry)
            inputs[entry['case']['id']] = str(target)
    # Exact integer seed audit against source manifests, not substring matches.
    historical = {s for data in (base, inventory) for e in data['cases'] for s in e['seeds']}
    if historical & {e['seeds'][0] for e in cases}:
        raise ValueError('Seed collision with source cohorts')
    methods = {}
    for name in ('bf16', 'fp8', 'awq_reference'):
        original = base['methods'][name]
        spec = {'config': copy.deepcopy(original['config'])}
        for key in ('baseline_adapter', 'scope'):
            if key in original:
                spec[key] = copy.deepcopy(original[key])
        spec['config'].update(fuse_ffn=False, share_pack=False)
        spec.update(performance_eligible=False, source_method_manifest_sha256=sha(sources['methods']))
        methods[name] = spec
    for name, wp, ap, dual in VARIANTS:
        mask = sum(1 << (w * 4 + a) for w in range(wp) for a in range(ap)
                   if not (dual and w == a == 1))
        cfg = copy.deepcopy(methods['bf16']['config'])
        cfg['attention_backend'] = 'fp4'
        cfg['attention']['default'].update(coverage=1., terms=4, sparse_pack=False)
        methods[name] = dict(config=cfg, expansion=dict(weight_planes=wp, activation_planes=ap,
            mask=mask, terms=mask.bit_count(), dual=dual),
            # Old workers must fail closed instead of silently dispatching BF16.
            baseline_adapter=dict(method='fp4_expansion'), performance_eligible=False)
    study = dict(schema=1, frozen=True, model=base['model'], model_name='edge', methods=methods,
        steps=35, debug='off', resource_mode='shared-calibration', performance_eligible=False,
        selection_eligible=False, role='exploratory_mechanism_demo', cases=cases, adapters={},
        expansion_assets=str(root/'offline_weights'), implementation_lock=str(root/'implementation.lock.json'),
        sources={k: dict(path=str(p), sha256=sha(p)) for k,p in sources.items()},
        selection=dict(salt=salt, categories=groups, method='lowest SHA256(salt/case_id) per category',
            seed_audit='Exact integer comparison to both pinned source cohorts; fresh seeds, previously used scenes'),
        protocol=dict(height=192,width=320,frames=61,fps=24,steps=35,cfg=6.,flow_shift=3.,
            execution='graph',ar_cache=True,first_frame=True,fuse_ffn=False,
            scope='All generation projections and FFN; FP4 variants also expand all noncausal QK/PV. '
                  'AR prefill and nonlinearities retain BF16. FP8/AWQ use their historical BF16 attention.',
            metric_reference='Same-seed BF16 raw float32 decoded video',
            cost='Term multiplier applies to controlled GEMMs only, not whole-model compute or latency'),
        video_export=dict(ffmpeg='/usr/bin/ffmpeg',ffprobe='/usr/bin/ffprobe',fps=24),input_copies=inputs,
        statistics=dict(unit='scene, one paired seed per scene',bootstrap_seed=210921,replicates=10000,
            inferential=False,interpretation='Exploratory ten-demo trend; no final quality-gate claim'))
    write(manifest,study)
    write(root/'cohort.json',dict(frozen=True,manifest=str(manifest),manifest_sha256=sha(manifest),cases=cases))
    return dict(manifest=str(manifest),cases=len(cases),methods=len(methods),videos=len(cases)*len(methods))


def smoke():
    import torch
    from expansion_kernels import pack_linear, linear, attention, term_mask
    torch.manual_seed(123)
    with torch.inference_mode():
        x=torch.randn(129,256,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(256,256,device='cuda',dtype=torch.bfloat16)
        wc,ws,wg,_=pack_linear(w,4)
        y=linear(x,(wc,ws,wg),4,4,term_mask(4,4))
        q=torch.randn(1,129,4,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,145,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        z=attention(q,k,v,4,4,term_mask(4,4))
        torch.cuda.synchronize()
        assert torch.isfinite(y).all() and torch.isfinite(z).all()
    print(json.dumps(dict(complete=True,linear=list(y.shape),attention=list(z.shape),terms=16)),flush=True)


def freeze(root):
    from expansion_runtime import source_identity
    evidence=json.loads((root/'qualification/results.json').read_text())
    for key in ('pytest','memcheck','racecheck'):
        item=evidence[key]
        if item['exit_code'] != 0 or sha(item['log']) != item['sha256']:
            raise ValueError('Missing or changed qualification evidence: '+key)
    if evidence['source_sha256'] != source_identity():
        raise ValueError('Qualification does not match current sources')
    bank=root/'offline_weights/manifest.json';data=json.loads(bank.read_text())
    if not data['complete'] or data['kernel_sha256'] != sha(HERE/'expansion_kernels.py'):
        raise ValueError('Weights are incomplete or from another quantizer')
    value=dict(complete=True,source_sha256=source_identity(),weight_manifest_sha256=sha(bank),
        qualification_sha256=sha(root/'qualification/results.json'))
    path=root/'implementation.lock.json'
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError('Refusing to replace a frozen implementation')
    write(path,value)
    return value


def run(root, names, gpus):
    manifest=root/'edge/final/manifest.json';study=json.loads(manifest.read_text())
    names=names or list(study['methods'])
    if len(set(names))!=len(names) or set(names)-set(study['methods']):
        raise ValueError('Unknown or repeated method')
    if not gpus or set(gpus)-{'0','1'} or len(set(gpus))!=len(gpus):
        raise ValueError('Use one worker per GPU, restricted to GPU0/1')
    generation=root/'edge/final/generation';logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    pending=list(names);active={};done=[];failed=[]
    suite=generation/('suite.json' if set(names)==set(study['methods']) else 'suite_subset_'+'_'.join(gpus)+'.json')
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending or failed:continue
            name=pending.pop(0)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,TW_SHARED_CALIBRATION_GPUS='0,1',
                     TW_SHARED_CALIBRATION_LOCK_SCOPE='device',PYTHONUNBUFFERED='1')
            log=(logs/(name+'.log')).open('ab')
            command=[sys.executable,str(Path(__file__).resolve()),'worker','--root',str(root),'--method',name]
            log.write((json.dumps(dict(command=command,gpu=gpu,started=time.time()))+'\n').encode());log.flush()
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(name,process,log)
            print(json.dumps(dict(started=name,gpu=gpu,pid=process.pid)),flush=True)
        for gpu,(name,process,log) in list(active.items()):
            code=process.poll()
            if code is None:continue
            log.close();del active[gpu]
            if code:failed.append(dict(method=name,gpu=gpu,exit_code=code,log=str(logs/(name+'.log'))))
            else:done.append(name)
            print(json.dumps(dict(finished=name,gpu=gpu,exit_code=code)),flush=True)
        state=dict(complete=not pending and not active and not failed,completed=done,pending=pending,
            active={g:dict(method=n,pid=p.pid) for g,(n,p,l) in active.items()},failures=failed,
            paused=bool(failed),performance_eligible=False,manifest_sha256=sha(manifest))
        write(suite,state)
        if failed and not active:raise RuntimeError('Worker failed; progress preserved: '+str(failed))
        if pending or active:time.sleep(5)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['prepare','prepare-assets','smoke','freeze','worker','run'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--method');parser.add_argument('--methods',nargs='+')
    parser.add_argument('--gpus',nargs='+',default=['0','1'])
    args=parser.parse_args();root=args.root.resolve();manifest=root/'edge/final/manifest.json'
    if args.operation=='prepare':result=prepare(root)
    elif args.operation=='smoke':result=smoke()
    elif args.operation=='freeze':result=freeze(root)
    elif args.operation=='run':result=run(root,args.methods,args.gpus)
    elif args.operation=='prepare-assets':
        from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
        with shared_worker():
            select_shared_device()
            import torch
            torch.set_num_threads(8)
            from expansion_runtime import prepare_assets
            result=prepare_assets(json.loads(manifest.read_text()),root/'offline_weights')
        result=dict(complete=result['complete'],modules=len(result['modules']))
    else:
        if not args.method:parser.error('worker needs --method')
        from expansion_runtime import install_worker_adapter
        install_worker_adapter()
        from dataset_infer import run as worker
        result=worker(manifest,args.method,root/'edge/final/generation'/args.method)
    if result is not None:print(json.dumps(result),flush=True)


if __name__=='__main__':main()

"""Frozen same-cohort 1.5x/2x bilateral support ablation for Figure 2(a)."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from baselines import sha,write

HERE=Path(__file__).resolve().parent
DEFAULT_PARENT=HERE.parents[2]/'fp4_expansion_demo_20260921'
VARIANTS={'fp4_both_1p5':1,'fp4_both_2p0':2}


def prepare(root,parent):
    target=root/'edge/final/manifest.json'
    if target.exists():raise ValueError('Already frozen; resume without prepare')
    origin=parent/'edge/final/manifest.json';base=json.loads(origin.read_text())
    study=copy.deepcopy(base);methods={}
    for name,num in VARIANTS.items():
        spec=copy.deepcopy(base['methods']['fp4_three'])
        spec['config']['compute_budget']=1+num/2
        spec['sparse_refinement']=dict(budget=1+num/2,numerator=num,denominator=4,
            a_fraction=num/4,w_fraction=num/4,integer_quota='floor(fraction * available_blocks)',
            linear_support='A1 represented energy * offline W0 K-block energy; offline W1 block energy',
            attention_qk_support='Current Q1/K0 and Q0/K1 squared-Frobenius norm products',
            attention_pv_support='Current V0 norm for P1V0; V1 norm for P0V1',
            selector_mma=0,cross=False,phase_policy='constant for all 35 steps and both CFG branches',
            full_residual_packing=True,compressed_weights=False,
            fitted_to_cohort=False,selection='Stable descending score; low index breaks exact ties')
        spec['baseline_adapter']={'method':'fp4_sparse_expansion'}
        methods[name]=spec
    study.update(methods=methods,role='exploratory_strict_sparse_extension',
        extension=dict(parent_root=str(parent),manifest_sha256=sha(origin),
            parent_metrics_sha256=sha(parent/'report/results.json'),source_sha256=sha(__file__),
            reuse='Original same-input/seed BF16 reference and frozen original single-side/full expansions'),
        sparse_implementation_lock=str(root/'qualification/results.json'),
        sparse_support_directory=str(root/'offline_supports'),
        preregistration=dict(hypothesis='Do sparse bilateral corrections outperform full A-only/W-only at 2x or lower MMA?',
            no_assumed_outcome=True,primary='LPIPS to original paired BF16',
            guardrails=['SSIM','PSNR','relative_l2'],all_results_retained=True,
            budget='Actual padded MMA over controlled generation GEMM/QK/PV; packing is not discounted',
            no_quality_tuning=True,controls_root=str(HERE.parents[2]/'fp4_expansion_two_term_20260922')))
    for entry in study['cases']:
        if sha(entry['case']['image'])!=entry['image_sha256']:raise ValueError('Input changed')
    write(target,study)
    write(root/'cohort.json',dict(frozen=True,manifest=str(target),manifest_sha256=sha(target),cases=study['cases']))
    return dict(complete=True,cases=10,methods=list(methods),new_videos=20)


def qualify(root):
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    from expansion_sparse_runtime import source_identity
    folder=root/'qualification';folder.mkdir(parents=True,exist_ok=True)
    test=HERE/'test_expansion_sparse.py'
    commands={'pytest':[sys.executable,'-m','pytest',str(test),'-x','-q']}
    for tool in ('memcheck','racecheck'):
        commands[tool]=['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',
                       sys.executable,str(Path(__file__).resolve()),'smoke','--root',str(root)]
    result=dict(complete=False,source_sha256=source_identity(),test_sha256=sha(test))
    with shared_worker():
        result['resource']=select_shared_device()
        for name,command in commands.items():
            log=folder/(name+'.log')
            with log.open('w') as stream:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
            result[name]=dict(exit_code=code,log=str(log),sha256=sha(log),command=command)
            write(folder/'results.json',result)
            print(json.dumps({name:result[name]}),flush=True)
            if code:raise RuntimeError('Qualification failed: '+name)
    result['complete']=True;write(folder/'results.json',result)
    return result


def smoke():
    import torch
    import expansion_sparse_kernels as sparse
    from expansion_kernels import pack_linear
    torch.manual_seed(9222600)
    with torch.inference_mode():
        x=torch.randn(129,512,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(256,512,device='cuda',dtype=torch.bfloat16)
        wc,ws,wg,_=pack_linear(w,2)
        q=torch.randn(1,129,4,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,513,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        for num in (1,2):
            wm,sensitivity=sparse.weight_support((wc,ws,wg),num)
            y=sparse.linear(x,(wc,ws,wg),wm,sensitivity,num)
            z=sparse.attention(q,k,v,num)
            torch.cuda.synchronize()
            assert torch.isfinite(y).all() and torch.isfinite(z).all()
    return dict(complete=True,budgets=[1.5,2.])


def run(root,gpu):
    # One serial worker on the chosen GPU, with resource admission/locks in
    # dataset_infer. Completed requests survive failures and can be resumed.
    from expansion_sparse_runtime import verify
    study=json.loads((root/'edge/final/manifest.json').read_text())
    for name in VARIANTS:verify(study,name)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    state=dict(complete=False,completed=[],pending=list(VARIANTS),active={},failures=[],
               performance_eligible=False,manifest_sha256=sha(root/'edge/final/manifest.json'))
    suite=root/'edge/final/generation/suite.json'
    for name in VARIANTS:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TW_SHARED_CALIBRATION_GPUS='0,1',
                 TW_SHARED_CALIBRATION_LOCK_SCOPE='device',PYTHONUNBUFFERED='1')
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--root',str(root),'--method',name]
        with (logs/(name+'.log')).open('a') as stream:
            stream.write(json.dumps(dict(command=command,gpu=gpu,started=time.time()))+'\n');stream.flush()
            proc=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            state['active']={str(gpu):dict(method=name,pid=proc.pid)};state['pending'].remove(name);write(suite,state)
            print(json.dumps(dict(started=name,pid=proc.pid,gpu=gpu)),flush=True)
            code=proc.wait()
        state['active']={}
        if code:
            state.update(paused=True,failures=[dict(method=name,exit_code=code)])
            write(suite,state);raise RuntimeError('Worker failed; completed data preserved')
        state['completed'].append(name);write(suite,state)
        print(json.dumps(dict(finished=name)),flush=True)
    state['complete']=True;write(suite,state);return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['prepare','qualify','smoke','run','worker','metrics'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    p.add_argument('--gpu',type=int,choices=(0,1),default=1);p.add_argument('--method',choices=list(VARIANTS))
    a=p.parse_args();root=a.root.resolve()
    if a.operation=='prepare':r=prepare(root,a.parent.resolve())
    elif a.operation=='qualify':r=qualify(root)
    elif a.operation=='smoke':r=smoke()
    elif a.operation=='run':r=run(root,a.gpu)
    elif a.operation=='metrics':
        from expansion_two_term import metrics
        r=metrics(root)
    else:
        if not a.method:p.error('worker requires --method')
        from expansion_sparse_runtime import install_worker_adapter
        install_worker_adapter()
        from dataset_infer import run as worker
        r=worker(root/'edge/final/manifest.json',a.method,root/'edge/final/generation'/a.method)
    print(json.dumps(r),flush=True)


if __name__=='__main__':main()

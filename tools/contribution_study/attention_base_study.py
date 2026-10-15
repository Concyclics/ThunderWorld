"""Matched ten-scene rerun and stratified 72-scene extension without attention repair."""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from baselines import sha,write

HERE=Path(__file__).resolve().parent;WORK=HERE.parents[2]
PARENT=WORK/'fp4_expansion_demo_20260921'
SIDES=WORK/'fp4_expansion_two_term_20260922'
SPARSE=WORK/'fp4_expansion_sparse_matched_20260922'
NAMES=['bf16','fp4_base','linear_a','linear_w','linear_both_1p5','linear_both_2p0']
SALT='attention-base-validation-72-v1'


def frozen_methods():
    base=json.loads((PARENT/'edge/final/manifest.json').read_text())
    sides=json.loads((SIDES/'edge/final/manifest.json').read_text())
    sparse=json.loads((SPARSE/'edge/final/manifest.json').read_text())
    methods={'bf16':copy.deepcopy(base['methods']['bf16']),'fp4_base':copy.deepcopy(base['methods']['fp4_base']),
        'linear_a':copy.deepcopy(sides['methods']['fp4_two_a']),'linear_w':copy.deepcopy(sides['methods']['fp4_two_w']),
        'linear_both_1p5':copy.deepcopy(sparse['methods']['fp4_both_1p5']),
        'linear_both_2p0':copy.deepcopy(sparse['methods']['fp4_both_2p0'])}
    for n,spec in methods.items():
        if n=='bf16':continue
        budget={'fp4_base':1.,'linear_both_1p5':1.5}.get(n,2.)
        spec.update(attention_expansion=dict(weight_planes=1,activation_planes=1,mask=1,terms=1),linear_budget=budget)
        spec['config'].update(precision='fp4',compute_budget=budget)
    return base,methods


def audit_seeds():
    files=subprocess.check_output(['rg','--files','--hidden','--no-ignore','-g','manifest.json',
        '-g','!**/artifacts/**','-g','!**/.git/**','-g','!**/.venvs/**','-g','!**/node_modules/**',
        '-g','!**/offline_weights/**','-g','!contribution_baselines_20260917/**'],cwd=WORK,text=True).splitlines()
    used=set();sources={}
    for file in files:
        path=WORK/file
        if path.stat().st_size>20_000_000:continue
        try:data=json.loads(path.read_text())
        except (ValueError,UnicodeError):continue
        if not isinstance(data,dict) or not isinstance(data.get('cases'),list):continue
        sources[str(path)]=sha(path)
        for e in data['cases']:
            if isinstance(e,dict):used.update(s for s in e.get('seeds',[]) if type(s) is int)
    # Catch outputs whose seeds were not represented by a cohort manifest.
    outputs=subprocess.check_output(['rg','--files','--hidden','--no-ignore','-g','seed*.npz',
        '-g','!**/.git/**','-g','!**/.venvs/**'],cwd=WORK,text=True).splitlines()
    for f in outputs:
        token=Path(f).stem[4:]
        if token.isdecimal():used.add(int(token))
    start=92231001
    while used&set(range(start,start+144)):start+=10000
    return start,dict(cohort_manifests=sources,output_archives_scanned=len(outputs),unique_historical_seeds=len(used),
        new_range=[start,start+143],collision=False)


def select_cases(base,inventory):
    excluded_ids={e['case']['id'] for e in base['cases']};pixels={e['pixel_sha256'] for e in base['cases']}
    groups=collections.defaultdict(list)
    for e in inventory['cases']:
        if e.get('prior_scene_roles') or e['case']['id'] in excluded_ids or e['pixel_sha256'] in pixels:continue
        groups[e['benchmark'],e['category']].append(e)
    selected=[];quotas={}
    for benchmark in ('rbench','paibench'):
        cats=sorted(c for b,c in groups if b==benchmark)
        extra=sorted(cats,key=lambda c:hashlib.sha256((SALT+'/category/'+c).encode()).hexdigest())[:36%len(cats)]
        for category in cats:
            count=36//len(cats)+(category in extra);quotas[benchmark+'/'+category]=count
            pool=sorted(groups[benchmark,category],key=lambda e:hashlib.sha256((SALT+'/'+e['case']['id']).encode()).hexdigest())
            accepted=[]
            for e in pool:
                if e['pixel_sha256'] in pixels:continue
                accepted.append(copy.deepcopy(e));pixels.add(e['pixel_sha256'])
                if len(accepted)==count:break
            if len(accepted)!=count:raise ValueError('Not enough distinct images in stratum')
            selected.extend(accepted)
    if len(selected)!=72:raise ValueError('Wrong new cohort size')
    return selected,quotas


def prepare(root):
    from PIL import Image
    from attention_base_runtime import source_identity
    if (root/'study.json').exists():raise ValueError('Already frozen; do not prepare again')
    base,methods=frozen_methods();invpath=WORK/'judge_full_i2v_20260919/edge/final/manifest.json'
    inventory=json.loads(invpath.read_text());fresh,quotas=select_cases(base,inventory)
    start,audit=audit_seeds()
    for i,e in enumerate(fresh):e['seeds']=[start+2*i,start+2*i+1]
    root.mkdir(parents=True,exist_ok=True)
    write(root/'seed_audit.json',audit)
    sources={str(p):sha(p) for p in (PARENT/'edge/final/manifest.json',SIDES/'edge/final/manifest.json',
             SPARSE/'edge/final/manifest.json',invpath)}
    for panel,entries in [('matched10',copy.deepcopy(base['cases'])),('expanded72',fresh)]:
        study=copy.deepcopy(base);inputs={};seen=set()
        for e in entries:
            image=Path(e['case']['image'])
            if sha(image)!=e['image_sha256']:raise ValueError('Image byte hash changed')
            with Image.open(image) as im:pixel=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            if pixel!=e['pixel_sha256'] or pixel in seen:raise ValueError('Changed/duplicate decoded pixels')
            seen.add(pixel)
            # Keep old ten paths identical for bitwise reference audit.
            if panel=='expanded72':
                target=root/'inputs'/(e['case']['id']+image.suffix);target.parent.mkdir(exist_ok=True)
                shutil.copyfile(image,target);e['case']['image']=str(target)
            e['case'].update(height=192,width=320,frames=61);inputs[e['case']['id']]=e['case']['image']
        study.update(cases=entries,methods=copy.deepcopy(methods),input_copies=inputs,
            role='fixed_policy_out_of_cohort_validation' if panel=='expanded72' else 'paired_attention_ablation',
            selection_eligible=False,attention_base_lock=str(root/'implementation.lock.json'),
            sparse_implementation_lock=str(SPARSE/'qualification/results.json'),
            sparse_support_directory=str(root/'offline_supports'),
            sources=sources,selection=dict(salt=SALT,stratum_quotas=quotas if panel=='expanded72' else None,
                excludes='Original ten scenes, prior inventoried cohort roles, duplicate decoded images',
                no_output_based_selection=True,scope='New to this ablation; public benchmark scenarios, not a generalization claim'),
            statistics=dict(unit='scene; first average paired seeds within scene',bootstrap_seed=922310,
                replicates=10000,strata=['benchmark','category'],primary_panel='expanded72',
                comparisons=[['linear_w','linear_a'],['linear_both_1p5','linear_a'],['linear_both_1p5','linear_w'],
                             ['linear_both_2p0','linear_a'],['linear_both_2p0','linear_w']],
                intervals='Paired scene-stratified 95%; simultaneous primary LPIPS intervals use Bonferroni 99%',
                no_policy_fitting=True),
            attention_ablation=dict(all_fp4_attention_repairs=False,base_blocks_preserved=True,
                original_report=str(SPARSE/'report/results.json'),original_report_sha256=sha(SPARSE/'report/results.json')))
        study['protocol']['scope']='All generation projections/FFN use indicated repair; every FP4 method uses identical one-plane QK/PV; common BF16 AR/nonlinearities.'
        study['protocol']['cost']='1.5x/2x are linear-only MMA caps. Whole-request controlled MMA is lower and reported separately.'
        if panel=='matched10':
            hashes={}
            for name in ('bf16','fp4_base'):
                hashes[name]={e['case']['id']:{str(r['seed']):r['sha256'] for r in json.loads((PARENT/'edge/final/generation'/name/e['case']['id']/'result.json').read_text())['rows']} for e in entries}
            path=root/'matched10/reference_signal_hashes.json';write(path,hashes);study['source_output_hashes']=str(path)
        write(root/panel/'edge/final/manifest.json',study)
    write(root/'implementation.lock.json',dict(complete=True,source_sha256=source_identity(),
        kernel_qualification=dict(parent=str(PARENT/'qualification/results.json'),sparse=str(SPARSE/'qualification/results.json')),
        qualification_scope='No new GPU kernels; all-base attention endpoint already qualified against decoded reference and original kernels'))
    value=dict(frozen=True,panels={'matched10':dict(scenes=10,seeds=10,videos=60),'expanded72':dict(scenes=72,seeds=144,videos=864)},
        methods=NAMES,attention_repair=False,performance_eligible=False,selection_eligible=False,
        manifests={p:sha(root/p/'edge/final/manifest.json') for p in ('matched10','expanded72')},
        no_quality_tuning=True,scope='Matched ten-scene attribution followed by category-stratified expansion, same frozen linear policies')
    write(root/'study.json',value);return value


def run(root,gpu):
    state=dict(complete=False,completed=[],active=None,paused=False,performance_eligible=False)
    for panel in ('matched10','expanded72'):
        study=root/panel;logs=study/'logs';logs.mkdir(parents=True,exist_ok=True)
        suite=dict(complete=False,completed=[],paused=False,performance_eligible=False)
        for name in NAMES:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TW_SHARED_CALIBRATION_GPUS='0,1',
                     TW_SHARED_CALIBRATION_LOCK_SCOPE='device',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--root',str(root),'--panel',panel,'--method',name]
            with (logs/(name+'.log')).open('a') as log:
                log.write(json.dumps(dict(command=cmd,gpu=gpu,started=time.time()))+'\n');log.flush()
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['active']=dict(panel=panel,method=name,pid=proc.pid);write(root/'workflow.json',state)
                print(json.dumps(state['active']),flush=True);code=proc.wait()
            if code:
                suite.update(paused=True,error=f'{name} exited {code}');state.update(paused=True,error=suite['error'])
                write(study/'edge/final/generation/suite.json',suite);write(root/'workflow.json',state)
                raise RuntimeError('Generation paused; existing results preserved')
            for folder in (study/'edge/final/generation'/name).iterdir():
                if folder.is_dir() and not (folder/'generation.log').exists():(folder/'generation.log').symlink_to(logs/(name+'.log'))
            suite['completed'].append(name);write(study/'edge/final/generation/suite.json',suite)
            state['completed'].append(panel+'/'+name);write(root/'workflow.json',state)
            print(json.dumps(dict(finished=panel+'/'+name)),flush=True)
        suite['complete']=True;write(study/'edge/final/generation/suite.json',suite)
    state.update(complete=True,active=None);write(root/'workflow.json',state);return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['prepare','run','worker'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,choices=(0,1),default=1)
    p.add_argument('--panel',choices=('matched10','expanded72'));p.add_argument('--method',choices=NAMES)
    a=p.parse_args();root=a.root.resolve()
    if a.operation=='prepare':result=prepare(root)
    elif a.operation=='run':result=run(root,a.gpu)
    else:
        if not a.panel or not a.method:p.error('worker requires panel and method')
        from attention_base_runtime import install
        install()
        from dataset_infer import run as worker
        study=root/a.panel
        result=worker(study/'edge/final/manifest.json',a.method,study/'edge/final/generation'/a.method)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

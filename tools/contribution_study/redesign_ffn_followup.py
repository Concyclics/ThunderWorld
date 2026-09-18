"""Serialize a full-FFN-aware development candidate after the first frozen experiment.

Unchanged baseline outputs/metrics are imported with their original provenance;
this reuses evaluation evidence, never old steps inside a new candidate request.
"""
import argparse,copy,json,os,shutil,subprocess,sys,time
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--reference-study',type=Path);p.add_argument('--output-subdir',default='ffn_followup');p.add_argument('--axes',nargs='+',choices=('n','k'),default=['n','k'])
    p.add_argument('--profile',choices=('protected','balanced'),default='protected');a=p.parse_args()
    while True:
        path=a.study/'ffn_calibration/manifest.json';state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('paused'):raise RuntimeError('Full FFN calibration paused')
        if state.get('complete'):break
        time.sleep(5)
    old_root=a.reference_study or a.study/'development'
    while True:
        state=json.loads((old_root/'workflow.json').read_text()) if (old_root/'workflow.json').exists() else {}
        if state.get('paused'):raise RuntimeError('Previous development paused')
        if state.get('complete'):break
        time.sleep(5)
    here=Path(__file__).resolve().parent;root=a.study/a.output_subdir;root.mkdir(exist_ok=True)
    shutil.copyfile(a.study/'cohort.json',root/'cohort.json')
    if not (root/'candidates/candidates.json').exists():
        combined=[]
        for axis in a.axes:
            target=root/(axis+'_candidates')
            if not (target/'candidates.json').exists():
                subprocess.run([sys.executable,str(here/'redesign_plan.py'),'--cohort',str(a.study/'cohort.json'),'--calibration',str(a.study/'calibration'),
                    '--previous',str(a.previous),'--output',str(target),'--ffn-calibration',str(a.study/'ffn_calibration'),'--weight-axis',axis,'--profile',a.profile],check=True)
            combined+=json.loads((target/'candidates.json').read_text())['candidates']
        write(root/'candidates/candidates.json',dict(complete=True,candidates=combined,cohort_sha256=sha(a.study/'cohort.json')))
    old=json.loads((old_root/'manifest.json').read_text());new=copy.deepcopy(old)
    cm=json.loads((root/'candidates/candidates.json').read_text())
    new['methods'].update({c['name']:c for c in cm['candidates']});new['candidates_sha256']=sha(root/'candidates/candidates.json')
    new.update(previous_development_sha256=sha(old_root/'manifest.json'),scope='Second development proposal using full FFN interventions. Same scenes/seeds; unchanged baseline quality evidence is imported. No final-test use.')
    run=root/'development';run.mkdir(exist_ok=True);manifest=run/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=new:raise ValueError('Changed followup experiment')
    else:write(manifest,new)
    digest=sha(manifest)
    def link(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha(source)!=sha(target):raise ValueError('Changed imported artifact')
        else:os.link(source,target)
    for name in old['methods']:
        if new['methods'][name]!=old['methods'][name]:raise ValueError('Cannot import changed method')
        progress=json.loads((old_root/'generation'/name/'progress.json').read_text())
        if not progress['complete']:raise ValueError('Incomplete baseline')
        for entry in new['cases']:
            case=entry['case']['id'];source=old_root/'generation'/name/case;folder=run/'generation'/name/case
            receipt=json.loads((source/'result.json').read_text())
            if not receipt['complete'] or receipt['identity']['case']!=entry['case'] or receipt['identity']['seeds']!=entry['seeds']:raise ValueError('Unmatched baseline inputs')
            for row in receipt['rows']:
                if sha(source/row['outputs'])!=row['outputs_sha256']:raise ValueError('Corrupt baseline output')
                link(source/row['outputs'],folder/row['outputs'])
            if (source/'compute.json').exists():link(source/'compute.json',folder/'compute.json')
            receipt['imported_from']=dict(path=str(source/'result.json'),sha256=sha(source/'result.json'),scope='Unchanged configuration and outputs; not a new timing measurement')
            receipt['identity']['manifest_sha256']=digest;write(folder/'result.json',receipt)
        progress['identity']['manifest_sha256']=digest;progress['imported_from']=dict(path=str(old_root/'generation'/name/'progress.json'),sha256=sha(old_root/'generation'/name/'progress.json'))
        write(run/'generation'/name/'progress.json',progress)
    old_metrics=json.loads((old_root/'metrics/progress.json').read_text());identity=dict(old_metrics['identity'],manifest_sha256=digest)
    if identity['source_sha256']!=sha(here/'dataset_metrics.py') or identity['quality_sha256']!=sha(here.parents[1]/'packages/fp4-repair/src/thunderworld_fp4/cosmos/quality.py'):raise ValueError('Scorer changed; recompute imported metrics')
    for name in old['methods']:
        if name=='bf16':continue
        for entry in new['cases']:
            case=entry['case']['id'];file=old_root/'metrics'/name/(case+'.json');value=json.loads(file.read_text())
            if not value['complete'] or value['provenance']['identity']!=old_metrics['identity']:raise ValueError('Unqualified previous metrics')
            for side,method in (('reference','bf16'),('candidate',name)):
                if value['provenance'][side+'_sha256']!=sha(old_root/'generation'/method/case/'result.json'):raise ValueError('Changed source generation')
            value['imported_from']=dict(path=str(file),sha256=sha(file),provenance=value['provenance'])
            value['provenance']=dict(identity=identity,reference_sha256=sha(run/'generation/bf16'/case/'result.json'),candidate_sha256=sha(run/'generation'/name/case/'result.json'))
            write(run/'metrics'/name/(case+'.json'),value)
    status=run/'workflow.json';state=dict(complete=False,manifest_sha256=digest,performance_eligible=False);write(status,state)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    with (run/'metrics.log').open('a') as log:
        metrics=subprocess.Popen([sys.executable,str(here/'dataset_metrics.py'),str(manifest),'--generation',str(run/'generation'),'--output',str(run/'metrics'),'--follow'],env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            with (run/'generation.log').open('a') as gl:
                subprocess.run([sys.executable,str(here/'dataset_suite.py'),str(manifest),'--output',str(run/'generation')],stdout=gl,stderr=subprocess.STDOUT,check=True)
            if metrics.wait()!=0:raise RuntimeError('Followup metric process failed')
            subprocess.run([sys.executable,str(here/'redesign_summarize.py'),'--study',str(root),'--archive-label',a.output_subdir],check=True)
            state['complete']=True;write(status,state)
        except Exception as error:
            if metrics.poll() is None:metrics.terminate();metrics.wait()
            state.update(paused=True,error=repr(error));write(status,state);raise


if __name__=='__main__':main()

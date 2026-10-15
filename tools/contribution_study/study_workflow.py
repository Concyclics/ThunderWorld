"""Serialized validation -> immutable selection -> final study using existing runners."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','cohort','baseline-root','legacy-template','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    identity={key:sha(getattr(a,key)) for key in ('manifest','cohort','legacy_template')}
    identity['baseline_root']=str(a.baseline_root.resolve())
    identity['analysis_sha256']={name:sha(Path(__file__).with_name(name)) for name in
        ('quality.py','quality_gate.py','paired_stats.py','freeze_selection.py','study_workflow.py')}
    receipt=a.output/'workflow.json'
    if receipt.exists() and json.loads(receipt.read_text())['identity']!=identity:raise ValueError('Workflow changed during resume')
    state=dict(identity=identity,complete=False,performance_eligible=False);write(receipt,state)
    here=Path(__file__).resolve().parent
    def run(script,args,*,cpu=False):
        env=dict(os.environ)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        subprocess.run([sys.executable,str(here/script),*map(str,args)],env=env,check=True)
    def complete(path):return path.exists() and json.loads(path.read_text()).get('complete',False)
    def legacy(split,output,selection=None):
        if complete(output/'result.json'):return
        run('legacy_infer.py',['--template',a.legacy_template,'--cohort',a.cohort,'--split',split,'--output',output,
            *(['--frozen-selection',selection] if selection else [])])
    def quality(manifest,reference,legacy_folder,output,split,extra=None,exclude=()):
        if complete(output):return
        if output.exists():raise RuntimeError('Inspect incomplete metric result before resuming: '+str(output))
        m=json.loads(manifest.read_text());cohort=json.loads(a.cohort.read_text())
        args=['--reference',reference]
        for job in m['runs']:
            if job['name']=='bf16' or job['name'] in exclude:continue
            args+=['--run',job['name']+'='+str(Path(m['output'])/job['name'])]
        args+=['--run','legacy_2.5='+str(legacy_folder)]
        if extra:
            for name,folder in extra.items():
                if name not in exclude:args+=['--run',name+'='+str(folder)]
        run('quality.py',[*args,'--seeds',*cohort['seeds'][split],'--output',output],cpu=True)
    run('suite.py',[a.manifest])
    legacy_validation=a.output/'legacy_validation';legacy('validation',legacy_validation)
    validation=a.output/'validation_quality.json'
    native_batch=Path(json.loads(a.manifest.read_text())['output'])/'quality_native.json'
    if complete(native_batch) and not complete(validation):
        existing=json.loads(native_batch.read_text());extra_batch=a.output/'validation_external_quality.json'
        quality(a.manifest,a.baseline_root/'bf16',legacy_validation,extra_batch,'validation',
            {'fp8':a.baseline_root/'fp8'},exclude=set(existing['means']))
        run('merge_quality.py',[native_batch,extra_batch,'--output',validation],cpu=True)
    else:quality(a.manifest,a.baseline_root/'bf16',legacy_validation,validation,'validation',{'fp8':a.baseline_root/'fp8'})
    selection=a.output/'frozen'
    if not (selection/'selection.json').exists():
        run('freeze_selection.py',['--validation-manifest',a.manifest,'--quality',validation,'--cohort',a.cohort,
            '--baseline-root',a.baseline_root,'--legacy',legacy_validation,'--output',selection],cpu=True)
    frozen=json.loads((selection/'selection.json').read_text())
    if frozen['validation_quality_sha256']!=sha(validation):raise ValueError('Validation metrics changed after selection')
    run('suite.py',[selection/'manifest.json'])
    legacy_final=a.output/'legacy_final';legacy('final',legacy_final,selection/'selection.json')
    final=a.output/'final_quality.json'
    quality(selection/'manifest.json',selection/'generation/bf16',legacy_final,final,'final')
    contrasts=[]
    for row in frozen['primary_comparisons']+frozen['secondary_comparisons']:
        contrasts+=['--contrast',row['candidate']+'='+row['control']]
    run('quality_gate.py',['--quality',final,'--cohort',a.cohort,'--split','final',*contrasts,
        '--family-size',frozen['family_size'],'--output',a.output/'final_gates.json'],cpu=True)
    state.update(complete=True,validation_quality_sha256=sha(validation),selection_sha256=sha(selection/'selection.json'),
        final_quality_sha256=sha(final),final_gates_sha256=sha(a.output/'final_gates.json'))
    write(receipt,state)


if __name__=='__main__':main()

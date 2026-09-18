"""Run frozen 2.5x redesign diagnostics on new development scenes, serially."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','previous'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cohort=json.loads((a.study/'cohort.json').read_text());old=json.loads(a.previous.read_text())
    candidates=json.loads((a.study/'candidates/candidates.json').read_text())
    if not candidates['complete'] or candidates['cohort_sha256']!=sha(a.study/'cohort.json'):raise ValueError('Unfrozen candidate assets')
    selected=['b2.5_protected_n','b2.5_uniform_n','b2.5_energy_n','b2.5_old_quota_rtn']
    methods={n:old['methods'][n] for n in ('bf16','fp8','fp4_three','b2.5_marginal')}
    methods.update({c['name']:c for c in candidates['candidates'] if c['name'] in selected})
    if not set(selected).issubset(methods):raise ValueError('Missing planned control')
    root=a.study/'development';root.mkdir(exist_ok=True);manifest=root/'manifest.json'
    value=dict(schema=1,frozen=True,model=cohort['model'],cases=cohort['cases']['development'],methods=methods,
        cohort_sha256=sha(a.study/'cohort.json'),candidates_sha256=sha(a.study/'candidates/candidates.json'),steps=35,
        resource_mode='shared-calibration',debug='off',execution='graph',role='development',selection_eligible=True,
        scope='New development scenes only. One scene per category: exploratory means, no stratified significance claim. Same 35-step Edge generation settings.')
    if manifest.exists():
        if json.loads(manifest.read_text())!=value:raise ValueError('Changed frozen development experiment')
    else:write(manifest,value)
    here=Path(__file__).resolve().parent;status=root/'workflow.json'
    state=dict(complete=False,manifest_sha256=sha(manifest),performance_eligible=False);write(status,state)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    with (root/'metrics.log').open('a') as ml:
        metric=subprocess.Popen([sys.executable,str(here/'dataset_metrics.py'),str(manifest),'--generation',str(root/'generation'),
            '--output',str(root/'metrics'),'--follow'],env=env,stdout=ml,stderr=subprocess.STDOUT)
        try:
            with (root/'generation.log').open('a') as gl:
                subprocess.run([sys.executable,str(here/'dataset_suite.py'),str(manifest),'--output',str(root/'generation')],stdout=gl,stderr=subprocess.STDOUT,check=True)
            if metric.wait()!=0:raise RuntimeError('Metric process failed; inspect retained log')
            subprocess.run([sys.executable,str(here/'redesign_summarize.py'),'--study',str(a.study)],check=True)
            state['complete']=True;write(status,state)
        except Exception as error:
            if metric.poll() is None:metric.terminate();metric.wait()
            state.update(paused=True,error=repr(error));write(status,state);raise


if __name__=='__main__':main()

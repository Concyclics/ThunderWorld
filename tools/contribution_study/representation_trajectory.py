"""Paired development generation with only offline representation changed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from baselines import sha,write


def replace(cmd,flag,values):
    if flag not in cmd:
        cmd.extend([flag,*values]);return
    i=cmd.index(flag);j=i+1
    while j<len(cmd) and not cmd[j].startswith('--'):j+=1
    cmd[i+1:j]=values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('template','assets','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--name',default='candidate')
    a=p.parse_args()
    template=json.loads(a.template.read_text())
    source=next(r for r in template['runs'] if r['name']==a.name)
    command=list(source['command']);command[0]=sys.executable
    asset=json.loads((a.assets/'manifest.json').read_text())
    if not asset['complete'] or len(asset['modules'])!=168:raise ValueError('Complete all-layer assets required')
    plan_path=Path(command[command.index('--nblock-plan')+1])
    old_plan=json.loads(plan_path.read_text())
    plan={**old_plan,'calibration_sha256':sha(a.assets/'manifest.json')}
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'plan.json',plan)
    replace(command,'--nblock-plan',[str(a.output/'plan.json')])
    replace(command,'--assets',[str(a.assets)])
    replace(command,'--seeds',['7103','7104'])
    replace(command,'--warm-requests',['1'])
    replace(command,'--output',[str(a.output/'generation')])
    if '--reference' in command:
        i=command.index('--reference');del command[i:i+2]
    # This is shared quality generation; the runtime itself selects GPU0/3.
    if '--idle-check' in command:command.remove('--idle-check')
    manifest=dict(complete=False,performance_eligible=False,command=command,
        template_sha256=sha(a.template),assets_sha256=sha(a.assets/'manifest.json'),
        original_plan_sha256=sha(plan_path),plan_sha256=sha(a.output/'plan.json'),
        sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        role='trajectory',seeds=[7103,7104],scope='Development only; same native schedule, masks, packing, and attention. Only offline W representation changes.')
    write(a.output/'manifest.json',manifest)
    with (a.output/'generation.log').open('w') as stream:
        status=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
    manifest['exit_code']=status
    file=a.output/'generation/result.json'
    manifest['complete']=status==0 and file.exists() and json.loads(file.read_text())['complete']
    if manifest['complete']:manifest['result_sha256']=sha(file)
    write(a.output/'manifest.json',manifest)
    if not manifest['complete']:raise RuntimeError('Generation failed; logs and partial progress retained')


if __name__=='__main__':main()

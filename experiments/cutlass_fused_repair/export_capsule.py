"""Export four operator families/all phases. No model checkpoint or full capture."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

NAMES=['layers.0.self_attn.add_q_proj','layers.0.self_attn.add_k_proj',
       'layers.0.mlp_moe_gen.up_proj','layers.0.mlp_moe_gen.down_proj']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.config.read_text());capture=json.loads((a.capture/'manifest.json').read_text())
    assets=json.loads((a.assets/'manifest.json').read_text())
    if capture['model_identity']!=assets['model_identity'] or capture['cohort_sha256']!=assets['cohort_sha256']:
        raise ValueError('Capture/asset mismatch')
    plan=Path(config['budget_plan'])
    if not plan.is_absolute():plan=a.config.parent/plan
    shutil.copyfile(plan,a.output/'plan.json');config['budget_plan']='plan.json'
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    def copy(root,target,name):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
        dst=a.output/target/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,dst)
    selected=[]
    for name in NAMES:
        for ph in range(4):
            row=next(r for r in capture['rows'] if r['kind']=='linear' and r['name']==name and r['phase']==ph and r['branch']==0)
            copy(a.capture,'capture',row['file']);selected.append(row)
    capture['rows']=selected;capture['scope']='Operator retest subset; cannot calibrate full model'
    assets['modules']={name:assets['modules'][name] for name in NAMES}
    for row in assets['modules'].values():copy(a.assets,'assets',row['file'])
    (a.output/'capture/manifest.json').write_text(json.dumps(capture,indent=2)+'\n')
    (a.output/'assets/manifest.json').write_text(json.dumps(assets,indent=2)+'\n')
    files={str(f.relative_to(a.output)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
           for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'capsule.json').write_text(json.dumps(dict(scope='kernel-only; four families, four phases, branch 0',
        source_config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),files=files),indent=2)+'\n')


if __name__=='__main__':main()

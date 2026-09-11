"""Keep a half-A floor, then distribute remaining A/W work using calibration."""
import argparse
import hashlib
import json
from pathlib import Path
from allocate import allocate


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.input.read_text());assert r['complete']
    for name in r['policies']:
        fraction=int(name.removeprefix('auto'))/100
        tasks={task:allocate([u for u in r['units'] if u['task']==task],fraction,
                            start_tier='a50',allowed={'a50','act2','a50w','dual'})
               for task in dict.fromkeys(u['task'] for u in r['units'])}
        r['policies'][name]=dict(assignments={k:v for d in tasks.values() for k,v in d['assignments'].items()},by_task=tasks)
    r['activation_floor']=.5
    r['parent_policy_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    r['guard_scope']='Exploratory guard introduced after unrestricted allocation failed trajectory validation. Same calibration data; no new fitting to evaluation activations. Evaluate with new seeds; not a blinded research process.'
    a.output.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()

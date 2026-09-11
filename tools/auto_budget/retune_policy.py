import argparse
import hashlib
import json
from pathlib import Path
from allocate import allocate


def main():
    p=argparse.ArgumentParser();p.add_argument('policy',type=Path);p.add_argument('benchmark',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.policy.read_text());b=json.loads(a.benchmark.read_text());assert b['complete'] and b['model']==r['model']
    costs={x['shape']+'|'+x['tier']:x['optimized_ms'] for x in b['rows']}
    probes=json.loads((a.policy.parent/'local_errors.json').read_text())
    shapes={x['unit_key']:f'{x["m"]}:{x["n"]}:{x["k"]}' for x in probes}
    for key,c in r['costs'].items():c['ms']=costs[key]
    # Unit shape is reconstructed exactly from its original cost choice.
    for u in r['units']:
        shape=u['shape']=shapes[u['key']]
        for tier,option in u['options'].items():option['ms']=costs[shape+'|'+tier]
    for name,old in r['policies'].items():
        fraction=int(name.removeprefix('auto'))/100
        tasks={task:allocate([u for u in r['units'] if u['task']==task],fraction) for task in dict.fromkeys(u['task'] for u in r['units'])}
        r['policies'][name]=dict(assignments={k:v for v0 in tasks.values() for k,v in v0['assignments'].items()},by_task=tasks)
    r['launches']=b['launches'];r['benchmark_sha256']=hashlib.sha256(a.benchmark.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()

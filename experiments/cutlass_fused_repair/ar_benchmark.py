"""AR cache: six configurations, five isolated randomized rounds, three seeds."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASES=('bf16','joint125','cutlass150')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilots',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    sources=None
    for base in BASES:
        gate=json.loads((a.pilots/f'pilot_{base}/result.json').read_text())
        if not gate['complete'] or not gate['all_bitwise_equal']:raise RuntimeError('Pilot equivalence gate failed')
        source=gate['receipt']['source_sha256']
        if sources is not None and sources!=source:raise RuntimeError('Mixed pilot source versions')
        sources=source
    basepath=ROOT/'runs5090/joint_e2e_20260914';jobs=[];rng=random.Random(914)
    for repeat in range(5):
        names=[(base,cached) for base in BASES for cached in (False,True)];rng.shuffle(names)
        for base,cached in names:
            name=base+('_ar' if cached else '_original');dest=a.output/f'performance_{repeat:02d}_{name}'
            cmd=[sys.executable,'-m','thunderworld_fp4.cosmos','benchmark','--model',str(ROOT/'models/Cosmos3-Edge-a9d944e'),
                '--config',str(basepath/f'configs/{base}.json'),'--tuning',str(basepath/f'profiles/{base}.json'),
                '--case',str(ROOT/'tools/joint_inference/cases/robot_observe.json'),'--output',str(dest),
                '--steps','35','--seeds','93001','93002','93003']
            if cached:cmd+=['--ar-cache']
            if base!='bf16':cmd+=['--assets',str(ROOT/f'data_local/cutlass5090_rebuilt_20260914/i2v_b{base[-3:]}')]
            jobs.append(dict(base=base,cached=cached,name=name,round=repeat,output=str(dest),command=cmd))
    manifest=dict(complete=False,random_order_seed=914,pilot_source_sha256=sources,jobs=jobs)
    path=a.output/'performance_matrix.json'
    def save():path.write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    for job in jobs:
        dest=Path(job['output']);result=dest/'result.json'
        if result.exists():
            if not json.loads(result.read_text()).get('complete'):raise RuntimeError(f'Inspect incomplete job: {dest}')
        else:
            print('Running',dest,flush=True)
            with dest.with_suffix('.log').open('w') as log:
                job['exit_code']=subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            save()
            if job['exit_code']:raise RuntimeError(f'Failed job: {dest}')
        if json.loads(result.read_text())['source_sha256']!=sources:raise RuntimeError('Source changed after pilot')
    manifest['complete']=True;save()


if __name__=='__main__':main()

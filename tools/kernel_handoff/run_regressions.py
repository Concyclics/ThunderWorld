"""Real held-out steps and tails using settings selected on the tuning panel.

FFN step indices 1/2 were not used for tuning. Attention step 0/34 were not used
for tuning. --profile omits timing; every case retains decoded and BF16 errors.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--tuning',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    configs=json.loads(a.tuning.read_text())['configs']
    jobs=[]
    for key,settings in configs.items():
        kind,asset,m,mode=key.split(':')
        if int(m)!=128:continue
        variants=([(asset,128,1),(asset,128,2),(asset,129,2)] if kind=='ffn' else
                  [(asset.replace('step17','step0'),128,0),
                   (asset.replace('step17','step34'),128,0),
                   (asset.replace('step17','step34'),129,0)])
        for file,rows,step in variants:
            cmd=[sys.executable,str(Path(__file__).with_name('bench.py')),
                 '--data',str(a.data),'--weights',str(a.weights),'--kind',kind,
                 '--asset',file,'--m',str(rows),'--preset',mode,'--step-index',str(step),'--profile','--compare-baseline']
            for name,value in settings.items():
                flag='--'+name.replace('_','-')
                if isinstance(value,bool):
                    if value:cmd.append(flag)
                else:cmd.extend([flag,str(value)])
            target=a.output/f'{len(jobs):03d}_{kind}_{Path(file).stem}_{mode}_m{rows}_step{step}.json'
            cmd.extend(['--output',str(target)])
            jobs.append(cmd)
    (a.output/'commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
    for i,cmd in enumerate(jobs):
        print(f'[{i+1}/{len(jobs)}] {cmd[-1]}',flush=True)
        with (a.output/f'{i:03d}.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('All held-out-step/tail regressions passed',flush=True)


if __name__=='__main__':main()

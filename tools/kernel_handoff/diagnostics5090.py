"""Keep prepacked GEMM and repeated-row stress measurements out of the main table."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    for name in ['data','weights','tuning','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    configs=json.loads(a.tuning.read_text())['configs'];jobs=[]
    plans=[('inner','ffn',f'ffn_l0_{kind}.pt','a75_w6',m,m) for kind in ['gate','up','down'] for m in [128,512]]
    plans += [('stress','ffn',f'ffn_l0_{kind}.pt',mode,2048,512) for kind in ['gate','down'] for mode in ['base','a75_w6']]
    plans += [('stress','attention',f'attention_{kind}_step17.pt','quality',2048,560) for kind in ['av','robot']]
    for label,kind,asset,mode,m,lookup_m in plans:
        settings=configs[f'{kind}:{asset}:{lookup_m}:{mode}']
        target=a.output/f'{label}_{Path(asset).stem}_{mode}_m{m}.json'
        cmd=[sys.executable,str(Path(__file__).with_name('bench.py')),'--data',str(a.data),'--weights',str(a.weights),
             '--kind',kind,'--asset',asset,'--preset',mode,'--m',str(m),'--compare-baseline','--output',str(target)]
        if label=='inner':cmd.append('--inner-only')
        for key,value in settings.items():
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:cmd.append(flag)
            else:cmd.extend([flag,str(value)])
        jobs.append(cmd)
    (a.output/'commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
    for i,cmd in enumerate(jobs):
        print(f'[{i+1}/{len(jobs)}] {plans[i][:5]}',flush=True)
        with (a.output/f'{i:03d}.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()

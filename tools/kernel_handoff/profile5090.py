"""Sequential full Nsight profiles of representative baseline/optimized pairs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    for name in ['data','weights','tuning','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[2]
    ncu=shutil.which('ncu')
    if ncu is None:raise RuntimeError('Nsight Compute not found')
    plans=[('down','ffn','ffn_l0_down.pt','a75_w6',128),
           ('gate','ffn','ffn_l0_gate.pt','a75_w6',512),
           ('attention','attention','attention_av_step17.pt','full_dual',560)]
    commands=[]
    for label,kind,asset,mode,m in plans:
        for original in [True,False]:
            tag=label+('_original' if original else '_optimized')
            script=(root/'data_local/baseline_source' if original else root)/'tools/kernel_handoff/bench.py'
            cmd=[ncu,'--clock-control','none','--set','full','--nvtx','--nvtx-include','steady/',
                 '-o',str(a.output/tag),sys.executable,str(script),'--data',str(a.data),'--weights',str(a.weights),
                 '--kind',kind,'--asset',asset,'--preset',mode,'--m',str(m),'--tile-m','32','--profile',
                 '--output',str(a.output/(tag+'.json'))]
            if not original:cmd += ['--tuning',str(a.tuning)]
            commands.append(cmd)
            print('Profiling',tag,flush=True)
            with (a.output/(tag+'.log')).open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            for page,suffix,extra in [('raw','_metrics.csv',['--csv']),('details','_details.txt',[])]:
                with (a.output/(tag+suffix)).open('w') as out:
                    subprocess.run([ncu,'--import',str(a.output/(tag+'.ncu-rep')),'--page',page,*extra],stdout=out,check=True)
    (a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')


if __name__=='__main__':main()

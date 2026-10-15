"""Sequential local GPU validation, timing, and profiling for AR caching."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    def run(script,args,log):
        print('Running',script,*map(str,args),flush=True)
        with (root/log).open('w') as f:
            subprocess.run([sys.executable,str(ROOT/'experiments/cutlass_fused_repair'/script),*map(str,args)],
                cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    for base in ('joint125','bf16','cutlass150'):
        dest=root/f'pilot_{base}'
        if not (dest/'result.json').exists():run('ar_probe.py',['--mode','pilot','--name',base,'--output',dest],f'pilot_{base}.log')
        assert json.loads((dest/'result.json').read_text())['complete']
    run('ar_benchmark.py',['--output',root/'performance','--pilots',root],'performance.log')
    run('summarize_ar.py',[root/'performance'],'summary.log')
    dest=root/'profile_joint125_ar'
    if not (dest/'receipt.json').exists():
        run('pipeline_probe.py',['--name','joint125','--ar-cache','--output',dest],'profile_joint125_ar.log')
    run('analyze_pipeline_trace.py',[dest/'trace.json','--output',dest],'profile_breakdown.log')
    print('Complete',root,flush=True)


if __name__=='__main__':main()

"""Run remaining qualification, randomized full-request benchmark and CUPTI trace serially."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs5090/next_tuning_20260914'
def run(script,*args,log):
    print('Running',script,args,flush=True)
    with (OUT/log).open('w') as f:subprocess.run([sys.executable,'experiments/cutlass_fused_repair/'+script,*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    for base in ('bf16','cutlass150'):
        run('next_tuning_probe.py','--name',base,'--variants','ar','best','--changed-inputs','--output',OUT/f'gate_{base}',log=f'gate_{base}.log')
    run('next_benchmark.py','--pilots',OUT,'--output',OUT/'performance',log='performance.log')
    run('summarize_next.py',OUT/'performance',log='summary.log')
    run('pipeline_probe.py','--name','joint125','--ar-cache','--first-frame','--codec-graph','--fuse-norm','--fuse-pointwise','--static-rotary','--fuse-calibrated','--output',OUT/'profile_best',log='profile_best.log')
    run('analyze_pipeline_trace.py',OUT/'profile_best/trace.json','--output',OUT/'profile_best',log='analyze.log')
if __name__=='__main__':main()

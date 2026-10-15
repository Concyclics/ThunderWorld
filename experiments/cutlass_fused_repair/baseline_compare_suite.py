"""Serial full-request baseline measurement followed by separate CPU analysis."""
import subprocess,sys
from baseline_compare import ROOT,OUT

def main():
    for script,args,log in [('baseline_compare.py',['--stage','performance'],'performance.log'),
                            ('summarize_baselines.py',['--stage','performance'],'summary.log'),
                            ('summarize_baselines.py',['--stage','quality'],'quality.log'),
                            ('report_baselines.py',[],'report.log')]:
        print('Running',script,args,flush=True)
        with (OUT/log).open('w') as f:subprocess.run([sys.executable,'experiments/cutlass_fused_repair/'+script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()

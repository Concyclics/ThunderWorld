"""Collect per-kernel hardware counters without shadowing Python's profile module."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

METRICS=['gpu__time_duration.sum','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
 'sm__warps_active.avg.pct_of_peak_sustained_active','dram__throughput.avg.pct_of_peak_sustained_elapsed',
 'sm__throughput.avg.pct_of_peak_sustained_elapsed','launch__registers_per_thread',
 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']


def main():
    p=argparse.ArgumentParser();p.add_argument('benchmarks',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    ncu='/usr/local/cuda-12.9/bin/ncu'
    for source in a.benchmarks:
        report=json.loads(source.read_text());assert report['complete']
        # Down projections, I2V and policy; dynamic partial and full dual buckets.
        rows=[r for r in report['rows'] if r['task'] in ('i2v','policy') and int(r['shape'].split(':')[1])<int(r['shape'].split(':')[2]) and r['tier'] in ('a50','dual')]
        for row in rows:
            tag=report['model']+'_'+row['shape'].replace(':','_')+'_'+row['tier']
            cmd=[ncu,'--clock-control','none','--nvtx','--nvtx-include','steady/','--metrics',','.join(METRICS),
                 '-o',str(a.output/tag),sys.executable,str(Path(__file__).with_name('bench.py')),
                 '--payload',row['payload'],'--tier',row['tier'],'--bm',str(row['winner']['bm']),
                 '--stages',str(row['winner']['num_stages'])]
            with (a.output/(tag+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            path=a.output/(tag+'.csv')
            with path.open('w') as f:subprocess.run([ncu,'--import',str(a.output/(tag+'.ncu-rep')),'--page','raw','--csv'],stdout=f,check=True)
            with path.open() as f:
                reader=csv.DictReader(f);units=None;kernels=[]
                for r in reader:
                    if not r.get('ID','').isdigit():units=r;continue
                    d=dict(kernel=r['Kernel Name'])
                    for metric in METRICS:
                        value=r.get(metric,'').replace(',','')
                        d[metric]=float(value) if value else None
                    unit=units.get('gpu__time_duration.sum','ns')
                    d['duration_ns']=d['gpu__time_duration.sum']*{'ns':1,'us':1000,'ms':1e6,'s':1e9}[unit]
                    kernels.append(d)
            assert kernels and any('gemm' in k['kernel'] for k in kernels)
            records.append(dict(tag=tag,benchmark=row,command=cmd,kernels=kernels))
            print('Profiled',tag,flush=True)
    (a.output/'hardware.json').write_text(json.dumps(dict(complete=True,profiles=records,
        scope='Nsight kernel replay, default cache control, clocks unchanged. Tensor elapsed activity is not a measured percentage of FP4 FLOP peak. Use separate steady-state timing for speed.'),indent=2)+'\n')


if __name__=='__main__':main()

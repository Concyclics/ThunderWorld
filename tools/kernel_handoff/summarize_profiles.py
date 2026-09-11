"""Compact measured hardware counters from Nsight Compute raw CSV exports."""
import argparse
import csv
import json
from pathlib import Path

METRICS={
    'duration_ns':'gpu__time_duration.sum',
    'registers_per_thread':'launch__registers_per_thread',
    'achieved_occupancy_percent':'sm__warps_active.avg.pct_of_peak_sustained_active',
    'tensor_elapsed_percent':'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm_throughput_percent':'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram_active_percent':'dram__cycles_active.avg.pct_of_peak_sustained_elapsed',
    'l2_throughput_percent':'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'local_load_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
    'local_store_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
    'spill_instructions':'sass__inst_executed_register_spilling',
    'long_scoreboard_cycles_per_issue':'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'short_scoreboard_cycles_per_issue':'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
    'wait_cycles_per_issue':'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio',
}


def main():
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report={'scope':'Nsight replay/cache-flush measurements; clocks unchanged; not steady-state operator timing','profiles':{}}
    for path in a.csv:
        kernels=[]
        with path.open() as f:
            units={}
            for row in csv.DictReader(f):
                if not row.get('ID','').isdigit():
                    units=row
                    continue
                d={name:row.get(name) for name in ['Kernel Name','Grid Size','Block Size']}
                for name,key in METRICS.items():
                    value=row.get(key,'')
                    try:
                        number=float(value.replace(',',''))
                        unit=units.get(key,'')
                        if name=='duration_ns':
                            number*={'ns':1,'us':1000,'ms':1000000,'s':1000000000}[unit]
                        elif name in ('local_load_sectors','local_store_sectors','spill_instructions'):
                            if unit.startswith('K'):number*=1000
                            elif unit.startswith('M'):number*=1000000
                            elif unit.startswith('G'):number*=1000000000
                        d[name]=number
                    except ValueError:d[name]=None
                kernels.append(d)
        report['profiles'][path.stem]=kernels
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

"""Inspect fixed-scene transfer, per-role errors, and packing after the frozen run."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from structure_repair import COVERAGES, write
from structure_repair_study import paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--completed-only',action='store_true',help='Report completed cohorts and explicitly list paused/missing cohorts')
    a=p.parse_args();protocol=json.loads((a.root/'protocol.json').read_text())
    output=[];scene_output=[];checks=[];skipped=[]
    for cohort,job in protocol['jobs'].items():
        result_path=a.root/cohort/'result.json'
        run=json.loads(result_path.read_text()) if result_path.exists() else None
        if run is None or not run['complete']:
            if not a.completed_only:raise ValueError('Wait for the complete frozen cohort')
            skipped.append(dict(cohort=cohort,completed_requests=len(run['rows']) if run else 0,reason='Not completed; see progress and worker log'))
            continue
        all_rows=[]
        for request in run['rows']:
            case,seed=request['case'],request['seed']
            file=a.root/cohort/f'bf16_{case}_seed{seed}_r0_heads.repair.json'
            records=json.loads(file.read_text())['rows']
            all_rows.extend(dict(r,case=case,seed=seed) for r in records)
            grouped=defaultdict(list)
            for r in records:grouped[r['step'],r['layer'],r['coverage']].append(r)
            for (step,layer,coverage),rows in grouped.items():
                if coverage in COVERAGES:
                    assert len({r['total_mma'] for r in rows})==1
                    assert len({r['selected_tiles'] for r in rows})==1
                    assert len(rows)==12
        checks.append(dict(cohort=cohort,requests=len(run['rows']),rows=len(all_rows),
            exact_mma_and_tile_count=True,random_repeats=8))
        metrics=['nmse']+sorted({role for r in all_rows for role in r['role_nmse']})
        for coverage in COVERAGES:
            for metric in metrics:
                panels=defaultdict(list)
                for r in all_rows:
                    if r['coverage']!=coverage:continue
                    if metric!='nmse' and metric not in r['role_nmse']:continue
                    method='random' if r['method'].startswith('random_') else r['method']
                    value=r['nmse'] if metric=='nmse' else r['role_nmse'][metric]
                    panels[r['case'],r['seed'],method,r['step'],r['layer']].append(value)
                requests=defaultdict(list)
                for key,values in panels.items():requests[key[:3]].append(float(np.mean(values)))
                def table(method):
                    return np.array([[np.mean(requests[c,s,method]) for s in protocol['evaluation_seeds']] for c in job['cases']])
                ours=table('structure')
                for baseline in ('random','window','structure_no_head','structure_no_step'):
                    other=table(baseline)
                    value=paired(ours,other,protocol['evaluation_seeds'])
                    output.append(dict(cohort=cohort,coverage=coverage,nominal_attention_mma=1+2*coverage,
                        metric=metric,baseline=baseline,**value))
                    for i,case in enumerate(job['cases']):
                        scene_output.append(dict(cohort=cohort,case=case,used_for_prior=case==job['development'],
                            coverage=coverage,metric=metric,baseline=baseline,
                            structure_nmse=float(ours[i].mean()),baseline_nmse=float(other[i].mean()),
                            reduction_pct=float(100*(other[i].mean()-ours[i].mean())/other[i].mean()),
                            request_wins=int((ours[i]<other[i]).sum()),requests=len(protocol['evaluation_seeds'])))
    directory=a.root/'report';directory.mkdir(exist_ok=True)
    write(directory/'detailed_comparisons.json',dict(complete=not skipped,skipped=skipped,comparisons=output,scenes=scene_output,checks=checks,
        interpretation='Positive reduction means lower error. Role/ablation intervals are descriptive, not additional multiplicity-adjusted primary claims. Random is the mean of eight fixed seeded controls.'))
    for name,rows in [('comparisons.csv',output),('scene_transfer.csv',scene_output)]:
        with (directory/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(complete=not skipped,skipped=skipped,checks=checks,comparison_rows=len(output),scene_rows=len(scene_output))))


if __name__=='__main__':main()

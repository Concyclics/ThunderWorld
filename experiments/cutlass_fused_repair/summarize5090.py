"""Archive the 5090 follow-up and verify unchanged complete-trajectory outputs."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=a.output/'records';records.mkdir(exist_ok=True)
    rows=[];summary=[];sources={}
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    for panel in ('i2v_b125','i2v_b150','policy_b125'):
        for tile in (128,64):
            path=a.runs/f'batched_{panel}_n{tile}.json';d=json.loads(path.read_text())
            assert d['complete'] and len(d['rows'])==16
            assert d['timing_protocol']['graph_batch']==32 and d['timing_protocol']['rounds']==9
            shutil.copy2(path,records/path.name);sources[str(path)]=digest(path)
            group=[]
            for r in d['rows']:
                assert r['direct_pack_bitwise']
                for value in r['timing_ms'].values():assert len(value['samples'])==9
                t={k:v['median'] for k,v in r['timing_ms'].items()}
                assert abs(r['direct_pack_speedup']-t['bridge_total']/t['bridge_direct_total'])<1e-12
                spec=json.loads(r['repair'])
                row=dict(panel=panel,tile_n=tile,module=r['module'],phase=r['phase'],
                    m=r['shape'][0],n=r['shape'][1],k=r['shape'][2],
                    a_blocks=spec['activation_blocks'],w_blocks=len(spec['weight_support']),
                    native_us=1000*t['native_total'],cutlass_us=1000*t['bridge_total'],
                    direct_us=1000*t['bridge_direct_total'],native_over_cutlass=r['diagnostic_native_over_bridge'],
                    direct_speedup=r['direct_pack_speedup'],bitwise=True)
                rows.append(row);group.append(row)
            gains=[r['direct_speedup'] for r in group];native=[r['native_over_cutlass'] for r in group]
            summary.append(dict(panel=panel,tile_n=tile,cases=16,native_min=min(native),native_max=max(native),
                direct_min=min(gains),direct_max=max(gains),direct_median=statistics.median(gains),
                faster=sum(v>1 for v in gains),above_1_03=sum(v>1.03 for v in gains)))
    # Preserve the noisy first attempt; it is not substituted for confirmation.
    for path in sorted(a.runs.glob('direct_*.json')):
        shutil.copy2(path,records/path.name);sources[str(path)]=digest(path)
    trajectory=[]
    import numpy as np
    for panel in ('i2v_b125','i2v_b150'):
        runs=[]
        for mode in ('original','direct'):
            path=a.runs/f'trajectory_{panel}_{mode}/result.json';d=json.loads(path.read_text())
            assert d['complete'] and len(d['rows'])==2
            with gzip.open(records/f'{path.parent.name}.json.gz','wt') as out:json.dump(d,out)
            sources[str(path)]=digest(path);runs.append((path.parent,d))
        original,direct=runs
        assert original[1]['image_sha256']==direct[1]['image_sha256']
        assert original[1]['model_identity']==direct[1]['model_identity']
        assert original[1]['prepared_weight_bytes']==direct[1]['prepared_weight_bytes']
        for old,new in zip(original[1]['rows'],direct[1]['rows']):
            assert old['seed']==new['seed'] and old['steps']==new['steps']==35
            assert old['sha256']==new['sha256']
            with np.load(original[0]/old['outputs']) as x,np.load(direct[0]/new['outputs']) as y:
                assert set(x.files)==set(y.files)
                for key in x.files:
                    assert np.isfinite(x[key]).all() and np.array_equal(x[key],y[key]),(panel,key)
            lo,ln=old['compute_ledger'],new['compute_ledger']
            assert lo['feasible'] and ln['feasible'] and lo['total_mma']==ln['total_mma']
            assert lo['plan_sha256']==ln['plan_sha256']
            assert ln['packing_traffic']['cutlass_scale_read_bytes']==0
            assert ln['packing_traffic']['cutlass_scale_conversion_launches']==0
            assert old['performance_eligible'] is False and new['performance_eligible'] is False
            trajectory.append(dict(panel=panel,seed=old['seed'],steps=35,bitwise=True,
                mma_ratio=ln['ratio'],removed_scale_read_bytes=lo['packing_traffic']['cutlass_scale_read_bytes'],
                removed_launches=lo['packing_traffic']['cutlass_scale_conversion_launches'],
                weight_bytes=direct[1]['prepared_weight_bytes'],sha256=new['sha256'],
                scope='Development trajectory equality, not final quality or E2E performance'))
    for name,data in [('operators.csv',rows),('summary.csv',summary)]:
        with (a.output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    result=dict(complete=True,upstream='23bd2a9',gpu='RTX 5090',operator_cases=len(rows),
        bitwise_operator_cases=sum(r['bitwise'] for r in rows),summary=summary,trajectory=trajectory,
        scope='Layer-0 four-phase diagnostic panels; no multiplication into E2E speedups',sources=sources)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# RTX 5090 CUTLASS follow-up measurements','',
        '32 operators per captured graph, 9 alternating rounds. Full operator includes selection, packing, scale layout and GEMM. Ranges span 16 shapes/phases per row, not confidence intervals. Native is the fixed BM128/stages2 Triton baseline, not a new best-of-all-Triton search. Actual release tensors are I2V M960 and Policy M316; the older M416 panel description is inaccurate. M416 is covered by correctness tests, not this policy timing table.','',
        '| Panel | CUTLASS N | Native / original CUTLASS | Direct-pack speedup range | Median direct-pack speedup | >1.03× cases |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['panel']} | {r['tile_n']} | {r['native_min']:.3f}–{r['native_max']:.3f}× | {r['direct_min']:.3f}–{r['direct_max']:.3f}× | {r['direct_median']:.3f}× | {r['above_1_03']}/16 |")
    lines+=['','All 96 direct-pack outputs match the same CUTLASS baseline bitwise. Both budgets and both development seeds match across complete 35-step trajectories. Pipeline timings are intentionally not promoted to a speed claim. First-attempt host-replay timing remains in records/direct_*.json.','']
    (a.output/'TABLE.md').write_text('\n'.join(lines))
    print(json.dumps({'operator_cases':len(rows),'trajectory_pairs':len(trajectory),'complete':True}))


if __name__=='__main__':main()

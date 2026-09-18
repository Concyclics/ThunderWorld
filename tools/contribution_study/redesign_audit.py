"""Audit frozen low-budget allocations without fitting to the evaluation outputs."""
import argparse,collections,json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.study/'manifest.json').read_text());quality=json.loads((a.study/'report/quality.json').read_text())
    if not quality['complete']:raise ValueError('Complete frozen evidence required')
    rows=[]
    for name,method in manifest['methods'].items():
        if not method['config'].get('budget_plan'):continue
        plan=Path(method['config']['budget_plan']);ledger_path=a.study/'generation'/name/manifest['cases'][0]['case']['id']/'compute.json'
        ledger=json.loads(ledger_path.read_text());terms=collections.Counter();by_phase={};unions={};full={};off=collections.Counter()
        for row in ledger['rows']:
            ph=by_phase.setdefault(str(row['phase']),collections.Counter())
            for dtype,items in row['mma_by_dtype'].items():
                for term,value in items.items():
                    key=row['kind']+'.'+dtype+'.'+term;terms[key]+=value*row['calls'];ph[key]+=value*row['calls']
            if row['kind']=='linear':
                name_=row['name'];full[name_]=row['shape'][1]*row['shape'][2]//128**2
                unions.setdefault(name_,set()).update(row['w_blocks'])
                if not row['a_blocks']:off['A_off']+=1
                if not row['w_blocks']:off['W_off']+=1
        density=sum(map(len,unions.values()))/sum(full.values())
        rows.append(dict(name=name,budget=ledger['budget'],ratio=ledger['ratio'],
            terms_over_C0={k:v/ledger['base_mma'] for k,v in terms.items()},
            phases_over_C0={ph:{k:v/ledger['base_mma'] for k,v in val.items()} for ph,val in by_phase.items()},
            w1_union_density=density,w1_union_bytes=ledger['w1_support_union_bytes'],
            generated_weight_bits_excluding_indices_auxiliary=4.5*(1+density),
            disabled_linear_groups=dict(off),group_scope='operator x phase x CFG, not unique model layers',
            plan_sha256=sha(plan),compute_sha256=sha(ledger_path)))
    value=dict(complete=True,source_manifest_sha256=sha(a.study/'manifest.json'),rows=rows,performance_eligible=False,
        observations=['Full three/four use RTN identity assets; old low-budget candidates also changed representation.',
            'Old base calibration optimized no-repair local loss on one scene; it did not optimize the deployed sparse objective.',
            'Independent local NMSE sums are not whole-trajectory or cross-domain error bounds.',
            'Head statistics propose interventions; attention repair alone cannot recover the FFN/projection repair budget.'])
    write(a.output,value)


if __name__=='__main__':main()

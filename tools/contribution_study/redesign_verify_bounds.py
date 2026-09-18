"""Enumerate padded key lengths, including selector power-of-two jumps, on CPU."""
import argparse,json
from fractions import Fraction
from pathlib import Path
from baselines import sha,write
from thunderworld_fp4 import RepairConfig
from thunderworld_fp4.cosmos.budget import RepairPlan,attention_cost


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    results=[]
    for catalog in a.catalog:
        for candidate in json.loads(catalog.read_text())['candidates']:
            path=Path(candidate['config']['budget_plan']);plan=RepairPlan.load(path);old=json.loads((path.parent/'compute.json').read_text())
            linears=[r for r in old['rows'] if r['kind']=='linear'];attention=[r for r in old['rows'] if r['kind']=='attention']
            base=sum(r['base_mma']*r['calls'] for r in linears);total=sum(r['total_mma']*r['calls'] for r in linears)
            cap=Fraction(str(plan.value['budget']));checks=[]
            for nt in range(8,17):
                n=max(961,(nt-1)*128+1);b=base;c=total
                for row in attention:
                    m,_,h,hk,_=row['shape'];cfg=plan.attention_config('i2v',row['branch'],row['phase'],row['name'],RepairConfig())
                    part=attention_cost(m,n,h,hk,qk_counts=cfg.head_counts(h,n),pv_counts=cfg.pv_counts(h,n),qk_terms=cfg.terms,pv_terms=cfg.pv_terms or cfg.terms)
                    b+=part['base_mma']*row['calls'];c+=part['total_mma']*row['calls']
                if c*cap.denominator>b*cap.numerator:raise ValueError(f'Budget exceeds cap: {candidate["name"]} at N{n}')
                checks.append(dict(n=n,key_tiles=nt,ratio=c/b))
            results.append(dict(name=candidate['name'],plan_sha256=sha(path),passed=True,checks=checks,maximum_ratio=max(r['ratio'] for r in checks)))
    write(a.output,dict(complete=True,results=results,scope='M960, H16/Hkv8, current Edge input shape; every padded K tile count for N961..2048. Includes BF16 selector jumps. Runtime still rechecks actual inputs.'))


if __name__=='__main__':main()

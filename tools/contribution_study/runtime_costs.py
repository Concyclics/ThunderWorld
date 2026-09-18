"""Audit every request cap and report actual resident memory, separately from logical work.

External sparse/dense FLOPs are not padded issued-MMA bounds and are never
converted into our FP4 budget ratios. Wall times are deliberately excluded.
"""
import argparse,json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();selection=json.loads((a.study/'frozen/selection.json').read_text())
    generation=a.study/'frozen/generation';reference=json.loads((generation/'b2.5_marginal/result.json').read_text())
    template=reference['rows'][0]['compute_ledger']['rows']
    projection=sum(2*r['shape'][0]*r['shape'][1]*r['shape'][2]*r['calls'] for r in template if r['kind']=='linear')
    attention=sum(4*r['shape'][0]*r['shape'][1]*r['shape'][2]*r['shape'][4]*r['calls'] for r in template if r['kind']=='attention')
    from thunderworld_fp4.cosmos.budget import linear_cost,attention_cost
    rows=[]
    for name in selection['methods']:
        path=(a.study/'legacy_final/result.json') if name=='legacy_2.5' else generation/name/'result.json'
        value=json.loads(path.read_text())
        if not value['complete'] or len(value['rows'])!=16:raise ValueError('Incomplete final run: '+name)
        runtime=value.get('runtime');row=dict(name=name,receipt=str(path),receipt_sha256=sha(path),performance_eligible=False)
        if runtime:
            row['memory']={k:runtime[k] for k in ('prepared_weight_bytes','support_buffer_bytes','resident_model_state_bytes')}
            row['memory'].update(peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in value['rows']),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in value['rows']))
            row['warmup_gqa_union_density']=runtime['warmup_gqa_union_density']
        ledgers=[r.get('compute_ledger') for r in value['rows']]
        if all(ledgers):
            if any(not x['feasible'] or x['ratio']>x['budget'] for x in ledgers):raise ValueError('Per-request cap violated')
            row.update(requests_verified=16,issued_mma={k:v for k,v in ledgers[0].items() if k!='rows'},
                max_request_ratio=max(x['ratio'] for x in ledgers))
        elif name.startswith('b') and name[1:2].isdigit():raise ValueError('Missing budget ledger')
        elif name.startswith('fp4_'):
            terms={'fp4_base':1,'fp4_three':3,'fp4_four':4}[name];total=base=0
            launches={s['module']:s['launch']['bm'] for s in runtime['observed_shapes']}
            for r in template:
                if r['kind']=='linear':
                    m,n,k=r['shape'];cost=linear_cost(m,n,k,bm=launches[r['name']],a_blocks=k//128 if terms>1 else 0,
                        w_blocks=list(range(n//128*(k//128))) if terms>1 else [],cross=terms==4)
                else:
                    m,n,h,hk,d=r['shape'];quota=[(n+127)//128 if terms>1 else 0]*h
                    cost=attention_cost(m,n,h,hk,bm=32,qk_counts=quota,pv_counts=quota,qk_terms=max(terms,3),pv_terms=max(terms,3))
                total+=r['calls']*cost['total_mma'];base+=r['calls']*cost['base_mma']
            row['issued_mma']=dict(base_mma=base,total_mma=total,ratio=total/base,scope='Native anchor launch tiles, full/empty masks; not a candidate within the four caps')
        else:
            logical=dict(bf16_attention_flops=attention,scope='Generation projections/attention only; excludes common AR prefill, codec and non-MMA work. Logical MAC counts, no hardware padding qualification.')
            if name in ('bf16','awq'):logical['bf16_projection_flops']=projection
            elif name=='fp8':logical['fp8_projection_flops']=projection
            elif name=='arcquant':logical['fp4_dense_projection_flops']=projection*1.5
            elif name=='sharq':logical.update(fp4_dense_projection_flops=projection,fp4_structured_sparse_nonzero_flops=projection//2,
                structured_sparse_dense_equivalent_flops=projection,scope=logical['scope']+' Sparse 2:4 instructions have a different peak; do not add dense-equivalent and nonzero counts.')
            else:logical=dict(scope='Historical optimized backend: use its recorded configuration and separate previous issued-cost audit; no new cost inferred here.')
            row['logical_work_not_budget_qualified']=logical
        rows.append(row)
    write(a.output,dict(complete=True,rows=rows,selection_sha256=sha(a.study/'frozen/selection.json'),performance_eligible=False,
        scope='Measured native resident tensors and allocator peaks; caps checked on all final requests. External logical work is explicitly not padded MMA accounting. Shared times excluded.'))


if __name__=='__main__':main()

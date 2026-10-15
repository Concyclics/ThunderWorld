"""Freeze one experimental candidate per cap before any final generation."""
import argparse,json
from pathlib import Path
from baselines import sha,write
from representation_trajectory import replace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('validation-manifest','quality','cohort','baseline-root','legacy','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Selection is immutable: '+str(a.output))
    manifest=json.loads(a.validation_manifest.read_text());quality=json.loads(a.quality.read_text());cohort=json.loads(a.cohort.read_text())
    if not quality['complete'] or set(quality['seeds'])!=set(cohort['seeds']['validation']) or cohort['seed_status']!='frozen':
        raise ValueError('Complete frozen validation split required')
    decisions=[];chosen=[]
    costs={}
    for job in manifest['runs']:
        if '--runtime-config' not in job['args']:continue
        config=Path(job['args'][job['args'].index('--runtime-config')+1])
        ledger=json.loads((config.parent/'compute.json').read_text());packing=ledger['packing_traffic']
        if not ledger['feasible']:raise ValueError('Infeasible candidate in validation')
        writes=sum(packing.get(key,0) for key in ('base_write_bytes','residual_write_bytes','base_code_scale_write_bytes',
            'base_factor_write_bytes','selected_residual_write_bytes_upper'))
        costs[job['name']]=(ledger['total_mma'],writes,ledger['w1_support_union_bytes'])
    for cap in (1.25,1.5,2.,2.5):
        control=f'b{cap:g}_control';reference=quality['means'][control]
        pool=[f'b{cap:g}_marginal',f'b{cap:g}_head_coordinate']
        guard=[name for name in pool if quality['means'][name]['ssim_mean']>=reference['ssim_mean']
               and quality['means'][name]['relative_l2']<=reference['relative_l2']]
        best=min(guard or pool,key=lambda name:(quality['means'][name]['lpips'],*costs[name],name))
        chosen.append(best);decisions.append(dict(budget=cap,selected=best,control=control,
            validation_guardrails=best in guard,considered=pool,
            selected_cost=dict(zip(('mma','packing_write_bytes_upper','w1_union_bytes'),costs[best])),
            scope='At most one experimental candidate frozen; failing validation guardrails is retained as a negative test, not recommended.'))
    runs=[row for row in manifest['runs'] if row['name'] in chosen or row['name'].endswith('_control') or not row['name'].startswith('b')]
    roots={row['name']:Path(manifest['output'])/row['name'] for row in runs}
    for name in ('bf16','fp8'):
        roots[name]=a.baseline_root/name;runs.append(dict(name=name,args=['--method',name]))
    roots['legacy_2.5']=a.legacy
    digests={}
    for name,folder in roots.items():
        receipt=json.loads((folder/'result.json').read_text())
        if not receipt['complete'] or receipt['admission'].get('cohort_sha256')!=sha(a.cohort) or receipt['admission']['split']!='validation':
            raise ValueError('Unqualified validation receipt: '+name)
        if {r['seed'] for r in receipt['rows']}!=set(cohort['seeds']['validation']):raise ValueError('Incomplete validation seeds')
        digests[name]=receipt['configuration_sha256']
    value=dict(complete=True,cohort_sha256=sha(a.cohort),validation_quality_sha256=sha(a.quality),
        methods=list(roots),configuration_sha256=digests,decisions=decisions,
        primary_comparisons=[dict(candidate=row['selected'],control=other) for row in decisions for other in (row['control'],'fp8')],
        secondary_comparisons=[dict(candidate=name,control='fp8') for name in ('fp4_three','fp4_four')],family_size=10)
    a.output.mkdir();write(a.output/'selection.json',value)
    common=list(manifest['common_args']);replace(common,'--split',['final']);replace(common,'--seeds',list(map(str,cohort['seeds']['final'])))
    common+=['--frozen-selection',str((a.output/'selection.json').resolve())]
    write(a.output/'manifest.json',dict(output=str((a.output/'generation').resolve()),common_args=common,runs=runs))


if __name__=='__main__':main()

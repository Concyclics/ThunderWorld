"""Observed quality/MMA/asset frontier, never a claim of global or latency optimality."""
import argparse,json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('quality','control-storage','candidate-storage','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();quality=json.loads(a.quality.read_text())
    if not quality['complete']:raise ValueError('Complete quality results required')
    banks={kind:json.loads(path.read_text()) for kind,path in (('control',a.control_storage),('marginal',a.candidate_storage))}
    rows=[]
    for kind,bank in banks.items():
        if not bank['complete']:raise ValueError('Incomplete physical asset report')
        for entry in bank['rows']:
            if not entry['name'].endswith('_grow_mixed'):continue
            name=f'b{entry["budget"]:g}_{kind}';metric=quality['means'][name]
            control=quality['means'][f'b{entry["budget"]:g}_control']
            rows.append(dict(name=name,budget=entry['budget'],mma_ratio=entry['ratio'],asset_payload_bytes=entry['bytes']['payload_bytes'],
                metrics=metric,feasible=True,mean_guardrails=metric['ssim_mean']>=control['ssim_mean'] and metric['relative_l2']<=control['relative_l2'],
                objective=[metric['lpips'],entry['ratio'],entry['bytes']['payload_bytes']]))
    eligible=[row for row in rows if row['mean_guardrails']]
    for row in rows:
        row['nondominated_in_searched_feasible_set']=row['mean_guardrails'] and not any(
            all(x<=y for x,y in zip(other['objective'],row['objective'])) and any(x<y for x,y in zip(other['objective'],row['objective']))
            for other in eligible)
    write(a.output,dict(complete=True,quality_sha256=sha(a.quality),control_storage_sha256=sha(a.control_storage),
        candidate_storage_sha256=sha(a.candidate_storage),rows=rows,
        scope='Observed LPIPS/MMA/asset-payload frontier with SSIM/L2 mean guardrails against the paired control. '
              'No latency claim, no whole-model VRAM claim, and no significance/FP8 acceptance inferred from dominance.'))


if __name__=='__main__':main()

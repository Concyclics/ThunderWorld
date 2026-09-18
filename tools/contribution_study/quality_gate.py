"""Paired contribution contrasts with explicit split and family-wise correction."""
import argparse,json
from pathlib import Path
from baselines import sha,write
from paired_stats import paired


def contrast(quality,candidate,control,*,family_size):
    rows={name:{r['seed']:r['metrics']['video'] for r in quality['rows'] if r['name']==name}
          for name in (candidate,control)}
    if any(set(v)!=set(quality['seeds']) for v in rows.values()):raise ValueError('Incomplete paired metrics')
    metrics={metric:paired({s:v[metric] for s,v in rows[candidate].items()},
        {s:v[metric] for s,v in rows[control].items()},family_size=family_size,lower_is_better=metric!='ssim_mean')
        for metric in ('lpips','ssim_mean','relative_l2')}
    passed=metrics['lpips']['familywise_ci'][1]<0 and metrics['ssim_mean']['mean_delta']>=0 and metrics['relative_l2']['mean_delta']<=0
    return dict(candidate=candidate,control=control,metrics=metrics,paired_gate_passed=passed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('quality','cohort','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--split',choices=('trajectory','validation','final'),required=True)
    p.add_argument('--contrast',action='append',required=True,help='candidate=control')
    p.add_argument('--family-size',type=int,required=True);a=p.parse_args()
    quality=json.loads(a.quality.read_text());cohort=json.loads(a.cohort.read_text())
    if not quality['complete'] or cohort['seed_status']!='frozen' or set(quality['seeds'])!=set(cohort['seeds'][a.split]):
        raise ValueError('Incomplete or wrong split')
    if a.family_size<len(a.contrast):raise ValueError('Correction cannot be smaller than the declared comparison family')
    rows=[contrast(quality,*pair.split('='),family_size=a.family_size) for pair in a.contrast]
    write(a.output,dict(complete=True,split=a.split,quality_sha256=sha(a.quality),cohort_sha256=sha(a.cohort),
        contrasts=rows,final_quality_claim_eligible=a.split=='final',
        scope='Paired BF16-agreement metrics. Validation contrasts are not final confirmation; no task-success or generalization claim.'))


if __name__=='__main__':main()

"""Add clip PSNR to historical per-video metrics without rewriting frozen results."""
import argparse,json,math,statistics
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--quality',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    value=json.loads(a.quality.read_text())
    if not value['complete']:raise ValueError('Incomplete metrics')
    rows=[]
    for row in value['rows']:
        rmse=row['metrics']['video']['rmse']
        if not math.isfinite(rmse) or rmse<0:raise ValueError('Invalid original RMSE')
        rows.append(dict(name=row['name'],seed=row['seed'],mse=rmse**2,psnr_db=-20*math.log10(rmse) if rmse else None,psnr_exact=rmse==0))
    means={name:dict(mean_psnr_db=statistics.mean(r['psnr_db'] for r in rows if r['name']==name) if all(r['psnr_db'] is not None for r in rows if r['name']==name) else None,
        exact_clips=sum(r['psnr_exact'] for r in rows if r['name']==name)) for name in value['means']}
    write(a.output,dict(complete=True,source_sha256=sha(a.quality),rows=rows,means=means,
        definition='Per-video -20*log10(float64 RMSE), data_range=1, then equal-weight mean across videos. Exact match is null (+infinity), never clamped. Historical outputs/metrics unchanged.'))


if __name__=='__main__':main()

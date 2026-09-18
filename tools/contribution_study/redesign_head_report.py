"""Descriptive within-panel head feature ranking against native repair interventions."""
import argparse,collections,json,warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cm=json.loads((a.calibration/'manifest.json').read_text())
    if not cm['complete']:raise ValueError('Complete calibration required')
    correlations=collections.defaultdict(list)
    for case,digest in cm['cases'].items():
        path=a.calibration/(case+'.json')
        if sha(path)!=digest:raise ValueError('Changed calibration case')
        data=json.loads(path.read_text())
        for rows in data['attention'].values():
            for row in rows:
                for side in ('qk','pv'):
                    gain=np.asarray(row['errors'][side+'0'])-np.asarray(row['errors']['three'])
                    for feature in ('ar_mass','normalized_entropy','concentration','q_energy'):
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore');rho=float(spearmanr(row[feature],gain).statistic)
                        if np.isfinite(rho):correlations[side+'.'+feature].append(rho)
    summary={key:dict(panel_count=len(vals),median=float(np.median(vals)),mean=float(np.mean(vals)),
        q25_q75=np.quantile(vals,[.25,.75]).tolist(),positive_fraction=float(np.mean(np.asarray(vals)>0))) for key,vals in correlations.items()}
    write(a.output,dict(complete=True,calibration_sha256=sha(a.calibration/'manifest.json'),summary=summary,
        scope='Within-panel Spearman over heads. Descriptive calibration only; repeated layer/phase panels are not independent inferential samples. No causal or final-quality claim.'))


if __name__=='__main__':main()

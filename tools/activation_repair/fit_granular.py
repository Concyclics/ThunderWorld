"""Fit smaller groups with early phases only; report held phases without filtering."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/activation_repair_20260911'
a=pd.DataFrame(json.loads((OUT/'calibration.json').read_text())['rows']);a['gain']=1.
b=pd.DataFrame(json.loads((OUT/'calibration_gain.json').read_text())['rows']);b['res']=1.
d=pd.concat([a,b]).drop_duplicates();ids=['model','module','part','mode','task','branch'];d['candidate']=d[['side','base','res','gain']].astype(str).agg('|'.join,axis=1)
s=d.groupby(ids+['candidate','split']).loss.mean().unstack('split').reset_index();rows=[]
for key,g in s.groupby(ids):
 best=g.sort_values(['fit','candidate']).iloc[0];base=g[g.candidate=='A|1.0|1.0|1.0'].iloc[0]
 rows.append(dict(zip(ids,key),candidate=best.candidate,fit_gain=1-best['fit']/base['fit'],held_gain=1-best.held_phase/base.held_phase))
r=pd.DataFrame(rows);r.to_csv(OUT/'granular_calibration.csv',index=False)
(OUT/'granular_sources.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUT/'calibration.json',OUT/'calibration_gain.json']},indent=2)+'\n')
print(r.groupby(['model','mode']).agg(groups=('candidate','size'),changed=('candidate',lambda x:(x!='A|1.0|1.0|1.0').sum()),mean_fit_gain=('fit_gain','mean'),mean_held_gain=('held_gain','mean')).to_string())

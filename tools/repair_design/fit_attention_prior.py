"""Freeze descriptive priors using development cases only, not test quality."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/repair_design_20260911'
for model,runname in [('edge','findings_edge_sampling'),('nano','findings_nano_sampling_complete')]:
 run=ROOT/'runs5090'/runname;meta=json.loads((run/'result.json').read_text());dev={c['id'] for c in meta['cases'] if c['split']=='development'};values=[];source={}
 for r in meta['rows']:
  if r['case'] not in dev:continue
  f=run/f'bf16_{r["case"]}_seed{r["seed"]}_r{r["repeat"]}_heads.npz';source[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
  with np.load(f) as z:
   # Each phase gets its own head/layer matrix. Average only where roles exist.
   m=z['mass'].astype(np.float64);counts=z['counts']
   d=m/np.where(counts[:,None,None,None,:]>0,counts[:,None,None,None,:],np.nan)[None]
   # Collapse CFG branches as a deliberate prototype simplification; task/CFG
   # conditioning remains a follow-up, not an inferred capability.
   phases=[]
   for lo,hi in [(0,7),(7,18),(18,28),(28,35)]:
    finite=np.isfinite(d[lo:hi]);den=finite.sum((0,1));num=np.nansum(d[lo:hi],axis=(0,1));phases.append(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0))
   values.append(np.stack(phases))
 v=np.stack(values);den=np.isfinite(v).sum(0);num=np.nansum(v,axis=0);atlas=np.divide(num,den,out=np.zeros_like(num),where=den>0)
 np.savez_compressed(OUT/f'{model}_attention_prior.npz',density=atlas)
 (OUT/f'{model}_attention_prior.json').write_text(json.dumps(dict(model=model,cases=sorted(dev),seeds=[6101,6102],shape=list(atlas.shape),source_sha256=source,scope='Descriptive per-token density, averaged over available development trajectories and CFG branches; not measured repair benefit. Query-role and key-role are directional. No validation inputs used to fit.'),indent=2)+'\n')
 print(model,atlas.shape)

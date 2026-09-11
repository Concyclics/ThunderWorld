"""Reproducible figures and flat tables; keep local probes and trajectories separate."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/activation_repair_20260911'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=160,bbox_inches='tight')
 plt.close(fig)
b=json.loads((OUT/'benchmark.json').read_text());assert b['complete'];rows=[]
for r in b['rows']:
 for layout in ['gather','direct']:rows.append(dict(model=r['model'],shape=r['shape'],tier=r['tier'],layout=layout,speedup=r['speedups'][layout],ms=r['median_ms'][layout],original_ms=r['median_ms']['original']))
d=pd.DataFrame(rows);d.to_csv(OUT/'speedups.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(14,7),layout='constrained')
for ax,model in zip(axs,['edge','nano']):
 part=d[d.model==model];shapes=list(part['shape'].drop_duplicates());v=np.array([[part[(part['shape']==s)&(part.tier==t)&(part.layout=='direct')].speedup.iloc[0] for t in ['a50','act2','a50w','dual']] for s in shapes]);im=ax.imshow(v,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=min(.7,d.speedup.min()),vcenter=1.,vmax=max(1.3,d.speedup.max())),aspect='auto')
 ax.set(yticks=range(len(shapes)),yticklabels=shapes,xticks=range(4),xticklabels=['A50','A100','A50+W','A100+W'],title=model.title()+' · direct paired packing')
 for i in range(len(shapes)):
  for j in range(4):ax.text(j,i,f'{v[i,j]:.2f}×',ha='center',va='center')
fig.suptitle('Activation-paired GEMM speedup over matched tuned original\nCurrent-input selector + online packing + GEMM; M:N:K; five randomized graph rounds');save(fig,'activation_speedups')
c=json.loads((OUT/'calibration.json').read_text());assert c['complete'];pd.DataFrame(c['rows']).to_csv(OUT/'calibration_probes.csv',index=False);rows=[]
for model,policies in c['policies'].items():
 for mode,parts in policies.items():
  for part,v in parts.items():rows.append(dict(model=model,mode=mode,part=part,side=v['selected']['side'],base=v['selected']['base'],res=v['selected']['res'],fit_gain=v['fit_gain'],held_gain=v['held_gain']))
pd.DataFrame(rows).to_csv(OUT/'calibration_choices.csv',index=False)
# Show the response surfaces, not empty bars when the selected gain is zero.
gain=json.loads((OUT/'calibration_gain.json').read_text());assert gain['complete']
pd.DataFrame(gain['rows']).to_csv(OUT/'gain_probes.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
for ax,study,keys,title in [(axs[0],c,[(b,r) for b in [.75,.875,1.] for r in [.75,1.,1.25]],'Range multipliers: base / residual'),(axs[1],gain,[(1.,g) for g in [.875,.9375,1.,1.0625,1.125]],'Post-quantization residual gain')]:
 vals=[];labels=[]
 for model in ['edge','nano']:
  for mode,side in [('act2','A'),('dual','A'),('dual','W')]:
   rr=[r for r in study['rows'] if (r['model'],r['mode'],r['side'],r['split'])==(model,mode,side,'held_phase')];field='res' if study is c else 'gain'
   ref=np.mean([r['loss'] for r in rr if r['base']==1. and r[field]==1.]);vals.append([np.mean([r['loss'] for r in rr if (r['base'],r[field])==k])/ref for k in keys]);labels.append(model+' '+mode+' '+side)
 v=np.array(vals);ax.imshow(v,cmap='YlOrRd',vmin=1,vmax=max(2.,v.max()),aspect='auto');ax.set(yticks=range(len(labels)),yticklabels=labels,xticks=range(len(keys)),xticklabels=[f'{a:g}\n{b:g}' if study is c else f'{b:g}' for a,b in keys],title=title)
 for i in range(len(labels)):
  for j in range(len(keys)):ax.text(j,i,f'{v[i,j]:.2f}',ha='center',va='center',fontsize=8)
fig.suptitle('Held-phase output MSE / original MSE · lower is better · each cell pools sampled projections');save(fig,'residual_calibration')
import sys
if '--local-only' in sys.argv:print('Local figures complete');raise SystemExit
records=[];checks=[]
for model in ['edge','nano']:
 path=ROOT/f'runs5090/activation_{model}_20260911';r=json.loads((path/'result.json').read_text());assert r['complete']
 for row in r['rows']:
  stem=f"{row['mode']}_{row['case']}_seed{row['seed']}_r{row['repeat']}_outputs.npz"
  with np.load(path/stem) as z:assert {k:hashlib.sha256(z[k].tobytes()).hexdigest() for k in z.files}==row['sha256']
  records.append(dict(model=model,mode=row['mode'],case=row['case'],seed=row['seed'],wall_seconds=row['wall_seconds'],video_ssim=row.get('vs_bf16',{}).get('video',{}).get('ssim_mean',1.),action_l2=row.get('vs_bf16',{}).get('action',{}).get('relative_l2'),sound_latent_l2=row.get('vs_bf16',{}).get('sound_latent',{}).get('relative_l2')))
 # Check each pair against its own same-tier parent, including all signals.
 for mode,parent in [('ffn_act_pair','ffn_act2'),('ffn_dual_pair','ffn_dual')]:
  for pair in [x for x in r['rows'] if x['mode']==mode]:
   ref=next(x for x in r['rows'] if (x['mode'],x['case'],x['seed'])==(parent,pair['case'],pair['seed']))
   assert pair['sha256']==ref['sha256'],('Trajectory changed',model,mode,pair['case'])
 checks.append(dict(pair_outputs_bitwise=True,path=str((path/'result.json').relative_to(ROOT)),sha256=hashlib.sha256((path/'result.json').read_bytes()).hexdigest(),rows=len(r['rows']),receipts=r['activation_repair']['receipts']))
pd.DataFrame(records).to_csv(OUT/'trajectory_screen.csv',index=False);(OUT/'trajectory_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(records).to_string(index=False))
# Separate finer-grained calibration screen, with repeated anchors audited.
gpath=ROOT/'runs5090/activation_edge_granular_20260911';gr=json.loads((gpath/'result.json').read_text());assert gr['complete']
main=json.loads((ROOT/'runs5090/activation_edge_20260911/result.json').read_text());grows=[];anchors=[]
for row in gr['rows']:
 stem=f"{row['mode']}_{row['case']}_seed{row['seed']}_r{row['repeat']}_outputs.npz"
 with np.load(gpath/stem) as z:assert {k:hashlib.sha256(z[k].tobytes()).hexdigest() for k in z.files}==row['sha256']
 if row['mode'] in ['bf16','ffn_act2']:
  other=next(x for x in main['rows'] if (x['mode'],x['case'],x['seed'])==(row['mode'],row['case'],row['seed']));assert other['sha256']==row['sha256'];anchors.append(dict(mode=row['mode'],case=row['case'],bitwise=True))
 if row['mode']!='bf16':grows.append(dict(mode=row['mode'],case=row['case'],ssim=row['vs_bf16']['video']['ssim_mean'],video_mse=row['vs_bf16']['video']['mse'],action_l2=row['vs_bf16'].get('action',{}).get('relative_l2')))
pd.DataFrame(grows).to_csv(OUT/'granular_trajectory.csv',index=False)
(OUT/'granular_trajectory_audit.json').write_text(json.dumps(dict(source=str((gpath/'result.json').relative_to(ROOT)),sha256=hashlib.sha256((gpath/'result.json').read_bytes()).hexdigest(),duplicate_anchors=anchors,unique_new_rows=2),indent=2)+'\n')
print(pd.DataFrame(grows).to_string(index=False))

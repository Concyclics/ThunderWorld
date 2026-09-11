"""Separate complete-trajectory effects from local proxy error attribution."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/repair_design_20260911';OUT.mkdir(exist_ok=True)
paths=[ROOT/'results/findings_20260910/ablation/ablation.csv']
a=pd.read_csv(paths[0]);modes=['ffn_base','ffn_act2','ffn_wgt2','ffn_dual','ffn_four']
rows=[]
for model in ['edge','nano']:
 for mode in modes:
  s=a[(a.model==model)&(a['mode']==mode)]
  v=s[s['case']=='robot_table'];p=s[s['case']=='policy_table'];av=s[s['case']=='av_pottery']
  rows.append(dict(model=model,mode=mode,i2v_ssim=v.ssim_bf16.mean(),i2v_lpips=v.lpips_bf16.mean(),action_l2=p.action_rel_l2_bf16.max(),audio_l2=av.audio_rel_l2_bf16.max() if len(av) else None))
pd.DataFrame(rows).to_csv(OUT/'term_quality.csv',index=False)
local=[]
for model in ['edge','nano']:
 path=ROOT/f'runs5090/auto_budget_{model}_calibration/local_errors.json';paths.append(path)
 for r in json.loads(path.read_text()):
  o=r['options'];base=o['base']['loss'];A=o['act2']['loss'];W=o['wgt2']['loss'];D=o['dual']['loss']
  local.append(dict(model=model,part=r['name'].split('.')[1],layer=int(r['name'].split('.')[0]),task=r['task'],step=r['step'],branch=r['branch'],base=base,A=A,W=W,dual=D,A_gain=(base-A)/base,W_gain=(base-W)/base,W_after_A=(A-D)/A,A_after_W=(W-D)/W,A_better=A<W))
l=pd.DataFrame(local);l.to_csv(OUT/'local_attribution.csv',index=False)
s=l.groupby(['model','part']).agg(units=('A_gain','size'),A_median_gain=('A_gain','median'),W_median_gain=('W_gain','median'),A_better_fraction=('A_better','mean'),W_after_A=('W_after_A','median'),A_after_W=('A_after_W','median'));s.to_csv(OUT/'local_attribution_summary.csv');print(s.to_string())
# Use the same FFN anchor to isolate attention degradation.
att=[]
for model in ['edge','nano']:
 for case in ['robot_table','policy_table','av_pottery']:
  for mode in ['attn_base','attn_act2','attn_dual50','attn_dual100','attn_four100']:
   sub=a[(a.model==model)&(a['case']==case)&(a['mode']==mode)]
   if len(sub):att.append(dict(model=model,case=case,mode=mode,n=len(sub),video_ssim_to_ffn_anchor=sub.ssim_parent.mean(),action_l2_to_ffn_anchor=sub.action_rel_l2_parent.max(),audio_l2_to_ffn_anchor=sub.audio_rel_l2_parent.max(),bf16_ssim=sub.ssim_bf16.mean()))
pd.DataFrame(att).to_csv(OUT/'attention_isolated.csv',index=False)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,model in zip(axs,['edge','nano']):
 t=s.loc[model];x=np.arange(len(t))
 ax.bar(x-.18,t.A_median_gain*100,.34,label='Activation A1W0',color='#007f86');ax.bar(x+.18,t.W_median_gain*100,.34,label='Weight A0W1',color='#be7c22')
 ax.set(xticks=x,xticklabels=t.index,title=model.title(),ylabel='Median local squared-error reduction (%)',ylim=(0,105));ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'local_AW_benefit.{ext}',dpi=160,bbox_inches='tight')
(OUT/'term_sources.json').write_text(json.dumps({'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'scope':'Full-trajectory seeds 6103/6104; local 64-row calibration proxies from seed 7101. Neither proxy rows nor attention heads are independent quality samples.'},indent=2)+'\n')

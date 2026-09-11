"""Audited, source-backed figures for the repair-path study."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/repair_design_20260911'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[ROOT/f'runs5090/{s}/result.json' for s in ['repair_design_edge_20260911','repair_design_edge_attention_20260911','repair_design_nano_20260911']]
unique={};duplicates=[];sources={};signals={}
for path in paths:
 r=json.loads(path.read_text());sources[str(path.relative_to(ROOT))]={'sha256':digest(path),'campaign_complete':r['complete'],'completed_rows':len(r['rows']),'error':r.get('error')}
 if 'nano' in path.parent.name:assert r['complete']
 for row in r['rows']:
  key=r['model'],row['mode'],row['case'],row['seed'],row['repeat']
  stem=f"{row['mode']}_{row['case']}_seed{row['seed']}_r{row['repeat']}_outputs.npz"
  with np.load(path.parent/stem) as z:
   assert {k:hashlib.sha256(z[k].tobytes()).hexdigest() for k in z.files}==row['sha256']
  if key in unique:
   assert unique[key]['sha256']==row['sha256'],('Duplicate anchors differ',key)
   duplicates.append(dict(key=key,hashes_identical=True));continue
  unique[key]=row;signals[key]=path.parent/stem
records=[]
for key,r in unique.items():
 model,mode,case,seed,repeat=key
 row=dict(model=model,mode=mode,case=case,seed=seed,repeat=repeat,source=str(signals[key].relative_to(ROOT)),wall_seconds=r['wall_seconds'],transformer_seconds=r['transformer_seconds'])
 for anchor in ['bf16','ffn_act2']:
  data=r.get('vs_'+anchor,{})
  for signal,metric in [('video','ssim_mean'),('action','relative_l2'),('sound_latent','relative_l2')]:
   row[f'{signal}_{metric}_vs_{anchor}']=data.get(signal,{}).get(metric,(1. if signal=='video' else 0.) if mode==anchor and signal in r['sha256'] else None)
 records.append(row)
df=pd.DataFrame(records);df.to_csv(OUT/'trajectory_screen.csv',index=False)
mergechecks=[]
for key in unique:
 model,mode,case,seed,repeat=key
 if mode!='ffn_split_aw_merged':continue
 other=(model,'ffn_split_aw',case,seed,repeat)
 with np.load(signals[key]) as a,np.load(signals[other]) as b:
  mergechecks.append(dict(model=model,case=case,signals={k:dict(bitwise=np.array_equal(a[k],b[k]),relative_l2=float(np.linalg.norm(a[k].astype(float)-b[k])/max(np.linalg.norm(b[k].astype(float)),1e-12))) for k in a.files}))
(OUT/'trajectory_audit.json').write_text(json.dumps(dict(sources=sources,unique_rows=len(unique),duplicate_anchor_checks=duplicates,merged_output_checks=mergechecks,scope='One fresh seed 8501, 35 steps, existing evaluation scenes. Edge initial campaign stopped after 12 completed rows at a causal-AR guard; completion campaign reused four bitwise-identical anchors. Timings are diagnostic eager requests. Equality checks do not establish merged-kernel execution; consult execution receipts.'),indent=2)+'\n')
# All shapes and tiers, including slowdowns.
bpath=OUT/'merged_weight_benchmark.json';b=json.loads(bpath.read_text());assert b['complete'];br=[]
for r in b['rows']:
 for layout in ['plane','pair']:br.append(dict(model=r['model'],shape=r['shape'],tier=r['tier'],layout=layout,speedup=r['speedups'][layout],**{f'{k}_ms':v for k,v in r['median_ms'].items()}))
bdf=pd.DataFrame(br);bdf.to_csv(OUT/'merge_speedups.csv',index=False)
fig,ax=plt.subplots(figsize=(10,6.5),layout='constrained');labels=[];values=[]
for model,shape in bdf[['model','shape']].drop_duplicates().itertuples(index=False):
 labels.append(model.title()+'  '+shape);values.append([float(bdf[(bdf.model==model)&(bdf['shape']==shape)&(bdf.tier==tier)&(bdf.layout==layout)].speedup.iloc[0]) for tier in ['wgt2','a50w','dual'] for layout in ['plane','pair']])
v=np.array(values);im=ax.imshow(v,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=.75,vcenter=1.,vmax=1.4),aspect='auto');ax.set(yticks=range(len(labels)),yticklabels=labels,xticks=range(6),xticklabels=['W / plane','W / pair','A50+W / plane','A50+W / pair','A+W / plane','A+W / pair'],title='Merged layout speedup over the existing fused kernel\nOnline packing + GEMM; M:N:K; >1 is faster')
for i in range(v.shape[0]):
 for j in range(v.shape[1]):ax.text(j,i,f'{v[i,j]:.2f}×',ha='center',va='center',fontsize=10)
fig.colorbar(im,ax=ax,shrink=.7,label='Speedup');save(fig,'merge_speedups')
# Comparable FFN variants, BF16 attention fixed.
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained');modes=['ffn_act2','ffn_wgt2','ffn_split_aw'];labels=['A everywhere','W everywhere','A up/gate; W down'];colors=['#007f86','#be7c22','#6654a3']
for ax,model,case,metric,title in zip(axs,['edge','nano','nano'],['robot_table','robot_table','policy_table'],['video_ssim_mean_vs_bf16','video_ssim_mean_vs_bf16','action_relative_l2_vs_bf16'],['Edge I2V · SSIM ↑','Nano I2V · SSIM ↑','Nano action · relative L2 ↓']):
 vals=[df[(df.model==model)&(df['case']==case)&(df['mode']==m)][metric].iloc[0] for m in modes]
 ax.bar(range(3),vals,color=colors);ax.set(xticks=range(3),xticklabels=['A','W','Split A/W'],title=title,ylim=(0,1 if 'ssim' in metric else max(vals)*1.2));ax.bar_label(ax.containers[0],fmt='%.3f');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Same two-product FFN budget · seed 8501 · 35 steps · output consistency, not task success');save(fig,'ffn_screen')
# Attention relative to its own fixed A-only FFN anchor, not BF16 mix.
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained');amodes=['attn_act2','attn_energy_a50','attn_struct_a50']
for ax,case,title in zip(axs,['robot_table','policy_table','av_pottery'],['I2V','Policy video','Audio-video']):
 for j,model in enumerate(['edge','nano']):
  vals=[df[(df.model==model)&(df['case']==case)&(df['mode']==m)].video_ssim_mean_vs_ffn_act2 for m in amodes]
  if any(len(v)==0 for v in vals):continue
  vv=[v.iloc[0] for v in vals];bars=ax.bar(np.arange(3)+(j-.5)*.36,vv,.34,label=model.title(),color=['#007f86','#be7c22'][j]);ax.bar_label(bars,fmt='%.3f',fontsize=8)
 ax.set(xticks=range(3),xticklabels=['Full A','Energy A50','Structure A50'],title=title,ylim=(0,1.08),ylabel='SSIM to fixed FFN anchor ↑');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False)
fig.suptitle('Feature-only half-tile repair is a candidate, not a demonstrated quality guarantee');save(fig,'attention_screen')
# Exact weight fusion semantics; code-native schematic.
fig,ax=plt.subplots(figsize=(12,4.2));ax.axis('off')
boxes=[(.02,.53,.27,.33,'OFFLINE\nKeep W0 and W1 + scales\nPack along expanded reduction'),(.37,.53,.27,.33,'ONLINE · native FP4 MMA\n[W0 W1] × [A0; A0]\nOne FP32 accumulator'),(.72,.53,.26,.33,'OUTPUT\nOne epilogue + BF16 store\nNo external sum kernel')]
from matplotlib.patches import FancyBboxPatch
for x,y,w,h,t in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',fc='#e5f2f1',ec='#007f86'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=11)
for x1,x2 in [(.30,.36),(.65,.71)]:ax.annotate('',(x2,.69),(x1,.69),arrowprops={'arrowstyle':'->','lw':2})
ax.text(.5,.29,'Exact residual representation still needs 2K products. A0 loads can alias the same buffer.',ha='center',fontsize=12)
ax.text(.5,.13,'Requantizing W0 + W1 into one FP4 plane gives K work, but introduces new quantization error.',ha='center',fontsize=11,color='#98512e');save(fig,'merge_architecture')
print(df[['model','mode','case','video_ssim_mean_vs_bf16','action_relative_l2_vs_bf16','video_ssim_mean_vs_ffn_act2']].to_string(index=False))

audio_path=ROOT/'runs5090/repair_design_nano_20260911/audio_decode.json'
audio=json.loads(audio_path.read_text());assert audio['complete']
pd.DataFrame([dict(mode=r['mode'],case=r['case'],seed=r['seed'],waveform_l2_vs_bf16=r.get('vs_bf16',{}).get('relative_l2'),waveform_l2_vs_ffn_act2=r.get('vs_ffn_act2',{}).get('relative_l2'),log_spectral_l1_vs_bf16=r.get('vs_bf16',{}).get('log_spectral_l1')) for r in audio['rows']]).to_csv(OUT/'audio_screen.csv',index=False)
(OUT/'audio_source.json').write_text(json.dumps({'path':str(audio_path.relative_to(ROOT)),'sha256':digest(audio_path),'scope':audio['scope']},indent=2)+'\n')

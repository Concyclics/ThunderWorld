"""Reproducible publication figures from archived ThunderWorld measurements.
Run with data_local/e2e_venv/bin/python. No model execution or image editing.
"""
from pathlib import Path
import csv, json, hashlib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import PercentFormatter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'assets/figures'; OUT.mkdir(exist_ok=True)
DATA=HERE/'data/visuals'; DATA.mkdir(exist_ok=True)
INK='#173747'; TEAL='#007F86'; BLUE='#3C65A5'; GOLD='#BE7C22'; GREY='#8B99A3'; LIGHT='#EAF4F4'; PALE='#FCF2E1'
COL={'edge':TEAL,'nano':BLUE}; TIERS=['ffn_act2','ffn_auto50','ffn_auto75']; TCOL=[TEAL,BLUE,GOLD]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.edgecolor':'#ACBAC1',
 'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
manifest={'as_of':'2026-09-11','sources':[],'figures':[],'notes':'Charts use archived observations; schematic geometry is not measured timing. No inference rerun.'}
seen=set()
def source(path):
 path=Path(path)
 if path not in seen:
  seen.add(path);manifest['sources'].append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 return path

def save(fig,name,kind,description):
 for ext in ['svg','pdf','png']:
  fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.16)
 plt.close(fig)
 manifest['figures'].append({'id':name,'kind':kind,'description':description,'files':[f'assets/figures/{name}.{x}' for x in ['svg','pdf','png']]})

def clean(ax,axis='y'):
 ax.grid(axis=axis,color='#E3E9EC',lw=.7); ax.set_axisbelow(True)

def canvas():
 fig,ax=plt.subplots(figsize=(12.6,5.3));ax.set(xlim=(0,12.6),ylim=(0,5.3));ax.axis('off');return fig,ax

def box(ax,x,y,w,h,title,body='',color=TEAL,planned=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=.09',fc=PALE if planned else LIGHT,ec=color,lw=1.5,linestyle='--' if planned else '-'))
 ax.text(x+w/2,y+h*.68 if body else y+h/2,title,ha='center',va='center',fontsize=12,fontweight='bold',color=color)
 if body:ax.text(x+w/2,y+h*.30,body,ha='center',va='center',fontsize=10.5,linespacing=1.4)

def arrow(ax,a,b,label=None,color=INK,rad=0):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.4,color=color,connectionstyle=f'arc3,rad={rad}'))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=10,color=color)

# 00: research logic.
fig,ax=canvas()
for x,title,body in [(0.1,'Deployment need','Lower latency and memory\nunder task-quality constraints'),(4.4,'Numerical opportunity','Complete FP4 base\n+ selected FP4 corrections'),(8.7,'Executable system','Regular Tensor Core work\n+ low scheduling overhead')]:box(ax,x,3.3,3.7,1.45,title,body)
arrow(ax,(3.85,4),(4.35,4));arrow(ax,(8.15,4),(8.65,4))
box(ax,.1,.65,3.7,1.45,'C1 · FP4 representation','Repair-aware first pass\nRESEARCH / current pass: RTN',GOLD,True)
box(ax,4.4,.65,3.7,1.45,'C2 · Repair allocation','Structure → measured repair benefit\nPARTIAL / uniform fallback',GOLD,True)
box(ax,8.7,.65,3.7,1.45,'C3 · Hardware execution','FP4 FFN · shape tuning · Graph\nIMPLEMENTED / two models')
for x in [1.95,6.25,10.55]:arrow(ax,(x,3.25),(x,2.15))
ax.text(6.3,.12,'Validation target: complete-trajectory quality × latency × memory; energy remains unmeasured.',ha='center',fontsize=11)
save(fig,'00_research_logic','schematic','Motivation, candidate algorithm contributions, and implemented system status.')

# 02: product expansion with selected supports.
fig,ax=canvas()
box(ax,.1,3.5,3.4,1.35,'Quantize the current input','A ≈ A₀ + A₁\nW ≈ W₀ + W₁ (offline)')
box(ax,4.4,3.5,3.5,1.35,'Choose residual support','Aᵣ = Sₐ ⊙ A₁\nWᵣ = Sw ⊙ W₁')
box(ax,8.8,3.5,3.6,1.35,'Retain high precision','FP32 accumulation\nBF16 output / nonlinear ops')
arrow(ax,(3.55,4.15),(4.35,4.15));arrow(ax,(7.95,4.15),(8.75,4.15))
for x,t,b,c,plan in [(0.1,'A₀W₀','Complete base\nalways executed',TEAL,False),(3.3,'AᵣW₀','Activation correction\nselected blocks',TEAL,False),(6.5,'A₀Wᵣ','Weight correction\nselected blocks',TEAL,False),(9.7,'AᵣWᵣ','Omitted cross term\nfour-term reference only',GOLD,True)]:
 box(ax,x,1.1,2.7,1.5,t,b,c,plan)
for x in [2.98,6.18]:ax.text(x,1.85,'+',ha='center',fontsize=25)
ax.text(6.2,.47,'Ŷ = A₀W₀ + AᵣW₀ + A₀Wᵣ     |     nominal cost ≈ 1 + ρA + ρW',ha='center',fontsize=15,fontweight='bold')
ax.text(6.2,.02,'Logical products may share a native GEMM. FP4 operands are not materialized as complete BF16 matrices.',ha='center',fontsize=10.5)
save(fig,'02_residual_products','schematic','Logical residual terms and the distinction between executed and omitted terms.')

# 03: allocator flow and exact phase lengths.
fig,ax=canvas()
for x,t,b in [(.1,'Calibration probes','Local output error\n+ measured shape latency'),(4.4,'Budget allocation','Positive benefit / extra time\n+ half-A floor'),(8.7,'Trajectory quality gate','Video + action + audio\nvs. equal-budget uniform')]:box(ax,x,3.45,3.7,1.5,t,b)
arrow(ax,(3.85,4.2),(4.35,4.2));arrow(ax,(8.15,4.2),(8.65,4.2))
box(ax,8.7,1.4,3.7,1.1,'Freeze and export','3 retained / 7 uniform fallbacks')
arrow(ax,(10.55,3.4),(10.55,2.55))
box(ax,.1,1.4,7.95,1.1,'Online: task × layer × phase × CFG','Look up tier; select K128 blocks from current activation energy')
arrow(ax,(8.65,1.95),(8.1,1.95))
start=.1
for count,label,col in [(7,'Phase 1 · 7 steps',TEAL),(11,'Phase 2 · 11 steps',BLUE),(10,'Phase 3 · 10 steps',TEAL),(7,'Phase 4 · 7 steps',BLUE)]:
 w=12.3*count/35;ax.add_patch(Rectangle((start,.25),w,.65,fc=col,ec='white'));ax.text(start+w/2,.575,label,color='white',ha='center',va='center',fontsize=10.5);start+=w
ax.text(6.2,1.02,'25 / 50 / 75% budgets = 0.5 / 1 / 1.5 extra FP4 products on average',ha='center',fontsize=10.5)
source(ROOT/'docs/AUTO_BUDGET_5090_20260910.md')
save(fig,'03_budget_flow','schematic','Offline allocation and full-trajectory gates; online phase and current-input selection.')

# 04: execution dependencies (no invented timings).
fig,ax=canvas()
ax.text(.1,4.95,'EAGER: recurring host submission',fontsize=12,fontweight='bold')
for x,t in [(.1,'Quantize / pack'),(3.3,'Native FP4 GEMM'),(6.5,'Activation'),(9.7,'amax / next GEMM')]:
 box(ax,x,3.8,2.7,.8,t)
for x in [2.85,6.05,9.25]:arrow(ax,(x,4.2),(x+.4,4.2))
ax.text(.1,3.22,'GRAPH + FUSION: capture once, replay with fresh inputs',fontsize=12,fontweight='bold')
box(ax,.1,1.55,2.7,1.25,'Current-step input','Copy input + timestep\nselect phase / CFG graph')
box(ax,3.3,1.55,5.9,1.25,'Captured Transformer execution','FP4 FFN + BF16 attention\nExact pointwise / amax fusion where supported')
box(ax,9.7,1.55,2.7,1.25,'Current-step output','Independent output\nupdated budget counters')
arrow(ax,(2.85,2.17),(3.25,2.17));arrow(ax,(9.25,2.17),(9.65,2.17))
ax.text(6.3,.75,'Same arithmetic and repair budget • no previous-step output reuse',ha='center',fontsize=13,fontweight='bold',color=TEAL)
ax.text(6.3,.2,'Schematic, not a measured timeline. Graph removes launch overhead; fusion preserves BF16 rounding before amax.',ha='center',fontsize=10.5)
source(ROOT/'docs/GRAPH_FUSION_5090_20260911.md')
save(fig,'04_execution_flow','schematic','Eager dependencies versus captured execution; geometry is not duration.')

# Extract compact, auditable plot inputs from all archived observation trajectories.
step_rows=[];depth_rows=[];head_rows=[];direction_rows=[];observations={}
for model,dirname in [('edge','findings_edge_sampling'),('nano','findings_nano_sampling_complete')]:
 run=ROOT/'runs5090'/dirname;meta=json.loads(source(run/'result.json').read_text());ars=[];fs=[];hfs=[]
 summary=json.loads(source(ROOT/f'results/findings_20260910/{model}/sampling_summary.json').read_text())
 for row in meta['rows']:
  path=source(run/f'bf16_{row["case"]}_seed{row["seed"]}_r{row["repeat"]}_heads.npz')
  with np.load(path) as z:
   m=z['mass'][:,0];f=z['features'][:,0];n=z['counts'][0].astype(float).copy();n[0]=0
   ar=np.nansum(m[...,0]*n,axis=-1)/n.sum();q=f[...,2,:]
   ars.append(ar.mean(axis=2));fs.append(q)
   for s,values in enumerate(ar.mean(axis=2)):
    for l,value in enumerate(values):depth_rows.append(dict(model=model,case=row['case'],seed=row['seed'],step=s+1,layer=l,value=float(value)))
   if row['case']=='robot_observe':hfs.append(q)
   for target,idx in [('action',4),('sound',3)]:
    counts=z['counts'][0]
    if counts[idx]>0:
     fw=m[...,idx,2].mean(axis=(1,2));rv=m[...,2,idx].mean(axis=(1,2))
     ratio=(fw/counts[2])/np.maximum(rv/counts[idx],1e-20)
     for s,v in enumerate(ratio):direction_rows.append(dict(model=model,case=row['case'],seed=row['seed'],role=target,step=s+1,density_ratio=float(v)))
  print('read',model,row['case'],row['seed'],flush=True)
 arr=np.stack(ars);feat=np.stack(fs)
 overall=summary['overall'];quart=max(1,arr.shape[2]//4)
 assert np.isclose(arr[:,:,:quart].mean(),overall['ar_shallow'],atol=2e-7)
 assert np.isclose(arr[:,:,-quart:].mean(),overall['ar_deep'],atol=2e-7)
 assert np.isclose(feat[:,1:,...,6].mean(),overall['output_rel_l2'],atol=2e-7)
 observations[model]={'depth':arr.mean(axis=0),'features':feat.mean(axis=(0,2,3)),'head':np.stack(hfs).mean(axis=0),'overall':overall,'n':len(ars)}
 for s,vals in enumerate(observations[model]['features']):
  if s:
   step_rows.append(dict(model=model,step=s+1,dominant_stability=float(vals[10]),output_rel_l2=float(vals[6]),support_jaccard=float(vals[7])))
 for s,v in enumerate(observations[model]['head']):
  for h,val in enumerate(v[v.shape[0]//2,:,6]):
   if s:head_rows.append(dict(model=model,step=s+1,layer=v.shape[0]//2,head=h,output_rel_l2=float(val)))
pd.DataFrame(depth_rows).to_csv(DATA/'depth_by_trajectory.csv',index=False)
pd.DataFrame(step_rows).to_csv(DATA/'step_means.csv',index=False)
pd.DataFrame(direction_rows).to_csv(DATA/'direction_steps.csv',index=False)
pd.DataFrame(head_rows).to_csv(DATA/'head_steps.csv',index=False)

# 05: whole depth/step field + endpoint summaries.
fig,axs=plt.subplots(1,3,figsize=(12.6,4.8),gridspec_kw={'width_ratios':[1,1,.85]},layout='constrained')
for ax,model in zip(axs[:2],COL):
 v=observations[model]['depth'].T
 im=ax.imshow(v,origin='lower',aspect='auto',extent=[.5,35.5,-.5,v.shape[0]-.5],vmin=0,vmax=1,cmap='viridis')
 ax.set(title=f'{model.title()} · {observations[model]["n"]} trajectories',xlabel='Denoising step',ylabel='Layer (zero-based)',xticks=[1,7,18,28,35])
fig.colorbar(im,ax=list(axs[:2]),label='DM ← AR attention mass',shrink=.85)
ax=axs[2]
for model in COL:
 o=observations[model]['overall'];ys=[o['ar_shallow'],o['ar_deep']]
 ax.plot([0,1],ys,'o-',color=COL[model],lw=2,label=model.title())
 for x,y in zip([0,1],ys):ax.annotate(f'{y:.4f}',(x,y),xytext=(0,9 if model=='nano' else -17),textcoords='offset points',ha='center',color=COL[model],fontsize=10)
ax.set(xticks=[0,1],xticklabels=['Shallow\nquarter','Deep\nquarter'],ylim=(0,.48),xlim=(-.25,1.25),ylabel='Mean attention mass',title='Depth summary');ax.legend(frameon=False);clean(ax)
save(fig,'05_depth_mass','data','Equal trajectory means, query-count-weighted DM mass; all layers, heads and 35 steps, conditional branch.')

# 06: individual trajectory ratios and actual step trend.
fig,axs=plt.subplots(1,3,figsize=(12.6,4.8),gridspec_kw={'width_ratios':[.9,.9,1.25]},layout='constrained')
rel=[]
for model,target in [('edge','action'),('nano','action'),('nano','sound')]:
 df=pd.read_csv(source(ROOT/f'results/findings_20260910/{model}/trajectory_summary.csv'))
 for _,r in df.dropna(subset=[target+'_density_ratio']).iterrows():
  rel.append(dict(model=model,role=target,case=r['case'],seed=r['seed'],raw=r[target+'_forward_row_mass']/r[target+'_reverse_row_mass'],density=r[target+'_density_ratio']))
rel=pd.DataFrame(rel);rel.to_csv(DATA/'direction_trajectories.csv',index=False)
for ax,key,title in zip(axs[:2],['raw','density'],['Raw row-mass ratio','Per-token density ratio']):
 for i,(model,target) in enumerate([('edge','action'),('nano','action'),('nano','sound')]):
  vals=rel[(rel.model==model)&(rel.role==target)][key].to_numpy();jitter=np.linspace(-.10,.10,len(vals))
  ax.scatter(vals,np.full(len(vals),i)+jitter,s=38,color=COL[model],marker='o' if target=='action' else 's')
 ax.set(yticks=[0,1,2],yticklabels=['Edge action','Nano action','Nano sound'],xlabel='Forward / reverse',title=title,xlim=(0,32) if key=='raw' else (0,1.9));ax.invert_yaxis();ax.axvline(1,ls='--',color=GREY,lw=1);clean(ax,'x')
dirs=pd.DataFrame(direction_rows)
for case,df in dirs[(dirs.model=='nano')&(dirs.role=='sound')].groupby('case'):
 curve=df.groupby('step').density_ratio.mean();axs[2].plot(curve.index,curve,label=case.replace('av_','').title(),lw=2,ls='-' if 'waterfall' in case else '--',color=BLUE if 'waterfall' in case else GOLD)
axs[2].axhline(1,color=GREY,ls=':',label='Symmetric');axs[2].set(xlabel='Denoising step',ylabel='Density ratio',title='Nano sound ← vision / reverse',xticks=[1,7,18,28,35]);axs[2].legend(frameon=False,fontsize=9);clean(axs[2])
save(fig,'06_directionality','data','Dots are whole-trajectory ratios, not confidence intervals; Nano sound curves average layers/heads within a trajectory then two seeds.')

# 07: numerical drift and head-specific structure.
fig,axs=plt.subplots(2,2,figsize=(12.6,6.4),layout='constrained')
for model in COL:
 f=observations[model]['features'];steps=np.arange(2,36)
 axs[0,0].plot(steps,f[1:,6],label=model.title(),color=COL[model],lw=2,ls='-' if model=='edge' else '--')
 axs[0,1].plot(steps,f[1:,7],label=model.title(),color=COL[model],lw=2,ls='-' if model=='edge' else '--')
for ax,title,ylabel in [(axs[0,0],'Output values still change','Adjacent-step relative L2'),(axs[0,1],'Mass-ranked supports are stable','Top-50% tile Jaccard')]:
 ax.set(title=title,xlabel='Denoising step',ylabel=ylabel,xticks=[2,7,18,28,35]);ax.legend(frameon=False,fontsize=9);clean(ax)
axs[0,0].set_ylim(bottom=0);axs[0,1].set_ylim(0,1.02)
vmax=max(observations[m]['head'][1:,observations[m]['head'].shape[1]//2,:,6].max() for m in COL)
for ax,model in zip(axs[1],COL):
 f=observations[model]['head'];layer=f.shape[1]//2;v=f[1:,layer,:,6].T
 im=ax.imshow(v,origin='lower',aspect='auto',extent=[1.5,35.5,-.5,v.shape[0]-.5],vmin=0,vmax=vmax,cmap='magma')
 ax.set(title=f'{model.title()} · layer {layer} · individual heads',xlabel='Denoising step',ylabel='Q-head',xticks=[2,7,18,28,35]);fig.colorbar(im,ax=ax,label='Output relative L2',shrink=.8)
save(fig,'07_temporal','data','Top: all observation trajectories. Bottom: robot_observe, two seeds, middle layer, common heatmap scale. First step has no predecessor and is omitted.')

# Remaining charts from archived CSVs.
exe=pd.read_csv(source(ROOT/'results/execution_20260911/summary.csv'))
abl=pd.read_csv(source(ROOT/'results/findings_20260910/ablation/ablation.csv'))
models=['edge','nano'];modes=['ffn_base','ffn_act2','ffn_wgt2','ffn_dual','ffn_four']
print('ablation modes:',list(abl['mode'].unique()),flush=True)
# Known mode identifiers are checked below, never filled with synthetic rows.
for short in ['ffn_dual','ffn_four']:
 if short not in abl['mode'].unique():
  replacement='ffn_dual_full' if short=='ffn_dual' else 'ffn_four_full'
  if replacement in abl['mode'].unique():modes[modes.index(short)]=replacement
assert set(modes)<=set(abl['mode'])
labels=['Base','+ A','+ W','Dual','Four terms']
qrows=[]
for model in models:
 for mode in modes:
  video=abl[(abl.model==model)&(abl['mode']==mode)&(abl['case']=='robot_table')]
  policy=abl[(abl.model==model)&(abl['mode']==mode)&(abl['case']=='policy_table')]
  audio=abl[(abl.model==model)&(abl['mode']==mode)&(abl['case']=='av_pottery')]
  assert len(video)==2 and len(policy)==2
  qrows.append(dict(model=model,mode=mode,ssim=video.ssim_bf16.mean(),lpips=video.lpips_bf16.mean(),action=policy.action_rel_l2_bf16.max(),audio=audio.audio_rel_l2_bf16.max() if len(audio) else None))
q=pd.DataFrame(qrows);q.to_csv(DATA/'quality_metrics.csv',index=False)
fig,axs=plt.subplots(2,2,figsize=(12.6,6.4),layout='constrained')
for ax,metric,title in zip(axs.flat,['ssim','lpips','action','audio'],['I2V video SSIM ↑','I2V perceptual distance (LPIPS) ↓','Policy action: worst relative L2 ↓','Nano audio: worst waveform L2 ↓']):
 for model in models:
  if metric=='audio' and model=='edge':continue
  vals=q[q.model==model][metric].to_numpy();offset=(-.18 if model=='edge' else .18) if metric!='audio' else 0
  bars=ax.bar(np.arange(5)+offset,vals,width=.34 if metric!='audio' else .55,color=COL[model],label=model.title())
  ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
 ax.set(xticks=np.arange(5),xticklabels=labels,title=title);clean(ax);ax.legend(frameon=False,fontsize=9)
 ax.set_ylim(0,1.25 if metric=='ssim' else max(q[metric].dropna())*1.23)
save(fig,'09_quality_ablation','data','Seeds 6103/6104, BF16 attention. Video mean over two seeds; action/audio maxima. Reference is same-seed BF16.')

# 08: cache quality vs own uncached reference, two models with distinct case sets.
cache_modes=['reuse_mask50','reuse_delta','reuse_delta_middle','reuse_output','reuse_output_middle','reuse_output_stable']
cache_modes=[m for m in abl['mode'].unique() if 'reuse' in m]
print('cache modes',cache_modes,flush=True)
cache=[]
for model in models:
 for mode in cache_modes:
  sub=abl[(abl.model==model)&(abl['mode']==mode)]
  if len(sub):cache.append(dict(model=model,mode=mode,ssim=sub.ssim_parent.min(),speedup=float(np.exp(np.log(sub.speedup_parent).mean()))))
cache=pd.DataFrame(cache);cache.to_csv(DATA/'cache_metrics.csv',index=False)
name_map={'mask_reuse2':'Mask','delta_reuse2':'Delta','delta_reuse2_middle':'Delta / middle','output_reuse2':'Output','output_reuse2_middle':'Output / middle','output_reuse2_stable':'Output / stable GQA','reuse_mask':'Mask','reuse_mask50':'Mask','reuse_delta':'Delta','reuse_delta_mid':'Delta / middle','reuse_delta_middle':'Delta / middle','reuse_output':'Output','reuse_output_mid':'Output / middle','reuse_output_middle':'Output / middle','reuse_output_stable':'Output / stable GQA'}
fig,axs=plt.subplots(1,2,figsize=(12.6,4.7),layout='constrained')
for ax,model in zip(axs,models):
 sub=cache[cache.model==model]
 for j,(_,r) in enumerate(sub.iterrows()):
  label=name_map.get(r['mode'],r['mode'].replace('reuse_','').replace('_',' / '))
  ax.scatter(r.speedup,r.ssim,color=COL[model],marker=['o','s','^','D','P','X'][j%6],s=65,label=label)
 ax.axvline(1,color=GREY,ls='--');ax.set(title=model.title(),xlabel='Speedup vs. own uncached reference →',ylabel='Worst video SSIM ↑',xlim=(.9,1.03),ylim=(.35,1.02));clean(ax);ax.legend(frameon=False,fontsize=9,loc='lower left')
save(fig,'08_cache_tradeoff','data','Exploratory eager requests; worst SSIM and paired geometric-mean speedup against own uncached reference. Different model case sets.')

# 10: all 15 FP4 operating points, with a common BF16 Graph reference.
cases=[('edge','robot_table','Edge I2V'),('edge','policy_table','Edge policy'),('nano','robot_table','Nano I2V'),('nano','policy_table','Nano policy'),('nano','av_pottery','Nano AV')]
fig,axs=plt.subplots(1,2,figsize=(12.6,5.5),layout='constrained')
for ax,metric,title,lim in [(axs[0],'fused_speedup_bf16','Request speedup vs. BF16 ↑',(0,1.52)),(axs[1],'ssim_mean','Video SSIM against BF16 ↑',(0,1.07))]:
 for t,(mode,c) in enumerate(zip(TIERS,TCOL)):
  vals=[float(exe[(exe.model==m)&(exe['case']==case)&(exe['mode']==mode)].iloc[0][metric]) for m,case,_ in cases]
  pos=np.arange(5)+(t-1)*.23
  bars=ax.barh(pos,vals,height=.21,color=c,label=mode.replace('ffn_','').upper())
  ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
 ax.set(yticks=np.arange(5),yticklabels=[x[2] for x in cases],xlim=lim,title=title);ax.invert_yaxis();clean(ax,'x');ax.axvline(1,color=GREY,ls='--',lw=1)
 axs[0].set_xlabel('Paired geometric-mean speedup');axs[1].set_xlabel('Mean over two seeds')
axs[0].legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=9)
exe.to_csv(DATA/'execution_summary.csv',index=False)
save(fig,'10_end_to_end','data','All 15 FP4 configurations from the latest execution cohort. Two seeds, 35 steps, Graph+fusion, common BF16 Graph reference.')

# 11: keep Graph contribution separate from BF16 speedup; resident W1 memory.
fig,axs=plt.subplots(1,2,figsize=(12.6,5.2),layout='constrained')
for t,(mode,c) in enumerate(zip(TIERS,TCOL)):
 vals=[float(exe[(exe.model==m)&(exe['case']==case)&(exe['mode']==mode)].iloc[0].graph_speedup) for m,case,_ in cases]
 bars=axs[0].barh(np.arange(5)+(t-1)*.23,vals,height=.21,color=c,label=mode.replace('ffn_','').upper());axs[0].bar_label(bars,fmt='%.2f×',padding=3,fontsize=9)
axs[0].set(yticks=np.arange(5),yticklabels=[x[2] for x in cases],xlim=(0,2.4),title='Graph vs. the same FP4 eager tier',xlabel='Paired geometric-mean speedup');axs[0].invert_yaxis();axs[0].axvline(1,color=GREY,ls='--');axs[0].legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=9);clean(axs[0],'x')
mem=[]
for j,(mode,c) in enumerate([('bf16',GREY),('ffn_auto75',TEAL)]):
 vals=[float(exe[(exe.model==m)&(exe['case']=='robot_table')&(exe['mode']==mode)].iloc[0].fused_peak_gib) for m in models]
 bars=axs[1].bar(np.arange(2)+(j-.5)*.32,vals,width=.30,color=c,label='BF16' if j==0 else 'AUTO75');axs[1].bar_label(bars,fmt='%.3f',padding=4,fontsize=10)
 mem.append(vals)
for i,model in enumerate(models):axs[1].text(i,max(mem[0])*1.15,f'−{100*(1-mem[1][i]/mem[0][i]):.1f}%',ha='center',fontweight='bold',color=TEAL,fontsize=14)
axs[1].set(xticks=[0,1],xticklabels=['Edge I2V','Nano I2V'],ylim=(0,39),ylabel='Peak allocated memory (GiB)',title='Memory with W1 resident');axs[1].legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.17));clean(axs[1])
save(fig,'11_execution_memory','data','Graph speedups versus own eager tier; I2V peak allocated memory with resident W1, not weight-only compression.')

# 12: ideal arithmetic upper bound and measured pipelining failures.
fig,axs=plt.subplots(1,2,figsize=(12.6,4.8),layout='constrained')
cost=np.array([1,1.5,2,2.5,3]);ideal=4/cost
bars=axs[0].barh(np.arange(5),ideal,color=TEAL);axs[0].bar_label(bars,fmt='%.2f×',padding=3)
axs[0].set(yticks=np.arange(5),yticklabels=['Base','Half A','Full A','Half A + full W','Full A + full W'],xlim=(0,4.7),title='Illustrative GEMM ceiling (r = 4)',xlabel='4 / (1 + ρA + ρW)');axs[0].invert_yaxis();clean(axs[0],'x')
tuned=pd.read_csv(source(ROOT/'results/execution_20260911/tuned_double_buffer.csv'))
print('pipeline columns',tuned.columns.tolist(),flush=True)
# Plot the archived best-over-launch configurations, not the individual candidates.
speedcol=next(c for c in ['speedup','best_speedup'] if c in tuned.columns)
chunkcol=next(c for c in ['chunk','chunk_m'] if c in tuned.columns)
# num_chunks is recorded by the benchmark; no inference from labels is needed.
for multi,col,label in [(True,GOLD,'Multiple chunks'),(False,GREY,'One chunk (no overlap)')]:
 vals=np.sort(tuned.loc[(tuned['num_chunks']>1)==multi,speedcol].to_numpy())
 axs[1].scatter(np.arange(1,len(vals)+1),vals,color=col,label=label,s=35,marker='o' if multi else 's')
axs[1].axhline(1,color=INK,ls='--');axs[1].set(title='Double buffering: best launch per case',xlabel='Configurations ranked within each group',ylabel='Speedup vs. complete GEMM',ylim=(.6,1.08));axs[1].legend(frameon=False,fontsize=9);clean(axs[1])
tuned.to_csv(DATA/'tuned_double_buffer.csv',index=False)
save(fig,'12_compute_limits','data_and_model','Left: idealized r=4 cost model, not measured speedup. Right: measured retuned cases, separating true multichunk execution from one-chunk cases.')

# 13: concrete next implementation and demo flow.
fig,ax=canvas()
for x,t,b in [(.1,'P0 / P1 · Quantization','Full-M calibration context\nrepair-aware first pass'),(4.4,'P2 / P3 · Allocation','Repair benefit vs. structure\nattention QK / PV validation'),(8.7,'P4 / P5 · Deployment','Pruned residual assets\nunseen scenes + energy')]:box(ax,x,3.4,3.7,1.45,t,b,GOLD,True)
arrow(ax,(3.85,4.1),(4.35,4.1));arrow(ax,(8.15,4.1),(8.65,4.1))
ax.text(.1,2.92,'PROPOSED DEMO · only validated configurations enter the default menu',fontsize=12,fontweight='bold')
box(ax,.1,.8,2.7,1.6,'Fixed request','Input / seed / task\nquality tier')
box(ax,3.3,.8,5.9,1.6,'Synchronized comparison','BF16  |  uniform repair  |  frozen auto policy\nVideo + actions + audio; label uniform fallback')
box(ax,9.7,.8,2.7,1.6,'Measured outcome','Latency / peak memory\nactual phase budget')
arrow(ax,(2.85,1.6),(3.25,1.6));arrow(ax,(9.25,1.6),(9.65,1.6))
ax.text(6.3,.2,'Research view: layer × step budgets, normalized relations, and failure cases. Integrated UI remains planned.',ha='center',fontsize=10.5)
save(fig,'13_roadmap_demo','schematic','Implementation priorities and proposed side-by-side demo; not a screenshot of an existing UI.')

manifest['validation']={'depth_and_drift_reconcile_to_archived_summaries':True,'observation_trajectories':sum(o['n'] for o in observations.values()),'execution_rows':len(exe),'quality_rows':len(q),'cache_rows':len(cache),'generated_figures':len(manifest['figures'])}
(HERE/'data/visual_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest['validation'],indent=2))

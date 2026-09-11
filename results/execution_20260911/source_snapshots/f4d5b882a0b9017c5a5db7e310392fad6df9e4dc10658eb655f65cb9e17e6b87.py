"""Aggregate paired trajectories; figures are descriptive, never head-level CIs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROLES = ('ar', 'vision_clean', 'vision_noisy', 'sound', 'action', 'special')


def mean(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(x, axis=axis)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=json.loads((a.input/'result.json').read_text())
    files=sorted(a.input.glob('*_heads.npz'))
    if not report['complete']:
        raise ValueError('Sampling must finish before aggregation')
    rows=[];bycase={};headrows=[];policy_scores=[];direction_steps={};branchrows=[]
    case_map={c['id']:c for c in report['cases']}
    for r in report['rows']:
        case,seed=r['case'],r['seed']
        path=a.input/f'bf16_{case}_seed{seed}_r{r["repeat"]}_heads.npz'
        z=np.load(path);mass=z['mass'];features=z['features'];counts=z['counts']
        # Conditional branch only in figures; all branches retained in raw data.
        m=mass[:,0];f=features[:,0];n=counts[0].astype(float);n[0]=0
        valid=np.flatnonzero(n>0)
        assert np.allclose(np.nansum(m,axis=-1)[...,valid],1,atol=3e-5)
        ar=np.nansum(m[...,0]*n,axis=-1)/n.sum()
        qs=f[...,2,:] if counts[0,2]>0 else f[...,4,:]
        layers=ar.shape[1];quart=max(1,layers//4)
        rec=dict(model=report['model'],case=case,seed=seed,split=case_map[case]['split'],
                 ar_shallow=float(mean(ar[:,:quart])),ar_deep=float(mean(ar[:,-quart:])),
                 depth_pearson=float(np.corrcoef(np.arange(layers),mean(ar,axis=(0,2)))[0,1]),
                 dominant_stability=float(mean(qs[1:,...,10])),
                 output_cos=float(mean(qs[1:,...,5])),output_rel_l2=float(mean(qs[1:,...,6])),
                 output_rel_l2_early=float(mean(qs[1:len(qs)//2,...,6])),
                 output_rel_l2_late=float(mean(qs[len(qs)//2:,...,6])),
                 support_jaccard=float(mean(qs[1:,...,7])),
                 previous_support_mass=float(mean(qs[1:,...,8])),current_support_mass=float(mean(qs[1:,...,9])))
        qr=2 if counts[0,2]>0 else 4
        density=m[...,qr,:]/np.where(counts[0]>0,counts[0],np.inf)
        dominant=np.argmax(density,axis=-1)
        rec['density_dominant_stability']=float(mean(dominant[1:]==dominant[:-1]))
        for b in range(mass.shape[1]):
            bc=counts[b].astype(float).copy();bc[0]=0
            ba=np.nansum(mass[:,b,...,0]*bc,axis=-1)/bc.sum()
            branchrows.append(dict(case=case,seed=seed,branch='conditional' if b==0 else 'unconditional',
                ar_shallow=float(mean(ba[:,:quart])),ar_deep=float(mean(ba[:,-quart:])),
                noisy_vision_output_rel_l2=float(mean(features[1:,b,...,2,6]))))
        relation=mean(m,axis=(0,1,2))
        for target in ('action','sound','vision_clean'):
            i,j=ROLES.index(target),ROLES.index('vision_noisy')
            if counts[0,i] and counts[0,j]:
                forward=relation[i,j];reverse=relation[j,i]
                rec[target+'_forward_row_mass']=float(forward)
                rec[target+'_reverse_row_mass']=float(reverse)
                # Paired directions contain identical numbers of score entries.
                # Density compares mean probability per edge, NOT raw row sums.
                rec[target+'_density_ratio']=float((forward/counts[0,j]) / max(reverse/counts[0,i],1e-20))
                rec[target+'_total_mass_ratio']=float((forward*counts[0,i]) / max(reverse*counts[0,j],1e-20))
                local=(m[...,i,j]/counts[0,j])/np.maximum(m[...,j,i]/counts[0,i],1e-20)
                rec[target+'_head_layer_step_fraction_forward_stronger']=float(mean(local>1))
                rec[target+'_head_layer_step_ratio_median']=float(np.nanmedian(local))
                step=(mean(m[...,i,j],axis=(1,2))/counts[0,j])/np.maximum(mean(m[...,j,i],axis=(1,2))/counts[0,i],1e-20)
                direction_steps.setdefault((case,target),[]).append(step)
        rows.append(rec)
        bycase.setdefault(case,[]).append((ar,qs,relation,z['timestep'],counts[0]))
        for l in range(layers):
            for h in range(ar.shape[2]):
                headrows.append(dict(case=case,seed=seed,layer=l,head=h,
                    ar_mass=float(mean(ar[:,l,h])),dominant_stability=float(mean(qs[1:,l,h,10])),
                    late_output_rel_l2=float(mean(qs[len(qs)//2:,l,h,6])),
                    late_support_jaccard=float(mean(qs[len(qs)//2:,l,h,7]))))
        if case_map[case]['split']=='development':
            # Protect every present query role, including action/audio.
            policy_scores.append(np.nanmax(mean(f[int(np.ceil(len(f)*.2)):int(np.floor(len(f)*.8)),...,6],axis=0),axis=-1))
    def csv_out(name,values):
        keys=list(dict.fromkeys(k for r in values for k in r))
        with (a.output/name).open('w') as out:
            writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader();writer.writerows(values)
    csv_out('trajectory_summary.csv',rows);csv_out('head_summary.csv',headrows);csv_out('cfg_branch_summary.csv',branchrows)
    cfg=report['model_config'];ratio=cfg['num_attention_heads']//cfg['num_key_value_heads']
    scores=mean(np.stack(policy_scores),axis=0).reshape(cfg['num_hidden_layers'],cfg['num_key_value_heads'],ratio).max(-1)
    # At most 25% of groups globally, ranked on development scenes only.
    order=np.argsort(scores.ravel(),kind='stable')[:scores.size//4]
    selected={str(l):[] for l in range(len(scores))}
    for flat in order:
        l,g=divmod(int(flat),scores.shape[1]);selected[str(l)].append(g)
    policy=dict(model=report['model'],kind='middle_every_other_step_output_reuse',active_fraction=[.2,.8],
                calibration_split='development',criterion='middle-step mean relative output drift; maximum over query roles and heads within GQA group',
                group_ids_per_layer={k:sorted(v) for k,v in selected.items()},
                fraction=len(order)/scores.size,source_sha256=hashlib.sha256((a.input/'result.json').read_bytes()).hexdigest())
    (a.output/'stable_group_policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=list(bycase);cols=2;nr=(len(names)+1)//2
    styles={}
    for case in names:
        task=case_map[case]['task'];evaluation=case_map[case]['split']=='evaluation'
        styles[case]=dict(color='#2563eb' if task=='i2v' else '#1e3a8a' if task=='policy' else '#d97706' if task=='t2v' else '#92400e',
                          linestyle='--' if evaluation else '-',linewidth=1.5,
                          marker='.' if task in ('policy','av') else None,markevery=5)
    fig,axs=plt.subplots(nr,cols,figsize=(12,3.1*nr),squeeze=False,constrained_layout=True)
    for ax,case in zip(axs.flat,names):
        ar=mean(np.stack([x[0] for x in bycase[case]]),axis=(0,3))
        im=ax.imshow(ar.T,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis')
        ax.set(title=case,xlabel='Denoising step',ylabel='Layer')
    for ax in list(axs.flat)[len(names):]:ax.set_visible(False)
    fig.colorbar(im,ax=list(axs.flat),label='DM -> AR mean row mass')
    fig.suptitle(report['model']+': all-layer / all-step attention (conditional branch)')
    fig.savefig(a.output/'depth_step_mass.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(nr,cols,figsize=(12,3.4*nr),squeeze=False,constrained_layout=True)
    for ax,case in zip(axs.flat,names):
        rel=mean(np.stack([x[2] for x in bycase[case]]),axis=0)
        im=ax.imshow(rel,vmin=0,vmax=1,cmap='magma')
        ax.set(xticks=range(6),xticklabels=ROLES,yticks=range(6),yticklabels=ROLES,title=case,xlabel='Key / value role',ylabel='Query role')
        ax.tick_params(axis='x',rotation=45)
        for i in range(6):
            for j in range(6):
                if np.isfinite(rel[i,j]):ax.text(j,i,f'{rel[i,j]:.2f}',ha='center',va='center',color='white' if rel[i,j]<.55 else 'black',fontsize=8)
    for ax in list(axs.flat)[len(names):]:ax.set_visible(False)
    fig.suptitle(report['model']+': relation row mass; token-count-adjusted ratios in CSV')
    fig.savefig(a.output/'directed_relations.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,target in zip(axs,['vision_clean','action','sound']):
        found=False
        for (case,role),values in direction_steps.items():
            if role==target:
                ax.plot(mean(np.stack(values),axis=0),label=case,**styles[case]);found=True
        ax.axhline(1,color='black',ls='--',lw=1)
        ax.set(title=target+' -> noisy vision / reverse',xlabel='Denoising step',ylabel='Token-count-adjusted density ratio')
        ax.set_yscale('log');ax.grid(alpha=.2)
        if found:ax.legend(fontsize=7)
        else:ax.text(.5,.5,'Modality absent',transform=ax.transAxes,ha='center')
    fig.suptitle(report['model']+': directed relations across 35 steps (1 = symmetric)\nConditional branch; mean over layers / heads / 2 seeds; log y scales')
    fig.savefig(a.output/'direction_step_trends.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    metrics=[(10,'Dominant relation unchanged',True),(6,'Previous-output relative L2',False),(7,'Top-half key-tile support Jaccard',True),(5,'Previous-output cosine',True)]
    for ax,(fidx,title,bounded) in zip(axs.flat,metrics):
        for case in names:
            f=mean(np.stack([x[1] for x in bycase[case]]),axis=(0,2,3))
            ax.plot(np.arange(1,len(f)),f[1:,fidx],label=case,**styles[case])
        ax.set(title=title,xlabel='Denoising step',ylabel='Mean over layers / heads / seeds')
        if bounded:ax.set_ylim(0,1.01)
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.suptitle(report['model']+': adjacent-step head statistics\nConditional noisy-vision queries; mean over layers / heads / 2 seeds')
    fig.savefig(a.output/'step_trends.png',dpi=170);plt.close(fig)
    # Individual heads at representative depths: no smoothing across steps.
    case='robot_observe' if 'robot_observe' in names else names[0]
    f=mean(np.stack([x[1] for x in bycase[case]]),axis=0)
    ls=[0,f.shape[1]//2,f.shape[1]-1]
    drift_max=float(np.nanmax(f[:,ls,:,6]))
    fig,axs=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    for ri,l in enumerate(ls):
        for ci,(idx,title) in enumerate([(0,'AR mass'),(6,'Previous-output relative L2')]):
            im=axs[ri,ci].imshow(f[:,l,:,idx].T,aspect='auto',origin='lower',cmap='viridis',vmin=0,vmax=1 if idx==0 else drift_max)
            axs[ri,ci].set(title=f'Layer {l}: {title}',xlabel='Step',ylabel='Query head');fig.colorbar(im,ax=axs[ri,ci])
    fig.suptitle(report['model']+' / '+case+': individual-head trajectories\nConditional noisy-vision queries; mean of 2 seeds; shared color scale across depths')
    fig.savefig(a.output/'head_step_trends.png',dpi=170);plt.close(fig)
    summary=dict(model=report['model'],trajectories=len(rows),cases=len(names),steps=report['args']['steps'],
        sampling_queries_per_role=report['args']['samples_per_role'],
        units='Each CSV row is one whole trajectory; no head-level confidence interval',
        overall={k:float(mean([r[k] for r in rows])) for k in ['ar_shallow','ar_deep','depth_pearson','dominant_stability','output_cos','output_rel_l2','output_rel_l2_early','output_rel_l2_late','support_jaccard']},
        trajectories_summary=rows)
    (a.output/'sampling_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='trajectories_summary'},indent=2))


if __name__=='__main__':main()

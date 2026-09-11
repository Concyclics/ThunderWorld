"""Paired same-budget quality, deployed budgets and hardware bottlenecks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PARENTS={'ffn_auto25':'ffn_uniform25','ffn_auto50':'ffn_act2','ffn_auto75':'ffn_uniform75'}
TIERS={'base':(0,0),'a50':(.5,0),'act2':(1,0),'wgt2':(0,1),'a50w':(.5,1),'dual':(1,1)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--hardware',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);table=[];pairs=[];allocation=[]
    for directory in a.runs:
        r=json.loads((directory/'result.json').read_text());assert r['complete'];policy=r['budget_policy'];model=r['model']
        lookup={(x['mode'],x['case'],x['seed']):x for x in r['rows']}
        perc=json.loads((directory/'perceptual.json').read_text()) if (directory/'perceptual.json').exists() else None
        lpips={(x['mode'],x['case'],x['seed']):x['vs_bf16']['mean'] for x in perc['rows']} if perc else {}
        for row in r['rows']:
            m=row['mode'];key=m,row['case'],row['seed'];b=lookup['bf16',row['case'],row['seed']]
            values=row.get('budget_actual',{})
            fraction=values.get('extra_flops',0)/max(2*values.get('base_flops',0),1)
            if m in PARENTS:
                ceiling=int(m.removeprefix('ffn_auto'))/100
                assert fraction<=ceiling+1e-6,(key,fraction,ceiling)
            if m!='bf16':assert sum(values['tier_calls'].values())==row['ffn_calls']
            rec=dict(model=model,mode=m,case=row['case'],seed=row['seed'],wall_seconds=row['wall_seconds'],
                transformer_seconds=row['transformer_seconds'],peak_allocated_gib=row['peak_allocated_bytes']/2**30,
                actual_repair_fraction=fraction,speedup_bf16=b['wall_seconds']/row['wall_seconds'],
                ssim_bf16=row.get('vs_bf16',{}).get('video',{}).get('ssim_mean',1),lpips_bf16=lpips.get(key),
                action_l2=row.get('vs_bf16',{}).get('action',{}).get('relative_l2'),
                audio_latent_l2=row.get('vs_bf16',{}).get('sound_latent',{}).get('relative_l2'))
            table.append(rec)
            if m in PARENTS:
                parent=lookup[PARENTS[m],row['case'],row['seed']]
                pairs.append(dict(model=model,case=row['case'],seed=row['seed'],mode=m,parent=PARENTS[m],
                    ssim_gain=rec['ssim_bf16']-parent['vs_bf16']['video']['ssim_mean'],
                    lpips_gain=lpips.get((PARENTS[m],row['case'],row['seed']),float('nan'))-lpips.get(key,float('nan')),
                    speedup=parent['wall_seconds']/row['wall_seconds'],actual_repair_fraction=fraction))
        for mode,pol in policy['policies'].items():
            for u in policy['units']:
                tier=pol['assignments'][u['key']];ac,wc=TIERS[tier]
                allocation.append(dict(model=model,mode=mode,task=u['task'],branch=u['branch'],phase=u['phase'],name=u['name'],
                    layer=int(u['name'].split('.')[0]),tier=tier,a=ac,w=wc,weight=u['weight'],ms=u['options'][tier]['ms'],loss=u['options'][tier]['loss']))
        tasks=list(dict.fromkeys(u['task'] for u in policy['units']))
        fig,axs=plt.subplots(len(tasks),3,figsize=(12,3.4*len(tasks)),squeeze=False,layout='constrained')
        for ri,task in enumerate(tasks):
            for ci,mode in enumerate(['auto25','auto50','auto75']):
                vals=np.full((r['model_config']['num_hidden_layers'],4),np.nan)
                for l in range(len(vals)):
                    for ph in range(4):
                        selected=[x for x in allocation if x['model']==model and x['mode']==mode and x['task']==task and x['layer']==l and x['phase']==ph]
                        vals[l,ph]=sum((x['a']+x['w'])*x['weight'] for x in selected)/sum(2*x['weight'] for x in selected)
                im=axs[ri,ci].imshow(vals,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='cividis')
                axs[ri,ci].set(title=f'{task} / {mode}',xlabel='Denoising phase',ylabel='Layer',xticks=range(4),xticklabels=['0–20%','20–50%','50–80%','80–100%'])
        fig.colorbar(im,ax=list(axs.flat),label='Extra products / full dual-repair products')
        fig.suptitle(model+': automatic repair allocation by layer and phase\nFLOP-weighted over FFN projections and CFG branches; policies calibrated before evaluation')
        fig.savefig(a.output/f'{model}_allocation.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(1,len(tasks),figsize=(5*len(tasks),4),squeeze=False,layout='constrained')
        case_tasks={c['id']:c['task'] for c in r['cases']}
        for ax,task in zip(axs.flat,tasks):
            for row in [x for x in table if x['model']==model and case_tasks[x['case']]==task]:
                auto=row['mode'].startswith('ffn_auto')
                color='#2563eb' if auto else '#d97706'
                ax.scatter(row['wall_seconds'],row['ssim_bf16'],color=color,marker='o' if auto else 'x')
                ax.annotate(row['mode'].removeprefix('ffn_'),(row['wall_seconds'],row['ssim_bf16']),fontsize=7,xytext=(3,3),textcoords='offset points')
            ax.set(title=task,xlabel='Generator seconds (eager)',ylabel='Video SSIM vs paired BF16',ylim=(0,1.02));ax.grid(alpha=.2)
        fig.suptitle(model+': complete 35-step quality / latency\nBlue circles: automatic; orange crosses: uniform or reference; each point is one seed')
        fig.savefig(a.output/f'{model}_frontier.png',dpi=160);plt.close(fig)
    for filename,rows in [('e2e.csv',table),('paired_budget.csv',pairs),('allocation.csv',allocation)]:
        with (a.output/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    hw=json.loads(a.hardware.read_text());assert hw['complete'];hardware=[]
    for profile in hw['profiles']:
        for k in profile['kernels']:
            hardware.append(dict(tag=profile['tag'],kernel=k['kernel'],duration_us=k['duration_ns']/1000,
                tensor_elapsed_pct=k['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'],
                occupancy_pct=k['sm__warps_active.avg.pct_of_peak_sustained_active'],
                dram_throughput_pct=k['dram__throughput.avg.pct_of_peak_sustained_elapsed'],
                registers=k['launch__registers_per_thread'],
                local_load_sectors=k['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'],
                local_store_sectors=k['l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']))
    with (a.output/'hardware.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(hardware[0]));writer.writeheader();writer.writerows(hardware)
    summaries=[]
    for model in dict.fromkeys(x['model'] for x in pairs):
        for mode in PARENTS:
            xs=[x for x in pairs if x['model']==model and x['mode']==mode]
            summaries.append(dict(model=model,mode=mode,n=len(xs),ssim_gain_mean=float(np.mean([x['ssim_gain'] for x in xs])),
                ssim_gain_min=min(x['ssim_gain'] for x in xs),speedup_geomean=float(np.exp(np.mean(np.log([x['speedup'] for x in xs]))))))
    (a.output/'summary.json').write_text(json.dumps(dict(passed_budget_checks=True,paired=summaries,trajectories=len(table)),indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

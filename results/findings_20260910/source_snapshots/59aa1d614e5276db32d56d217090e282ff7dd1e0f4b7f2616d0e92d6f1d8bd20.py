"""Quality-first summaries; all latency ratios stay within model/case/seed."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    table=[];reports=[]
    for path in a.inputs:
        report=json.loads((path/'result.json').read_text())
        if not report['complete']:raise ValueError(str(path)+' incomplete')
        reports.append(report)
        base={(r['case'],r['seed'],r['repeat']):r for r in report['rows'] if r['mode']=='bf16'}
        anchor={(r['case'],r['seed'],r['repeat']):r for r in report['rows'] if r['mode']=='ffn_act2'}
        repair={(r['case'],r['seed'],r['repeat']):r for r in report['rows'] if r['mode']=='attn_dual50'}
        for r in report['rows']:
            key=r['case'],r['seed'],r['repeat'];b=base[key]
            parent=repair.get(key) if r['mode'].startswith(('mask_reuse','delta_reuse')) else anchor.get(key)
            rec=dict(model=report['model'],case=r['case'],mode=r['mode'],seed=r['seed'],repeat=r['repeat'],
                wall_seconds=r['wall_seconds'],transformer_seconds=r['transformer_seconds'],
                peak_allocated_gib=r['peak_allocated_bytes']/2**30,speedup_bf16=b['wall_seconds']/r['wall_seconds'],
                parent_mode=parent['mode'] if parent else 'bf16',
                speedup_parent=parent['wall_seconds']/r['wall_seconds'] if parent else b['wall_seconds']/r['wall_seconds'],
                ssim_bf16=r.get('vs_bf16',{}).get('video',{}).get('ssim_mean',1),
                video_rel_l2_bf16=r.get('vs_bf16',{}).get('video',{}).get('relative_l2',0),
                action_rel_l2_bf16=r.get('vs_bf16',{}).get('action',{}).get('relative_l2'),
                audio_rel_l2_bf16=r.get('vs_bf16',{}).get('sound',{}).get('relative_l2'),
                ssim_parent=r.get('vs_current_repair',r.get('vs_ffn_act2',{})).get('video',{}).get('ssim_mean'),
                action_rel_l2_parent=r.get('vs_current_repair',r.get('vs_ffn_act2',{})).get('action',{}).get('relative_l2'),
                audio_rel_l2_parent=r.get('vs_current_repair',r.get('vs_ffn_act2',{})).get('sound',{}).get('relative_l2'),
                reused_output=r['counts']['reused_output'],reused_delta=r['counts']['reused_delta'],reused_mask=r['counts']['reused_mask'])
            table.append(rec)
    with (a.output/'ablation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    summaries=[]
    for model in dict.fromkeys(r['model'] for r in table):
        modelrows=[r for r in table if r['model']==model]
        modes=list(dict.fromkeys(r['mode'] for r in modelrows))
        for mode in modes:
            rows=[r for r in modelrows if r['mode']==mode]
            summaries.append(dict(model=model,mode=mode,requests=len(rows),
                ssim_bf16_mean=float(np.mean([r['ssim_bf16'] for r in rows])),
                ssim_bf16_min=float(min(r['ssim_bf16'] for r in rows)),
                speedup_parent_geomean=float(np.exp(np.mean(np.log([r['speedup_parent'] for r in rows])))),
                action_rel_l2_bf16_max=max((r['action_rel_l2_bf16'] for r in rows if r['action_rel_l2_bf16'] is not None),default=None),
                audio_rel_l2_bf16_max=max((r['audio_rel_l2_bf16'] for r in rows if r['audio_rel_l2_bf16'] is not None),default=None)))
        cases=list(dict.fromkeys(r['case'] for r in modelrows))
        fig,axs=plt.subplots(len(cases),2,figsize=(15,4*len(cases)),squeeze=False,constrained_layout=True)
        for axpair,case in zip(axs,cases):
            rows=[r for r in modelrows if r['case']==case]
            for ax,metric,title in zip(axpair,['ssim_bf16','wall_seconds'],['Paired video SSIM vs BF16','Full Generator seconds (eager)']):
                vals=[[r[metric] for r in rows if r['mode']==m] for m in modes]
                med=[np.median(v) if v else np.nan for v in vals]
                ax.barh(np.arange(len(modes)),med,color=['#9ca3af' if m=='bf16' else '#2563eb' if m.startswith('ffn') else '#0d9488' if 'reuse' not in m else '#d97706' for m in modes])
                for i,v in enumerate(vals):
                    if v:ax.plot([min(v),max(v)],[i,i],color='black',lw=1.2)
                ax.set(yticks=range(len(modes)),yticklabels=modes,title=f'{case}: {title}')
                ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
                if metric=='ssim_bf16':ax.set_xlim(0,1.01)
        fig.suptitle(model+': complete 35-step trajectories; bars=median, lines=observed range')
        fig.savefig(a.output/f'{model}_ablation.png',dpi=160);plt.close(fig)
    (a.output/'ablation_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    lines=['# Full-trajectory repair and reuse ablation','',
           'All runs use the same eager execution and input within each comparison. Attention ablations share RTN W4 + full ACT2 FFNs. Sampling overhead is excluded by running a separate process. Numbers are diagnostic eager latency, not the earlier CUDA Graph production benchmark.','',
           '| Model | Mode | Requests | Mean / min BF16 SSIM | Max action rel L2 | Max audio rel L2 | Parent speedup |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        act='—' if r['action_rel_l2_bf16_max'] is None else f"{r['action_rel_l2_bf16_max']:.4f}"
        aud='—' if r['audio_rel_l2_bf16_max'] is None else f"{r['audio_rel_l2_bf16_max']:.4f}"
        lines.append(f"| {r['model']} | {r['mode']} | {r['requests']} | {r['ssim_bf16_mean']:.4f} / {r['ssim_bf16_min']:.4f} | {act} | {aud} | {r['speedup_parent_geomean']:.3f}× |")
    lines+=['','Parent = current attn_dual50 for mask/delta reuse; ffn_act2 for output reuse and other modes where available. Per-case values and exact parent labels are in ablation.csv. Action/audio are paired numerical differences, not task success or perceptual quality scores.']
    (a.output/'ABLATION.md').write_text('\n'.join(lines)+'\n')
    print('requests',len(table),'modes',len(summaries))


if __name__=='__main__':main()

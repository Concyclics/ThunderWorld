"""Show error propagation on fixed 256-coordinate prediction sketches."""
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
    rows=[]
    selected=['ffn_base','ffn_act2','ffn_dual','ffn_four','mask_reuse2','delta_reuse2_middle','output_reuse2_stable']
    styles={m:dict(color='#2563eb' if m.startswith('ffn') else '#d97706',
                   linestyle=['-', '--', ':', '-.'][i if i<4 else i-4]) for i,m in enumerate(selected)}
    for path in a.inputs:
        report=json.loads((path/'result.json').read_text());assert report['complete']
        cases={c['id']:c for c in report['cases']};refs={};anchors={};repairs={}
        for r in report['rows']:
            mode=r['mode'];key=r['case'],r['seed'],r['repeat']
            with np.load(path/f'{mode}_{key[0]}_seed{key[1]}_r{key[2]}_outputs.npz') as z:
                sketch=z['trajectory'].astype(np.float64)
            branches=1 if cases[key[0]]['task']=='policy' else 2
            sketch=sketch.reshape(35,branches,3,256)[:,0]
            if mode=='bf16':refs[key]=sketch
            if mode=='ffn_act2':anchors[key]=sketch
            if mode=='attn_dual50':repairs[key]=sketch
            parent=repairs.get(key) if mode.startswith(('mask_reuse','delta_reuse')) else anchors.get(key)
            parentname='attn_dual50' if mode.startswith(('mask_reuse','delta_reuse')) else 'ffn_act2'
            for comparator,reference in [('bf16',refs[key]),(parentname,parent)]:
                if reference is None:continue
                for mi,modality in enumerate(['vision','sound','action']):
                    if modality=='sound' and cases[key[0]]['task']!='av':continue
                    if modality=='action' and cases[key[0]]['task']!='policy':continue
                    norm=np.linalg.norm(reference[:,mi],axis=-1)
                    err=np.linalg.norm(sketch[:,mi]-reference[:,mi],axis=-1)/np.maximum(norm,1e-12)
                    for step,value in enumerate(err):
                        rows.append(dict(model=report['model'],case=key[0],seed=key[1],repeat=key[2],mode=mode,
                            comparator=comparator,modality=modality,step=step,sketch_relative_l2=float(value)))
        fig,axs=plt.subplots(len(cases),3,figsize=(14,3.4*len(cases)),squeeze=False,layout='constrained')
        for ri,case in enumerate(cases):
            for ci,modality in enumerate(['vision','sound','action']):
                ax=axs[ri,ci];found=False
                for mode in selected:
                    chosen=[r for r in rows if r['model']==report['model'] and r['case']==case and r['modality']==modality and r['mode']==mode and r['comparator']=='bf16']
                    if not chosen:continue
                    means=[np.mean([r['sketch_relative_l2'] for r in chosen if r['step']==s]) for s in range(35)]
                    ax.plot(means,label=mode,**styles[mode]);found=True
                ax.set(title=f'{case} / {modality}',xlabel='Denoising step',ylabel='Prediction sketch relative L2 vs BF16');ax.grid(alpha=.2)
                if not found:ax.text(.5,.5,'Modality absent',transform=ax.transAxes,ha='center')
                if ri==0 and ci==0:ax.legend(fontsize=7)
        fig.suptitle(report['model']+': free-running prediction error across steps\nConditional branch; fixed 256 coordinates per modality; mean of 2 seeds; not full-tensor error')
        fig.savefig(a.output/f'{report["model"]}_prediction_error.png',dpi=150);plt.close(fig)
    with (a.output/'prediction_error.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':main()

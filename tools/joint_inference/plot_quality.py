"""Export the paired confidence intervals for predeclared primary candidates."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edge',type=Path,required=True);p.add_argument('--nano',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data={model:json.loads(path.read_text()) for model,path in [('Edge',a.edge),('Nano',a.nano)]}
    selection=[('Edge','robot_table','dual'),('Edge','policy_table','dual'),
               ('Nano','robot_table','four'),('Nano','policy_table','four_dual_attention'),('Nano','av_pottery','four')]
    fig,axes=plt.subplots(3,1,figsize=(10,8),gridspec_kw={'height_ratios':[5,2,1]},layout='constrained')
    for ax,metric,title in zip(axes,['video.lpips','action.relative_l2','waveform.log_spectral_l1'],
        ['Video LPIPS','Action relative L2','Waveform log-spectral error']):
        labels=[]
        for model,case,config in selection:
            decision=next(x for x in data[model]['decisions'] if x['case']==case and x['config']==config)
            if metric not in decision['metrics']:continue
            stats=decision['metrics'][metric];mu=stats['mean_paired_difference'];lo,hi=stats['ci95']
            y=len(labels);labels.append(model+' '+{'robot_table':'I2V','policy_table':'policy','av_pottery':'AV'}[case])
            ax.errorbar(mu,y,xerr=[[mu-lo],[hi-mu]],fmt='o',capsize=4,
                color='#087f5b' if hi<0 else '#d97706')
        ax.axvline(0,color='#6b7280',linestyle='--',linewidth=1)
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
        ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Candidate minus FP8 error (lower is better); paired 95% bootstrap CI')
    fig.suptitle('35-step BF16 consistency: held-out seeds 91001–91008',fontsize=14)
    fig.savefig(a.output,dpi=180)


if __name__=='__main__':main()

"""Absolute-memory figures for the frozen DM and isolated AR FFN comparisons."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    import numpy as np
    from ar_ffn_report import METHODS
    rows=json.loads(a.input.read_text())
    extended=rows[0]['scope']=='DM + AR FFN'
    # Figure 3 (head_depth_structure) uses DejaVu Sans and blue/gold/gray.
    # A translucent blue keeps the common base distinct from the extra layout.
    parts=(('base_payload_bytes','Base weights','#245D91',.24),
           ('base_metadata_bytes','Scales & metadata','#9AA5AD',1.),
           ('extra_scale_layout_bytes','Extra scale layout','#245D91',1.),
           ('w1_bytes','W1 & indices','#B37521',1.))
    methods=tuple(m for m in METHODS if m not in ('bf16','fp8'))
    labels=('W4A4','ARCQuant\n(50%)','SharQ',
            'Ours\n'+r'($\beta=1.5$)','Ours\n'+r'($\beta=2$)')
    references=(('bf16','BF16','#717D87',(0,(8,4))),
                ('fp8','FP8','#245D91',(0,(3,2))))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,
                         'mathtext.fontset':'dejavusans',
                         'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,
                         'text.color':'#263647','axes.labelcolor':'#263647',
                         'xtick.color':'#000000','ytick.color':'#000000',
                         'axes.linewidth':1.1,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(14.6,4.8))
    fig.subplots_adjust(left=.066,right=.987,bottom=.27,top=.83,wspace=.19)
    fig.legend(handles=[Patch(facecolor=c,alpha=alpha,label=label) for _,label,c,alpha in parts],
        loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,frameon=False,
        fontsize=15,columnspacing=1.5,handlelength=1.5)
    for ax,model in zip(axes,('edge','nano')):
        model_rows={r['method']:r for r in rows if r['model']==model}
        for r in model_rows.values():
            if sum(r[k] for k,_,_,_ in parts)!=r['target_total_bytes']:
                raise AssertionError('Stack mismatch')
        selected=[model_rows[m] for m in methods]
        x=np.arange(len(methods));bottom=np.zeros(len(methods))
        for key,_,color,alpha in parts:
            values=np.array([r[key] for r in selected])/2**30
            ax.bar(x,values,.64,bottom=bottom,color=color,alpha=alpha,
                   edgecolor='white',linewidth=.45,zorder=3)
            bottom+=values
        for i,r in enumerate(selected):
            ax.annotate(f'{bottom[i]:.2f} GiB',(i,bottom[i]),xytext=(0,6),textcoords='offset points',
                        ha='center',va='bottom',fontsize=14,zorder=5,
                        bbox={'facecolor':'white','edgecolor':'none','pad':.4})
            value=r['base_payload_bytes']/2**30
            ax.text(i,value/2,f'{value:.2f}\nGiB',ha='center',va='center',color='#263647',
                    fontsize=14,linespacing=1.05,fontweight='semibold')
        reference_values=[]
        for method,label,color,dashes in references:
            value=model_rows[method]['target_total_bytes']/2**30
            reference_values.append(value)
            ax.axhline(value,color=color,linestyle=dashes,linewidth=1.6,zorder=2)
            ax.annotate(f'{label}: {value:.2f} GiB',(.99,value),
                        xycoords=('axes fraction','data'),xytext=(0,5),
                        textcoords='offset points',ha='right',va='bottom',
                        fontsize=14,color=color,zorder=5,
                        bbox={'facecolor':'white','edgecolor':'none','pad':.4})
        ax.set_ylim(0,max(max(bottom),*reference_values)*1.14)
        ax.set_xlim(-.6,len(methods)-.4)
        ax.set_xticks(x,labels)
        ax.set_xlabel(model.title(),fontweight='semibold',labelpad=10)
        ax.set_ylabel(('DM + AR FFN' if extended else 'DM')+' weights (GiB)')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6,steps=[1,2,2.5,5,10]))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax.grid(axis='y',color='#E1E6EA',linewidth=.7,zorder=0)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#000000')
        ax.tick_params(axis='both',length=4,width=1.1,color='#000000')
    stem=str(a.input).removesuffix('.chart-data.json')
    for ext in ('png','pdf','svg'):fig.savefig(stem+'.'+ext,dpi=200,facecolor='white')
    plt.close(fig)
    print(stem+'.png')


if __name__=='__main__':main()

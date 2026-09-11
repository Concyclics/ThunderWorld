"""All-frame paired LPIPS and fixed-frame visual comparisons of completed runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import lpips
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(8)
    torch.hub.set_dir(str(root / 'data_local/perceptual_cache'))
    metric = lpips.LPIPS(net='alex', version='0.1').eval().cuda()
    def distance(video, reference):
        assert video.shape == reference.shape
        values = []
        for start in range(0, len(video), 8):
            x = torch.from_numpy(video[start:start+8]).permute(0,3,1,2).float().cuda()*2-1
            y = torch.from_numpy(reference[start:start+8]).permute(0,3,1,2).float().cuda()*2-1
            values.extend(metric(x,y).flatten().cpu().tolist())
        assert np.isfinite(values).all()
        return dict(mean=float(np.mean(values)), max=float(np.max(values)), per_frame=values)
    for directory in a.inputs:
        report = json.loads((directory/'result.json').read_text())
        assert report['complete']
        bases={}; anchors={}; repairs={}; rows=[]; panels={}
        panel_modes=['bf16','ffn_base','ffn_act2','ffn_dual','ffn_four',
                     'attn_dual50','mask_reuse2','delta_reuse2_middle','output_reuse2_stable']
        panel_seed=min(report['args']['seeds'])
        for row in report['rows']:
            mode=row['mode']; key=row['case'],row['seed'],row['repeat']
            stem=f'{mode}_{key[0]}_seed{key[1]}_r{key[2]}'
            with np.load(directory/(stem+'_outputs.npz')) as z:
                video=z['video']
            assert video.min()>=0 and video.max()<=1 and np.isfinite(video).all()
            if mode=='bf16': bases[key]=video
            if mode=='ffn_act2': anchors[key]=video
            if mode=='attn_dual50': repairs[key]=video
            rec=dict(model=report['model'],case=key[0],seed=key[1],repeat=key[2],mode=mode)
            rec['vs_bf16']=distance(video,bases[key]) if mode!='bf16' else dict(mean=0.,max=0.,per_frame=[0.]*len(video))
            if key in anchors: rec['vs_ffn_act2']=distance(video,anchors[key])
            if mode.startswith(('mask_reuse','delta_reuse')): rec['vs_current_repair']=distance(video,repairs[key])
            if mode in panel_modes and key[1]==panel_seed and key[2]==0:
                ids=[len(video)//4,len(video)//2,len(video)-1]
                panels.setdefault(key[0],{})[mode]=(video[ids],ids,rec['vs_bf16']['mean'])
            rows.append(rec)
            print(report['model'],stem,round(rec['vs_bf16']['mean'],5),flush=True)
        weights=[root/'data_local/perceptual_cache/checkpoints/alexnet-owt-7be5be79.pth',
                 Path(lpips.__file__).parent/'weights/v0.1/alex.pth']
        result=dict(complete=True,metric='LPIPS AlexNet v0.1; RGB [-1,1]; native resolution; all frames equally weighted',
            scope='Paired perceptual distance, not task success or absolute quality. Lower means closer to reference.',
            input=str(directory.resolve()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            weight_sha256={str(w):hashlib.sha256(w.read_bytes()).hexdigest() for w in weights},rows=rows)
        (directory/'perceptual.json').write_text(json.dumps(result,indent=2)+'\n')
        for case,variants in panels.items():
            modes=[m for m in panel_modes if m in variants]
            fig,axs=plt.subplots(len(modes),3,figsize=(9,1.85*len(modes)),squeeze=False,layout='constrained')
            for ri,mode in enumerate(modes):
                frames,ids,score=variants[mode]
                for ci,frame in enumerate(frames):
                    ax=axs[ri,ci];ax.imshow(frame);ax.set_xticks([]);ax.set_yticks([])
                    if ri==0:ax.set_title(f'Frame {ids[ci]}')
                    if ci==0:ax.set_ylabel(f'{mode}\nLPIPS {score:.3f}',fontsize=8)
            fig.suptitle(f'{report["model"]} / {case} / seed {panel_seed}: fixed frames\nLPIPS averages all frames vs BF16; identical frame indices for every variant',fontsize=11)
            fig.savefig(a.output/f'{report["model"]}_{case}_frames.png',dpi=140)
            plt.close(fig)


if __name__=='__main__': main()

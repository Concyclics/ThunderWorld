"""CPU reconstruction diagnostics of the actual offline FP4 weight planes.

Uses 64 evenly spaced output rows per projection, all K columns and the saved
full-matrix factor. It never recomputes scales on sampled input rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from baselines import sha, write


def run(root):
    torch.set_num_threads(8)
    study=json.loads((root/'edge/final/manifest.json').read_text())
    bank=root/'offline_weights';manifest=json.loads((bank/'manifest.json').read_text())
    model=Path(study['model'])/'transformer'
    index=json.loads((model/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
    rows=[]
    with torch.inference_mode():
        for name,entry in manifest['modules'].items():
            path=bank/entry['file']
            if sha(path)!=entry['sha256']:raise ValueError('Changed offline weight plane')
            asset=torch.load(path,map_location='cpu',weights_only=True)
            n,k=entry['shape'];selected=torch.linspace(0,n-1,min(n,64)).round().long().unique()
            key=name+'.weight'
            with safe_open(model/index[key],framework='pt',device='cpu') as stream:
                gold=stream.get_tensor(key)[selected].float()
            norm=float(gold.square().sum());reconstruction=torch.zeros_like(gold);planes=[]
            for p in range(4):
                codes=asset['codes'][p,selected].to(torch.long)
                scales=asset['scales'][p,selected].float()
                values=table[torch.stack([codes&15,codes>>4],dim=-1).flatten(-2)]
                reconstruction.add_(values*scales.repeat_interleave(16,dim=-1)*asset['factor'])
                sse=float((gold-reconstruction).square().sum())
                planes.append(dict(planes=p+1,squared_error=sse,nmse=sse/max(norm,1.e-30),
                    nonzero_code_fraction=float((values!=0).float().mean()),
                    floor_scale_fraction=float((scales==1/512).float().mean())))
            rows.append(dict(module=name,original_shape=[n,k],sampled_rows=selected.tolist(),
                reference_squared_norm=norm,sampled_elements=gold.numel(),planes=planes))
    summary=[]
    for p in range(4):
        total=sum(r['sampled_elements'] for r in rows)
        summary.append(dict(planes=p+1,
            pooled_nmse=sum(r['planes'][p]['squared_error'] for r in rows)/sum(r['reference_squared_norm'] for r in rows),
            mean_module_nmse=float(np.mean([r['planes'][p]['nmse'] for r in rows])),
            nonzero_code_fraction=sum(r['planes'][p]['nonzero_code_fraction']*r['sampled_elements'] for r in rows)/total,
            floor_scale_fraction=sum(r['planes'][p]['floor_scale_fraction']*r['sampled_elements'] for r in rows)/total))
    result=dict(complete=True,weight_manifest_sha256=sha(bank/'manifest.json'),source_sha256=sha(__file__),
        scope='64 evenly spaced output rows per projection, all K columns; stored full-matrix scales unchanged. '
              'FP32 reconstruction versus original stored BF16 weights; not an output quality metric.',
        summary=summary,modules=rows)
    write(root/'weight_reconstruction.json',result)
    print(json.dumps(dict(complete=True,modules=len(rows),summary=summary)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root.resolve())

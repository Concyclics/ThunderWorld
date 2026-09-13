"""Transfer calibration recipes; regenerate FP4 weights offline from the pinned HF model.

Every reconstructed tensor is checked byte-for-byte against the original asset.
Only the no-outlier calibrated encoder is supported; unsupported recipes fail closed.
"""
import argparse
import hashlib
import json
from pathlib import Path

ENCODED=('codes_base','codes_residual','scales_base','scales_residual')


def tensor_hash(t):
    import torch
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['export','rebuild'])
    p.add_argument('--assets',type=Path)
    p.add_argument('--recipe',type=Path)
    p.add_argument('--model',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.operation=='export':
        if a.assets is None:p.error('--assets required')
        import torch
        manifest=json.loads((a.assets/'manifest.json').read_text())
        a.output.mkdir(parents=True,exist_ok=False);modules={}
        for name,row in manifest['modules'].items():
            asset=torch.load(a.assets/row['file'],map_location='cpu',weights_only=True)
            if asset.get('mandatory_w_blocks') or asset.get('outlier_threshold') is not None:
                raise ValueError('Outlier asset needs explicit removal support; transfer the original asset instead')
            if asset.get('storage_schema')!=2:raise ValueError('Explicit compact support required')
            filename=f'recipe_{len(modules):03}.pt'
            small={k:v for k,v in asset.items() if k not in ENCODED}
            torch.save(small,a.output/filename)
            modules[name]=dict(file=filename,sha256=hashlib.sha256((a.output/filename).read_bytes()).hexdigest(),
                reconstructed={k:tensor_hash(asset[k]) for k in ENCODED},factor_sha256=tensor_hash(asset['factor'].reshape(1)))
        result=dict(schema_version=1,encoder='calibrated_triton_no_outliers',asset_manifest=manifest,modules=modules)
        (a.output/'recipe.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    if a.recipe is None or a.model is None:p.error('--recipe and --model required')
    recipe=json.loads((a.recipe/'recipe.json').read_text())
    if recipe['schema_version']!=1 or recipe['encoder']!='calibrated_triton_no_outliers':raise ValueError('Unsupported recipe')
    manifest=recipe['asset_manifest']
    for name,digest in manifest['model_identity'].items():
        if hashlib.sha256((a.model/name).read_bytes()).hexdigest()!=digest:raise ValueError('Wrong HF model revision')
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        select_shared_device()
        import torch
        from safetensors import safe_open
        from thunderworld_fp4._calibrated import pack
        from thunderworld_fp4.cosmos.assets import compact
        torch.set_num_threads(8)
        a.output.mkdir(parents=True,exist_ok=False)
        manifest={**manifest,'complete':False,'modules':dict(manifest['modules'])}
        def save(): (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        save()
        weight_map=json.loads((a.model/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
        with torch.inference_mode():
            for name,row in recipe['modules'].items():
                path=a.recipe/row['file']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Corrupt recipe')
                small=torch.load(path,map_location='cpu',weights_only=True)
                key=name+'.weight'
                with safe_open(a.model/'transformer'/weight_map[key],framework='pt',device='cpu') as handle:w=handle.get_tensor(key).cuda()
                d=small['prescale'].cuda();base,res=small.get('weight_scale',(1.,1.))
                c,s,g,_=pack((w.float()/d).bfloat16().contiguous(),torch.ones_like(d),True,True,w,base=base,res=res)
                if tensor_hash(g.cpu().reshape(1))!=row['factor_sha256']:raise ValueError(f'Factor mismatch: {name}')
                expanded={k:v for k,v in small.items() if k not in ('storage_schema','weight_support_fraction','residual_indices')}
                expanded.update(codes=c.cpu(),scales=s.cpu())
                rebuilt=compact(expanded,small['residual_indices'].tolist())
                for field,digest in row['reconstructed'].items():
                    if tensor_hash(rebuilt[field])!=digest:raise ValueError(f'Encoder mismatch: {name}.{field}; original asset required')
                filename=manifest['modules'][name]['file'];torch.save(rebuilt,a.output/filename)
                manifest['modules'][name]={**manifest['modules'][name],'sha256':hashlib.sha256((a.output/filename).read_bytes()).hexdigest()}
                save();del w,d,c,s,g,rebuilt,expanded
        manifest['complete']=True;manifest['recipe_sha256']=hashlib.sha256((a.recipe/'recipe.json').read_bytes()).hexdigest();save()


if __name__=='__main__':main()

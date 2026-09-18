"""Export compact calibration recipes; reconstruct byte-identical assets from HF weights.

No model weights or residual planes are transferred. Restore is CPU-only and
checks every packed tensor, including empty supports and FP8 scale bytes.
Sparse byte corrections preserve source GPU rounding at quantization ties.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
from baselines import sha,write

PACKED=('codes_base','scales_base','codes_residual','scales_residual')


def tensor_hash(value):
    value=value.detach().cpu().contiguous()
    header=json.dumps([str(value.dtype),list(value.shape)]).encode()
    return hashlib.sha256(header+value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def rebuild(weight,metadata,encoded=None):
    from thunderworld_fp4.cosmos.calibration import quantize_weight
    from thunderworld_fp4.cosmos.assets import compact
    if metadata.get('mandatory_w_blocks'):raise ValueError('Outlier assets require their removal mask; recipe v1 refuses them')
    codes,scales,factor=encoded if encoded is not None else quantize_weight(weight,metadata['prescale'],base=metadata['weight_scale'][0],residual=metadata['weight_scale'][1])
    base={k:v for k,v in metadata.items() if k not in ('storage_schema','weight_support_fraction','residual_indices')}
    base.update(codes=codes,scales=scales,factor=factor)
    return compact(base,metadata['residual_indices'].tolist())


def export(a):
    import torch
    torch.set_num_threads(4)
    selection=json.loads(a.selection.read_text());study=json.loads(a.manifest.read_text())
    if not selection.get('complete'):raise ValueError('Frozen selection required')
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(a.selection,a.output/'selection.json')
    common=study['common_args'];case_path=Path(common[common.index('--case')+1]);case=json.loads(case_path.read_text())
    if case.get('image'):
        image=(case_path.parent/case['image']).resolve();name='input'+image.suffix
        shutil.copyfile(image,a.output/name);case['image']=name
    write(a.output/'case.json',case)
    result=dict(schema=1,complete=False,selection_sha256=sha(a.selection),jobs=[],
        model_repo='nvidia/Cosmos3-Edge',revision='a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba',
        scope='Frozen selected policies and same-budget controls. Rebuild offline; exact packed tensor hashes required. No quality selection on timing seeds.')
    chosen={r['selected'] for r in selection['decisions']}|{r['control'] for r in selection['decisions']}
    for job in study['runs']:
        if job['name'] not in chosen and job['name'] not in ('bf16','fp8','fp4_base','fp4_three','fp4_four'):continue
        folder=a.output/job['name'];folder.mkdir();args=job['args'];row=dict(name=job['name'],execution_type='native')
        receipt_path=Path(study['output'])/job['name']/'result.json';receipt=json.loads(receipt_path.read_text())
        if not receipt['complete']:raise ValueError('Complete frozen generation required for portable quality verification')
        row['quality_signal_hashes']={str(r['seed']):r['sha256'] for r in receipt['rows']}
        row['quality_receipt_sha256']=sha(receipt_path)
        if '--runtime-config' in args:
            config_path=Path(args[args.index('--runtime-config')+1]);config=json.loads(config_path.read_text())
            shutil.copyfile(config['budget_plan'],folder/'plan.json');config['budget_plan']='plan.json'
            bank=Path(args[args.index('--assets')+1]);manifest=json.loads((bank/'manifest.json').read_text())
            row['asset_manifest']=manifest;row['modules']={};row['_source_bank']=str(bank)
            (folder/'recipes').mkdir()
            for name,entry in manifest['modules'].items():
                if sha(bank/entry['file'])!=entry['sha256']:raise ValueError('Changed source asset')
                asset=torch.load(bank/entry['file'],weights_only=True,map_location='cpu')
                if asset.get('mandatory_w_blocks'):raise ValueError('Unsupported outlier recipe')
                meta={k:v for k,v in asset.items() if k not in PACKED}
                target=folder/'recipes'/entry['file'];torch.save(meta,target)
                row['modules'][name]=dict(file=str(target.relative_to(a.output)),sha256=sha(target),
                    tensor_hashes={k:tensor_hash(v) for k,v in asset.items() if isinstance(v,torch.Tensor)})
        else:
            from thunderworld_fp4.cosmos import RunConfig
            from thunderworld_fp4.cosmos.settings import LinearSpec
            from thunderworld_fp4 import RepairConfig,RepairPolicy
            name=job['name']
            if name in ('bf16','fp8'):config=RunConfig(precision=name,attention_backend='bf16').to_dict()
            else:
                coverage=0. if name=='fp4_base' else 1.;terms=4 if name=='fp4_four' else 3
                config=RunConfig(precision='fp4',attention_backend='fp4',ffn=LinearSpec(coverage,coverage,terms==4),
                    projection=LinearSpec(coverage,coverage,terms==4),attention=RepairPolicy(RepairConfig(
                    coverage=coverage,terms=terms,query_tile=32,transpose_v=True,sparse_pack=True))).to_dict()
        write(folder/'config.json',config);row['config']=str((folder/'config.json').relative_to(a.output));result['jobs'].append(row)
    # CPU and CUDA division can straddle an FP4 rounding midpoint. Transfer only
    # sparse byte corrections to the CPU recipe, never silently change assets.
    from safetensors import safe_open
    from thunderworld_fp4.cosmos.calibration import quantize_weight
    from thunderworld_fp4.cosmos.assets import compact
    model=Path(common[common.index('--model')+1]);index=json.loads((model/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    for name in sorted({name for row in result['jobs'] for name in row.get('modules',{})}):
        with safe_open(model/'transformer'/index[name+'.weight'],framework='pt',device='cpu') as handle:weight=handle.get_tensor(name+'.weight').bfloat16()
        cached={}
        for row in result['jobs']:
            if name not in row.get('modules',{}):continue
            entry=row['modules'][name];meta=torch.load(a.output/entry['file'],weights_only=True)
            key=(tensor_hash(meta['prescale']),tuple(meta['weight_scale']))
            if key not in cached:cached[key]=quantize_weight(weight,meta['prescale'],base=key[1][0],residual=key[1][1])
            codes,scales,factor=cached[key]
            raw={k:v for k,v in meta.items() if k not in ('storage_schema','weight_support_fraction','residual_indices')}
            raw.update(codes=codes,scales=scales,factor=factor);restored=compact(raw,meta['residual_indices'].tolist())
            source=torch.load(Path(row['_source_bank'])/row['asset_manifest']['modules'][name]['file'],weights_only=True)
            patch={}
            for k,v in restored.items():
                if not isinstance(v,torch.Tensor):continue
                x=v.reshape(-1).view(torch.uint8);y=source[k].contiguous().reshape(-1).view(torch.uint8)
                indices=(x!=y).nonzero().flatten()
                if indices.numel():patch[k]=(indices,y[indices].clone())
            patch_path=(a.output/entry['file']).with_suffix('.patch.pt');torch.save(patch,patch_path)
            entry['rounding_patch']=str(patch_path.relative_to(a.output))
            entry['correction_bytes']=sum(v[0].numel()*8+v[1].numel() for v in patch.values())
        print(json.dumps(dict(recipe=name)),flush=True)
    for row in result['jobs']:row.pop('_source_bank',None)
    result['files']={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()}
    result['complete']=True;write(a.output/'package.json',result)


def restore(a):
    import torch
    from safetensors import safe_open
    from thunderworld_fp4.cosmos.runner import model_identity
    torch.set_num_threads(a.threads)
    recipe=json.loads((a.package/'package.json').read_text())
    if not recipe['complete']:raise ValueError('Incomplete package')
    for name,digest in recipe['files'].items():
        if sha(a.package/name)!=digest:raise ValueError('Changed package file: '+name)
    a.output.mkdir(parents=True,exist_ok=False)
    case=json.loads((a.package/'case.json').read_text())
    if case.get('image'):case['image']=str((a.package/case['image']).resolve())
    write(a.output/'case.json',case)
    identity=model_identity(a.model);index=json.loads((a.model/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    jobs=[];banks={}
    for row in recipe['jobs']:
        folder=a.output/row['name'];folder.mkdir();config=json.loads((a.package/row['config']).read_text())
        job=dict(name=row['name'],execution_type='native',config=str((folder/'config.json').resolve()),
            quality_signal_hashes=row['quality_signal_hashes'],quality_receipt_sha256=row['quality_receipt_sha256'])
        if 'asset_manifest' in row:
            if row['asset_manifest']['model_identity']!=identity:raise ValueError('Wrong model revision')
            plan=a.package/row['name']/'plan.json';shutil.copyfile(plan,folder/'plan.json');config['budget_plan']=str((folder/'plan.json').resolve())
            job['budget_plan_sha256']=sha(folder/'plan.json');bank=folder/'assets';bank.mkdir()
            manifest={**row['asset_manifest'],'complete':False,'modules':{}}
            banks[row['name']]=(bank,manifest);job['assets']=str(bank.resolve())
        write(folder/'config.json',config);job['config_sha256']=sha(folder/'config.json');jobs.append(job)
    names=sorted({name for row in recipe['jobs'] for name in row.get('modules',{})})
    for count,name in enumerate(names):
        key=name+'.weight'
        with safe_open(a.model/'transformer'/index[key],framework='pt',device='cpu') as handle:weight=handle.get_tensor(key).to(torch.bfloat16)
        cache={}
        from thunderworld_fp4.cosmos.calibration import quantize_weight
        for row in recipe['jobs']:
            if name not in row.get('modules',{}):continue
            entry=row['modules'][name];meta=torch.load(a.package/entry['file'],weights_only=True,map_location='cpu')
            # Share encoding only; each policy retains its exact support and metadata.
            cache_key=(tensor_hash(meta['prescale']),tuple(meta['weight_scale']))
            if cache_key not in cache:cache[cache_key]=quantize_weight(weight,meta['prescale'],base=cache_key[1][0],residual=cache_key[1][1])
            asset=rebuild(weight,meta,cache[cache_key])
            if entry.get('rounding_patch'):
                asset={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in asset.items()}
                patch=torch.load(a.package/entry['rounding_patch'],weights_only=True,map_location='cpu')
                for key,(indices,values) in patch.items():asset[key].reshape(-1).view(torch.uint8)[indices]=values
            hashes={k:tensor_hash(v) for k,v in asset.items() if isinstance(v,torch.Tensor)}
            if hashes!=entry['tensor_hashes']:raise ValueError('Reconstructed tensor mismatch: '+row['name']+'/'+name)
            bank,manifest=banks[row['name']];path=bank/Path(entry['file']).name;torch.save(asset,path)
            manifest['modules'][name]=dict(file=path.name,sha256=sha(path))
        print(json.dumps(dict(reconstructed=name,completed=count+1,total=len(names))),flush=True)
    for job in jobs:
        if job['name'] in banks:
            bank,manifest=banks[job['name']];manifest['complete']=True;write(bank/'manifest.json',manifest);job['assets_sha256']=sha(bank/'manifest.json')
    write(a.output/'retest.json',dict(frozen=True,quality_selection_sha256=recipe['selection_sha256'],
        package_sha256=sha(a.package/'package.json'),model=str(a.model.resolve()),case=str((a.output/'case.json').resolve()),jobs=jobs,
        scope='All packed tensor bytes matched the frozen source. Only paths and serialization hashes changed. Per-device tuning still required.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    e=sub.add_parser('export');e.add_argument('--manifest',type=Path,required=True);e.add_argument('--selection',type=Path,required=True)
    r=sub.add_parser('restore');r.add_argument('--package',type=Path,required=True);r.add_argument('--model',type=Path,required=True);r.add_argument('--threads',type=int,default=8)
    for parser in (e,r):parser.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(export if a.operation=='export' else restore)(a)


if __name__=='__main__':main()

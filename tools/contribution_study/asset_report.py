"""Measure physical offline assets; do not confuse payload bytes with GPU peaks."""
import argparse,json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidates',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    candidates=json.loads(a.candidates.read_text())
    if not candidates['complete']:raise ValueError('Incomplete candidates')
    result=dict(complete=False,candidates_sha256=sha(a.candidates),rows=[],performance_eligible=False,
        scope='Measured files and tensor payload, not allocator reserved or whole-model peak VRAM. '
              'Full-W1 control preserves the exact support and arithmetic but retains unused residual blocks.')
    for candidate in candidates['candidates']:
        root=Path(candidate['path']);bank=root/'assets';manifest=json.loads((bank/'manifest.json').read_text())
        plan=json.loads((root/'plan.json').read_text());ledger=json.loads((root/'compute.json').read_text())
        totals=dict(file_bytes=0,base_bytes=0,residual_bytes=0,index_bytes=0,auxiliary_bytes=0,full_residual_bytes=0,mandatory_blocks=0)
        all_blocks=kept_blocks=0
        for name,entry in manifest['modules'].items():
            path=bank/entry['file']
            if sha(path)!=entry['sha256']:raise ValueError('Asset hash mismatch')
            asset=torch.load(path,weights_only=True,map_location='cpu')
            if 'codes' in asset or 'scales' in asset:raise ValueError('Hidden full residual in compact asset')
            union=set(i for key,row in plan['linears'].items() if key.split('|',2)[-1]==name for i in row['w_blocks'])
            if set(asset['residual_indices'].tolist())!=union:raise ValueError('Asset is not the exact phase union')
            for key,row in plan['linears'].items():
                if key.split('|',2)[-1]==name and not set(asset.get('mandatory_w_blocks',())).issubset(row['w_blocks']):
                    raise ValueError('Mandatory support absent in a phase')
            all_blocks+=asset['score'].numel();kept_blocks+=len(union)
            totals['file_bytes']+=path.stat().st_size
            totals['mandatory_blocks']+=len(asset.get('mandatory_w_blocks',()))
            for key,value in asset.items():
                if not isinstance(value,torch.Tensor):continue
                field='base_bytes' if key in ('codes_base','scales_base') else 'residual_bytes' if key in ('codes_residual','scales_residual') else 'index_bytes' if key=='residual_indices' else 'auxiliary_bytes'
                totals[field]+=value.numel()*value.element_size()
            totals['full_residual_bytes']+=sum(asset[k].numel()*asset[k].element_size() for k in ('codes_base','scales_base'))
        totals['payload_bytes']=sum(totals[k] for k in ('base_bytes','residual_bytes','index_bytes','auxiliary_bytes'))
        totals['same_support_full_w1_payload_bytes']=totals['payload_bytes']+totals['full_residual_bytes']-totals['residual_bytes']
        result['rows'].append(dict(name=candidate['name'],budget=candidate['budget'],ratio=candidate['ratio'],
            union_density=kept_blocks/all_blocks,bytes=totals,compute_ledger=ledger,
            plan_sha256=sha(root/'plan.json'),assets_sha256=sha(bank/'manifest.json'),quality_qualified=False))
        write(a.output,result)
    result['complete']=True;write(a.output,result)


if __name__=='__main__':main()

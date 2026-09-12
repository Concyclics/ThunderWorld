"""Bind quality/performance to input contents, model, projection scope and assets."""
import argparse
import hashlib
import json
from pathlib import Path


def provenance(run):
    case=dict(run['case']);image=case.pop('image',None)
    if image and not run.get('image_sha256'):raise ValueError('Missing input image content hash')
    return dict(config=run['config'],model_identity=run['model_identity'],targets=run['targets'],
        case=case,image_sha256=run.get('image_sha256'),asset_identity=run.get('asset_identity'),
        ffn_policy_identity=run.get('ffn_policy_identity'),
        numerical_environment={k:run.get('environment',{}).get(k) for k in ('torch','triton','cuda','kernels')})


def main():
    p=argparse.ArgumentParser(description='Add input provenance without recomputing or changing any metrics')
    p.add_argument('matrix',type=Path);p.add_argument('quality',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    matrix=json.loads(a.matrix.read_text());quality=json.loads(a.quality.read_text())
    if not matrix['complete'] or not quality['complete']:raise ValueError('Completed inputs required')
    if quality['matrix_sha256']!=hashlib.sha256(a.matrix.read_bytes()).hexdigest():raise ValueError('Different original matrix')
    quality['metric_source_sha256']=hashlib.sha256(a.quality.read_bytes()).hexdigest()
    quality['provenance']={j['case']+'|'+j['config']:provenance(json.loads((Path(j['output'])/'result.json').read_text()))
        for j in matrix['jobs'] if j['config']!='bf16'}
    a.output.write_text(json.dumps(quality,indent=2)+'\n')


if __name__=='__main__':main()

"""Bind a new Thor expansion campaign to immutable v8 assets/inputs."""
import argparse
import copy
import json
from pathlib import Path
from expand_worker import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixed-root',type=Path,required=True)
    p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for model in ('edge','nano'):
        path=a.fixed_root/model/'fixed-model.json'
        fixed=json.loads(path.read_text())
        fixed=copy.deepcopy(fixed)
        fixed['expanded_parent_manifest_sha256']=sha(path)
        for name,spec in fixed['methods'].items():
            if spec['config']['linear_backend']=='cutlass-fused':
                spec['config']['cutlass_library']=str(a.variant/'cutlass_sm110.so')
                spec['config']['cutlass_small_library']=None
        target=a.output/'manifests'/(model+'.json')
        if target.exists():raise FileExistsError(target)
        write(target,fixed)
    jobs=[(model,method,profile) for model in ('edge','nano') for method,profile in
        [('bf16','original'),('bf16','trim'),('b1.5_protected_n','original'),
         ('b1.5_protected_n','trim'),('b1.5_protected_n','ar4_embed2'),
         ('b1.5_protected_n','full2'),('b2_joint','original'),('b2_joint','full2')]]
    write(a.output/'jobs.json',jobs)
    print(json.dumps(dict(output=str(a.output),jobs=len(jobs),quality_evaluation=False)))


if __name__=='__main__':main()

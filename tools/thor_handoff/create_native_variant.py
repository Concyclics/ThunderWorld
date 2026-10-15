"""Create an isolated, hash-recorded Thor package after kernel qualification.

The installed package and any active historical benchmark remain untouched.
The variant includes native-attention M128 in the actual MMA/padding ledger.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import time
from build_baselines import write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--name',default='transposed-attention-v2')
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--gemm-qualification',type=Path)
    a=p.parse_args()
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.name):raise ValueError('Invalid variant name')
    kernel=Path(__file__).with_name('native_attention.py')
    qualified=json.loads(a.qualification.read_text())
    if not qualified['complete'] or qualified['kernel_sha256']!=sha(kernel):raise ValueError('Current native kernel is not qualified')
    tile=qualified['query_tile'];warps=qualified['num_warps'];transpose_v=qualified.get('transpose_v',True)
    gemm_qualified=None
    gemm_kernel=Path(__file__).with_name('tma_gemm.py')
    if a.gemm_qualification:
        gemm_qualified=json.loads(a.gemm_qualification.read_text())
        if (not gemm_qualified['complete'] or not gemm_qualified.get('tma') or
            gemm_qualified['kernel_sha256']!=sha(gemm_kernel) or len(gemm_qualified['rows'])<35 or
            not gemm_qualified.get('current_inputs_graph') or not gemm_qualified.get('zero_input')):
            raise ValueError('Current TMA GEMM does not have complete qualification')
    source=a.root/'src/ThunderWorld/packages/fp4-repair/src/thunderworld_fp4'
    target=a.root/'variants'/a.name/'thunderworld_fp4'
    if target.parent.exists():raise FileExistsError(target.parent)
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so'))
    changed={}
    def edit(rel,old,new):
        path=target/rel
        text=path.read_text()
        if text.count(old)!=1:raise ValueError(f'Unexpected source pattern {rel}: {old[:60]}')
        changed.setdefault(rel,text)
        path.write_text(text.replace(old,new))
    shutil.copyfile(kernel,target/'_sm110_attention.py')
    if gemm_qualified:
        shutil.copyfile(gemm_kernel,target/'_sm110_gemm.py')
        edit('_native.py','class Native:',
             '_reference_gemm=gemm\nfrom ._sm110_gemm import gemm\n\nclass Native:')
    edit('config.py','self.query_tile not in (16, 32, 64)','self.query_tile not in (16, 32, 64, 128)')
    edit('config.py','query_tile must be None, 16, 32, or 64; BM128 failed validation',
         'query_tile must be None, 16, 32, 64, or Thor-native 128')
    edit('runtime.py','    _validate(q, k, v)\n    if not isinstance(config, RepairConfig):',
         f'    _validate(q, k, v)\n    if torch.cuda.get_device_capability(q.device)!=(11,0):\n        raise RuntimeError("Isolated Thor-native attention variant requires SM110")\n    if config.query_tile!={tile}:\n        raise ValueError("Thor-native attention requires the qualified query tile {tile}")\n    if not isinstance(config, RepairConfig):')
    edit('runtime.py','from ._kernels import pack_inputs, select_tiles, select_tiles_head, streaming_attention, combine',
         'from ._kernels import pack_inputs, select_tiles, select_tiles_head, combine\nfrom ._sm110_attention import streaming_attention')
    edit('cosmos/engine.py','        return cfg\n',f'        return replace(cfg,query_tile={tile},num_warps={warps},transpose_v={transpose_v},reload_q=True)\n')
    edit('cosmos/engine.py','for b in (16,32,64) for w in (4,8)', f'for b in ({tile},) for w in ({warps},)')
    edit('cosmos/budget.py','bm not in (16,32,64)', 'bm not in (16,32,64,128)')
    edit('cosmos/tuning.py',"paths = list(root.glob('_*.py'))", "paths = [root/'config.py',root/'runtime.py'] + list(root.glob('_*.py'))")
    patch=''.join(''.join(difflib.unified_diff(old.splitlines(True),(target/rel).read_text().splitlines(True),
        fromfile='a/'+rel,tofile='b/'+rel)) for rel,old in changed.items())
    (target.parent/'variant.patch').write_text(patch)
    receipt=dict(created=time.time(),scope='Thor speed only; native FP4 arithmetic, frozen repair selections and weights',
        source=str(source),target=str(target),qualification=dict(path=str(a.qualification),sha256=sha(a.qualification)),
        kernel_sha256=sha(kernel),original_hashes={rel:sha(source/rel) for rel in changed},
        variant_hashes={str(path.relative_to(target)):sha(path) for path in sorted(target.rglob('*.py'))},
        accumulation_note='Native NVFP4 replaces BF16-emulated dot_scaled; numerical parity, not bitwise identity',
        query_tile=tile,num_warps=warps,transpose_v=transpose_v,
        gemm='tma-double-buffer' if gemm_qualified else 'unchanged',quality_evaluation=False)
    if gemm_qualified:
        receipt['gemm_kernel_sha256']=sha(gemm_kernel)
        receipt['gemm_qualification']=dict(path=str(a.gemm_qualification),sha256=sha(a.gemm_qualification))
    write(target.parent/'variant.json',receipt)
    print(json.dumps(dict(variant=str(target.parent),kernel_sha256=sha(kernel))),flush=True)


if __name__=='__main__':main()

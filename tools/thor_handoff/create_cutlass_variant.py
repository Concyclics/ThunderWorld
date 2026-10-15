"""Build a qualified, isolated Thor CUTLASS/attention package.

Never changes the installed package, frozen weights, policies or old results.
The library and all Python sources are pinned in the deployment receipt.
"""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import time
from build_baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--memcheck',type=Path,required=True)
    p.add_argument('--racecheck',type=Path,required=True)
    p.add_argument('--attention-qualification',type=Path)
    p.add_argument('--attention-memcheck',type=Path)
    p.add_argument('--attention-racecheck',type=Path)
    p.add_argument('--fused',action='store_true',help='Use the single CUTLASS mainloop for base and repairs')
    a=p.parse_args()
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in a.name):raise ValueError('Invalid name')
    here=Path(__file__).parent
    old=json.loads((a.parent/'variant.json').read_text())
    source=a.parent/'thunderworld_fp4'
    for name,digest in old['variant_hashes'].items():
        if sha(source/name)!=digest:raise ValueError(f'Parent changed: {name}')
    qualified=json.loads(a.qualification.read_text())
    if bool(qualified.get('fused'))!=a.fused:raise ValueError('Qualification backend mismatch')
    if not qualified['complete'] or len(qualified['rows'])<42 or len(qualified['sparse_packing'])<18:
        raise ValueError('Incomplete GEMM qualification')
    if sum(row.get('current_inputs_graph',False) for row in qualified['rows'])<6:
        raise ValueError('Missing GEMM Graph qualification')
    if a.fused and not all(row.get('previous_bitwise_equal') for row in qualified['rows']):
        raise ValueError('Mainloop fusion must be bitwise equivalent to the qualified split implementation')
    for name,digest in qualified['sources'].items():
        if sha(here/name)!=digest:raise ValueError(f'Unqualified source: {name}')
    if sha(a.library)!=qualified['library_sha256']:raise ValueError('Unqualified library')
    build=json.loads(a.library.with_suffix('.build.json').read_text())
    for name,digest in build['sources'].items():
        if qualified['sources'].get(name)!=digest:raise ValueError('Build and qualification sources differ')
    backend='cutlass-mainloop-fused' if a.fused else 'cutlass-base-fp32-repair-tail'
    def sanitizer(path,source_key=None,source_hash=None):
        if path is None:raise ValueError('Sanitizer receipt required')
        row=json.loads(path.read_text())
        if not row['complete']:raise ValueError('Incomplete sanitizer run')
        if source_key and row[source_key]!=source_hash:raise ValueError('Sanitizer source mismatch')
        log=path.with_suffix('.log').read_text()
        expected='RACECHECK SUMMARY: 0 hazards' if 'racecheck' in path.name else 'ERROR SUMMARY: 0 errors'
        if expected not in log:raise ValueError(f'Missing clean sanitizer summary: {path}')
    for path in (a.memcheck,a.racecheck):
        sanitizer(path,'library_sha256',sha(a.library))
        if json.loads(path.read_text())['sources']!=qualified['sources']:raise ValueError('Sanitizer source mismatch')
    attn=None
    if a.attention_qualification:
        attn=json.loads(a.attention_qualification.read_text())
        kernel=here/'pipelined_attention.py'
        if not attn['complete'] or attn['kernel_sha256']!=sha(kernel) or len(attn['tests'])<24:
            raise ValueError('Incomplete attention qualification')
        if not all(r.get('native_v3_bitwise_equal') for r in attn['tests']):raise ValueError('Attention must be bitwise equivalent')
        for path in (a.attention_memcheck,a.attention_racecheck):sanitizer(path,'kernel_sha256',sha(kernel))
    target=a.root/'variants'/a.name/'thunderworld_fp4'
    if target.parent.exists():raise FileExistsError(target.parent)
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so'))
    changed={}
    def edit(rel,before,after):
        path=target/rel;text=path.read_text()
        if text.count(before)!=1:raise ValueError(f'Unexpected source: {rel}: {before[:80]}')
        changed.setdefault(rel,text);path.write_text(text.replace(before,after))
    lib=target.parent/'cutlass_sm110.so'
    shutil.copyfile(a.library,lib)
    shutil.copyfile(a.library.with_suffix('.build.json'),lib.with_suffix('.build.json'))
    shutil.copyfile(here/('cutlass_thor_fused_backend.py' if a.fused else 'cutlass_thor_backend.py'),target/'_sm110_cutlass.py')
    if not a.fused:shutil.copyfile(here/'cutlass_repair_tail.py',target/'_sm110_repair_tail.py')
    edit('cosmos/linears.py','        self.ops[key]=op;self.bound=op',
        '''        from .._sm110_cutlass import CutlassThor
        from pathlib import Path
        op=CutlassThor(op,m,Path(__file__).resolve().parents[2]/'cutlass_sm110.so')
        if self.ops:op.w0_scales=next(iter(self.ops.values())).w0_scales
        # Represented-gain selection retains its existing native packing.
        if self.repair_score!='residual_energy':op.direct_scale_pack=False
        self.controller.cutlass_artifact=op.artifact
        self.ops[key]=op;self.bound=op''')
    edit('cosmos/linears.py',
        '[(128,2),(32,2),(32,3),(64,2),(64,3),(128,3)]],apply,lambda:self._compute(flat))',
        '[(128,2)]],apply,lambda:self._compute(flat))')
    edit('cosmos/engine.py',"backend='cutlass-fused' if hasattr(op,'tile_n') else 'triton'",
        f"backend='{backend}' if hasattr(op,'tile_n') else 'triton'")
    edit('cosmos/engine.py',"schedule='TMA cooperative + sparse cp.async'",
        "schedule='CUTLASS TMA base/repair stages, shared TMEM accumulator'" if a.fused else "schedule='CUTLASS TMA base + FP32-seeded FP4 repair kernel'")
    edit('cosmos/engine.py',"cost['execution_backend']='cutlass-fused' if hasattr(native,'tile_n') else 'triton'",
        f"cost['execution_backend']='{backend}' if hasattr(native,'tile_n') else 'triton'")
    edit('cosmos/engine.py',"                        cost['cutlass_fp32_gmem_scratch_bytes']=0",'''                        repaired=native.na>0 or native.hasw
                        cost['cutlass_fp32_gmem_scratch_bytes']=4*m*n if repaired else 0
                        cost['cutlass_fp32_intermediate_read_write_bytes']=8*m*n if repaired else 0
                        cost['gemm_kernel_launches']=2 if repaired else 1
                        cost['cutlass_workspace_bytes']=native.workspace.numel()
                        cost['accumulation_order']='base then A/W/cross; one BF16 epilogue' ''')
    if a.fused:
        edit('cosmos/engine.py',"cost['cutlass_fp32_gmem_scratch_bytes']=4*m*n if repaired else 0", "cost['cutlass_fp32_gmem_scratch_bytes']=0")
        edit('cosmos/engine.py',"cost['cutlass_fp32_intermediate_read_write_bytes']=8*m*n if repaired else 0", "cost['cutlass_fp32_intermediate_read_write_bytes']=0")
        edit('cosmos/engine.py',"cost['gemm_kernel_launches']=2 if repaired else 1", "cost['gemm_kernel_launches']=1")
    if attn:
        changed['_sm110_attention.py']=(target/'_sm110_attention.py').read_text()
        shutil.copyfile(here/'pipelined_attention.py',target/'_sm110_attention.py')
    patch=''.join(''.join(difflib.unified_diff(before.splitlines(True),(target/name).read_text().splitlines(True),
        fromfile='a/'+name,tofile='b/'+name)) for name,before in changed.items())
    (target.parent/'variant.patch').write_text(patch)
    receipt=dict(created=time.time(),scope='Thor speed only; current activation and frozen repair/weight assets',
        parent=dict(path=str(a.parent),sha256=sha(a.parent/'variant.json')),
        backend=backend,kernel_sha256=sha(target/'_sm110_attention.py'),
        qualification=dict(path=str(a.qualification),sha256=sha(a.qualification)),
        sanitizer_receipts=[dict(path=str(p),sha256=sha(p)) for p in (a.memcheck,a.racecheck)],
        library_sha256=sha(lib),build_receipt_sha256=sha(lib.with_suffix('.build.json')),
        variant_hashes={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*.py'))},
        accumulation_note=qualified['accumulation'],query_tile=old['query_tile'],num_warps=old['num_warps'],
        transpose_v=old['transpose_v'],quality_evaluation=False,
        attention='grouped asynchronous FP4 MMA' if attn else 'native v3 unchanged')
    if attn:receipt['attention_qualification']=dict(path=str(a.attention_qualification),sha256=sha(a.attention_qualification))
    write(target.parent/'variant.json',receipt)
    print(json.dumps(dict(variant=str(target.parent),backend=receipt['backend'])),flush=True)


if __name__=='__main__':main()

"""Load the archived handoff kernels for honest same-process comparisons.

Only immutable prepared FFN weights are shared. The three unchanged activation
producer sources are checked byte-for-byte before imports share that package.
"""
import hashlib
from pathlib import Path
from bench import ROOT, load_module


def load():
    root=ROOT/'data_local/baseline_source'
    if not root.exists():
        raise FileNotFoundError('Create data_local/baseline_source with git archive 9b51f0d first; see RTX5090_RESULTS.md')
    package=Path('packages/fp4-repair/src/thunderworld_fp4')
    for name in ['linear.py','_linear_kernels.py','_quantize.py']:
        if (ROOT/package/name).read_bytes()!=(root/package/name).read_bytes():
            raise ValueError(f'Frozen baseline requires its own producer implementation after changing {name}')
    kernels=load_module('thunderworld_fp4._frozen_kernels',root/package/'_kernels.py')
    runtime=load_module('thunderworld_fp4._frozen_runtime',root/package/'runtime.py')
    for name in ['pack_inputs','select_tiles','select_tiles_head','streaming_attention','combine']:
        setattr(runtime,name,getattr(kernels,name))
    native=load_module('frozen_ffn_native',root/'experiments/ffn_block_repair_20260907/native.py')
    sources=[root/'experiments/ffn_block_repair_20260907/native.py',*(root/package).glob('*.py')]
    return native,runtime,{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def share_weights(op,baseline_class):
    original=object.__new__(baseline_class)
    original.__dict__.update(op.__dict__)
    original.bm=32
    original.interleave=True
    return original

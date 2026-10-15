"""Thor CUTLASS NVFP4 mainloop-fused sparse repair; no global FP32 intermediate.

Construction and weight-scale conversion are offline. Replay reads current
activations and masks. No BF16 GEMM fallback or online weight quantization.
"""
import ctypes
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import torch
import triton
from thunderworld_fp4.cutlass.layout import swizzle
from thunderworld_fp4.cutlass.packing import CutlassPacked,pack_input
class RepairIO(ctypes.Structure):
    _fields_=[(name,ctypes.c_void_p) for name in ('ac','ascale','wc','ws','wr','wrs','wi','off','wloc','mask')]+[
        (name,ctypes.c_int) for name in ('m','mp','n','k','na','hasw','cross','shared','dense','bank_blocks')]


@lru_cache(maxsize=4)
def load_library(path):
    p=Path(path)
    meta=json.loads(p.with_suffix('.build.json').read_text())
    if meta['architecture']!='sm_110a' or meta['tile']!=[128,128,128] or meta['backend']!='cutlass-sm110-fused-mainloop':
        raise ValueError('Expected fixed M128/N128/K128 Thor build')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['library_sha256']:
        raise ValueError('CUTLASS artifact changed')
    lib=ctypes.CDLL(str(p))
    lib.tw_workspace.argtypes=[ctypes.c_int]*3;lib.tw_workspace.restype=ctypes.c_size_t
    lib.tw_fused.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*8+[ctypes.POINTER(RepairIO)]
    lib.tw_fused.restype=ctypes.c_int
    lib.tw_status.argtypes=[ctypes.c_int];lib.tw_status.restype=ctypes.c_char_p
    return lib,meta


class CutlassThor:
    def __init__(self,op,m,library):
        if torch.cuda.get_device_capability(op.wc.device)!=(11,0):raise ValueError('SM110 required')
        if op.b!=128 or op.bm!=128 or op.split_k!=1:raise ValueError('Fixed M/N/K128 required')
        self.op=op;self.m=m;self.direct_scale_pack=True;self.tile_n=128;self.mainloop_fused=True
        self.lib,meta=load_library(str(library))
        self.artifact={k:meta[k] for k in ('backend','library_sha256','tile','sources')}
        self.w0_scales=swizzle(op.ws)
        self.workspace=torch.empty(max(1,self.lib.tw_workspace(m,op.n,op.k)),device=op.wc.device,dtype=torch.uint8)

    def __getattr__(self,name):return getattr(self.op,name)

    def __call__(self,x):return self.run_packed(x,pack_input(x,self.op))

    def run_packed(self,x,packed):
        op=self.op
        if x.shape!=(self.m,op.k) or x.dtype!=torch.bfloat16 or x.device!=op.wc.device:
            raise ValueError('Unprepared shape/dtype/device')
        direct=packed if isinstance(packed,CutlassPacked) else None
        ac,sc,ag,mp=direct.native if direct else packed
        if mp!=triton.cdiv(self.m,16)*16 or ac.shape[1:]!=(mp,op.k//2):
            raise ValueError('Invalid packed input shape')
        if ac.dtype!=torch.uint8 or sc.dtype!=torch.float8_e4m3fn or sc.shape!=(ac.shape[0],mp,op.k//16):
            raise ValueError('Invalid code/scale dtype or shape')
        if op.na and ac.shape[0]<2:raise ValueError('Missing current activation repair plane')
        if any(t.device!=x.device or not t.is_contiguous() for t in (ac,sc,ag)):
            raise ValueError('Packed buffers must be contiguous on the prepared device')
        if direct:
            scales,alpha=direct.scales,direct.alpha
        else:
            alpha=torch.empty((),device=x.device,dtype=torch.float32)
            scales=swizzle(sc[0],ag,op.wg,alpha)
        if scales.shape!=(triton.cdiv(self.m,128)*128,op.k//16):raise ValueError('Invalid CUTLASS scale layout shape')
        output=torch.empty((self.m,op.n),device=x.device,dtype=torch.bfloat16)
        args=RepairIO(*[t.data_ptr() for t in (ac,sc,op.wc,op.ws,op.wr,op.wrs,op.wi,op.off,op.wloc,op.am)],
            self.m,mp,op.n,op.k,op.na,int(op.hasw),int(op.cross),int(op.shared_bank),int(op.dense_w),op.wr.shape[0])
        status=self.lib.tw_fused(self.m,op.n,op.k,ac.data_ptr(),op.wc.data_ptr(),scales.data_ptr(),self.w0_scales.data_ptr(),
            output.data_ptr(),alpha.data_ptr(),self.workspace.data_ptr(),torch.cuda.current_stream(x.device).cuda_stream,ctypes.byref(args))
        if status:raise RuntimeError(self.lib.tw_status(status).decode())
        return output

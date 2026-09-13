"""Single CUTLASS launch: dense TMA mainloop, sparse CuTe MMA, one epilogue."""
import ctypes
import json
from pathlib import Path
import torch
import triton
from .layout import swizzle


class RepairIO(ctypes.Structure):
    _fields_=[(name,ctypes.c_void_p) for name in ('ac','ascale','wc','ws','wr','wrs','wi','off','wloc','mask')]+[
        (name,ctypes.c_int) for name in ('m','mp','n','k','na','hasw','cross','shared','dense')]


class CutlassFused:
    def __init__(self,op,m,library):
        if op.b!=128 or op.bm!=128 or op.split_k!=1:
            raise ValueError('Fused CUTLASS requires M128/K128 and unsplit accumulation')
        if torch.cuda.get_device_capability(op.wc.device)!=(12,0):raise ValueError('SM120 required')
        self.op=op;self.m=m
        meta=json.loads(Path(library).with_suffix('.build.json').read_text())
        self.tile_n=meta['tile'][1]
        self.artifact={k:meta[k] for k in ('backend','library_sha256','sources','tile')}
        self.lib=ctypes.CDLL(str(library))
        self.lib.tw_workspace.argtypes=[ctypes.c_int]*3;self.lib.tw_workspace.restype=ctypes.c_size_t
        self.lib.tw_fused.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*8+[ctypes.POINTER(RepairIO)]
        self.lib.tw_fused.restype=ctypes.c_int
        self.lib.tw_status.argtypes=[ctypes.c_int];self.lib.tw_status.restype=ctypes.c_char_p
        self.w0_scales=swizzle(op.ws)
        self.workspace=torch.empty(max(1,self.lib.tw_workspace(m,op.n,op.k)),device=op.wc.device,dtype=torch.uint8)

    def __getattr__(self,name):return getattr(self.op,name)

    def __call__(self,x):return self.run_packed(x,self.op.pack(x))

    def _validate(self,packed):
        ac,sc,ag,mp=packed;op=self.op
        planes=2 if op.na else 1
        if mp!=triton.cdiv(self.m,16)*16 or ac.ndim!=3 or ac.shape[0]<planes:
            raise ValueError('Invalid packed M or missing activation residual plane')
        if ac.shape[1:]!=(mp,op.k//2) or sc.shape!=(ac.shape[0],mp,op.k//16):
            raise ValueError('Packed code/scale shape mismatch')
        if ac.dtype!=torch.uint8 or sc.dtype!=torch.float8_e4m3fn or ag.dtype!=torch.float32 or ag.numel()!=1:
            raise ValueError('Invalid packed dtypes/factor')
        if any(t.device!=op.wc.device or not t.is_contiguous() for t in (ac,sc,ag)):
            raise ValueError('Packed inputs must be contiguous on the prepared device')

    def run_packed(self,x,packed):
        if x.shape!=(self.m,self.op.k) or x.dtype!=torch.bfloat16 or x.device!=self.op.wc.device:
            raise ValueError('Input must match prepared BF16 shape/device')
        self._validate(packed)
        ac,sc,ag,mp=packed;op=self.op
        alpha=torch.empty((),device=x.device,dtype=torch.float32)
        scales=swizzle(sc[0],ag,op.wg,alpha)
        output=torch.empty((self.m,op.n),device=x.device,dtype=torch.bfloat16)
        args=RepairIO(*[t.data_ptr() for t in (ac,sc,op.wc,op.ws,op.wr,op.wrs,op.wi,op.off,op.wloc,op.am)],
            self.m,mp,op.n,op.k,op.na,int(op.hasw),int(op.cross),int(op.shared_bank),int(op.dense_w))
        status=self.lib.tw_fused(self.m,op.n,op.k,ac.data_ptr(),op.wc.data_ptr(),scales.data_ptr(),self.w0_scales.data_ptr(),
            output.data_ptr(),alpha.data_ptr(),self.workspace.data_ptr(),torch.cuda.current_stream(x.device).cuda_stream,ctypes.byref(args))
        if status:raise RuntimeError(self.lib.tw_status(status).decode())
        return output

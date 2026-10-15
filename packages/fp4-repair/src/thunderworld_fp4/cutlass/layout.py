"""Exact byte permutation to CUTLASS block-scale layout, with fused alpha."""
import torch
import triton
import triton.language as tl

@triton.jit
def _swizzle(S,O,AG,WG,ALPHA,ROWS:tl.constexpr,COLS:tl.constexpr,SIZE:tl.constexpr,BLOCK:tl.constexpr,FACTOR:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//(128*COLS)*128+(i//4%4)*32+i//16%32
    col=i//512%(COLS//4)*4+i%4
    v=tl.load(S+row*COLS+col,(i<SIZE)&(row<ROWS),0)
    tl.store(O+i,v,i<SIZE)
    if FACTOR:
        if tl.program_id(0)==0:tl.store(ALPHA,tl.load(AG)*tl.load(WG))


def swizzle(scales,ag=None,wg=None,alpha=None):
    if scales.dtype!=torch.float8_e4m3fn or not scales.is_contiguous():
        raise ValueError('Contiguous E4M3 scales required')
    rows,cols=scales.shape
    if cols%4:raise ValueError('Scale columns must be divisible by four')
    out=torch.empty((triton.cdiv(rows,128)*128,cols),device=scales.device,dtype=scales.dtype)
    _swizzle[(triton.cdiv(out.numel(),256),)](scales.view(torch.uint8),out.view(torch.uint8),ag,wg,alpha,rows,cols,out.numel(),256,alpha is not None)
    return out

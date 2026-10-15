"""Record the actual Thor architecture, FP8 reference parity and native FP4 PTX."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


def decode(codes, scales):
    import torch
    ids = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0,.5,1,1.5,2,3,4,6], device=codes.device)
    return lut[ids & 7] * torch.where(ids >= 8, -1, 1) * scales.float().repeat_interleave(16,-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    import triton
    from thunderworld_fp4.linear import _pack
    from thunderworld_fp4._native import gemm, Native
    from vllm import _custom_ops as ops
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(813)
    prop = torch.cuda.get_device_properties(0)
    if (prop.major, prop.minor) != (11,0):
        raise RuntimeError('Not Thor SM110')
    report = dict(complete=False, timestamp=time.time(), machine=platform.machine(),
        torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
        gpu=prop.name, capability=[prop.major,prop.minor], multiprocessors=prop.multi_processor_count,
        torch_total_memory_bytes=prop.total_memory, tests=[], performance_eligible=False)
    a.output.parent.mkdir(exist_ok=True, parents=True)
    with torch.inference_mode():
        x = torch.randn(129,384,device='cuda',dtype=torch.bfloat16)
        w = torch.randn(256,384,device='cuda',dtype=torch.bfloat16)
        y = x@w.T
        assert torch.isfinite(y).all()
        report['tests'].append(dict(name='bf16_gemm', passed=True))
        q,sa = ops.scaled_fp8_quant(x,scale=None,use_per_token_if_dynamic=True)
        r,sb = ops.scaled_fp8_quant(w,scale=None)
        y = ops.cutlass_scaled_mm(q,r.T,scale_a=sa,scale_b=sb,out_dtype=torch.bfloat16)
        ref = (q.float()*sa)@(r.float()*sb).T
        error = float((y.float()-ref).norm()/ref.norm())
        assert error < .004
        report['tests'].append(dict(name='native_fp8_decoded_reference', relative_l2=error, passed=True))
        d = torch.ones(384,device='cuda')
        wc,ws,wg,_ = _pack(w,d,True,True,w)
        for m in (129,416):
            x = torch.randn(m,384,device='cuda',dtype=torch.bfloat16)
            ac,asc,ag,mp = _pack(x,d,True,True,x)
            av = decode(ac,asc)[:,:m];wv = decode(wc,ws)
            for amode,wmode,cross in ((0,0,False),(1,0,False),(0,1,False),(1,1,False),(1,1,True),(.5,.5,True)):
                am = torch.zeros(((m+127)//128,384),device='cuda',dtype=torch.bool)
                wm = torch.zeros_like(w,dtype=torch.bool)
                am[:,:384 if amode==1 else 128 if amode else 0] = True
                wm[:,:384 if wmode==1 else 256 if wmode else 0] = True
                aa = av[1]*am.repeat_interleave(128,0)[:m]
                ww = wv[1]*wm
                reference = av[0]@wv[0].T
                if amode: reference += aa@wv[0].T
                if wmode: reference += av[0]@ww.T
                if cross: reference += aa@ww.T
                reference *= ag*wg
                for bm in (32,64,128):
                    op = Native(wc,ws,wg,d,am,wm,bm=bm,cross=cross,num_stages=1)
                    y = op(x)
                    error = float((y.float()-reference).norm()/reference.norm())
                    assert torch.isfinite(y).all() and error < .004, (m,amode,wmode,cross,bm,error)
                    report['tests'].append(dict(name='fp4_decoded_reference', m=m, a=amode, w=wmode,
                        cross=cross, bm=bm, relative_l2=error, passed=True))
        # Keep evidence of an actual native FP4 instruction from our kernel.
        x = torch.randn(128,384,device='cuda',dtype=torch.bfloat16)
        ac,asc,ag,mp = _pack(x,d,False,False,x)
        y = torch.empty((128,256),device='cuda',dtype=torch.bfloat16)
        idx = torch.empty(0,device='cuda',dtype=torch.int32)
        mask = torch.zeros((1,3),device='cuda',dtype=torch.bool)
        kernel = gemm[(1,2,1)](ac,asc,wc[0],ws[0],wg,ag,idx,mask,wc[1],ws[1],idx,idx,y,
            128,256,384,mp,0,False,False,128,128,True,1,False,
            WLOC=idx,SHARED_BANK=False,num_warps=4,num_stages=1)
        torch.cuda.synchronize()
        ptx = kernel.asm['ptx']
        instructions = sorted(set(line.strip() for line in ptx.splitlines()
                                  if 'tcgen05.mma' in line and 'mxf4nvf4' in line))
        assert instructions, 'No native FP4 Tensor Core instructions'
        path = a.output.with_suffix('.ptx')
        path.write_text(ptx)
        report['fp4_instruction_evidence'] = dict(file=path.name,
            sha256=hashlib.sha256(ptx.encode()).hexdigest(), instructions=instructions)
    report['complete'] = True
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True, tests=len(report['tests']), output=str(a.output))),flush=True)


if __name__ == '__main__':
    main()

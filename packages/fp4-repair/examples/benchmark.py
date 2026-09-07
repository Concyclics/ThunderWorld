"""CUDA Graph latency, including online activation quantization.

Run on a permitted, preferably idle SM120 device. These are operator timings,
not a model speedup claim. The FP4Linear weights are prepared offline by RTN.
"""
import argparse
import json
import torch
import torch.nn.functional as F
import triton
import triton.testing
from thunderworld_fp4 import FP4Linear, LinearRepairConfig, fp4_attention, preset


def timed(fn):
    return triton.testing.do_bench_cudagraph(fn, rep=100)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    args=parser.parse_args()
    torch.manual_seed(42)
    q=torch.randn(1,128,8,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,1024,2,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    x=torch.randn(128,1024,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(1024,1024,device='cuda',dtype=torch.bfloat16)
    layers={str(c):FP4Linear.from_float(w,config=LinearRepairConfig(coverage=c)) for c in (0,.5,1)}
    result={'device':torch.cuda.get_device_name(),'torch':torch.__version__, 'triton':triton.__version__,
            'attention_shape':{'M':128,'N':1024,'Hq':8,'Hkv':2,'D':128},
            'linear_shape':{'M':128,'N':1024,'K':1024},'units':'ms',
            'attention':{},'linear':{},'scope':'graph operator timings, activation packing included; shared GPU'}
    # Use expanded GQA heads outside timed region; force actual Flash SDPA.
    kr=k.repeat_interleave(4,dim=2).transpose(1,2).contiguous()
    vr=v.repeat_interleave(4,dim=2).transpose(1,2).contiguous()
    qr=q.transpose(1,2)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result['attention']['bf16_flash']=timed(lambda:F.scaled_dot_product_attention(qr,kr,vr))
    for name in ('fp4_base','balanced','quality','full_act2'):
        c=preset(name)
        result['attention'][name]=timed(lambda:fp4_attention(q,k,v,c))
    result['linear']['bf16']=timed(lambda:F.linear(x,w))
    for name,layer in layers.items():result['linear'][name]=timed(lambda:layer(x))
    from thunderworld_fp4._linear_kernels import linear_gemm
    from thunderworld_fp4._kernels import streaming_attention
    result['native_fp4_ptx']={}
    for name,kernel in [('linear',linear_gemm),('attention',streaming_attention)]:
        compiled=list(kernel.device_caches[torch.cuda.current_device()][0].values())
        result['native_fp4_ptx'][name]=any('mma.sync.aligned' in obj.asm['ptx'] and 'e2m1' in obj.asm['ptx'] for obj in compiled)
    print(json.dumps(result,indent=2))
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()

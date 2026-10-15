"""Thor same-shape BF16/FP8/FP4 operator comparison, with explicit cache policy.

Uses existing cuBLAS / vLLM CUTLASS / frozen v6 operators, without tuning them.
Pure GEMM and BF16-input online-packing scopes are measured separately. Random
inputs explain throughput and implementation costs, not generation quality.
"""
import argparse
import fcntl
import json
from pathlib import Path
import random
import statistics
import sys
from build_baselines import sha,write
from compare_speed import idle_guard

SHAPES=[(129,256,384),(416,2048,2048),(960,1024,2048),(960,2048,2048),
    (960,9216,2048),(960,2048,9216),(4096,4096,4096),(8192,8192,8192)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--variant',type=Path,default=Path('/home/chenhan/thunderworld/variants/cutlass-mainloop-v6'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--small',action='store_true')
    a=p.parse_args()
    lock=(a.root/'logs/gpu-test.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle_guard()
    version=json.loads((a.variant/'variant.json').read_text())
    for name,digest in version['variant_hashes'].items():
        if sha(a.variant/'thunderworld_fp4'/name)!=digest:raise ValueError('Changed variant')
    sys.path.insert(0,str(a.variant))
    import torch,triton,triton.testing
    from vllm import _custom_ops as ops
    from thunderworld_fp4._native import Native
    from thunderworld_fp4.linear import _pack
    from thunderworld_fp4.cutlass.packing import pack_input
    from thunderworld_fp4._sm110_cutlass import CutlassThor
    from doctor import decode
    torch.set_num_threads(4);torch.manual_seed(230924)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability()==(11,0)
    props=torch.cuda.get_device_properties(0)
    l2=getattr(props,'L2_cache_size',getattr(props,'l2_cache_size',32*2**20))
    report=dict(complete=False,rows=[],quality_evaluation=False,performance_eligible=False,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        variant_sha256=sha(a.variant/'variant.json'),source_sha256=sha(Path(__file__)),
        l2_bytes=l2,nominal_dense_tflops=dict(fp4=1035,fp8=517,fp16=258),
        hardware_source='https://dam-cdn.nvd.orangelogic.com/AssetLink/v686u306w26ftuk863xjeimilimtx55e.pdf',
        limitations=['MAXN with DVFS, no claimed hardware peak attainment.',
            'Input BF16, W row-major with transpose view, BF16 output, alpha=1/beta=0, no bias; existing library accumulation settings retained.',
            'FP4 preserves production K16 E4M3 block scale/E2M1 codes; FP8 W per-tensor and A per-token.',
            'No residual repairs in this operator probe. Real repaired execution is measured by full requests.',
            'Hot means identical buffers; rotating workspaces span at least 2 x L2 unless capped.',
            'Footprints are logical minimum I/O, not measured DRAM bytes; cold transient inference is excluded.',
            'Random input operator test does not assess video quality.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    rng=random.Random(230924)
    try:
        with torch.inference_mode():
            for m,n,k in (SHAPES[:1] if a.small else SHAPES):
                # FP4 footprint includes codes, K16 scales and BF16 output.
                minimum_bytes=((m+n)*k*9+15)//16+2*m*n
                count=min(16,max(2,(2*l2+minimum_bytes-1)//minimum_bytes))
                cases=[]
                for i in range(count):
                    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
                    d=torch.ones(k,device='cuda')
                    wc,ws,wg,_=_pack(w,d,True,True,w)
                    am=torch.zeros((triton.cdiv(m,128),k),device='cuda',dtype=torch.bool)
                    wm=torch.zeros_like(w,dtype=torch.bool)
                    native=Native(wc,ws,wg,d,am,wm,bm=128,interleave=True,num_stages=2)
                    op=CutlassThor(native,m,a.variant/'cutlass_sm110.so')
                    packed=pack_input(x,native)
                    fw,fs=ops.scaled_fp8_quant(w,scale=None)
                    fa,fas=ops.scaled_fp8_quant(x,scale=None,use_per_token_if_dynamic=True)
                    def bf16(x=x,w=w):return torch.nn.functional.linear(x,w)
                    def fp8(fa=fa,fw=fw,fas=fas,fs=fs):
                        return ops.cutlass_scaled_mm(fa,fw.T,scale_a=fas,scale_b=fs,out_dtype=torch.bfloat16)
                    def fp8_pack(x=x,fw=fw,fs=fs):
                        av,sv=ops.scaled_fp8_quant(x,scale=None,use_per_token_if_dynamic=True)
                        return ops.cutlass_scaled_mm(av,fw.T,scale_a=sv,scale_b=fs,out_dtype=torch.bfloat16)
                    def fp4(x=x,op=op,packed=packed):return op.run_packed(x,packed)
                    def fp4_pack(x=x,op=op):return op(x)
                    def official(packed=packed,op=op,m=m):
                        return ops.cutlass_scaled_fp4_mm(packed.native[0][0,:m],op.wc,
                            packed.scales,op.w0_scales,packed.alpha.reshape(1),torch.bfloat16)
                    def official_pack(x=x,op=op,m=m):
                        pt=pack_input(x,op.op)
                        return ops.cutlass_scaled_fp4_mm(pt.native[0][0,:m],op.wc,
                            pt.scales,op.w0_scales,pt.alpha.reshape(1),torch.bfloat16)
                    if i==0:
                        out=fp4();other=official()
                        av=decode(packed.native[0],packed.native[1])[0,:min(m,16)]
                        wv=decode(wc,ws)[0]
                        oracle=(av@wv.T)*(packed.native[2]*wg)
                        fp4_error=float((out[:min(m,16)].float()-oracle).norm()/oracle.norm().clamp_min(1.e-12))
                        fp8_oracle=(fa[:min(m,16)].float()*fas[:min(m,16)])@(fw.float()*fs).T
                        fp8_error=float((fp8()[:min(m,16)].float()-fp8_oracle).norm()/fp8_oracle.norm().clamp_min(1.e-12))
                        official_error=float((out.float()-other.float()).norm()/out.float().norm().clamp_min(1.e-12))
                        assert max(fp4_error,fp8_error,official_error)<.004,(fp4_error,fp8_error,official_error)
                        checks=dict(fp4_decoded_relative_l2=fp4_error,fp8_decoded_relative_l2=fp8_error,
                            official_cutlass_relative_l2=official_error,official_bitwise_equal=torch.equal(out,other))
                        del av,wv,oracle,fp8_oracle,out,other
                    cases.append(dict(bf16_cublas=bf16,fp8_cutlass=fp8,fp8_with_pack=fp8_pack,
                        fp4_v6=fp4,fp4_v6_with_pack=fp4_pack,
                        fp4_library=official,fp4_library_with_pack=official_pack))
                    del wc,ws,wg,am,wm,native,packed,op,x,w,d,fw,fs,fa,fas
                names=list(cases[0])
                for cache,num in [('hot',1),('rotating',count)]:
                    measured={name:[] for name in names}
                    for repeat in range(3):
                        order=names.copy();rng.shuffle(order)
                        for name in order:
                            fns=[r[name] for r in cases[:num]]
                            def run_all(fns=fns):
                                for fn in fns:fn()
                            idle_guard()
                            measured[name].append(triton.testing.do_bench_cudagraph(run_all,rep=40)/num)
                    row=dict(shape=[m,n,k],cache=cache,workspace_count=num,
                        fp4_logical_footprint_bytes=minimum_bytes*num,
                        exceeds_twice_l2=minimum_bytes*num>=2*l2,
                        useful_flops=2*m*n*k,fp4_issued_flops=2*triton.cdiv(m,128)*128*n*k,
                        checks=checks,measurements={name:dict(samples_ms=times,median_ms=statistics.median(times),
                            useful_tflops=2*m*n*k/statistics.median(times)/1.e9) for name,times in measured.items()})
                    report['rows'].append(row);write(a.output,report);print(json.dumps(row),flush=True)
                del cases,bf16,fp8,fp8_pack,fp4,fp4_pack,official,official_pack,fns,run_all
                torch.cuda.empty_cache()
        report['complete']=True
    except Exception as e:
        report['error']=repr(e);raise
    finally:write(a.output,report)


if __name__=='__main__':main()

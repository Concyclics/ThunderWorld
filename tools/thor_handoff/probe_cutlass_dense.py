"""Thor dense NVFP4 CUTLASS diagnostic with the existing exact packed values.

No repair terms are removed from a repair method: this probe only measures
the plain W4A4 base. It neither installs a backend nor claims repaired speed.
"""
import argparse
import fcntl
import json
from pathlib import Path
import statistics

from build_baselines import sha, write
from compare_speed import idle_guard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    lock = Path('/home/chenhan/thunderworld/logs/gpu-test.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    idle_guard()
    import torch, triton, triton.testing
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
    from thunderworld_fp4.linear import _pack
    from thunderworld_fp4._native import Native
    from doctor import decode
    torch.set_num_threads(4)
    torch.manual_seed(220927)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (11,0)
    report = dict(complete=False, rows=[], performance_eligible=False,
        quality_evaluation=False, source_sha256=sha(Path(__file__)),
        scope='Dense base only; identical FP4 codes/scales; no repair deployment',
        supports_fp4=ops.cutlass_scaled_mm_supports_fp4(110))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not report['supports_fp4']:
            report.update(complete=True, skipped='Installed vLLM does not advertise native NVFP4 on SM110')
            return
        with torch.inference_mode():
            for m,n,k in [(129,256,384),(416,2048,2048),(960,2048,2048),
                          (960,9216,2048),(960,2048,9216)]:
                x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                w = torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
                d = torch.ones(k,device='cuda')
                wc,ws,wg,_ = _pack(w,d,True,True,w)
                am = torch.zeros((triton.cdiv(m,128),k),device='cuda',dtype=torch.bool)
                wm = torch.zeros_like(w,dtype=torch.bool)
                op = Native(wc,ws,wg,d,am,wm,bm=128,interleave=True,num_stages=2)
                packed = op.pack(x)
                ac,sc,ag,mp = packed
                sw = swizzle_blockscale(op.ws)
                sa = swizzle_blockscale(sc[0,:m])
                alpha = (ag*wg).reshape(1)
                def cutlass():
                    return ops.cutlass_scaled_fp4_mm(ac[0,:m],op.wc,sa,sw,alpha,torch.bfloat16)
                def complete_cutlass():
                    ca,cs,cg,_ = op.pack(x)
                    return ops.cutlass_scaled_fp4_mm(ca[0,:m],op.wc,
                        swizzle_blockscale(cs[0,:m]),sw,(cg*wg).reshape(1),torch.bfloat16)
                ref = op.run_packed(x,packed)
                out = cutlass()
                error = float((out.float()-ref.float()).norm()/ref.float().norm().clamp_min(1.e-12))
                assert torch.isfinite(out).all() and error < .004, (m,n,k,error)
                row = dict(shape=[m,n,k],bitwise_equal=torch.equal(out,ref),relative_l2=error)
                if m == 129:
                    decoded = (decode(ac,sc)[0,:m] @ decode(wc,ws)[0].T)*(ag*wg)
                    derr = float((out.float()-decoded).norm()/decoded.norm().clamp_min(1.e-12))
                    assert derr < .004, derr
                    row['decoded_reference_relative_l2'] = derr
                    complete_cutlass();torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): gout=complete_cutlass()
                    graph.replay();saved=gout.clone()
                    x.normal_();graph.replay()
                    assert torch.equal(gout,complete_cutlass()) and not torch.equal(gout,saved)
                    row['current_input_graph']=True
                    x.zero_();graph.replay();assert torch.count_nonzero(gout)==0
                    row['zero_input']=True
                    x.normal_()
                    del graph,gout,saved
                fw,fs = ops.scaled_fp8_quant(w,scale=None)
                fa,fas = ops.scaled_fp8_quant(x,scale=None,use_per_token_if_dynamic=True)
                def fp8():
                    return ops.cutlass_scaled_mm(fa,fw.T,scale_a=fas,scale_b=fs,out_dtype=torch.bfloat16)
                def complete_fp8():
                    va,vs=ops.scaled_fp8_quant(x,scale=None,use_per_token_if_dynamic=True)
                    return ops.cutlass_scaled_mm(va,fw.T,scale_a=vs,scale_b=fs,out_dtype=torch.bfloat16)
                for name,fn in [('fp4_triton_gemm',lambda:op.run_packed(x,packed)),
                                ('fp4_cutlass_gemm',cutlass),('fp8_cutlass_gemm',fp8),
                                ('fp4_triton_with_pack',lambda:op(x)),
                                ('fp4_cutlass_with_pack_and_swizzle',complete_cutlass),
                                ('fp8_cutlass_with_pack',complete_fp8)]:
                    row[name+'_ms']=statistics.median(triton.testing.do_bench_cudagraph(fn,rep=30) for _ in range(3))
                report['rows'].append(row)
                write(a.output,report);print(json.dumps(row),flush=True)
        report['complete']=True
    except Exception as e:
        report['error']=repr(e)
        raise
    finally:
        write(a.output,report)


if __name__=='__main__':
    main()

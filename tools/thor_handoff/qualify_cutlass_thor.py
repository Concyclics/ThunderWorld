"""Numerical, budget-shape and Graph checks for CUTLASS + sparse FP4 repair."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
from build_baselines import sha,write
from compare_speed import idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--small',action='store_true')
    p.add_argument('--model-shapes',choices=('edge','nano'),default='edge')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--fused',action='store_true')
    p.add_argument('--reference-library',type=Path,help='Previous CUTLASS base + FP32 repair tail binary')
    a=p.parse_args()
    lock=Path('/home/chenhan/thunderworld/logs/gpu-test.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle_guard()
    import torch,triton,triton.testing
    from thunderworld_fp4.linear import _pack
    from thunderworld_fp4._native import Native
    from thunderworld_fp4._sparse_pack import pack_sparse
    from thunderworld_fp4.cutlass.packing import pack_input
    from cutlass_thor_backend import CutlassThor as SplitThor
    if a.fused:
        from cutlass_thor_fused_backend import CutlassThor
        if a.reference_library is None:p.error('--fused requires --reference-library')
    else:CutlassThor=SplitThor
    from doctor import decode
    torch.set_num_threads(4);torch.manual_seed(220928)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(__file__).parent
    sources=('cutlass_sm110_fused.cu','sm110_repair_collective.cuh','cutlass_thor_fused_backend.py') if a.fused else ('cutlass_sm110.cu','cutlass_thor_backend.py','cutlass_repair_tail.py')
    report=dict(complete=False,rows=[],sparse_packing=[],performance_eligible=False,quality_evaluation=False,
        library_sha256=sha(a.library),sources={n:sha(root/n) for n in sources},fused=a.fused,
        accumulation='FP32 base completed before A/W/cross; single final BF16 rounding')
    if a.reference_library:report['reference_library_sha256']=sha(a.reference_library)
    shapes=([(129,256,384),(416,2048,2048),(960,1024,2048),
        (960,2048,2048),(960,9216,2048),(960,2048,9216)] if a.model_shapes=='edge' else
        [(129,1024,4096),(416,4096,4096),(960,1024,4096),
         (960,4096,4096),(960,12288,4096),(960,4096,12288)])
    if a.small:shapes=shapes[:1]
    report['model_shapes']=a.model_shapes
    try:
        with torch.inference_mode():
            for m,n,k in shapes:
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
                d=torch.pow(2.,torch.linspace(-1,1,k,device='cuda'))
                wc,ws,wg,_=_pack(w,torch.ones_like(d),True,True,w)
                for af,wf,cross in [(0,0,False),(1,0,False),(0,1,False),(.375,.25,False),(1,1,False),(1,1,True),(.5,.5,True)]:
                    na=min(k//128,max(1,int(k//128*af))) if af else 0
                    nw=min(k//128,max(1,int(k//128*wf))) if wf else 0
                    am=torch.zeros((triton.cdiv(m,128),k),device='cuda',dtype=torch.bool)
                    wm=torch.zeros((n,k),device='cuda',dtype=torch.bool)
                    for i in range(am.shape[0]):
                        for j in range(na):am[i,((j+i)%(k//128))*128:((j+i)%(k//128)+1)*128]=True
                    for i in range(n//128):
                        for j in range(nw):wm[i*128:(i+1)*128,((j+i)%(k//128))*128:((j+i)%(k//128)+1)*128]=True
                    op=Native(wc,ws,wg,d,am,wm,bm=128,interleave=True,cross=cross,num_stages=2)
                    if op.hasw:
                        op.wloc=torch.arange(op.wi.numel()-1,-1,-1,device='cuda',dtype=torch.int32)
                        op.wr=op.wr.flip(0);op.wrs=op.wrs.view(torch.uint8).flip(0).view(torch.float8_e4m3fn)
                        op.shared_bank=True
                    fast=CutlassThor(op,m,a.library)
                    previous=SplitThor(op,m,a.reference_library) if a.reference_library else None
                    packed=op.pack(x);direct=pack_input(x,op)
                    assert all(torch.equal(v.reshape(-1).view(torch.uint8),u.reshape(-1).view(torch.uint8)) for v,u in zip(packed[:3],direct.native[:3]))
                    ref=op.run_packed(x,packed);out=fast.run_packed(x,direct)
                    err=float((out.float()-ref.float()).norm()/ref.float().norm().clamp_min(1.e-12))
                    assert torch.isfinite(out).all() and err<1.e-4,(m,n,k,af,wf,cross,err)
                    row=dict(shape=[m,n,k],a=af,w=wf,cross=cross,relative_l2=err,
                        bitwise_equal=torch.equal(out,ref),same_packed_values=True,base_tile=[128,128,128],
                        fp32_intermediate_bytes=0 if a.fused else 4*m*n if na or nw else 0)
                    if previous:
                        old=previous.run_packed(x,direct)
                        perr=float((out.float()-old.float()).norm()/old.float().norm().clamp_min(1.e-12))
                        assert perr<1.e-4,(m,n,k,af,wf,cross,perr)
                        row.update(previous_relative_l2=perr,previous_bitwise_equal=torch.equal(out,old))
                    if m==129:
                        ac,sc,ag,mp=packed;av=decode(ac,sc)[:,:m];wv=decode(wc,ws)
                        ar=av[1]*am.repeat_interleave(128,0)[:m] if na else torch.zeros_like(av[0])
                        wr=wv[1]*wm if nw else torch.zeros_like(wv[0])
                        decoded=av[0]@wv[0].T
                        if na:decoded+=ar@wv[0].T
                        if nw:decoded+=av[0]@wr.T
                        if cross:decoded+=ar@wr.T
                        decoded*=ag*wg
                        derr=float((out.float()-decoded).norm()/decoded.norm().clamp_min(1.e-12))
                        assert derr<.004,derr
                        row['decoded_reference_relative_l2']=derr
                    if not a.check_only:
                        benches=[('native',lambda:op(x)),('cutlass',lambda:fast(x))]
                        if previous:benches.append(('previous',lambda:previous(x)))
                        for label,fn in benches:
                            row[label+'_with_pack_ms']=statistics.median(triton.testing.do_bench_cudagraph(fn,rep=25) for _ in range(3))
                    report['rows'].append(row);write(a.output,report);print(json.dumps(row),flush=True)
                # Calibration and online selection must preserve both supports
                # and all represented values, including energy-capped selection.
                sensitivity=w.float().square().sum(0)
                for base,res,energy in [(1.,1.,1.),(.875,1.125,1.),(.75,.875,.8)]:
                    native=pack_sparse(x,d,sensitivity,op.am,op.na,base=base,res=res,energy=energy)
                    selected=op.am.clone()
                    direct=pack_input(x,op,sensitivity=sensitivity,base=base,res=res,energy=energy)
                    assert torch.equal(selected,op.am)
                    active=op.am.repeat_interleave(128,0)[:native[3]]
                    for i,repeat in [(0,64),(1,8)]:
                        lhs=native[i].view(torch.uint8);rhs=direct.native[i].view(torch.uint8)
                        assert torch.equal(lhs[0],rhs[0])
                        valid=active.repeat_interleave(repeat,1)
                        assert torch.equal(lhs[1][valid],rhs[1][valid])
                    lhs=op.run_packed(x,native);rhs=fast.run_packed(x,direct)
                    err=float((lhs.float()-rhs.float()).norm()/lhs.float().norm().clamp_min(1.e-12))
                    assert err<1.e-4
                    report['sparse_packing'].append(dict(shape=[m,n,k],base=base,residual=res,energy=energy,
                        identical_masks=True,identical_codes_scales=True,relative_l2=err))
                # Dynamic current-input/current-mask and independent output
                # storage checks; no replay may use the preceding request.
                fast(x);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):gout=fast(x)
                graph.replay();saved=gout.clone()
                x.normal_();op.am.copy_(op.am.roll(1,1));graph.replay()
                expected=fast(x)
                assert torch.equal(gout,expected) and not torch.equal(gout,saved)
                if a.fused:
                    op.am.zero_();graph.replay();empty=fast(x)
                    assert torch.equal(gout,empty)
                    row['empty_dynamic_support']=True
                x.zero_();graph.replay();assert torch.count_nonzero(gout)==0
                row.update(current_inputs_graph=True,current_mask_graph=True,zero_input=True,output_lifetime=True)
                del graph,gout,saved
        report['complete']=True
    except Exception as error:
        report['error']=repr(error);raise
    finally:write(a.output,report)


if __name__=='__main__':main()

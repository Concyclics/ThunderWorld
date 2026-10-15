"""Compare the opt-in Thor TMEM GEMM with the frozen packed arithmetic."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics

from build_baselines import write,sha
from compare_speed import idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--tiny',action='store_true')
    p.add_argument('--prefetch',action='store_true')
    p.add_argument('--warps',type=int,choices=(4,8),default=4)
    p.add_argument('--tma',action='store_true')
    p.add_argument('--small',action='store_true')
    a=p.parse_args()
    lock=Path('/home/chenhan/thunderworld/logs/gpu-test.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle_guard()
    import torch,triton,triton.testing
    from thunderworld_fp4.linear import _pack
    from thunderworld_fp4._native import Native,gemm as original
    if a.tma:from tma_gemm import gemm
    elif a.prefetch:from prefetch_gemm import gemm
    else:from tmem_gemm import gemm
    from doctor import decode
    assert torch.cuda.get_device_capability()==(11,0)
    torch.set_num_threads(4);torch.manual_seed(220923)
    implementation='tma_gemm.py' if a.tma else 'prefetch_gemm.py' if a.prefetch else 'tmem_gemm.py'
    report=dict(complete=False,rows=[],performance_eligible=False,prefetch=a.prefetch,tma=a.tma,
        kernel_sha256=sha(Path(__file__).with_name(implementation)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with torch.inference_mode():
            shapes=[(129,256,384)] if a.small else [(128,128,128),(128,128,256)] if a.tiny else [(960,2048,2048)] if a.quick else [(129,256,384),(416,256,384),(960,2048,2048),(960,9216,2048),(960,2048,9216)]
            for m,n,k in shapes:
                d=torch.ones(k,device='cuda')
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
                wc,ws,wg,_=_pack(w,d,True,True,w)
                packed=_pack(x,d,True,True,x)
                for amode,wmode,cross in ([(.375,0,False),(.5,.25,False)] if a.quick else [(0,0,False),(1,0,False),(0,1,False),(.375,.25,False),(1,1,False),(1,1,True),(.5,.5,True)]):
                    am=torch.zeros((triton.cdiv(m,128),k),device='cuda',dtype=torch.bool)
                    wm=torch.zeros_like(w,dtype=torch.bool)
                    na=min(k//128,max(1,int(k//128*amode))) if amode else 0
                    nw=min(k//128,max(1,int(k//128*wmode))) if wmode else 0
                    # Vary the selected blocks between M/N tiles, with a fixed quota.
                    for i in range(am.shape[0]):
                        for j in range(na):am[i,((j+i)%(k//128))*128:((j+i)%(k//128)+1)*128]=True
                    for i in range(n//128):
                        for j in range(nw):wm[i*128:(i+1)*128,((j+i)%(k//128))*128:((j+i)%(k//128)+1)*128]=True
                    op=Native(wc,ws,wg,d,am,wm,bm=128,interleave=True,cross=cross,num_stages=2)
                    if op.hasw:
                        op.wloc=torch.arange(op.wi.numel()-1,-1,-1,device='cuda',dtype=torch.int32)
                        op.wr=op.wr.flip(0)
                        op.wrs=op.wrs.view(torch.uint8).flip(0).view(torch.float8_e4m3fn)
                        op.shared_bank=True
                    ac,sc,ag,mp=packed
                    kernels={}
                    def run(kernel):
                        out=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                        obj=kernel[(triton.cdiv(m,128),n//128,1)](ac,sc,op.wc,op.ws,op.wg,ag,op.ai,op.am,op.wr,op.wrs,op.wi,op.off,out,
                            m,n,k,mp,op.na,op.hasw,op.cross,128,128,True,1,op.dense_w,WLOC=op.wloc,SHARED_BANK=op.shared_bank,
                            num_warps=4 if kernel is original else a.warps,num_stages=2 if kernel is original else 1)
                        kernels[kernel]=obj
                        return out
                    reference=run(original);out=run(gemm)
                    equal=torch.equal(reference,out)
                    error=float((out.float()-reference.float()).norm()/reference.float().norm())
                    if not equal:
                        print(json.dumps(dict(shape=[m,n,k],a=amode,w=wmode,cross=cross,error=error,out=out[:2,:8].float().tolist(),ref=reference[:2,:8].float().tolist())),flush=True)
                        for name,ker in [('original',original),('tmem',gemm)]:
                            a.output.with_name(f'failed-gemm-{name}.ptx').write_text(kernels[ker].asm['ptx'])
                    torch.testing.assert_close(out,reference,rtol=0,atol=0)
                    if m==129:
                        av=decode(ac,sc)[:,:m];wv=decode(wc,ws)
                        ref=av[0]@wv[0].T
                        aa=av[1]*am.repeat_interleave(128,0)[:m]
                        ww=wv[1]*wm
                        if na:ref+=aa@wv[0].T
                        if nw:ref+=av[0]@ww.T
                        if cross:ref+=aa@ww.T
                        ref*=ag*wg
                        assert float((out.float()-ref).norm()/ref.norm())<.004
                    row=dict(shape=[m,n,k],a=amode,w=wmode,cross=cross,bitwise_equal=equal,relative_l2=error,shared_bank=op.shared_bank)
                    for name,kernel in [('original',original),('tmem',gemm)]:
                        obj=kernels[kernel]
                        row[name]=dict(registers=obj.n_regs,spills=obj.n_spills,shared_bytes=obj.metadata.shared,
                            tmem_load_sites=obj.asm['ptx'].count('tcgen05.ld.'),tmem_store_sites=obj.asm['ptx'].count('tcgen05.st.'))
                        if not a.check_only:
                            row[name]['milliseconds']=statistics.median(triton.testing.do_bench_cudagraph(lambda:run(kernel),rep=30) for _ in range(3))
                        if m==960 and n==2048 and k==2048:
                            a.output.with_name(f'gemm-{name}-a{amode}-w{wmode}.ptx').write_text(obj.asm['ptx'])
                    report['rows'].append(row);write(a.output,report);print(json.dumps(row),flush=True)
            # No policy selection is performed by this kernel; changing masks
            # changes the actual current call, also inside a captured graph.
            original_out=out.clone();op.am.copy_(op.am.roll(1,1))
            # A partial quota is necessary to exercise runtime mask reads.
            if 0<op.na<k//128:
                changed=run(gemm);expected=run(original)
                assert torch.equal(changed,expected)
                assert not torch.equal(changed,original_out)
                report['current_mask_update']=True
            graph=torch.cuda.CUDAGraph()
            run(gemm);torch.cuda.synchronize()
            with torch.cuda.graph(graph):gout=run(gemm)
            graph.replay();saved=gout.clone()
            fresh=_pack(torch.randn_like(x),d,True,True,x)
            ac.copy_(fresh[0]);sc.copy_(fresh[1]);ag.copy_(fresh[2])
            graph.replay();expected=run(original)
            assert torch.equal(gout,expected) and not torch.equal(gout,saved)
            report['current_inputs_graph']=True
            # All-zero activations exercise outer-factor clamping and code zero.
            fresh=_pack(torch.zeros_like(x),d,True,True,x)
            ac.copy_(fresh[0]);sc.copy_(fresh[1]);ag.copy_(fresh[2]);graph.replay()
            assert torch.count_nonzero(gout)==0
            report['zero_input']=True
        report['complete']=True
    except Exception as error:
        report['error']=repr(error);raise
    finally:write(a.output,report)


if __name__=='__main__':main()

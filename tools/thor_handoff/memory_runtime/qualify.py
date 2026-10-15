"""Compact current-A addressing against the frozen v6 arithmetic and decoder."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/chenhan/thunderworld'))
    p.add_argument('--variant',type=Path,default=Path('/home/chenhan/thunderworld/variants/cutlass-memory-v7'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--small',action='store_true')
    a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from build_baselines import write,sha
    from compare_speed import idle_guard
    lock=(a.root/'logs/gpu-test.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    idle_guard();sys.path.insert(0,str(a.variant))
    import torch,triton
    from thunderworld_fp4.linear import _pack
    from thunderworld_fp4._native import Native
    from thunderworld_fp4._sm110_cutlass import CutlassThor
    from thunderworld_fp4.cutlass.packing import pack_input,pack_input_dense
    from thunderworld_fp4._sm110_compact_pack import compact_bytes
    from cutlass_thor_fused_backend import CutlassThor as Reference
    from doctor import decode
    torch.set_num_threads(4);torch.manual_seed(230927)
    torch.backends.cuda.matmul.allow_tf32=False
    shapes=[(129,256,384),(416,2048,2048),(960,1024,2048),(960,9216,2048),(960,2048,9216),
            (129,1024,4096),(416,4096,4096),(960,1024,4096),(960,12288,4096),(960,4096,12288),
            (1920,256,2048),(2160,256,4096),(2880,256,9216),(3840,256,12288)]
    if a.small:shapes=shapes[:1]+shapes[5:6]
    old=a.root/'variants/cutlass-mainloop-v6/cutlass_sm110.so'
    report=dict(complete=False,rows=[],graphs=[],library_sha256=sha(a.variant/'cutlass_sm110.so'),
        parent_library_sha256=sha(old),variant_receipt_sha256=sha(a.variant/'variant.json'),
        small=a.small,quality_evaluation=False,sources={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
    try:
        with torch.inference_mode():
            for m,n,k in shapes:
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
                d=torch.pow(2.,torch.linspace(-1,1,k,device='cuda'))
                wc,ws,wg,_=_pack(w,torch.ones_like(d),True,True,w)
                sensitivity=w.float().square().sum(0)
                cases=[(0,0,False,None,1.,1.,1.),(1,0,False,None,1.,1.,1.),
                       (0,1,False,None,1.,1.,1.),(.125,.25,False,None,1.,1.,1.),
                       (.375,.25,False,True,1.,1.,1.),(.375,.5,True,True,.875,1.125,1.),
                       (.5,.5,True,True,.75,.875,.8),(1,1,True,None,1.,1.,1.)]
                for af,wf,cross,sparse,base,res,energy in cases:
                    na=max(1,int(k//128*af)) if af else 0
                    nw=max(1,int(k//128*wf)) if wf else 0
                    am=torch.zeros((triton.cdiv(m,128),k),device='cuda',dtype=torch.bool)
                    wm=torch.zeros((n,k),device='cuda',dtype=torch.bool)
                    for mb in range(am.shape[0]):
                        for j in range(na):
                            kb=(mb+j*3)%(k//128)
                            # Use consecutive blocks so non-coprime K counts
                            # cannot accidentally reduce the declared quota.
                            kb=(mb+j)%(k//128)
                            am[mb,kb*128:(kb+1)*128]=True
                    for nb in range(n//128):
                        for j in range(nw):
                            kb=(nb+j)%(k//128);wm[nb*128:(nb+1)*128,kb*128:(kb+1)*128]=True
                    op=Native(wc,ws,wg,d,am,wm,bm=128,interleave=True,cross=cross,num_stages=2)
                    if op.hasw:
                        op.wloc=torch.arange(op.wi.numel()-1,-1,-1,device='cuda',dtype=torch.int32)
                        op.wr=op.wr.flip(0);op.wrs=op.wrs.view(torch.uint8).flip(0).view(torch.float8_e4m3fn)
                        op.shared_bank=True
                    fast=CutlassThor(op,m,a.variant/'cutlass_sm110.so')
                    previous=Reference(op,m,old)
                    kwargs=dict(base=base,res=res,energy=energy)
                    if sparse:kwargs['sensitivity']=sensitivity
                    dense=pack_input_dense(x,op,**kwargs);mask=op.am.clone()
                    packed=pack_input(x,op,**kwargs)
                    assert torch.equal(mask,op.am)
                    ac,sc,ag,mp=dense.native
                    for lhs,rhs in [(ac[0],packed.native[0][0]),(sc[0],packed.native[1][0]),
                                    (ag,packed.native[2]),(dense.scales,packed.scales),(dense.alpha,packed.alpha)]:
                        assert torch.equal(lhs.contiguous().reshape(-1).view(torch.uint8),rhs.contiguous().reshape(-1).view(torch.uint8))
                    compact=getattr(packed,'compact_repair',None)
                    costs=compact_bytes(m,k,op.na)
                    assert bool(compact is not None)==costs['enabled']
                    if compact is not None:
                        rc,rs,locations=compact
                        for mb,kb in op.am.nonzero().tolist():
                            slot=int(locations[mb,kb]);rows=min(128,mp-mb*128)
                            assert torch.equal(rc[slot,:rows],ac[1,mb*128:mb*128+rows,kb*64:(kb+1)*64])
                            assert torch.equal(rs[slot,:rows].view(torch.uint8),sc[1,mb*128:mb*128+rows,kb*8:(kb+1)*8].view(torch.uint8))
                    out=fast.run_packed(x,packed);ref=previous.run_packed(x,dense)
                    assert torch.equal(out,ref),(m,n,k,af,wf,cross,base,res,'v6 mismatch')
                    row=dict(shape=[m,n,k],a=af,w=wf,cross=cross,base=base,res=res,energy=energy,
                             same_mask=True,same_codes_scales=True,bitwise_equal=True,storage=costs)
                    if m==129:
                        av=decode(ac,sc)[:,:m];wv=decode(wc,ws)
                        decoded=av[0]@wv[0].T
                        if na:
                            ar=av[1]*op.am.repeat_interleave(128,0)[:m].repeat_interleave(128,1)
                            decoded+=ar@wv[0].T
                        if nw:
                            wr=wv[1]*wm;decoded+=av[0]@wr.T
                            if cross and na:decoded+=ar@wr.T
                        decoded*=ag*wg
                        err=float((out.float()-decoded).norm()/decoded.norm().clamp_min(1.e-12))
                        assert err<.004,err
                        row['decoded_reference_relative_l2']=err
                    report['rows'].append(row);write(a.output,report)
                # Use a partial, calibrated selector for Graph current-input
                # coverage, then an explicit mask for empty/support changes.
                op.na=max(1,k//128//4)
                op.am.zero_();op.am[:,:op.na]=True
                fast=CutlassThor(op,m,a.variant/'cutlass_sm110.so');previous=Reference(op,m,old)
                fn=lambda:fast.run_packed(x,pack_input(x,op,sensitivity=sensitivity,base=.875,res=1.125,energy=.8))
                fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): y=fn()
                g.replay();saved=y.clone();x.normal_();g.replay()
                expected=previous.run_packed(x,pack_input_dense(x,op,sensitivity=sensitivity,base=.875,res=1.125,energy=.8))
                assert torch.equal(y,expected) and not torch.equal(y,saved)
                del g,y,saved
                fast(x);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):y=fast(x)
                op.am.zero_();g.replay()
                assert torch.equal(y,previous.run_packed(x,pack_input_dense(x,op)))
                op.am[:,-op.na:]=True;x.zero_();g.replay();assert torch.count_nonzero(y)==0
                saved=y.clone();x.normal_();g.replay();assert torch.count_nonzero(saved)==0
                report['graphs'].append(dict(shape=[m,n,k],current_input=True,current_selection=True,empty_mask=True,zero=True,output_lifetime=True))
                del g,y,saved
                write(a.output,report)
                print(json.dumps(dict(shape=[m,n,k],passed=len(cases),graph=True)),flush=True)
        report['complete']=True
    except Exception as exc:
        report['error']=repr(exc);raise
    finally:write(a.output,report)


if __name__=='__main__':main()

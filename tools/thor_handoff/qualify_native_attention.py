"""Thor native-attention numerics, current-input graphs and sanitizer workload."""
import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import sys
import statistics

from build_baselines import write,sha
from compare_speed import idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--small',action='store_true')
    p.add_argument('--nano-heads',action='store_true',help='Validate Nano H32/KV8 in addition to the existing D128 attention semantics')
    p.add_argument('--query-tile',type=int,choices=(32,64,128),default=64)
    p.add_argument('--warps',type=int,choices=(4,8),default=4)
    p.add_argument('--row-major-v',action='store_true')
    p.add_argument('--kernel',choices=('native','pipelined'),default='native')
    p.add_argument('--terms',type=int,choices=(2,3),default=3)
    p.add_argument('--benchmark',action='store_true',help='Diagnostic full attention timings including packing/selection/combine')
    a=p.parse_args()
    lock=Path('/home/chenhan/thunderworld/logs/gpu-test.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle_guard()
    import torch,triton,triton.testing
    import thunderworld_fp4.runtime as runtime
    from thunderworld_fp4.config import RepairConfig
    from native_attention import streaming_attention as native_attention
    if a.kernel=='pipelined':
        from pipelined_attention import streaming_attention
    else:streaming_attention=native_attention
    assert torch.cuda.get_device_capability()==(11,0)
    torch.set_num_threads(4);torch.manual_seed(220924)
    torch.backends.cuda.matmul.allow_tf32=False
    original=runtime.streaming_attention
    class Dispatch:
        def __init__(self,kernel):self.kernel=kernel
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                args=list(args);args[18]=a.query_tile;kwargs['num_warps']=a.warps
                return self.kernel[(triton.cdiv(args[12],a.query_tile),grid[1],grid[2])](*args,**kwargs)
            return launch
    tested=Dispatch(streaming_attention)
    report=dict(complete=False,tests=[],quality_evaluation=False,performance_eligible=False,
        numerical_relative_l2_limit=.004,kernel_sha256=sha(Path(__file__).with_name(a.kernel+'_attention.py')),
        kernel=a.kernel,
        terms=a.terms,
        query_tile=a.query_tile,num_warps=a.warps,transpose_v=not a.row_major_v,nano_heads=a.nano_heads)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with torch.inference_mode():
            shapes=[(129,257,4,2),(416,960,4,1),(960,1504,16,8)]
            if a.nano_heads:shapes=[(m,n,32,8) for m,n,_,_ in shapes]
            if a.small:shapes=shapes[:1]
            for m,n,h,hk in shapes:
                q=torch.randn(1,m,h,128,device='cuda',dtype=torch.bfloat16)
                k=torch.randn(1,n,hk,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
                nt=triton.cdiv(n,128)
                modes=['empty','full','partial','qk_only','pv_only','explicit','four_terms','zero']
                for mode in modes:
                    if mode=='zero':q.zero_();k.zero_();v.zero_()
                    counts=[0]*h if mode in ('empty','pv_only') else [nt]*h if mode=='full' else [(i+1)% (nt+1) for i in range(h)]
                    pv=[0]*h if mode in ('empty','qk_only') else [nt]*h if mode=='full' else list(reversed([(i+2)%(nt+1) for i in range(h)]))
                    cfg=RepairConfig(head_quotas=tuple(counts),pv_head_quotas=tuple(pv),terms=4 if mode=='four_terms' else a.terms,
                        pv_terms=4 if mode=='four_terms' else a.terms,query_tile=32,num_warps=4,
                        transpose_v=not a.row_major_v,reload_q=True,sparse_pack=True)
                    quotas=(torch.tensor(counts,device='cuda',dtype=torch.int32),torch.tensor(pv,device='cuda',dtype=torch.int32))
                    mask=torch.rand(h,triton.cdiv(m,128),nt,device='cuda')>.5 if mode=='explicit' else None
                    kwargs=dict(counts_tensor=quotas,repair_mask=mask,return_mask=True)
                    runtime.streaming_attention=original
                    ref,refmask,_=runtime._forward(q,k,v,cfg,**kwargs)
                    runtime.streaming_attention=tested
                    out,outmask,_=runtime._forward(q,k,v,cfg,**kwargs)
                    err=float((out.float()-ref.float()).norm()/ref.float().norm().clamp_min(1.e-12))
                    assert torch.isfinite(out).all() and err<.004,(mode,m,n,err)
                    assert torch.equal(outmask,refmask)
                    row=dict(shape=[m,n,h,hk],mode=mode,relative_l2=err,identical_masks=True,
                        bitwise_equal=torch.equal(out,ref))
                    if a.kernel=='pipelined':
                        runtime.streaming_attention=Dispatch(native_attention)
                        baseline,bmask,_=runtime._forward(q,k,v,cfg,**kwargs)
                        assert torch.equal(out,baseline) and torch.equal(outmask,bmask),(mode,m,n,'MMA group changed output')
                        row['native_v3_bitwise_equal']=True
                        runtime.streaming_attention=tested
                    if a.benchmark and mode!='zero':
                        for label,kernel in [('native',native_attention),('tested',streaming_attention)]:
                            runtime.streaming_attention=Dispatch(kernel)
                            row[label+'_full_attention_ms']=statistics.median(triton.testing.do_bench_cudagraph(
                                lambda:runtime._forward(q,k,v,cfg,**kwargs),rep=25) for _ in range(3))
                        runtime.streaming_attention=tested
                    if a.row_major_v:
                        # Layout-only optimization must preserve the already
                        # qualified native arithmetic, not merely stay close
                        # to the older BF16-emulated reference.
                        baseline,bmask,_=runtime._forward(q,k,v,replace(cfg,transpose_v=True),**kwargs)
                        assert torch.equal(out,baseline) and torch.equal(outmask,bmask),(mode,m,n,'V layout changed output')
                        row['native_transposed_v_bitwise_equal']=True
                    if m==129 and mode in ('empty','full'):
                        sys.path.insert(0,str(Path(runtime.__file__).resolve().parents[2]/'tests'))
                        from oracle import reference
                        decoded=reference(q,k,v,outmask[0].to(torch.int32)*(cfg.terms-1)+1,
                            mode!='empty',terms=cfg.terms,pvflags=outmask[1].to(torch.int32)*(cfg.pv_terms-1)+1)
                        derr=float((out.float()-decoded.float()).norm()/decoded.float().norm().clamp_min(1.e-12))
                        assert derr<.006,(mode,derr)
                        row['decoded_reference_relative_l2']=derr
                    if mode=='explicit':
                        # Every input and explicit mask must be read on replay.
                        runtime._forward(q,k,v,cfg,**kwargs)
                        torch.cuda.synchronize()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            gout,gmask,held=runtime._forward(q,k,v,cfg,**kwargs)
                        graph.replay();saved=gout.clone()
                        q.normal_();k.normal_();v.normal_();mask.logical_not_()
                        graph.replay()
                        eager,emask,_=runtime._forward(q,k,v,cfg,**kwargs)
                        assert torch.equal(gout,eager) and torch.equal(gmask,emask)
                        assert not torch.equal(saved,gout)
                        row.update(current_input_graph=True,current_mask_graph=True,output_lifetime=True)
                        del graph,held,gout,gmask
                    report['tests'].append(row);write(a.output,report);print(json.dumps(row),flush=True)
        report['complete']=True
    except Exception as error:
        report['error']=repr(error);raise
    finally:
        runtime.streaming_attention=original;write(a.output,report)


if __name__=='__main__':main()

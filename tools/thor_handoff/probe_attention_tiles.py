"""Isolated SM110 launch experiment, including unqualified 128-row tiles.

Only this probe overrides the low-level launch; it does not relax the public
RepairConfig guard or change any frozen inference configuration.
"""
import argparse
import fcntl
import json
from pathlib import Path
import statistics

from build_baselines import write
from compare_speed import idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--native',action='store_true')
    p.add_argument('--skip-acc-init-pass',action='store_true')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--row-major-v',action='store_true')
    p.add_argument('--launch',type=int,nargs=2,metavar=('TILE','WARPS'))
    a=p.parse_args()
    lock=Path('/home/chenhan/thunderworld/logs/gpu-test.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle_guard()
    import torch,triton,triton.testing
    if a.skip_acc_init_pass:
        from triton.backends.nvidia.compiler import passes
        passes.ttgpuir.add_optimize_accumulator_init=lambda pm:None
    import thunderworld_fp4.runtime as runtime
    from thunderworld_fp4.config import RepairConfig
    assert torch.cuda.get_device_capability()==(11,0)
    torch.set_num_threads(4);torch.manual_seed(220922)
    original=runtime.streaming_attention
    if a.native:
        from native_attention import streaming_attention as tested
    else:tested=original
    report=dict(complete=False,rows=[],performance_eligible=False,transpose_v=not a.row_major_v,scope='Kernel correctness and speed; unchanged quantization, selection and split reduction')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    launch_choice=[32,4,True]
    kernels={}
    class ProbeLaunch:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                args=list(args);args[18]=launch_choice[0]
                kwargs['num_warps']=launch_choice[1]
                actual_grid=(triton.cdiv(args[12],launch_choice[0]),grid[1],grid[2])
                compiled=tested[actual_grid](*args,**kwargs)
                kernels[tuple(launch_choice[:2])]=compiled
                return compiled
            return launch
    try:
        with torch.inference_mode():
            for m,n in ([(960,1504)] if a.quick else [(960,1504),(129,416)]):
                h,hk=16,8;nt=triton.cdiv(n,128)
                q=torch.randn(1,m,h,128,device='cuda',dtype=torch.bfloat16)
                k=torch.randn(1,n,hk,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
                for mode in (['partial'] if a.quick else ['none','partial','full']):
                    counts=[0]*h if mode=='none' else [nt]*h if mode=='full' else [min(nt,(i%4+1)*nt//4) for i in range(h)]
                    pv=list(reversed(counts))
                    cfg=RepairConfig(head_quotas=tuple(counts),pv_head_quotas=tuple(pv),terms=3,pv_terms=3,
                        query_tile=32,num_warps=4,transpose_v=not a.row_major_v,reload_q=True,sparse_pack=True)
                    quotas=(torch.tensor(counts,device='cuda',dtype=torch.int32),torch.tensor(pv,device='cuda',dtype=torch.int32))
                    runtime.streaming_attention=original
                    reference=runtime._forward(q,k,v,cfg,counts_tensor=quotas,return_mask=True)[:2]
                    runtime.streaming_attention=ProbeLaunch()
                    choices=[a.launch] if a.launch else ([(32,4),(32,8),(64,4),(64,8),(128,4),(128,8)] if a.native else [(128,4),(128,8)] if a.skip_acc_init_pass else [(32,4),(64,4),(64,8),(128,4),(128,8)])
                    for bm,warps in choices:
                        launch_choice[:]=[bm,warps,not a.row_major_v]
                        try:
                            out,mask,_=runtime._forward(q,k,v,cfg,counts_tensor=quotas,return_mask=True)
                            error=float((out.float()-reference[0].float()).norm()/reference[0].float().norm())
                            equal=torch.equal(out,reference[0]);same_mask=torch.equal(mask,reference[1])
                            kernel=kernels[(bm,warps)]
                            fn=lambda:runtime._forward(q,k,v,cfg,counts_tensor=quotas)[0]
                            ms=None if a.check_only else statistics.median(triton.testing.do_bench_cudagraph(fn,rep=30) for _ in range(3))
                            row=dict(shape=[m,n,h,hk],mode=mode,bm=bm,warps=warps,bitwise_equal=equal,
                                identical_masks=same_mask,relative_l2=error,milliseconds=ms,
                                registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared,
                                native_fp4_sites=kernel.asm['ptx'].count('kind::mxf4nvf4'),
                                bf16_mma_sites=kernel.asm['ptx'].count('mma.sync.aligned.m16n8k16.row.col.f32.bf16'))
                            if a.native or a.skip_acc_init_pass:
                                assert same_mask and error<.004 and torch.isfinite(out).all(),row
                                assert row['native_fp4_sites']>0 and row['bf16_mma_sites']==0,row
                            if mode=='partial' and m==960:
                                a.output.with_name(f'attention-{"native-" if a.native else "skip-pass-" if a.skip_acc_init_pass else ""}bm{bm}-w{warps}.ptx').write_text(kernel.asm['ptx'])
                        except (triton.CompilationError,triton.OutOfResources) as error:
                            row=dict(shape=[m,n,h,hk],mode=mode,bm=bm,warps=warps,error=str(error)[-2000:])
                        report['rows'].append(row);write(a.output,report);print(json.dumps(row),flush=True)
        report['complete']=True
    finally:
        runtime.streaming_attention=original
        write(a.output,report)


if __name__=='__main__':main()

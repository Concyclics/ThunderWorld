"""Compact resident W1 union versus identical-support full CPU asset control."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from baselines import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch
        from representation import planes
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec
        from thunderworld_fp4.cosmos.tuning import LaunchTable
        torch.manual_seed(170917)
        w=torch.randn(256,256,device='cuda',dtype=torch.bfloat16)
        rep=planes(w)
        asset={k:rep[k].cpu() for k in ('codes','scales','factor')}
        asset.update(prescale=torch.ones(256),score=torch.ones(2,2),bias=None)
        specs=[LinearSpec(0,0,activation_blocks=1,weight_support=(0,)),
               LinearSpec(0,0,activation_blocks=1,weight_support=(2,))]
        current=[specs[0]]
        def controller(plan):
            return SimpleNamespace(warming=True,table=LaunchTable(),budget_plan=plan,
                linear_spec=lambda *args:current[0],record_linear=lambda *args:None)
        plan=SimpleNamespace(linears={'0|0|probe':dict(w_blocks=[0]),'1|1|probe':dict(w_blocks=[2])})
        source=torch.nn.Linear(256,256,bias=False,device='meta',dtype=torch.bfloat16)
        cc,rc=controller(plan),controller(None)
        compact=RepairLinear(source,cc,'probe',specs[0],asset)
        reference=RepairLinear(source,rc,'probe',specs[0],asset)
        if 'codes' in compact.asset or 'scales' in compact.asset:raise AssertionError('Full CPU W1 retained')
        result=dict(complete=False,performance_eligible=False,resource=resource,rows=[],
            union_density=.5,cpu_residual_tiles=compact.asset['codes_residual'].shape[0],
            scope='Same supports and numerical operations, compact CPU asset and shared GPU union; no latency claim.')
        with torch.inference_mode():
            for m in (129,416,960):
                x=torch.randn(m,256,device='cuda',dtype=torch.bfloat16)
                for phase,spec in enumerate(specs):
                    current[0]=spec;cc.warming=rc.warming=True
                    compact(x);reference(x)
                    cc.warming=rc.warming=False
                    torch.testing.assert_close(compact(x),reference(x),rtol=0,atol=0)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):compact(x)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):output=compact(x)
                    for zero in (False,True):
                        if zero:x.zero_()
                        else:x.copy_(torch.randn_like(x))
                        graph.replay()
                        torch.testing.assert_close(output,reference(x),rtol=0,atol=0)
                    result['rows'].append(dict(m=m,phase=phase,eager_exact=True,graph_current_input_exact=True))
                    write(a.output,result)
                    del graph,output
        for op in compact.ops.values():
            if op.wr.shape[0]!=2:raise AssertionError('GPU W1 is not the exact union')
        result.update(complete=True,gpu_residual_bytes=sum(t.numel()*t.element_size() for t in compact.weight_bank[:2]),
            full_residual_bytes=256*256*9//16)
        write(a.output,result)


if __name__=='__main__':main()

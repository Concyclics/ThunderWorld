"""Independent producer/GEMM fork-join: a concurrency probe, not E2E inference.

Uses distinct inputs and masks. It does not bypass a real model dependency or
implement a producer/consumer pipeline between layers/steps. Both serial and
parallel timings include one packed GEMM and one current-input selector+pack.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import torch
import triton
import triton.testing
from dynamic import BudgetLinear,TIERS,select_tiles
from thunderworld_fp4.linear import _pack


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('benchmarks',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--trace',action='store_true')
    p.add_argument('--torch-trace',action='store_true');a=p.parse_args()
    if a.torch_trace and not a.trace:raise ValueError('--torch-trace requires --trace')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    rows=[];side=torch.cuda.Stream()
    for source in a.benchmarks:
        b=json.loads(source.read_text());assert b['complete']
        selected=[r for r in b['rows'] if r['tier'] in ['a50','act2','a50w','dual'] and int(r['shape'].split(':')[1])<int(r['shape'].split(':')[2]) and r['task'] in ['i2v','policy']]
        if a.trace:selected=[r for r in selected if b['model']=='edge' and r['shape'].startswith('316:') and r['tier'] in ['a50','act2']]
        for r in selected:
            d=torch.load(r['payload'],weights_only=True,map_location='cpu')
            src=torch.nn.Linear(d['weight'].shape[1],d['weight'].shape[0],bias=False,dtype=torch.bfloat16)
            src.weight=torch.nn.Parameter(d['weight'],requires_grad=False);linear=BudgetLinear(src)
            x=d['x'].cuda();other=(x*.9375).contiguous();tier=r['tier'];ac,wc=TIERS[tier]
            linear.op.bm=r['winner']['bm'];linear.op.num_stages=r['winner']['num_stages'];linear.prepare(x,tier)
            packed=linear.op.pack(x);m,k=x.shape;mask=torch.empty_like(linear.op.am)
            def producer():
                if ac==.5:
                    select_tiles[(triton.cdiv(m,128),)](other,linear.op.d,mask,m,k,triton.next_power_of_2(k//128),k//256,num_warps=4)
                return _pack(other,linear.op.d,ac>0,ac==1,mask)
            def gemm():return linear.op.run_packed(x,packed)
            def serial():return gemm(),producer()
            def parallel():
                main=torch.cuda.current_stream();side.wait_stream(main)
                with torch.cuda.stream(side):produced=producer()
                y=gemm();main.wait_stream(side)
                return y,produced
            ref=serial();candidate=parallel();torch.cuda.synchronize()
            assert torch.equal(ref[0],candidate[0])
            assert all(torch.equal(x.reshape(-1).view(torch.uint8),y.reshape(-1).view(torch.uint8)) for x,y in zip(ref[1][:3],candidate[1][:3]))
            samples={name:[] for name in ['serial','parallel','gemm','producer']}
            fns=dict(serial=serial,parallel=parallel,gemm=gemm,producer=producer)
            if a.trace:
                prof=None
                if a.torch_trace:
                    prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]);prof.start()
                for name in ['serial','parallel']:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):captured=fns[name]()
                    graph.replay();torch.cuda.synchronize()
                    with torch.cuda.nvtx.range('overlap_probe/'+tier+'/'+name),torch.profiler.record_function('overlap_probe/'+tier+'/'+name):
                        for _ in range(5):graph.replay()
                        torch.cuda.synchronize()
                if prof is not None:
                    prof.stop();prof.export_chrome_trace(str(a.output.with_suffix('.'+tier+'.chrome.json')))
                rows.append(dict(model=b['model'],shape=r['shape'],tier=tier,outputs_bitwise_equal=True,trace_only=True))
                continue
            for repeat in range(5):
                order=['serial','parallel','gemm','producer']
                if repeat%2:order.reverse()
                for name in order:samples[name].append(triton.testing.do_bench_cudagraph(fns[name],rep=20))
            times={key:statistics.median(v) for key,v in samples.items()}
            rows.append(dict(model=b['model'],shape=r['shape'],tier=tier,bm=linear.op.bm,stages=linear.op.num_stages,
                times_ms=times,samples_ms=samples,speedup=times['serial']/times['parallel'],outputs_bitwise_equal=True,
                scope='Independent inputs and masks; producer for another work item plus prepacked GEMM for current item; not dependent same-input packing/GEMM or full generation'))
            print(b['model'],r['shape'],tier,'serial/parallel',round(rows[-1]['speedup'],3),flush=True)
            del ref,candidate,packed,linear,src,d,x,other,mask;torch.cuda.empty_cache()
    a.output.write_text(json.dumps(dict(complete=True,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        benchmark_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.benchmarks},
        scope='Five alternating-order CUDA Graph measurements per configuration. Concurrency probe, not proof of concurrent residency, achieved TC activity, production E2E speedup or a general upper bound.'),indent=2)+'\n')


if __name__=='__main__':main()

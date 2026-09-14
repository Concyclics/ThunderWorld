"""Actual frozen-policy panels, fixed CUTLASS launch, diagnostic timings only."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--correctness-only',action='store_true')
    p.add_argument('--compare-direct-pack',action='store_true',help='Matched original vs fused scale-producer timing and bitwise check')
    p.add_argument('--graph-batch',type=int,default=1,help='Operators per captured graph; reduces host replay gaps in diagnostic timing')
    p.add_argument('--timing-rounds',type=int,default=5)
    p.add_argument('--phases',type=int,nargs='+',default=[0],choices=range(4))
    p.add_argument('--resource-mode',choices=['shared-calibration','exclusive'],default='shared-calibration')
    p.add_argument('--device',choices=['0','3'],default='0')
    a=p.parse_args()
    if a.graph_batch<1 or a.timing_rounds<1:p.error('Positive graph batch and timing rounds required')
    if a.output.exists():raise FileExistsError(a.output)
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        if a.resource_mode=='shared-calibration':resource=select_shared_device()
        else:
            import os
            os.environ['CUDA_VISIBLE_DEVICES']=a.device
            from thunderworld_fp4.cosmos.cli import idle_guard
            idle_guard()
            resource=dict(device=int(a.device),resource_mode='exclusive',performance_eligible=False)
        import torch
        from fused import CutlassFused as CutlassBridge
        from thunderworld_fp4.cosmos.assets import expand
        from thunderworld_fp4.cosmos.budget import RepairPlan,linear_cost
        from thunderworld_fp4.cosmos.linears import RepairLinear
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        from thunderworld_fp4.cosmos.tuning import identity
        config=json.loads(a.config.read_text());plan_path=Path(config['budget_plan'])
        if not plan_path.is_absolute():plan_path=a.config.parent/plan_path
        plan=RepairPlan.load(plan_path)
        capture=json.loads((a.capture/'manifest.json').read_text());assets=json.loads((a.assets/'manifest.json').read_text())
        if capture['model_identity']!=assets['model_identity'] or capture['cohort_sha256']!=assets['cohort_sha256']:
            raise ValueError('Capture/asset provenance mismatch')
        result=dict(complete=False,resource=resource,performance_eligible=False,debug='timing' if not a.correctness_only else 'off',
            scope='Fused CUTLASS operator on calibration panels; not full trajectory quality or formal latency',
            config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
            assets_sha256=hashlib.sha256((a.assets/'manifest.json').read_bytes()).hexdigest(),
            library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
            torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),device_identity=identity(),rows=[])
        result['timing_protocol']=dict(graph_batch=a.graph_batch,rounds=a.timing_rounds,replays=max(8,100//a.graph_batch),alternating_order=True)
        def save():a.output.write_text(json.dumps(result,indent=2)+'\n')
        save()
        names=['layers.0.self_attn.add_q_proj','layers.0.self_attn.add_k_proj',
               'layers.0.mlp_moe_gen.up_proj','layers.0.mlp_moe_gen.down_proj']
        try:
            with torch.inference_mode():
                for ph,name in [(ph,name) for ph in a.phases for name in names]:
                    row=next(r for r in capture['rows'] if r['kind']=='linear' and r['name']==name and r['phase']==ph and r['branch']==0)
                    panel=torch.load(a.capture/row['file'],map_location='cpu',weights_only=True)
                    x=panel['x'].cuda();gold=panel['y'].cuda().float();m,k=x.shape;n=gold.shape[1]
                    spec=plan.linear(plan.value['task'],0,ph,name)
                    union=sorted(set(i for key,v in plan.linears.items() if key.split('|',2)[-1]==name for i in v['w_blocks']))
                    asset=expand(torch.load(a.assets/assets['modules'][name]['file'],map_location='cpu',weights_only=True),union)
                    source=torch.nn.Linear(k,n,bias=False,device='meta',dtype=torch.bfloat16)
                    controller=SimpleNamespace(budget_plan=plan,config=SimpleNamespace(kernel_variant='optimized'))
                    layer=RepairLinear(source,controller,name,spec,asset);layer.prepare_shape(m,spec);native=layer.bound
                    bridge=CutlassBridge(native,m,a.library)
                    def compute(op,direct=False):
                        if isinstance(op,CutlassBridge):op.direct_scale_pack=direct
                        layer.bound=op;return layer._compute(x)
                    captured=[];original=native.run_packed
                    def record(inp,packed):captured.append(packed);return original(inp,packed)
                    native.run_packed=record
                    expected=compute(native)
                    native.run_packed=original;packed=captured[-1]
                    actual=compute(bridge)
                    direct_equal=None
                    if a.compare_direct_pack:
                        direct_equal=torch.equal(actual,compute(bridge,True))
                        if not direct_equal:raise ValueError(f'Direct scale producer changed output: {name}, phase {ph}')
                    error=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-20))
                    if error>=.002:raise ValueError(f'Bridge numerical mismatch {name}: {error}')
                    nmse=lambda y:float((y.float()-gold).square().sum()/gold.square().sum().clamp_min(1e-20))
                    cost=linear_cost(m,n,k,bm=native.bm,a_blocks=native.na,w_blocks=native.weight_ids,cross=native.cross)
                    cost['launch']['bn']=bridge.tile_n
                    cost['execution_ctas']=((m+127)//128)*(n//bridge.tile_n)
                    cost['repair_block']=[128,128]
                    scratch=0
                    item=dict(module=name,phase=ph,branch=0,shape=[m,n,k],repair=spec.key,bridge_relative_l2_vs_native=error,
                        bitwise_equal=torch.equal(actual,expected),native_nmse=nmse(expected),bridge_nmse=nmse(actual),
                        cost=cost,extra_fp32_scratch_bytes=scratch,extra_fp32_traffic_bytes=2*scratch,
                        extra_weight_scale_bytes=bridge.w0_scales.numel(),workspace_bytes=bridge.workspace.numel(),timing_ms={})
                    if a.compare_direct_pack:
                        item['direct_pack_bitwise']=direct_equal
                        item['direct_pack_removed_scale_read_bytes']=packed[1][0].numel()
                        item['direct_pack_removed_launches']=1
                    if not a.correctness_only:
                        if a.resource_mode=='exclusive':idle_guard()
                        funcs={'native_total':lambda:compute(native),'bridge_total':lambda:compute(bridge),
                               'native_packed':lambda:native.run_packed(x,packed),'bridge_packed':lambda:bridge.run_packed(x,packed)}
                        if a.compare_direct_pack:funcs['bridge_direct_total']=lambda:compute(bridge,True)
                        graphs={};keepers=[]
                        for key,fn in funcs.items():
                            for _ in range(3):fn()
                            g=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):
                                for _ in range(a.graph_batch):y=fn()
                            graphs[key]=g;keepers.append(y)
                        measurements={key:[] for key in funcs}
                        # Fixed alternating order; no launch search or leaderboard update.
                        replays=max(8,100//a.graph_batch)
                        for repeat in range(a.timing_rounds):
                            keys=list(funcs)
                            if repeat%2:keys.reverse()
                            for key in keys:
                                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(replays):graphs[key].replay()
                                end.record();end.synchronize()
                                measurements[key].append(start.elapsed_time(end)/(replays*a.graph_batch))
                        item['timing_ms']={key:dict(median=statistics.median(v),samples=v) for key,v in measurements.items()}
                        item['diagnostic_native_over_bridge']=item['timing_ms']['native_total']['median']/item['timing_ms']['bridge_total']['median']
                        if a.compare_direct_pack:
                            item['direct_pack_speedup']=item['timing_ms']['bridge_total']['median']/item['timing_ms']['bridge_direct_total']['median']
                        if a.resource_mode=='exclusive':idle_guard()
                        del graphs,keepers
                    result['rows'].append(item);save();print(json.dumps(item),flush=True)
                    del layer,native,bridge,packed,expected,actual,x,gold,panel
                result['complete']=True;save()
        except Exception as error:
            result['error']=repr(error);save();raise


if __name__=='__main__':main()

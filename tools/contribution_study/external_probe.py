"""Official SharQ native/simulation comparison on actual generation projections."""
import argparse,json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('sources','capture','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();lock=json.loads(a.sources.read_text())['sources']['sharq']
    capture=json.loads((a.capture/'manifest.json').read_text())
    if not capture['complete']:raise ValueError('Incomplete capture')
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch
        from external import load_sharq,SharQLinear
        official=load_sharq(lock['path'],lock['commit'])
        result=dict(complete=False,resource=resource,performance_eligible=False,rows=[],
            source_commit=lock['commit'],lock_sha256=sha(a.sources),
            native_simulation_bitwise_equivalence_required=False,
            scope='Official native and official simulation are measured separately; numerical disagreement is reported, not hidden.')
        write(a.output,result)
        for name in list(capture['weights'])[:6]:
            weight=torch.load(a.capture/capture['weights'][name],weights_only=True).cuda()
            source=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,device='meta',dtype=weight.dtype)
            source.weight=torch.nn.Parameter(weight,requires_grad=False)
            native=SharQLinear(source,official,'SHARQ');simulation=SharQLinear(source,official,'SHARQ_SIM')
            r=next(r for r in capture['rows'] if r['kind']=='linear' and r['name']==name)
            panel=torch.load(a.capture/r['file'],weights_only=True)
            x=panel['x'].cuda();gold=panel['y'].cuda().float()
            with torch.inference_mode():
                y=native(x);s=simulation(x)
                upstream=native.official(x)
                batched=native(x.unsqueeze(0)).squeeze(0)
                if not torch.equal(y,upstream) or not torch.equal(y,batched):
                    raise ValueError('Cosmos wrapper differs from official native forward')
                nmse=lambda pred:float((pred.float()-gold).square().sum()/gold.square().sum().clamp_min(1e-20))
                row=dict(module=name,shape=[len(x),source.out_features,source.in_features],
                    native_nmse=nmse(y),simulation_nmse=nmse(s),
                    official_adapter_bitwise_equal=True,leading_dimension_bitwise_equal=True,
                    native_simulation_relative_l2=float((y.float()-s.float()).norm()/s.float().norm()),
                    finite=bool(y.isfinite().all()),shape_correct=list(y.shape)==list(gold.shape))
                if not row['finite'] or not row['shape_correct']:raise ValueError('Baseline correctness failed')
            result['rows'].append(row);write(a.output,result);print(json.dumps(row),flush=True)
            del weight,source,native,simulation,x,gold,y,s
            torch.cuda.empty_cache()
        result['complete']=True;write(a.output,result)


if __name__=='__main__':main()

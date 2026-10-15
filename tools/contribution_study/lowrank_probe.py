"""Official ModelOpt lite-SVDQuant operator control, not Nunchaku kernel timings."""
import argparse,copy,json,math
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','sources','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ranks',type=int,nargs='+',default=[32,64]);a=p.parse_args()
    capture=json.loads((a.capture/'manifest.json').read_text())
    if not capture['complete']:raise ValueError('Complete capture required')
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch
        import modelopt.torch.quantization as mtq
        torch.set_num_threads(8);torch.manual_seed(170917)
        result=dict(complete=False,performance_eligible=False,resource=resource,rows=[],
            sources_sha256=sha(a.sources),capture_sha256=sha(a.capture/'manifest.json'),
            scope='Official NVIDIA ModelOpt lite-SVDQuant numerical operator control. '
                  'Not a Nunchaku native Cosmos port; high-precision compensation listed separately.')
        with torch.inference_mode():
            for name in list(capture['weights'])[:6]:
                weight=torch.load(a.capture/capture['weights'][name],weights_only=True).cuda()
                rows=[r for r in capture['rows'] if r['kind']=='linear' and r['name']==name]
                panels=[torch.load(a.capture/r['file'],weights_only=True)['x'].cuda() for r in rows]
                for rank in [0,*a.ranks]:
                    linear=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,device='meta',dtype=weight.dtype)
                    linear.weight=torch.nn.Parameter(weight.clone(),requires_grad=False);model=torch.nn.Sequential(linear)
                    config=copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG if rank else mtq.NVFP4_DEFAULT_CFG)
                    if rank:config['algorithm']['lowrank']=rank
                    mtq.quantize(model,config,forward_loop=lambda current:[current(x) for x in panels])
                    errors=[]
                    for x in panels:
                        gold=torch.nn.functional.linear(x,weight).float();pred=model(x).float()
                        errors.append(float((pred-gold).square().sum()/gold.square().sum().clamp_min(1e-20)))
                    m,k=panels[0].shape;n=weight.shape[0]
                    row=dict(name=name,rank=rank,configuration=config,panel_nmse=errors,mean_nmse=sum(errors)/len(errors),
                        costs=dict(base_fp4_logical_flops=2*m*n*k,lowrank_bf16_logical_flops=2*m*rank*(k+n),
                            lowrank_weight_bytes=2*rank*(n+k),lowrank_activation_bytes=2*m*rank,
                            accounting='Logical operations of a numerical reference. Actual kernel padding is not qualified.'))
                    if not all(math.isfinite(x) for x in errors):raise ValueError('Nonfinite SVDQuant result')
                    result['rows'].append(row);write(a.output,result)
                    print(json.dumps(dict(name=name,rank=rank,mean_nmse=row['mean_nmse'])),flush=True)
                    del model,linear
            result['complete']=True;write(a.output,result)


if __name__=='__main__':main()

"""Explicit offline development capture; never used for timing or final seeds."""
import argparse,json,hashlib
from pathlib import Path
import torch
from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
from thunderworld_fp4.cosmos.settings import phase


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',nargs='+',type=int,default=[0,12,24,35]);p.add_argument('--seed',type=int,default=7101)
    a=p.parse_args()
    if a.seed in range(91001,91009) or a.seed in range(92001,92009):raise ValueError('Reserved final/validation seed')
    a.output.mkdir(parents=True,exist_ok=False)
    case=json.loads(a.case.read_text())
    if case.get('split')!='development':raise ValueError('Capture requires an explicit development case')
    if case.get('image'):case['image']=str((a.case.parent/case['image']).resolve())
    runner=CosmosRunner(a.model,RunConfig(precision='bf16',attention_backend='bf16',execution='eager'),audio=case['task']=='av').prepare()
    runner.warmup(case)
    rows=[];weights={};seen=set();handles=[]
    def save_row(kind,name,payload):
        e=runner.engine;ph=phase(e.step,e.steps);key=(kind,name,e.branch,ph)
        if key in seen:return
        seen.add(key);file=f'{len(rows):05d}.pt';torch.save(payload,a.output/file)
        rows.append(dict(kind=kind,name=name,branch=e.branch,phase=ph,step=e.step,file=file,
                         sha256=hashlib.sha256((a.output/file).read_bytes()).hexdigest()))
    def hook(name):
        def callback(module,args,out):
            x=args[0].reshape(-1,module.in_features)
            if name not in weights:
                file=f'weight_{len(weights):03d}.pt';torch.save(module.weight.detach().cpu(),a.output/file);weights[name]=file
            save_row('linear',name,dict(x=x.detach().cpu().contiguous(),y=out.reshape(-1,module.out_features).detach().cpu().contiguous()))
        return callback
    for name in runner.engine.targets:
        if int(name.split('.')[1]) in a.layers:handles.append(runner.pipe.transformer.get_submodule(name).register_forward_hook(hook(name)))
    def attention(layer,q,k,v,out):
        if layer in a.layers:save_row('attention',str(layer),dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),y=out.cpu()))
    runner.engine.capture_attention=attention
    with torch.inference_mode():runner.infer(case,a.seed)
    for h in handles:h.remove()
    (a.output/'manifest.json').write_text(json.dumps(dict(complete=True,split='development',seed=a.seed,
        case=case,model_identity=runner.identity,weights=weights,rows=rows,
        scope='Full current-input M, not 64-row crops; synchronous diagnostic capture, no timing claim'),indent=2)+'\n')


if __name__=='__main__':main()

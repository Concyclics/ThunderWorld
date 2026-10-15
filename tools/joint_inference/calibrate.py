"""Training-free prescale / represented-scale search on complete M contexts."""
import argparse,json,hashlib,math
from pathlib import Path
from types import SimpleNamespace
import torch
from thunderworld_fp4._calibrated import pack
from thunderworld_fp4.cosmos.linears import RepairLinear
from thunderworld_fp4.cosmos.settings import TIERS
from thunderworld_fp4.cosmos.tuning import LaunchTable
from thunderworld_fp4.cosmos.assets import expand, compact


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path)
    p.add_argument('--assets',type=Path,required=True,help='Prepared FP4 assets for the identical model')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tier',choices=list(TIERS),default='four')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    capture=json.loads((a.capture/'manifest.json').read_text());manifest=json.loads((a.assets/'manifest.json').read_text())
    if capture['split']!='development' or not capture['complete'] or capture['model_identity']!=manifest['model_identity']:
        raise ValueError('Calibration split/model mismatch')
    controller=SimpleNamespace(warming=True,table=LaunchTable(),linear_spec=lambda n,d:d,record_linear=lambda *a:None)
    decisions=[]
    for name,entry in manifest['modules'].items():
        asset=expand(torch.load(a.assets/entry['file'],weights_only=True,map_location='cpu'),1.)
        panels=[r for r in capture['rows'] if r['kind']=='linear' and r['name']==name]
        if panels:
            w=torch.load(a.capture/capture['weights'][name],weights_only=True,map_location='cpu').cuda()
            data=[torch.load(a.capture/r['file'],weights_only=True,map_location='cpu') for r in panels]
            source=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=torch.bfloat16,device='meta')
            source.weight=torch.nn.Parameter(w,requires_grad=False)
            mean=torch.stack([x['x'].float().abs().mean(0) for x in data]).mean(0).cuda().clamp_min(1e-6)
            wmax=w.float().abs().amax(0).clamp_min(1e-6)
            def loss(candidate):
                layer=RepairLinear(source,controller,name,TIERS[a.tier],candidate)
                errors=[]
                for panel in data:
                    x=panel['x'].cuda();gold=panel['y'].cuda().float();y=layer(x).float()
                    errors.append(float((y-gold).square().sum()/gold.square().sum().clamp_min(1e-20)))
                return sum(errors)/len(errors)
            best=asset;best_loss=loss(asset);tested=[]
            # Coordinate search: preserve the global factor from the complete X.
            for alpha in (0.,.25,.5,.75,1.):
                d=(mean.pow(alpha)/wmax.pow(1-alpha)).clamp(1/16,16)
                d=d/(d.min()*d.max()).sqrt()
                for base in (1.,.875):
                    c,s,g,_=pack((w.float()/d).bfloat16().contiguous(),torch.ones_like(d),True,True,w,base=base)
                    candidate={**asset,'codes':c.cpu(),'scales':s.cpu(),'factor':g.cpu(),'prescale':d.cpu()}
                    candidate['score']=s[1].float().reshape(w.shape[0]//128,128,w.shape[1]//128,8).square().sum((1,3)).cpu()
                    err=loss(candidate);tested.append(dict(alpha=alpha,weight_base_scale=base,loss=err))
                    if err<best_loss:best,best_loss=candidate,err
            for base,res in ((.875,1.),(1.,.875),(1.,1.125)):
                candidate={**best,'activation_scale':(base,res)}
                err=loss(candidate);tested.append(dict(activation_scale=[base,res],loss=err))
                if err<best_loss:best,best_loss=candidate,err
            asset=best;decisions.append(dict(module=name,loss=best_loss,candidates=tested))
        path=a.output/entry['file'];torch.save(compact(asset,TIERS[a.tier].weight),path)
        entry['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['calibration']=dict(source=str(a.capture),tier=a.tier,decisions=decisions,
        scope='Development teacher-input proxy; full trajectories must independently validate and select assets')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

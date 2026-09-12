"""All-layer zero-repair coordinate calibration. Actual native FP4 output objective."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
from capture_budget import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    capture=json.loads((a.capture/'manifest.json').read_text())
    if not capture['complete'] or capture['split']!='development':raise ValueError('Complete development capture required')
    with shared_worker():
        resource=select_shared_device()
        import torch
        from thunderworld_fp4.cosmos.calibration import prescale_candidates,panel_objective
        from thunderworld_fp4._calibrated import pack
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec
        from thunderworld_fp4.cosmos.tuning import LaunchTable
        from thunderworld_fp4.cosmos.assets import compact
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        spec=LinearSpec(0,0)
        controller=SimpleNamespace(warming=True,table=LaunchTable(),linear_spec=lambda n,d:d,record_linear=lambda *a:None)
        path=a.output/'manifest.json'
        manifest=dict(complete=False,precision='fp4',model_identity=capture['model_identity'],modules={},
            calibration_round=capture.get('calibration_round',0),cohort_sha256=capture['cohort_sha256'],capture_sha256=hashlib.sha256((a.capture/'manifest.json').read_bytes()).hexdigest(),
            calibration=dict(objective='mean panel NMSE + 0.25 * worst panel NMSE',budget=1.,
                weight_residual_scale=1.,residual_scope='No repair in base calibration; residual scales searched with repair',
                coordinate_rounds=1,resource=resource,performance_eligible=False),decisions=[])
        if a.resume and path.exists():
            previous=json.loads(path.read_text())
            for field in ('cohort_sha256','capture_sha256','model_identity'):
                if previous[field]!=manifest[field]:raise ValueError('Resume provenance mismatch')
            manifest=previous
        atomic_json(path,manifest)
        try:
            with torch.inference_mode():
                for index,(name,wfile) in enumerate(capture['weights'].items()):
                    if name in manifest['modules']:continue
                    rows=[r for r in capture['rows'] if r['kind']=='linear' and r['name']==name]
                    expected=len(capture['cohort']['seeds']['calibration'])*4*(1 if capture['case']['task']=='policy' else 2)
                    if len(rows)!=expected:raise ValueError(f'Expected {expected} panels: {name}')
                    data=[torch.load(a.capture/r['file'],map_location='cpu',weights_only=True) for r in rows]
                    rms=torch.stack([p['x'].float().square().mean(0) for p in data]).mean(0).sqrt().cuda()
                    w=torch.load(a.capture/wfile,map_location='cpu',weights_only=True).cuda()
                    source=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device='meta',dtype=torch.bfloat16)
                    source.weight=torch.nn.Parameter(w,requires_grad=False)
                    tested=[];best=None;best_loss=float('inf');best_meta=None
                    def evaluate(asset,meta):
                        nonlocal best,best_loss,best_meta
                        layer=RepairLinear(source,controller,name,spec,asset)
                        errors=[]
                        for panel in data:
                            x=panel['x'].cuda();gold=panel['y'].cuda().float()
                            pred=layer(x).float()
                            errors.append(float((pred-gold).square().sum()/gold.square().sum().clamp_min(1e-20)))
                        loss=panel_objective(errors)
                        tested.append(dict(**meta,loss=loss,mean_nmse=sum(errors)/len(errors),worst_nmse=max(errors),panel_nmse=errors))
                        if loss<best_loss:best,best_loss,best_meta=asset,loss,meta
                    for label,d in prescale_candidates(rms,w):
                        for base in (.75,.875,1.):
                            c,s,g,_=pack((w.float()/d).bfloat16().contiguous(),torch.ones_like(d),True,True,w,base=base)
                            asset=dict(codes=c.cpu(),scales=s.cpu(),factor=g.cpu(),prescale=d.cpu(),bias=None,
                                score=s[1].float().reshape(w.shape[0]//128,128,w.shape[1]//128,8).square().sum((1,3)).cpu(),
                                activation_scale=(1.,1.),weight_scale=(base,1.),
                                activation_sensitivity=(w.float()/d).square().sum(0).cpu(),selector='residual_sensitivity')
                            evaluate(asset,dict(prescale=label,weight_base=base,activation_base=1.))
                    selected_meta=dict(best_meta)
                    for base in (.75,.875):
                        candidate={**best,'activation_scale':(base,1.)}
                        evaluate(candidate,{**selected_meta,'activation_base':base})
                    outfile=f'{index:04d}.pt'
                    # Keep a full residual bank on disk for subsequent offline budget search.
                    # Each final budget asset is compacted separately before loading.
                    torch.save(compact(best,1.),a.output/outfile)
                    manifest['modules'][name]=dict(file=outfile,sha256=hashlib.sha256((a.output/outfile).read_bytes()).hexdigest())
                    identity=next(t for t in tested if t['prescale']=='identity' and t['weight_base']==1.)
                    manifest['decisions'].append(dict(module=name,best=best_meta,loss=best_loss,baseline_loss=identity['loss'],candidates=tested))
                    atomic_json(path,manifest)
                    print(json.dumps(dict(module=name,index=index,loss=best_loss,baseline=identity['loss'])),flush=True)
                    del w,source,data,best
            manifest['complete']=True;manifest.pop('error',None);atomic_json(path,manifest)
        except Exception as error:
            manifest.update(error=repr(error),paused=True);atomic_json(path,manifest);raise


if __name__=='__main__':main()

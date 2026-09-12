"""One offline coordinate pass over residual scales and budget-reserved outlier splits.

Only the intersection of already paid W1 blocks across all phases/CFG may remove
W0 outliers. These blocks are marked mandatory. W0 remains immutable per request.
"""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
from thunderworld_fp4.cosmos.budget import RepairPlan
from capture_budget import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','candidate','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--search-prescale',action='store_true',help='One additional teacher+student coordinate search with fixed block supports')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    capture=json.loads((a.capture/'manifest.json').read_text());plan=RepairPlan.load(a.candidate/'plan.json')
    source=a.candidate/'assets';manifest=json.loads((source/'manifest.json').read_text())
    if capture['model_identity']!=manifest['model_identity']:raise ValueError('Model mismatch')
    manifest={**manifest,'complete':False,'modules':dict(manifest['modules']),'refinement':[]}
    if a.resume and (a.output/'manifest.json').exists():
        previous=json.loads((a.output/'manifest.json').read_text())
        if previous.get('plan_sha256')!=manifest.get('plan_sha256'):raise ValueError('Resume plan mismatch')
        manifest=previous
    completed={r['name'] for r in manifest['refinement']}
    with shared_worker():
        resource=select_shared_device()
        import torch
        from thunderworld_fp4.cosmos.calibration import panel_objective,outlier_mask,quantize_weight,prescale_candidates
        from thunderworld_fp4.cosmos.assets import expand,compact
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.tuning import LaunchTable
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        manifest['resource']=resource
        manifest['refinement_options']=dict(search_prescale=a.search_prescale,calibration_round=capture.get('calibration_round',0),coordinate_passes=1)
        try:
            with torch.inference_mode():
                for name,entry in manifest['modules'].items():
                    if name in completed:continue
                    rules=[r for key,r in plan.linears.items() if key.split('|',2)[-1]==name]
                    union=sorted(set(i for r in rules for i in r['w_blocks']))
                    reserved=sorted(set.intersection(*(set(r['w_blocks']) for r in rules)))
                    has_a=any(r['a_blocks'] for r in rules)
                    asset=expand(torch.load(source/entry['file'],weights_only=True,map_location='cpu'),union)
                    if not union and not has_a and not a.search_prescale:
                        outfile=a.output/entry['file'];torch.save(compact(asset,union),outfile)
                        entry['sha256']=hashlib.sha256(outfile.read_bytes()).hexdigest()
                        manifest['refinement'].append(dict(name=name,skipped='No residual computation in any phase'))
                        atomic_json(a.output/'manifest.json',manifest);continue
                    rows=[r for r in capture['rows'] if r['kind']=='linear' and r['name']==name]
                    panels=[{key:value.cuda() for key,value in torch.load(a.capture/r['file'],weights_only=True,map_location='cpu').items()} for r in rows]
                    w=torch.load(a.capture/capture['weights'][name],weights_only=True,map_location='cpu').cuda()
                    rms=torch.stack([p['x'].float().square().mean(0) for p in panels]).mean(0).sqrt().cuda()
                    d=asset['prescale'].cuda()
                    layer_source=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device='meta',dtype=torch.bfloat16)
                    layer_source.weight=torch.nn.Parameter(w,requires_grad=False)
                    current=[None]
                    controller=SimpleNamespace(warming=True,table=LaunchTable(),linear_spec=lambda n,d:current[0],record_linear=lambda *a:None)
                    tested=[]
                    def evaluate(candidate,label):
                        layer=RepairLinear(layer_source,controller,name,plan.linear(plan.value['task'],0,0,name),candidate);errors=[]
                        for row,panel in zip(rows,panels):
                            current[0]=plan.linear(plan.value['task'],row['branch'],row['phase'],name)
                            x=panel['x'].cuda();gold=panel['y'].cuda().float();pred=layer(x).float()
                            errors.append((pred-gold).square().sum()/gold.square().sum().clamp_min(1e-20))
                        loss=panel_objective(torch.stack(errors).cpu().tolist());tested.append(dict(label=label,loss=loss));return loss
                    best=asset;best_loss=evaluate(best,'original')
                    if a.search_prescale:
                        if capture.get('calibration_round')!=1:raise ValueError('Additional prescale search requires the single student calibration round')
                        if asset.get('mandatory_w_blocks'):raise ValueError('Prescale search requires an asset without removed W0 outliers')
                        # Fixed block supports and the current panel input are used for every candidate.
                        wb=asset.get('weight_scale',(1.,1.))[0]
                        for label,new_d in prescale_candidates(rms,w):
                            c,s,g=quantize_weight(w,new_d,base=wb,residual=asset.get('weight_scale',(1.,1.))[1])
                            candidate={**best,'codes':c.cpu(),'scales':s.cpu(),'factor':g.cpu(),'prescale':new_d.cpu(),
                                'activation_sensitivity':(w.float()/new_d).square().sum(0).cpu(),'mandatory_w_blocks':(),
                                'weight_scale':(wb,asset.get('weight_scale',(1.,1.))[1]),'prescale_search':label}
                            loss=evaluate(candidate,f'prescale={label},wbase={wb}')
                            if loss<best_loss:best,best_loss=candidate,loss
                        for wb in (.75,.875,1.):
                            new_d=best['prescale'].cuda()
                            c,s,g=quantize_weight(w,new_d,base=wb,residual=best['weight_scale'][1])
                            candidate={**best,'codes':c.cpu(),'scales':s.cpu(),'factor':g.cpu(),'weight_scale':(wb,best['weight_scale'][1])}
                            loss=evaluate(candidate,f'wbase={wb}')
                            if loss<best_loss:best,best_loss=candidate,loss
                        for ab in (.75,.875,1.):
                            candidate={**best,'activation_scale':(ab,best['activation_scale'][1])}
                            loss=evaluate(candidate,f'abase={ab}')
                            if loss<best_loss:best,best_loss=candidate,loss
                        d=best['prescale'].cuda();asset=best
                    # Offline weight residual scale; unchanged W0 with no removed elements.
                    weight_base=asset.get('weight_scale',(1.,1.))[0]
                    for threshold in ((None,2.,3.) if union else ()):
                        if threshold is not None and not reserved:continue
                        remove,mandatory=outlier_mask(w,d,rms,reserved,threshold)
                        for residual_scale in (.875,1.,1.125):
                            c,s,g=quantize_weight(w,d,base=weight_base,residual=residual_scale,remove=remove)
                            candidate={**best,'codes':c.cpu(),'scales':s.cpu(),'factor':g.cpu(),
                                'mandatory_w_blocks':mandatory,'outlier_threshold':threshold,'weight_scale':(weight_base,residual_scale)}
                            loss=evaluate(candidate,f'outlier={threshold},wres={residual_scale}')
                            if loss<best_loss:best,best_loss=candidate,loss
                    for scale in ((.875,1.,1.125) if has_a else ()):
                        candidate={**best,'activation_scale':(best['activation_scale'][0],scale)}
                        loss=evaluate(candidate,f'ares={scale}')
                        if loss<best_loss:best,best_loss=candidate,loss
                    outfile=a.output/entry['file'];torch.save(compact(best,union),outfile)
                    entry['sha256']=hashlib.sha256(outfile.read_bytes()).hexdigest()
                    manifest['refinement'].append(dict(name=name,reserved_w_blocks=reserved,
                        mandatory_w_blocks=list(best.get('mandatory_w_blocks',())),best_loss=best_loss,trials=tested))
                    atomic_json(a.output/'manifest.json',manifest)
                    print(json.dumps(dict(module=name,refined=len(manifest['refinement']),mandatory=len(best.get('mandatory_w_blocks',())))),flush=True)
            manifest['complete']=True;atomic_json(a.output/'manifest.json',manifest)
        except Exception as error:
            manifest.update(error=repr(error),paused=True);atomic_json(a.output/'manifest.json',manifest);raise


if __name__=='__main__':main()

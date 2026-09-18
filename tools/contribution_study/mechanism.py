"""Full-M, all-phase representation interventions on existing teacher captures."""
import json
from pathlib import Path
from baselines import sha, write


def run(args):
    capture=json.loads((args.capture/'manifest.json').read_text())
    assets=json.loads((args.assets/'manifest.json').read_text())
    if not capture['complete'] or not assets['complete'] or capture['model_identity']!=assets['model_identity']:
        raise ValueError('Complete, matching model capture and assets required')
    if capture['split']!='development' or capture.get('trajectory','teacher')!='teacher':
        raise ValueError('This mechanism study requires teacher development panels')
    names=args.modules or list(assets['modules'])
    if len(set(names))!=len(names) or not set(names).issubset(assets['modules']):
        raise ValueError('Unknown or duplicated module')
    provenance=dict(capture_sha256=sha(args.capture/'manifest.json'),
        assets_sha256=sha(args.assets/'manifest.json'),modules=names,threshold=args.threshold,
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    args.output.mkdir(parents=True,exist_ok=args.resume)
    path=args.output/'mechanism.json'
    result=dict(complete=False,performance_eligible=False,provenance=provenance,modules={},
        execution_type='independent_decoded_fp32_reference',native_qualified=False,
        scope='Full-M teacher-input mechanism evidence, not generation quality or native speed. '
              'Full W1 support is reserved for these 3/4-term diagnostic anchors; not a <=2.5 budget candidate. '
              'Joint diagonal selector is an activation-weighted group reconstruction surrogate; '
              'acceptance uses actual full-panel output error. Removed-outlier base/ACT2 results are invalid policies.')
    if args.resume and path.exists():
        result=json.loads(path.read_text())
        if result['provenance']!=provenance:raise ValueError('Resume provenance changed')
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        result['resource']=select_shared_device()
        import torch
        from thunderworld_fp4.cosmos.assets import expand
        from thunderworld_fp4.cosmos.calibration import panel_objective
        from representation import planes, removal_masks, terms
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False
        write(path,result)
        try:
            with torch.inference_mode():
                for name in names:
                    if name in result['modules']:continue
                    rows=[r for r in capture['rows'] if r['kind']=='linear' and r['name']==name]
                    if len(rows)!=16 or {(r['seed'],r['phase'],r['branch']) for r in rows}!={
                        (s,p,b) for s in (7101,7102) for p in range(4) for b in range(2)}:
                        raise ValueError('Expected 16 full-M calibration panels')
                    entry=assets['modules'][name]
                    if sha(args.assets/entry['file'])!=entry['sha256']:raise ValueError('Changed asset')
                    asset=expand(torch.load(args.assets/entry['file'],weights_only=True),1.)
                    d=asset['prescale'].cuda().float()
                    weight_file=args.capture/capture['weights'][name]
                    original_w=torch.load(weight_file,weights_only=True).cuda()
                    w=(original_w.float()/d).bfloat16().float()
                    panels=[]
                    for r in rows:
                        if sha(args.capture/r['file'])!=r['sha256']:raise ValueError('Changed panel')
                        panel=torch.load(args.capture/r['file'],weights_only=True)
                        x=(panel['x'].cuda().float()*d).bfloat16().float()
                        panels.append((x,panel['y'].cuda().float()))
                    rms=torch.stack([x.square().mean(0) for x,_ in panels]).mean(0).sqrt()
                    n,k=w.shape
                    allowed=torch.ones((n//128,k//128),device=w.device,dtype=torch.bool)
                    masks=removal_masks(w,rms,allowed,args.threshold)
                    variants={'rtn':planes(w),'four_six':planes(w,'four_six')}
                    variants.update({key:planes(w,remove=mask) for key,mask in masks.items()})
                    sums={key:{t:[] for t in ('base','act2','weight2','three','four')} for key in variants}
                    panel_details=[]
                    scaling_errors=[]
                    for row,(x,gold) in zip(rows,panels):
                        a=planes(x)
                        denominator=gold.square().sum().clamp_min(1e-20)
                        scaling_errors.append(float(((x@w.T)-gold).square().sum()/denominator))
                        for label,wp in variants.items():
                            values=terms(a,wp)
                            metrics={key:float((value-gold).square().sum()/denominator) for key,value in values.items()}
                            for key,error in metrics.items():sums[label][key].append(error)
                            panel_details.append(dict(seed=row['seed'],phase=row['phase'],branch=row['branch'],
                                method=label,nmse=metrics))
                    m=panels[0][0].shape[0]
                    baseline_mma=2*((m+127)//128*128)*n*k
                    out=dict(shape=[m,n,k],weight_sha256=sha(weight_file),
                        prescale_sha256=entry['sha256'],panel_provenance=rows,
                        scaling_nmse=scaling_errors,panels=panel_details,
                        objective={label:{t:panel_objective(errors) for t,errors in values.items()} for label,values in sums.items()},
                        mma_per_call={t:baseline_mma*c for t,c in dict(base=1,act2=2,weight2=2,three=3,four=4).items()},
                        removed={label:int(mask.sum()) for label,mask in masks.items()},
                        invalid_policies={label:['base','act2'] for label in masks},
                        weight_union_density=1.,performance_eligible=False)
                    result['modules'][name]=out
                    write(path,result)
                    print(json.dumps(dict(module=name,objective=out['objective'],removed=out['removed'])),flush=True)
                    del panels,variants,masks,w,original_w,asset,rms
                    torch.cuda.empty_cache()
            result['complete']=True
            result.pop('error',None);result.pop('paused',None)
            write(path,result)
        except Exception as error:
            result.update(error=repr(error),paused=True)
            write(path,result)
            raise

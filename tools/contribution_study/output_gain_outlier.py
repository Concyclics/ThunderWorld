"""Full-output-response outlier proposals, distinct from diagonal RMS heuristics.

One pre-reserved N128/K128 block per operator is a bounded intervention. Each
K16 group tests all 16 positions against the current complete output; combined
choices are reevaluated, never scored by adding isolated gains. This is an
independent decoded reference with BF16 output rounding, not native timing.
"""
import argparse,json
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','assets','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--modules',nargs='+');p.add_argument('--all-modules',action='store_true')
    a=p.parse_args();capture=json.loads((a.capture/'manifest.json').read_text());assets=json.loads((a.assets/'manifest.json').read_text())
    names=a.modules or (list(assets['modules']) if a.all_modules else list(assets['modules'])[:6])
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch
        from representation import planes,encode,terms,mask_from_indices
        from thunderworld_fp4.cosmos.calibration import decode,panel_objective
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        result=dict(complete=False,rows=[],performance_eligible=False,resource=resource,
            capture_sha256=sha(a.capture/'manifest.json'),assets_sha256=sha(a.assets/'manifest.json'),source_sha256=sha(__file__),
            scope='One mandatory N128/K128 block per operator; full M and all 16 teacher panels. '
                  'Full-W1 three/four-term anchors, not a low-budget deployed asset. '
                  'Single-group output-response proposals include BF16 rounding; combined choices are reencoded and reevaluated. '
                  'FP32 algebra reordering is a numerical reference, not bitwise native equivalence.')
        with torch.inference_mode():
            for name in names:
                entry=assets['modules'][name]
                if sha(a.assets/entry['file'])!=entry['sha256']:raise ValueError('Changed asset')
                asset=torch.load(a.assets/entry['file'],weights_only=True);d=asset['prescale'].cuda().float()
                original=torch.load(a.capture/capture['weights'][name],weights_only=True).cuda()
                w=(original.float()/d).bfloat16().float();wp=planes(w);n,k=w.shape
                # Fixed support selected before trying any outlier: residual-score block.
                support=int(asset['score'].flatten().argmax());nb,kb=divmod(support,k//128)
                ns=slice(nb*128,(nb+1)*128);ks=slice(kb*128,(kb+1)*128)
                z=w[ns,ks]/wp['factor'];old0=wp['base'][ns,ks];old1=wp['residual'][ns,ks]
                panels=[];baselines=[];rms=torch.zeros(k,device='cuda')
                rows=[r for r in capture['rows'] if r['kind']=='linear' and r['name']==name]
                if len(rows)!=16:raise ValueError('All teacher panels required')
                for row in rows:
                    if sha(a.capture/row['file'])!=row['sha256']:raise ValueError('Changed panel')
                    panel=torch.load(a.capture/row['file'],weights_only=True)
                    x=(panel['x'].cuda().float()*d).bfloat16().float();ap=planes(x);gold=panel['y'].cuda().float()
                    base=terms(ap,wp)
                    panels.append((ap,gold));baselines.append({key:base[key] for key in ('three','four')})
                    rms+=x.square().mean(0)/len(rows)
                rms=rms.sqrt();contribution=w[ns,ks].abs().reshape(128,8,16)*rms[ks].reshape(1,8,16)
                active=contribution.amax(-1)>2*contribution.square().mean(-1).sqrt()
                delta0=[];delta1=[]
                for index in range(16):
                    mask=mask_from_indices(torch.full((128,8),index,device='cuda',dtype=torch.long),active,(128,128))
                    c0,s0=encode(z.masked_fill(mask,0.));b0=decode(c0,s0,1.)
                    c1,s1=encode(z-b0);b1=decode(c1,s1,1.)
                    delta0.append((b0*wp['factor']-old0).reshape(128,8,16))
                    delta1.append((b1*wp['factor']-old1).reshape(128,8,16))
                delta0=torch.stack(delta0,2);delta1=torch.stack(delta1,2)
                scores={key:[] for key in ('three','four')};base_loss={key:[] for key in scores}
                for (ap,gold),base in zip(panels,baselines):
                    m=gold.shape[0];denominator=gold.square().sum(dtype=torch.float64).clamp_min(1e-20)
                    qa=ap['base'][:,ks].reshape(m,8,16);qr=ap['residual'][:,ks].reshape(m,8,16)
                    shift0=torch.einsum('mgk,ngck->ngmc',qa+qr,delta0)
                    for key in scores:
                        shift=shift0+torch.einsum('mgk,ngck->ngmc',qa+qr if key=='four' else qa,delta1)
                        rounded=base[key].bfloat16().float();error=(rounded-gold).square()
                        total=error.sum(dtype=torch.float64);own=error[:,ns].sum(0,dtype=torch.float64)
                        trial=(base[key][:,ns].T[:,None,:,None]+shift).bfloat16().float()
                        trial_error=(trial-gold[:,ns].T[:,None,:,None]).square().sum(2,dtype=torch.float64)
                        scores[key].append((total-own[:,None,None]+trial_error)/denominator)
                        base_loss[key].append(float(total/denominator))
                local_masks={
                    'magnitude':mask_from_indices(w[ns,ks].abs().reshape(128,8,16).argmax(-1),active,(128,128)),
                    'quant_error':mask_from_indices((w[ns,ks]-old0).abs().reshape(128,8,16).argmax(-1),active,(128,128)),
                }
                for key in scores:
                    table=torch.stack(scores[key]);objective=table.mean(0)+.25*table.amax(0)
                    best,indices=objective.min(-1)
                    local_masks['output_'+key]=mask_from_indices(indices,active & (best<panel_objective(base_loss[key])),(128,128))
                objective={'rtn':{key:panel_objective(value) for key,value in base_loss.items()}}
                for label,local in local_masks.items():
                    mask=torch.zeros_like(w,dtype=torch.bool);mask[ns,ks]=local
                    represented=planes(w,remove=mask);errors={key:[] for key in scores}
                    for ap,gold in panels:
                        predictions=terms(ap,represented);denominator=gold.square().sum(dtype=torch.float64).clamp_min(1e-20)
                        for key in errors:errors[key].append(float((predictions[key].bfloat16().float()-gold).square().sum(dtype=torch.float64)/denominator))
                    objective[label]={key:panel_objective(value) for key,value in errors.items()}
                result['rows'].append(dict(name=name,shape=[panels[0][1].shape[0],n,k],mandatory_w_blocks=[support],
                    objective=objective,removed={key:int(value.sum()) for key,value in local_masks.items()}))
                write(a.output,result);print(json.dumps(dict(name=name,objective=objective)),flush=True)
                del panels,baselines,delta0,delta1,scores,wp,w,original
                torch.cuda.empty_cache()
            result['complete']=True;write(a.output,result)


if __name__=='__main__':main()

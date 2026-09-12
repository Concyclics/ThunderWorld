"""Measure whole A/W combinations on all full-M calibration panels, checkpoint per operator."""
import argparse,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
from capture_budget import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','assets','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    capture=json.loads((a.capture/'manifest.json').read_text());assets=json.loads((a.assets/'manifest.json').read_text())
    if not capture['complete'] or not assets['complete'] or capture['model_identity']!=assets['model_identity']:
        raise ValueError('Complete matching capture/assets required')
    with shared_worker():
        resource=select_shared_device()
        import torch
        from thunderworld_fp4.cosmos.calibration import decode,panel_objective
        from thunderworld_fp4._calibrated import pack
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec
        from thunderworld_fp4.cosmos.tuning import LaunchTable
        from thunderworld_fp4.cosmos.assets import expand,compact
        from thunderworld_fp4.cosmos.budget import linear_cost,attention_cost,PHASES
        from thunderworld_fp4 import fp4_attention,RepairConfig
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        controller=SimpleNamespace(warming=True,table=LaunchTable(),linear_spec=lambda n,d:d,record_linear=lambda *a:None)
        output=a.output/'search.json'
        result=dict(complete=False,resource=resource,task=capture['case']['task'],branches=1 if capture['case']['task']=='policy' else 2,cohort_sha256=capture['cohort_sha256'],
                    model_identity=capture['model_identity'],groups={},modules={},attention={},
                    assets_sha256=hashlib.sha256((a.assets/'manifest.json').read_bytes()).hexdigest(),
                    weight_ranking='Exact native single-K-block interventions, independent output N blocks',
                    scope='Teacher-input local combinations only; no claim of output quality or latency Pareto optimality')
        if a.resume and output.exists():
            old=json.loads(output.read_text())
            if old['assets_sha256']!=result['assets_sha256']:raise ValueError('Resume asset mismatch')
            result=old
            result.setdefault('task',capture['case']['task']);result.setdefault('branches',1 if capture['case']['task']=='policy' else 2)
        atomic_json(output,result)
        def record(kind,name,rows,evaluations,shape):
            for branch in range(result['branches']):
                for ph in range(4):
                    panel_ids=[i for i,r in enumerate(rows) if r['phase']==ph and r['branch']==branch]
                    if len(panel_ids)!=2:raise ValueError('Missing seed/phase/branch calibration pair')
                    group=[];calls=PHASES[ph+1]-PHASES[ph]
                    for i,(setting,losses,cost) in enumerate(evaluations):
                        errors=[losses[j] for j in panel_ids]
                        group.append(dict(id=i,kind=kind,name=name,branch=branch,phase=ph,
                            loss=panel_objective(errors)*calls,mean_nmse=sum(errors)/2,
                            mma=cost['total_mma']*calls,base_mma=cost['base_mma']*calls,
                            **setting))
                    result['groups'][f'{kind}|{branch}|{ph}|{name}']=group
        try:
            with torch.inference_mode():
                for name,entry in assets['modules'].items():
                    if name in result['modules']:continue
                    asset=expand(torch.load(a.assets/entry['file'],weights_only=True,map_location='cpu'),1.)
                    rows=[r for r in capture['rows'] if r['kind']=='linear' and r['name']==name]
                    data=[{key:value.cuda() for key,value in torch.load(a.capture/r['file'],weights_only=True,map_location='cpu').items()} for r in rows]
                    w=torch.load(a.capture/capture['weights'][name],weights_only=True,map_location='cpu').cuda();n,k=w.shape
                    source=torch.nn.Linear(k,n,bias=False,device='meta',dtype=torch.bfloat16);source.weight=torch.nn.Parameter(w,requires_grad=False)
                    nb,kb=n//128,k//128
                    ranking=torch.zeros((nb,kb),device='cuda')
                    base=RepairLinear(source,controller,name,LinearSpec(0,0),asset)
                    # For fixed K block, repair every N block in one native GEMM.
                    # Output N blocks are independent: each slice is exactly the
                    # corresponding single-W1-block intervention, including BF16 rounding.
                    base_losses=[]
                    for panel in data:
                        x=panel['x'].cuda();gold=panel['y'].cuda().float();pred=base(x).float()
                        base_losses.append((pred-gold).square().reshape(len(x),nb,128).sum((0,2))/gold.square().sum().clamp_min(1e-20))
                    for block in range(kb):
                        spec=LinearSpec(0,0,activation_blocks=0,weight_support=tuple(nblock*kb+block for nblock in range(nb)))
                        intervention=RepairLinear(source,controller,name,spec,asset)
                        for panel,original_loss in zip(data,base_losses):
                            x=panel['x'].cuda();gold=panel['y'].cuda().float();pred=intervention(x).float()
                            loss=(pred-gold).square().reshape(len(x),nb,128).sum((0,2))/gold.square().sum().clamp_min(1e-20)
                            ranking[:,block]+=(original_loss-loss)/len(data)
                    del intervention,base_losses
                    order=torch.argsort(ranking.flatten(),descending=True,stable=True).cpu().tolist()
                    asset['score']=ranking.cpu();asset['support_order']=order
                    evaluations=[]
                    grid=[(0,0,False)]+[(f,0,False) for f in (.125,.25,.5,1.)]+[(0,f,False) for f in (.125,.25,.5,1.)]
                    grid += [(f,f,cross) for f in (.125,.25,.5,1.) for cross in (False,True)]
                    seen=set()
                    for af,wf,cross in grid:
                        na=math.ceil(kb*af);ids=order[:math.ceil(nb*kb*wf)]
                        key=(na,len(ids),cross)
                        if key in seen:continue
                        seen.add(key);spec=LinearSpec(0,0,cross,activation_blocks=na,weight_support=tuple(ids))
                        layer=RepairLinear(source,controller,name,spec,asset);losses=[]
                        for panel in data:
                            x=panel['x'].cuda();gold=panel['y'].cuda().float();pred=layer(x).float()
                            losses.append((pred-gold).square().sum()/gold.square().sum().clamp_min(1e-20))
                        cost=linear_cost(len(data[0]['x']),n,k,a_blocks=na,w_blocks=ids,cross=cross)
                        evaluations.append((dict(a_blocks=na,w_blocks=ids,cross=cross),torch.stack(losses).cpu().tolist(),cost))
                    outfile=f'linear_{len(result["modules"]):03d}.pt';torch.save(compact(asset,1.),a.output/outfile)
                    result['modules'][name]=dict(file=outfile,sha256=hashlib.sha256((a.output/outfile).read_bytes()).hexdigest(),shape=[len(data[0]['x']),n,k],support_order=order)
                    record('linear',name,rows,evaluations,[len(data[0]['x']),n,k]);atomic_json(output,result)
                    print(json.dumps(dict(linear=name,complete=len(result['modules']))),flush=True)
                    del data,w,source,base,layer
                for name in sorted({r['name'] for r in capture['rows'] if r['kind']=='attention'},key=int):
                    if name in result['attention']:continue
                    rows=[r for r in capture['rows'] if r['kind']=='attention' and r['name']==name]
                    data=[{key:value.cuda() for key,value in torch.load(a.capture/r['file'],weights_only=True,map_location='cpu').items()} for r in rows]
                    m,n,h,hk=data[0]['q'].shape[1],data[0]['k'].shape[1],data[0]['q'].shape[2],data[0]['k'].shape[2];nt=math.ceil(n/128)
                    evaluations=[]
                    # First measure per-head full QK/PV output benefit. It proposes which
                    # heads receive partial block quotas; combination losses are rerun.
                    head_gain=torch.zeros((2,h));base_errors=[]
                    for panel in data:
                        q,k,v=(panel[x].cuda() for x in ('q','k','v'));gold=panel['y'].cuda().float()
                        base=fp4_attention(q,k,v,RepairConfig(coverage=0.,query_tile=32,transpose_v=True)).float()
                        e=(base-gold).square().sum((0,1,3))/gold.square().sum().clamp_min(1e-20)
                        for side in (0,1):
                            cfg=RepairConfig(coverage=0.,head_quotas=(nt if side==0 else 0,)*h,pv_head_quotas=(nt if side==1 else 0,)*h,
                                             terms=3,pv_terms=3,sparse_pack=True,query_tile=32,transpose_v=True)
                            pred=fp4_attention(q,k,v,cfg).float()
                            head_gain[side]+=(e-(pred-gold).square().sum((0,1,3))/gold.square().sum().clamp_min(1e-20)).cpu()/len(data)
                    order=torch.argsort(head_gain,descending=True,stable=True).tolist()
                    configs=[((0,)*h,(0,)*h,3,3)]
                    for side in (0,1,2):
                        for frac in (.125,.25,.5,1.):
                            counts=[]
                            for which in (0,1):
                                c=[0]*h
                                if side==which or side==2:
                                    # Nonuniform heads: focus full block quotas on ranked heads.
                                    for head in order[which][:max(1,math.ceil(h*frac))]:c[head]=nt
                                counts.append(tuple(c))
                            configs.append((*counts,3,3))
                    # Also test block-wise repair inside selected heads, including ACT2.
                    for side in (0,1,2):
                        for hf,bf in ((.25,.25),(.5,.25),(.5,.5),(1.,.125),(1.,.25),(1.,.5)):
                            pair=[]
                            for which in (0,1):
                                counts=[0]*h
                                if side==which or side==2:
                                    for head in order[which][:max(1,math.ceil(h*hf))]:counts[head]=max(1,math.ceil(nt*bf))
                                pair.append(tuple(counts))
                            configs.append((*pair,3,3))
                            configs.append((*pair,2,2))
                    configs.append(((nt,)*h,(nt,)*h,4,4))
                    for qc,pc,qt,pt in configs:
                        cfg=RepairConfig(coverage=0.,head_quotas=qc,pv_head_quotas=pc,terms=qt,pv_terms=pt,sparse_pack=True,query_tile=32,transpose_v=True)
                        losses=[]
                        for panel in data:
                            q,k,v=(panel[x].cuda() for x in ('q','k','v'));gold=panel['y'].cuda().float();pred=fp4_attention(q,k,v,cfg).float()
                            losses.append((pred-gold).square().sum()/gold.square().sum().clamp_min(1e-20))
                        cost=attention_cost(m,n,h,hk,qk_counts=qc,pv_counts=pc,qk_terms=qt,pv_terms=pt)
                        evaluations.append((dict(qk_counts=list(qc),pv_counts=list(pc),qk_terms=qt,pv_terms=pt),torch.stack(losses).cpu().tolist(),cost))
                    result['attention'][name]=dict(shape=[m,n,h,hk],head_gain=head_gain.tolist())
                    record('attention',name,rows,evaluations,[m,n,h,hk]);atomic_json(output,result)
                    print(json.dumps(dict(attention=name)),flush=True)
            result['complete']=True;result.pop('error',None);atomic_json(output,result)
        except Exception as error:
            result.update(error=repr(error),paused=True);atomic_json(output,result);raise


if __name__=='__main__':main()

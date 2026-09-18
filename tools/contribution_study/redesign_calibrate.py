"""Full-M multi-scene native interventions anchored to the successful RTN three-term path.

No gradient or input-row sampling. Only small error/feature summaries are saved;
original teacher weights and quantized deployment weights are prepared once.
N128 W1 interventions retain the complete K reduction, a regular deployment unit.
"""
import argparse,json,math
from pathlib import Path
from types import SimpleNamespace
from baselines import sha,write
from dataset_infer import clear_case


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cohort',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cohort=json.loads(a.cohort.read_text())
    if not cohort['frozen']:raise ValueError('Frozen disjoint cohort required')
    a.output.mkdir(parents=True,exist_ok=True);path=a.output/'manifest.json'
    identity=dict(cohort_sha256=sha(a.cohort),source_sha256=sha(__file__),split='calibration')
    state=dict(complete=False,identity=identity,cases={},performance_eligible=False)
    if path.exists():
        state=json.loads(path.read_text())
        if state['identity']!=identity:raise ValueError('Changed calibration resume')
        if state['complete']:return
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        state['resource']=select_shared_device();write(path,state)
        import torch
        from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec,phase
        from thunderworld_fp4.cosmos.assets import compact
        from thunderworld_fp4.cosmos.tuning import LaunchTable
        from thunderworld_fp4 import fp4_attention,RepairConfig
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        config=RunConfig(precision='bf16',attention_backend='bf16',resource_mode='shared-calibration',execution='eager')
        runner=CosmosRunner(cohort['model'],config,ar_cache=True,first_frame=True).prepare()
        controller=SimpleNamespace(warming=True,table=LaunchTable(),current=LinearSpec(1,0),record_linear=lambda *args:None)
        controller.linear_spec=lambda name,default:controller.current
        bank=a.output/'rtn_assets';bank.mkdir(exist_ok=True)
        am=dict(complete=False,precision='fp4',model_identity=runner.identity,cohort_sha256=identity['cohort_sha256'],modules={})
        native={};handles=[];current={};seen=set();active=[False]
        with torch.inference_mode():
            for i,name in enumerate(runner.engine.targets):
                source=runner.pipe.transformer.get_submodule(name)
                native[name]=RepairLinear(source,controller,name,LinearSpec(1,1))
                asset=native[name].asset
                # Identity-prescale RTN bytes are unchanged from the full-three anchor.
                asset.update(selector='residual_sensitivity',activation_sensitivity=source.weight.float().square().sum(0).cpu(),identity_prescale=True,
                    activation_scale=(1.,1.),weight_scale=(1.,1.))
                file=bank/f'linear_{i:03d}.pt'
                if not file.exists():torch.save(compact(asset,1.),file)
                am['modules'][name]=dict(file=file.name,sha256=sha(file),shape=[source.out_features,source.in_features])
            am['complete']=True;write(bank/'manifest.json',am)
        def key(kind,name):
            e=runner.engine;return kind,name,phase(e.step,e.steps),e.branch
        def eligible(k):return active[0] and not runner.engine.warming and k not in seen
        def hook(name):
            def callback(module,args,out):
                k=key('linear',name)
                if not eligible(k):return
                seen.add(k);x=args[0].detach().reshape(-1,module.in_features).contiguous();gold=out.detach().reshape(-1,module.out_features).float()
                denominator=gold.square().sum().clamp_min(1e-20);errors={}
                for label,spec in (('act2',LinearSpec(1,0)),('three',LinearSpec(1,1))):
                    controller.current=spec;y=native[name](x).float()
                    errors[label]=((y-gold).square().reshape(len(x),-1,128).sum((0,2))/denominator).cpu().tolist()
                rows=current['linears'].setdefault(name,[])
                rows.append(dict(phase=k[2],branch=k[3],shape=[len(x),module.out_features,module.in_features],
                    errors_by_n128=errors,input_amax=float(x.abs().amax()),rms=x.float().square().mean(0).sqrt().cpu().tolist()))
            return callback
        def attention(layer,q,k,v,out):
            key_=key('attention',str(layer))
            if not eligible(key_):return
            seen.add(key_);m,n,h,hk=q.shape[1],k.shape[1],q.shape[2],k.shape[2];nt=math.ceil(n/128)
            full=(nt,)*h;zero=(0,)*h;half=(max(1,nt//2),)*h;gold=out.float();denominator=gold.square().sum().clamp_min(1e-20)
            settings={'three':(full,full),'qk0':(zero,full),'pv0':(full,zero),'qkhalf':(half,full),'pvhalf':(full,half)};errors={}
            for label,(qc,pc) in settings.items():
                cfg=RepairConfig(coverage=0.,head_quotas=qc,pv_head_quotas=pc,terms=3,pv_terms=3,query_tile=32,transpose_v=True,sparse_pack=True)
                y=fp4_attention(q,k,v,cfg).float();errors[label]=((y-gold).square().sum((0,1,3))/denominator).cpu().tolist()
            probability=(torch.einsum('bmhd,bnhd->bhmn',q.float(),k.repeat_interleave(h//hk,2).float())/math.sqrt(q.shape[-1])).softmax(-1)
            ar=n-m
            if ar<=0:raise ValueError('Expected current AR prefix plus DM tokens')
            rows=current['attention'].setdefault(str(layer),[])
            rows.append(dict(phase=key_[2],branch=key_[3],shape=[m,n,h,hk],errors=errors,
                half_quota=half[0],ar_tokens=ar,ar_mass=probability[...,:ar].sum(-1).mean((0,2)).cpu().tolist(),
                normalized_entropy=(-(probability*probability.clamp_min(1e-20).log()).sum(-1).mean((0,2))/math.log(n)).cpu().tolist(),
                concentration=probability.amax(-1).mean((0,2)).cpu().tolist(),q_energy=q.float().square().mean((0,1,3)).cpu().tolist()))
        try:
            for name in runner.engine.targets:handles.append(runner.pipe.transformer.get_submodule(name).register_forward_hook(hook(name)))
            runner.engine.capture_attention=attention
            for entry in cohort['cases']['calibration']:
                case_id=entry['case']['id'];target=a.output/(case_id+'.json')
                if case_id in state['cases']:
                    if sha(target)!=state['cases'][case_id]:raise ValueError('Changed completed calibration')
                    continue
                if sha(entry['case']['image'])!=entry['image_sha256']:raise ValueError('Changed calibration image')
                current=dict(complete=False,entry=entry,linears={},attention={},scope='Native RTN teacher-input interventions, current full M; not generation quality or performance')
                clear_case(runner);active[0]=False;runner.warmup(entry['case'],35)
                with torch.inference_mode():
                    for seed in entry['seeds']:
                        seen.clear();active[0]=True;runner.infer(entry['case'],seed,35);active[0]=False
                expected=8*len(entry['seeds'])
                if set(current['linears'])!=set(runner.engine.targets) or any(len(rows)!=expected for rows in current['linears'].values()):raise ValueError('Incomplete linear panels')
                if len(current['attention'])!=len(runner.pipe.transformer.layers) or any(len(rows)!=expected for rows in current['attention'].values()):raise ValueError('Incomplete attention panels')
                current['complete']=True;write(target,current);state['cases'][case_id]=sha(target);write(path,state)
                print(json.dumps(dict(complete_cases=len(state['cases']),total=len(cohort['cases']['calibration']))),flush=True)
            state.update(complete=True,assets_sha256=sha(bank/'manifest.json'),model_identity=runner.identity);write(path,state)
        except Exception as error:
            state.update(paused=True,error=repr(error));write(path,state);raise
        finally:
            for handle in handles:handle.remove()
            runner.engine.capture_attention=None;runner.engine.clear()


if __name__=='__main__':main()

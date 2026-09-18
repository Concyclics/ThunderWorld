"""Measure complete local repair combinations on current candidate trajectories.

All 16 calibration scenes, 4 phases, 2 CFG branches. Full M and amax are retained.
BF16 references consume exactly the same current input. W0 and the W1 bank stay
frozen; only compute schedules change. Summaries are calibration, never timings.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
from types import SimpleNamespace

from baselines import sha, write
from dataset_infer import clear_case
from schedule_common import CONTROLS, FFN_CHOICES, variants, attention_variants, resolve_attention, attention_setting


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','previous','importance'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--budgets',type=float,nargs='+',default=[1.5,2.,2.5])
    p.add_argument('--max-cases',type=int)
    a=p.parse_args()
    cohort=json.loads((a.previous/'cohort.json').read_text())
    catalog=json.loads((a.importance/'ffn_guarded/candidates.json').read_text())
    source={c['name']:c for c in catalog['candidates']}
    capture=json.loads((a.importance/'capture/manifest.json').read_text())
    root=a.study/'calibration';root.mkdir(parents=True,exist_ok=True)
    # Historical full-attention interventions propose nonuniform head support;
    # every proposal is remeasured below on the current candidate's trajectory.
    head_gains={};cm=json.loads((a.previous/'calibration/manifest.json').read_text())
    for entry in cohort['cases']['calibration']:
        file=a.previous/'calibration'/(entry['case']['id']+'.json')
        if sha(file)!=cm['cases'][entry['case']['id']]:raise ValueError('Changed head proposals')
        data=json.loads(file.read_text());weight=.5/(9 if entry['benchmark']=='rbench' else 7)
        for layer,rows in data['attention'].items():
            for row in rows:
                for side in ('qk','pv'):
                    key=f'{row["branch"]}|{row["phase"]}|{layer}|{side}'
                    gains=head_gains.setdefault(key,[0.]*16)
                    for i,(half,full) in enumerate(zip(row['errors'][side+'half'],row['errors']['three'])):
                        gains[i]+=weight*(half-full)
    del data
    write(root/'head_proposals.json',dict(complete=True,gains=head_gains,source_sha256=sha(a.previous/'calibration/manifest.json')))
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        resource=select_shared_device()
        import torch
        from thunderworld_fp4 import fp4_attention
        from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
        from thunderworld_fp4.cosmos.assets import expand
        from thunderworld_fp4.cosmos.budget import RepairPlan
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec,phase
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        for cap in a.budgets:
            candidate=source[CONTROLS[cap]];bank=Path(candidate['assets'])
            if sha(bank/'manifest.json')!=candidate['assets_sha256']:raise ValueError('Changed frozen weights')
            if sha(candidate['config']['budget_plan'])!=candidate['plan_sha256']:raise ValueError('Changed frozen baseline plan')
            assets=json.loads((bank/'manifest.json').read_text())
            plan=json.loads(Path(candidate['config']['budget_plan']).read_text())
            ledger=json.loads((Path(candidate['config']['budget_plan']).parent/'compute.json').read_text())
            shapes={'linears':{},'attention':{}}
            for row in ledger['rows']:
                shapes['linears' if row['kind']=='linear' else 'attention'][row['name']]=row['shape'][:3 if row['kind']=='linear' else 4]
            folder=root/f'b{cap:g}';folder.mkdir(exist_ok=True);path=folder/'manifest.json'
            identity=dict(cohort_sha256=sha(a.previous/'cohort.json'),control=candidate,source_sha256=sha(__file__),
                          choices_sha256=sha(Path(__file__).with_name('schedule_common.py')),
                          capture_sha256=sha(a.importance/'capture/manifest.json'),max_cases=a.max_cases)
            identity['head_proposals_sha256']=sha(root/'head_proposals.json')
            state=dict(complete=False,identity=identity,cases={},resource=resource,performance_eligible=False)
            if path.exists():
                state=json.loads(path.read_text())
                if state['identity']!=identity:raise ValueError('Calibration resume changed')
                if state['complete']:continue
            config=RunConfig.from_dict(dict(candidate['config'],execution='eager'))
            runner=CosmosRunner(cohort['model'],config,assets=str(bank),ar_cache=True,first_frame=True).prepare()
            if runner.identity!=assets['model_identity'] or runner.identity!=capture['model_identity']:raise ValueError('Model mismatch')
            controller=SimpleNamespace(warming=False,config=config,budget_plan=RepairPlan(plan))
            shadows={};weights={};ranked={};handles=[];definitions={};current={};seen=set();active=[False]
            with torch.inference_mode():
                for name,entry in assets['modules'].items():
                    file=bank/entry['file']
                    if sha(file)!=entry['sha256']:raise ValueError('Changed asset')
                    raw=torch.load(file,map_location='cpu',weights_only=True)
                    ids=raw['residual_indices'].tolist();ranked[name]=sorted(ids,key=lambda i:(-float(raw['score'].flatten()[i]),i))
                    spec=LinearSpec(0,0,activation_blocks=0,weight_support=tuple(ids))
                    shadows[name]=RepairLinear(runner.pipe.transformer.get_submodule(name),controller,name,spec,expand(raw,ids))
                    entry=capture['weights'][name];file=a.importance/'capture'/entry['file']
                    if sha(file)!=entry['sha256']:raise ValueError('Changed BF16 reference')
                    weights[name]=torch.load(file,map_location='cpu',weights_only=True).cuda()
            def linear_choices(name,branch,ph):
                return variants(plan['linears'][f'{branch}|{ph}|{name}'],shapes['linears'][name][2],ranked[name])
            for branch in range(2):
                for ph in range(4):
                    for name in assets['modules']:
                        if '.mlp_moe_gen.' in name:continue
                        definitions[f'projection|{branch}|{ph}|{name}']=dict(kind='projection',branch=branch,phase=ph,name=name,
                            shapes={name:shapes['linears'][name]},choices={label:{name:row} for label,row in linear_choices(name,branch,ph).items()})
                    for layer in range(28):
                        name=f'layers.{layer}.mlp_moe_gen';up=name+'.up_proj';down=name+'.down_proj'
                        uc,dc=linear_choices(up,branch,ph),linear_choices(down,branch,ph)
                        definitions[f'ffn|{branch}|{ph}|{name}']=dict(kind='ffn',branch=branch,phase=ph,name=name,
                            shapes={up:shapes['linears'][up],down:shapes['linears'][down]},
                            choices={label:{up:uc[u],down:dc[d]} for label,(u,d) in FFN_CHOICES.items()})
                        definitions[f'attention|{branch}|{ph}|{layer}']=dict(kind='attention',branch=branch,phase=ph,name=str(layer),
                            shapes={'attention':shapes['attention'][str(layer)]},
                            choices={label:{'attention':row} for label,row in attention_variants(plan['attention'][f'{branch}|{ph}|{layer}']).items()})
                        fractions={}
                        for side in ('qk','pv'):
                            gain=head_gains[f'{branch}|{ph}|{layer}|{side}']
                            top=set(sorted(range(16),key=lambda i:(-gain[i],i))[:8])
                            fractions[side]=[1. if i in top else .5 for i in range(16)]
                        choices=definitions[f'attention|{branch}|{ph}|{layer}']['choices']
                        choices.update(qk_heads75={'attention':attention_setting(fractions['qk'],[1.]*16)},
                                       pv_heads75={'attention':attention_setting([1.]*16,fractions['pv'])},
                                       both_heads75={'attention':attention_setting(fractions['qk'],fractions['pv'])})
            write(folder/'definitions.json',dict(complete=True,groups=definitions,shapes=shapes,control=candidate))
            state['definitions_sha256']=sha(folder/'definitions.json');write(path,state)

            def run_linear(name,x,row):
                op=shadows[name];spec=LinearSpec(0,0,row['cross'],activation_blocks=row['a_blocks'],weight_support=tuple(row['w_blocks']))
                flat=x.reshape(-1,op.in_features).contiguous();op.prepare_shape(len(flat),spec)
                return op._compute(flat).reshape(*x.shape[:-1],op.out_features)

            def get_key(kind,name):
                e=runner.engine;return f'{kind}|{e.branch}|{phase(e.step,e.steps)}|{name}'

            def eligible(key):return active[0] and not runner.engine.warming and key not in seen

            def observe(key,x,gold,predictions,actual):
                if not torch.equal(predictions['base'],actual):raise ValueError('Local baseline differs from deployed arithmetic: '+key)
                gold=gold.float();denom=gold.square().sum().clamp_min(1e-20)
                losses={label:float((pred.float()-gold).square().sum()/denom) for label,pred in predictions.items()}
                rms=x.detach().float().reshape(-1,x.shape[-1]).square().mean(0)
                current['groups'][key]=dict(losses=losses,step=runner.engine.step,input_amax=float(x.abs().amax()),
                    rms_k128=rms.reshape(-1,128).mean(1).sqrt().cpu().tolist(),baseline_bitwise_equal=True)

            def projection_hook(name):
                def callback(module,args,out):
                    key=get_key('projection',name)
                    if not eligible(key):return
                    seen.add(key);x=args[0]
                    gold=torch.nn.functional.linear(x,weights[name],module.bias)
                    values={label:run_linear(name,x,rows[name]) for label,rows in definitions[key]['choices'].items()}
                    observe(key,x,gold,values,out)
                return callback

            def ffn_hook(name):
                def callback(module,args,out):
                    key=get_key('ffn',name)
                    if not eligible(key):return
                    if module.hidden_act!='relu2':raise ValueError('Expected Edge ReLU2')
                    seen.add(key);x=args[0];up=name+'.up_proj';down=name+'.down_proj'
                    gold=torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x,weights[up],module.up_proj.bias)).square(),weights[down],module.down_proj.bias)
                    hidden={};values={}
                    for label,rows in definitions[key]['choices'].items():
                        uk=json.dumps(rows[up],sort_keys=True)
                        if uk not in hidden:hidden[uk]=torch.relu(run_linear(up,x,rows[up])).square()
                        values[label]=run_linear(down,hidden[uk],rows[down])
                    observe(key,x,gold,values,out)
                return callback

            def attention_hook(layer,q,k,v,out):
                key=get_key('attention',str(layer))
                if not eligible(key):return
                seen.add(key)
                gold=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),enable_gqa=True).transpose(1,2).contiguous()
                choices=definitions[key]['choices'];values={label:fp4_attention(q,k,v,resolve_attention(rows['attention'])) for label,rows in choices.items()}
                observe(key,q,gold,values,out)
                denom=gold.float().square().sum().clamp_min(1e-20)
                current['groups'][key].update(shape=[q.shape[1],k.shape[1],q.shape[2],k.shape[2]],
                    head_errors={label:((pred.float()-gold.float()).square().sum((0,1,3))/denom).cpu().tolist() for label,pred in values.items()},
                    q_head_energy=q.float().square().mean((0,1,3)).cpu().tolist())

            try:
                for name in assets['modules']:
                    if '.mlp_moe_gen.' not in name:handles.append(runner.pipe.transformer.get_submodule(name).register_forward_hook(projection_hook(name)))
                for layer in range(28):
                    name=f'layers.{layer}.mlp_moe_gen';handles.append(runner.pipe.transformer.get_submodule(name).register_forward_hook(ffn_hook(name)))
                runner.engine.capture_attention=attention_hook
                entries=cohort['cases']['calibration'][:a.max_cases] if a.max_cases else cohort['cases']['calibration']
                for entry in entries:
                    case=entry['case']['id'];file=folder/(case+'.json')
                    if case in state['cases']:
                        if sha(file)!=state['cases'][case]:raise ValueError('Changed completed calibration')
                        continue
                    if sha(entry['case']['image'])!=entry['image_sha256']:raise ValueError('Changed input')
                    clear_case(runner);active[0]=False;runner.warmup(entry['case'],35)
                    current=dict(complete=False,entry=entry,groups={});seen.clear();active[0]=True
                    with torch.inference_mode():runner.infer(entry['case'],entry['seeds'][0],35)
                    active[0]=False
                    if set(current['groups'])!=set(definitions):raise ValueError('Incomplete layer/phase/CFG coverage')
                    current['complete']=True;write(file,current);state['cases'][case]=sha(file);write(path,state)
                    print(json.dumps(dict(budget=cap,cases=len(state['cases']),expected=len(entries))),flush=True)
                state['complete']=True;state.pop('paused',None);state.pop('error',None);write(path,state)
            except Exception as error:
                state.update(paused=True,error=repr(error));write(path,state);raise
            finally:
                active[0]=False
                for handle in handles:handle.remove()
                runner.engine.capture_attention=None;runner.engine.clear()
                shadows.clear();weights.clear();del runner;gc.collect();torch.cuda.empty_cache()
    write(a.study/'calibration_complete.json',dict(complete=True,budgets=a.budgets,performance_eligible=False))


if __name__=='__main__':main()

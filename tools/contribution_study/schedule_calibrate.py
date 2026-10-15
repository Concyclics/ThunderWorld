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


def phase_sample_steps(count):
    """Fixed first/middle/last coverage; no timestep tensor readback."""
    from schedule_common import PHASES
    if count not in (1, 3):
        raise ValueError('Choose one or three observations per phase')
    return sorted({step for start, end in zip(PHASES, PHASES[1:])
                   for step in ([start] if count == 1 else [start, (start+end-1)//2, end-1])})


def aggregate_temporal_samples(samples, definitions, count):
    """Equal-weight local samples, not a full-trajectory or downstream objective."""
    result = {}
    for key in definitions:
        rows = [groups[key] for groups in samples.values() if key in groups]
        if len(rows) != count:
            raise ValueError('Incomplete within-phase coverage: '+key)
        result[key] = dict(losses={label: sum(r['losses'][label] for r in rows)/count for label in rows[0]['losses']},
            observed_steps=sorted(r['step'] for r in rows), baseline_bitwise_equal=all(r['baseline_bitwise_equal'] for r in rows))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','previous','importance'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--budgets',type=float,nargs='+',default=[1.5,2.,2.5])
    p.add_argument('--max-cases',type=int)
    p.add_argument('--case-indices',type=int,nargs='+')
    p.add_argument('--layers',type=int,nargs='+')
    p.add_argument('--steps-per-phase',type=int,choices=(1,3),default=1)
    p.add_argument('--project-attention',action='store_true',help='Additional offline BF16 Wo intervention; not a runtime change')
    p.add_argument('--after-generation',type=Path,help='Wait for a serial generation suite receipt before acquiring the GPU lock')
    a=p.parse_args()
    if a.after_generation:
        import time
        while True:
            status=json.loads(a.after_generation.read_text()) if a.after_generation.exists() else {}
            if status.get('paused'):raise RuntimeError('Preceding generation paused: '+status.get('error',''))
            if status.get('complete'):break
            time.sleep(5)
    cohort=json.loads((a.previous/'cohort.json').read_text())
    sampled_steps=phase_sample_steps(a.steps_per_phase)
    layers=list(range(28)) if a.layers is None else a.layers
    if not layers or len(set(layers))!=len(layers) or any(l<0 or l>=28 for l in layers):raise ValueError('Invalid layer pilot')
    if a.case_indices is not None and (a.max_cases or len(set(a.case_indices))!=len(a.case_indices) or any(i<0 or i>=len(cohort['cases']['calibration']) for i in a.case_indices)):
        raise ValueError('Invalid/disagreeing case selection')
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
            module_names=[name for name in assets['modules'] if int(name.split('.')[1]) in layers]
            plan=json.loads(Path(candidate['config']['budget_plan']).read_text())
            ledger=json.loads((Path(candidate['config']['budget_plan']).parent/'compute.json').read_text())
            shapes={'linears':{},'attention':{}}
            for row in ledger['rows']:
                shapes['linears' if row['kind']=='linear' else 'attention'][row['name']]=row['shape'][:3 if row['kind']=='linear' else 4]
            folder=root/f'b{cap:g}';folder.mkdir(exist_ok=True);path=folder/'manifest.json'
            identity=dict(cohort_sha256=sha(a.previous/'cohort.json'),control=candidate,source_sha256=sha(__file__),
                          choices_sha256=sha(Path(__file__).with_name('schedule_common.py')),
                          capture_sha256=sha(a.importance/'capture/manifest.json'),max_cases=a.max_cases)
            identity.update(case_indices=a.case_indices,layers=layers,sampled_steps=sampled_steps,project_attention=a.project_attention)
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
                for name in module_names:
                    entry=assets['modules'][name]
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
                    for name in module_names:
                        if '.mlp_moe_gen.' in name:continue
                        definitions[f'projection|{branch}|{ph}|{name}']=dict(kind='projection',branch=branch,phase=ph,name=name,
                            shapes={name:shapes['linears'][name]},choices={label:{name:row} for label,row in linear_choices(name,branch,ph).items()})
                    for layer in layers:
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

            def eligible(key):return active[0] and not runner.engine.warming and runner.engine.step in sampled_steps and (key,runner.engine.step) not in seen

            def observation(key):
                if a.steps_per_phase==1:return current['groups'][key]
                return current['samples'][str(runner.engine.step)][key]

            def observe(key,x,gold,predictions,actual):
                if not torch.equal(predictions['base'],actual):raise ValueError('Local baseline differs from deployed arithmetic: '+key)
                gold=gold.float();denom=gold.square().sum().clamp_min(1e-20)
                losses={label:float((pred.float()-gold).square().sum()/denom) for label,pred in predictions.items()}
                rms=x.detach().float().reshape(-1,x.shape[-1]).square().mean(0)
                storage=current['groups'] if a.steps_per_phase==1 else current['samples'].setdefault(str(runner.engine.step),{})
                storage[key]=dict(losses=losses,step=runner.engine.step,input_amax=float(x.abs().amax()),
                    rms_k128=rms.reshape(-1,128).mean(1).sqrt().cpu().tolist(),baseline_bitwise_equal=True)

            def projection_hook(name):
                def callback(module,args,out):
                    key=get_key('projection',name)
                    if not eligible(key):return
                    seen.add((key,runner.engine.step));x=args[0]
                    gold=torch.nn.functional.linear(x,weights[name],module.bias)
                    values={label:run_linear(name,x,rows[name]) for label,rows in definitions[key]['choices'].items()}
                    observe(key,x,gold,values,out)
                return callback

            def ffn_hook(name):
                def callback(module,args,out):
                    key=get_key('ffn',name)
                    if not eligible(key):return
                    if module.hidden_act!='relu2':raise ValueError('Expected Edge ReLU2')
                    seen.add((key,runner.engine.step));x=args[0];up=name+'.up_proj';down=name+'.down_proj'
                    gold=torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x,weights[up],module.up_proj.bias)).square(),weights[down],module.down_proj.bias)
                    hidden={};values={}
                    for label,rows in definitions[key]['choices'].items():
                        uk=json.dumps(rows[up],sort_keys=True)
                        if uk not in hidden:hidden[uk]=torch.relu(run_linear(up,x,rows[up])).square()
                        values[label]=run_linear(down,hidden[uk],rows[down])
                    observe(key,x,gold,values,out)
                return callback

            def attention_hook(layer,q,k,v,out):
                if layer not in layers:return
                key=get_key('attention',str(layer))
                if not eligible(key):return
                seen.add((key,runner.engine.step))
                gold=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),enable_gqa=True).transpose(1,2).contiguous()
                choices=definitions[key]['choices'];values={label:fp4_attention(q,k,v,resolve_attention(rows['attention'])) for label,rows in choices.items()}
                observe(key,q,gold,values,out)
                denom=gold.float().square().sum().clamp_min(1e-20)
                observation(key).update(shape=[q.shape[1],k.shape[1],q.shape[2],k.shape[2]],
                    head_errors={label:((pred.float()-gold.float()).square().sum((0,1,3))/denom).cpu().tolist() for label,pred in values.items()},
                    q_head_energy=q.float().square().mean((0,1,3)).cpu().tolist())
                if a.project_attention:
                    # Same original Wo for both paths isolates attention error;
                    # this is not a full Transformer/CFG/rollout quality metric.
                    w=weights[f'layers.{layer}.self_attn.to_add_out']
                    projected=torch.nn.functional.linear(gold.flatten(2),w)
                    pd=projected.float().square().sum().clamp_min(1e-20)
                    projected_losses={label:float((torch.nn.functional.linear(pred.flatten(2),w).float()-projected.float()).square().sum()/pd) for label,pred in values.items()}
                    # Head-isolated perturbations retain cancellation within a
                    # head, but deliberately do not claim additive head gains.
                    head_losses={}
                    for label in ('base','qk50','pv50'):
                        diff=values[label].float()-gold.float();items=[]
                        for h in range(q.shape[2]):
                            e=torch.nn.functional.linear(diff[:,:,h,:],w[:,h*q.shape[3]:(h+1)*q.shape[3]].float())
                            items.append(float(e.square().sum()/pd))
                        head_losses[label]=items
                    observation(key).update(projected_losses=projected_losses,projected_head_errors=head_losses)

            try:
                for name in module_names:
                    if '.mlp_moe_gen.' not in name:handles.append(runner.pipe.transformer.get_submodule(name).register_forward_hook(projection_hook(name)))
                for layer in layers:
                    name=f'layers.{layer}.mlp_moe_gen';handles.append(runner.pipe.transformer.get_submodule(name).register_forward_hook(ffn_hook(name)))
                runner.engine.capture_attention=attention_hook
                entries=cohort['cases']['calibration'][:a.max_cases] if a.max_cases else cohort['cases']['calibration']
                if a.case_indices is not None:entries=[cohort['cases']['calibration'][i] for i in a.case_indices]
                for entry in entries:
                    case=entry['case']['id'];file=folder/(case+'.json')
                    if case in state['cases']:
                        if sha(file)!=state['cases'][case]:raise ValueError('Changed completed calibration')
                        continue
                    if sha(entry['case']['image'])!=entry['image_sha256']:raise ValueError('Changed input')
                    clear_case(runner);active[0]=False;runner.warmup(entry['case'],35)
                    current=dict(complete=False,entry=entry,groups={},samples={});seen.clear();active[0]=True
                    with torch.inference_mode():runner.infer(entry['case'],entry['seeds'][0],35)
                    active[0]=False
                    if a.steps_per_phase==3:current['groups']=aggregate_temporal_samples(current['samples'],definitions,a.steps_per_phase)
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

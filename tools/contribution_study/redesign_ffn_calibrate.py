"""Native full-FFN weight-repair interventions, including ReLU2 and current down inputs."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
from baselines import sha,write
from dataset_infer import clear_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cohort','assets','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--after',type=Path);a=p.parse_args()
    if a.after:
        while True:
            state=json.loads(a.after.read_text()) if a.after.exists() else {}
            if state.get('paused'):raise RuntimeError('Previous GPU workflow paused')
            if state.get('complete'):break
            time.sleep(5)
    cohort=json.loads(a.cohort.read_text());assets=json.loads((a.assets/'manifest.json').read_text())
    a.output.mkdir(exist_ok=True);path=a.output/'manifest.json'
    identity=dict(cohort_sha256=sha(a.cohort),assets_sha256=sha(a.assets/'manifest.json'),source_sha256=sha(__file__))
    state=dict(complete=False,identity=identity,cases={},performance_eligible=False)
    if path.exists():
        state=json.loads(path.read_text())
        if state['identity']!=identity:raise ValueError('Changed FFN calibration')
        if state['complete']:return
    from thunderworld_fp4.cosmos.resources import shared_worker,select_shared_device
    with shared_worker():
        state['resource']=select_shared_device();write(path,state)
        import torch
        from thunderworld_fp4.cosmos import CosmosRunner,RunConfig
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.assets import expand
        from thunderworld_fp4.cosmos.settings import LinearSpec,phase
        from thunderworld_fp4.cosmos.tuning import LaunchTable
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
        runner=CosmosRunner(cohort['model'],RunConfig(precision='bf16',attention_backend='bf16',execution='eager',resource_mode='shared-calibration'),ar_cache=True,first_frame=True).prepare()
        if runner.identity!=assets['model_identity']:raise ValueError('Changed teacher model')
        c=SimpleNamespace(warming=True,table=LaunchTable(),current=LinearSpec(1,1),record_linear=lambda *args:None)
        c.linear_spec=lambda name,default:c.current
        native={};handles=[];active=[False];current={};seen=set()
        for name in runner.engine.targets:
            if 'mlp_moe_gen' not in name:continue
            entry=assets['modules'][name];file=a.assets/entry['file']
            if sha(file)!=entry['sha256']:raise ValueError('Changed weight plane')
            native[name]=RepairLinear(runner.pipe.transformer.get_submodule(name),c,name,LinearSpec(1,1),expand(torch.load(file,weights_only=True,map_location='cpu'),1.))
        def hook(name):
            def callback(module,args,out):
                e=runner.engine;key=(name,phase(e.step,e.steps),e.branch)
                if not active[0] or e.warming or key in seen:return
                if module.hidden_act!='relu2':raise ValueError('This intervention is qualified for Edge ReLU2 FFN only')
                seen.add(key);x=args[0];gold=out.float();denominator=gold.square().sum().clamp_min(1e-20)
                values={};ups={}
                for terms in (2,3):
                    c.current=LinearSpec(1,int(terms==3));ups[terms]=torch.relu(native[name+'.up_proj'](x)).square()
                for label,up,down in (('three',3,3),('without_up_w',2,3),('without_down_w',3,2),('act2',2,2)):
                    c.current=LinearSpec(1,int(down==3));pred=native[name+'.down_proj'](ups[up]).float()
                    values[label]=float((pred-gold).square().sum()/denominator)
                current['modules'].setdefault(name,[]).append(dict(phase=key[1],branch=key[2],errors=values,shape=list(x.shape)))
            return callback
        try:
            for layer in range(len(runner.pipe.transformer.layers)):
                name=f'layers.{layer}.mlp_moe_gen';handles.append(runner.pipe.transformer.get_submodule(name).register_forward_hook(hook(name)))
            for entry in cohort['cases']['calibration']:
                case_id=entry['case']['id'];target=a.output/(case_id+'.json')
                if case_id in state['cases']:
                    if sha(target)!=state['cases'][case_id]:raise ValueError('Changed FFN result')
                    continue
                current=dict(complete=False,entry=entry,modules={});clear_case(runner);active[0]=False;runner.warmup(entry['case'],35)
                with torch.inference_mode():
                    for seed in entry['seeds']:
                        seen.clear();active[0]=True;runner.infer(entry['case'],seed,35);active[0]=False
                if len(current['modules'])!=28 or any(len(r)!=8*len(entry['seeds']) for r in current['modules'].values()):raise ValueError('Incomplete FFN coverage')
                current['complete']=True;write(target,current);state['cases'][case_id]=sha(target);write(path,state)
                print(json.dumps(dict(complete_cases=len(state['cases']),total=len(cohort['cases']['calibration']))),flush=True)
            state['complete']=True;write(path,state)
        except Exception as error:
            state.update(paused=True,error=repr(error));write(path,state);raise
        finally:
            for handle in handles:handle.remove()
            runner.engine.clear()


if __name__=='__main__':main()

"""AR invariance audit and fresh-runner end-to-end cache equivalence pilot."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import torch
from thunderworld_fp4.cosmos.runner import CosmosRunner
from thunderworld_fp4.cosmos.settings import RunConfig
from thunderworld_fp4.cosmos.ar_cache import prefill
from thunderworld_fp4.cosmos.cli import idle_guard

ROOT=Path(__file__).resolve().parents[2]


def make(name, cached=False):
    base=ROOT/'runs5090/joint_e2e_20260914'
    runner=CosmosRunner(ROOT/'models/Cosmos3-Edge-a9d944e',
        RunConfig.from_dict(json.loads((base/f'configs/{name}.json').read_text())),
        tuning=base/f'profiles/{name}.json',ar_cache=cached,
        assets=None if name=='bf16' else ROOT/f'data_local/cutlass5090_rebuilt_20260914/i2v_b{name[-3:]}')
    runner.prepare();return runner


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['audit','pilot'],required=True)
    p.add_argument('--name',choices=['bf16','joint125','cutlass150'],default='joint125')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;idle_guard()
    cp=ROOT/'tools/joint_inference/cases/robot_observe.json';case=json.loads(cp.read_text());case['image']=str((cp.parent/case['image']).resolve())
    result=dict(complete=False,mode=a.mode,name=a.name)
    def save(): (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save();run=make(a.name);run.warmup(case,35);run.warm_request(case,35)
    if a.mode=='audit':
        run.engine.config=replace(run.engine.config,execution='eager')
        refs={};flags=[];state=dict(prefilling=False,branch=0);counts=dict(layer_output=0,k_before_rope=0,v=0,static_inputs=0)
        for i,layer in enumerate(run.pipe.transformer.layers):
            def check(module,args,out,_i=i,_kind='layer_output'):
                if state['prefilling']:return
                row=refs[state['branch']]['rows'][_i]
                ref=row[{'layer_output':'out','k_before_rope':'k_before_rope','v':'v'}[_kind]]
                value=out[0] if _kind=='layer_output' else out.reshape_as(ref)
                flags.append(torch.all(value==ref));counts[_kind]+=1
            layer.register_forward_hook(check)
            layer.self_attn.to_v.register_forward_hook(lambda m,x,y,_fn=check:_fn(m,x,y,_kind='v'))
            layer.self_attn.k_norm_und_for_gen.register_forward_hook(lambda m,x,y,_fn=check:_fn(m,x,y,_kind='k_before_rope'))
        previous=run.pipe.transformer.forward
        def observed(*args,**kw):
            branch=run.engine.call%run.engine.branches;state['branch']=branch
            if branch not in refs:
                state['prefilling']=True
                try:rows=prefill(run.pipe.transformer,kw)
                finally:state['prefilling']=False
                refs[branch]=dict(rows=rows,ids=kw['input_ids'].clone(),pos=kw['position_ids'].clone())
            else:
                flags.extend([torch.all(kw['input_ids']==refs[branch]['ids']),torch.all(kw['position_ids']==refs[branch]['pos'])]);counts['static_inputs']+=2
            return previous(*args,**kw)
        run.pipe.transformer.forward=observed
        _,row=run.infer(case,7103,35)
        assert bool(torch.stack(flags).all()),'AR prefix or layer output changed'
        old=json.loads((ROOT/f'runs5090/joint_e2e_20260914/trajectory_00_{a.name}/result.json').read_text())['rows'][0]
        assert row['sha256']==old['sha256'],'Eager audit differs from frozen graph output'
        result.update(complete=True,counts=counts,all_bitwise_equal=True,output_hash=row['sha256'],
            scope='Instrumented eager AR audit: 35 steps, both CFG branches, every layer output and DM-facing K before RoPE plus V; static full position/input IDs checked. Not performance timing.')
    else:
        # Same-length changed text tests current-request refill while keeping frozen shapes.
        other=dict(case);tokenizer=run.pipe.text_tokenizer
        length=len(tokenizer(case['prompt']).input_ids)
        for replacement in ['human','metal','wooden','small','large']:
            candidate=case['prompt'].replace('robot',replacement)
            if len(tokenizer(candidate).input_ids)==length:
                other['prompt']=candidate;break
        assert other['prompt']!=case['prompt']
        cases=[('original',case,7103),('original',case,7104),('changed_prompt',other,7103),('original_after_change',case,7103)]
        outputs={};rows=[]
        for cached in (False,True):
            if cached:
                del run;gc.collect();torch.cuda.empty_cache();run=make(a.name,True)
                run.warmup(case,35);run.warm_request(case,35)
            for label,c,seed in cases:
                run.warmup(c,35)
                _,row=run.infer(c,seed,35)
                key=(label,seed)
                if not cached:outputs[key]=row['sha256']
                else:assert row['sha256']==outputs[key],(label,seed,'cache changed output')
                rows.append(dict(row,cached=cached,probe_case=label));result['rows']=rows;save();print(cached,label,seed,row['wall_seconds'],flush=True)
            if cached:result['cache_receipt']=run.ar_cache.receipt()
        assert outputs['original',7103]!=outputs['original',7104]
        assert outputs['original',7103]!=outputs['changed_prompt',7103]
        assert outputs['original',7103]==outputs['original_after_change',7103]
        result.update(complete=True,rows=rows,changed_prompt=other['prompt'],all_bitwise_equal=True,
            scope='Fresh baseline/candidate runners; changed seeds and same-length changed prompt, then restored prompt. Pilot only, not randomized formal timing.')
    result['receipt']=run.receipt();save();print(a.output,flush=True)


if __name__=='__main__':main()

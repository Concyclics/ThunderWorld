"""CPU preflight of final request shapes/budgets; never generate final samples."""
import argparse
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

from baselines import sha, write


def preflight(root):
    from diffusers import Cosmos3OmniPipeline
    from transformers import AutoTokenizer
    from thunderworld_fp4 import RepairConfig
    from thunderworld_fp4.cosmos.budget import RepairPlan, attention_cost, linear_cost, compile_request
    manifest=root/'final/manifest.json';study=json.loads(manifest.read_text());model=Path(study['model'])
    model_cfg=json.loads((model/'model_index.json').read_text())
    vae=json.loads((model/'vae/config.json').read_text())
    transformer=json.loads((model/'transformer/config.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(model/'text_tokenizer',local_files_only=True)
    use_system=model_cfg.get('default_use_system_prompt',inspect.signature(Cosmos3OmniPipeline.__init__).parameters['default_use_system_prompt'].default)
    dummy=SimpleNamespace(config=SimpleNamespace(default_use_system_prompt=use_system),text_tokenizer=tokenizer,
        llm_special_tokens={'eos_token_id':tokenizer.eos_token_id,'start_of_generation':tokenizer.convert_tokens_to_ids('<|vision_start|>')},
        video_resolution_template='',inverse_video_resolution_template='')
    def dimensions(case):
        if case['task']!='i2v' or not case.get('image'):raise ValueError('Only fixed-image I2V preflight is qualified')
        frames,h,w=case.get('frames',61),case.get('height',192),case.get('width',320)
        sf,tf=vae['scale_factor_spatial'],vae['scale_factor_temporal'];patch=transformer['latent_patch_size']
        if (frames-1)%tf or h%sf or w%sf:raise ValueError('Unsupported latent rounding in CPU preflight')
        m=((frames-1)//tf+1)*math.ceil(h/sf/patch)*math.ceil(w/sf/patch)
        tokens=Cosmos3OmniPipeline.tokenize_prompt(dummy,case['prompt'],case.get('negative_prompt','distorted objects, flickering, blurry, low quality'),
            num_frames=frames,height=h,width=w,add_resolution_template=False,add_duration_template=False)
        return m,tuple(m+len(t) for t in tokens)
    smoke=json.loads((root/'smoke/manifest.json').read_text())
    checks={};all_rows=[]
    for name,spec in study['methods'].items():
        if not spec['config'].get('budget_plan'):continue
        source=root/'smoke/generation'/name/smoke['cases'][0]['case']['id']/'compute.json'
        measured=json.loads(source.read_text());m,ns=dimensions(smoke['cases'][0]['case'])
        for row in measured['rows']:
            if row['shape'][0]!=m:raise ValueError('CPU latent shape differs from actual warmup')
            if row['kind']=='attention' and row['shape'][1]!=ns[row['branch']]:
                raise ValueError('CPU prompt tokenization differs from actual warmup')
        plan=RepairPlan.load(spec['config']['budget_plan']);cache={}
        expected_modules={r['name'] for r in measured['rows'] if r['kind']=='linear'}
        expected_layers={r['name'] for r in measured['rows'] if r['kind']=='attention'}
        cases=[]
        for entry in study['cases']:
            if sha(entry['case']['image'])!=entry['image_sha256']:raise ValueError('Changed final image')
            current_m,current_ns=dimensions(entry['case'])
            if current_m!=m:raise ValueError('Unqualified projection shape; run development shape check first')
            key=tuple(math.ceil(n/128) for n in current_ns)
            if key not in cache:
                rows=[]
                for observed in measured['rows']:
                    branch,phase,module=observed['branch'],observed['phase'],observed['name']
                    if observed['kind']=='linear':
                        setting=plan.linears[f'{branch}|{phase}|{module}']
                        cost=linear_cost(*observed['shape'],bm=observed['launch']['bm'],**setting)
                    else:
                        _,_,heads,kvheads,_=observed['shape']
                        cfg=plan.attention_config('i2v',branch,phase,int(module),RepairConfig())
                        n=(key[branch]-1)*128+1
                        cost=attention_cost(m,n,heads,kvheads,bm=observed['launch']['bm'],
                            qk_counts=cfg.head_counts(heads,n),pv_counts=cfg.pv_counts(heads,n),
                            qk_terms=cfg.terms,pv_terms=cfg.pv_terms or cfg.terms)
                    rows.append(dict(cost,name=module,branch=branch,phase=phase))
                ledger=compile_request(rows,spec['config']['compute_budget'],expected_modules=expected_modules,expected_layers=expected_layers)
                cache[key]=dict(ratio=ledger['ratio'],base_mma=ledger['base_mma'],total_mma=ledger['total_mma'],feasible=ledger['feasible'])
            cases.append(dict(case=entry['case']['id'],m=m,n_by_cfg=current_ns,key_tiles_by_cfg=key,**cache[key]))
        checks[name]=dict(cases=cases,measured_ledger_sha256=sha(source),distinct_tile_shapes=len(cache),
            ratio_min=min(c['ratio'] for c in cases),ratio_max=max(c['ratio'] for c in cases),budget=spec['config']['compute_budget'])
    result=dict(complete=True,manifest_sha256=sha(manifest),source_sha256=sha(__file__),
        tokenizer_model=str(model/'text_tokenizer'),pipeline_source_sha256=sha(inspect.getfile(Cosmos3OmniPipeline)),
        default_use_system_prompt=use_system,checks=checks,final_outputs_generated=False,performance_eligible=False,
        scope='Exact pipeline tokenization and VAE/patch dimensions, validated against actual smoke ledgers; issued-tile budgets. Actual request warmup still recompiles before Graph capture.')
    write(root/'shape_preflight.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    preflight(p.parse_args().root)

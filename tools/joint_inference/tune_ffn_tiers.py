"""Observe a real request and tune every FFN repair tier at its actual M."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import torch
from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
from thunderworld_fp4.cosmos.settings import TIERS
from thunderworld_fp4.cosmos.cli import idle_guard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();idle_guard();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    model=json.loads((a.model/'transformer/config.json').read_text())
    case=json.loads(a.case.read_text())
    if case.get('image'):case['image']=str((a.case.parent/case['image']).resolve())
    parts=['up_proj','down_proj']+(['gate_proj'] if model['hidden_act']=='silu' else [])
    tiers=list(TIERS);assignments={}
    for layer in range(model['num_hidden_layers']):
        for phase in range(4):
            for branch in range(1 if case['task']=='policy' else 2):
                for part in parts:
                    assignments[f"{case['task']}|{branch}|{phase}|{layer}.{part}"]=tiers[(layer+phase)%len(tiers)]
    policy=a.output/'all_tiers_policy.json'
    policy.write_text(json.dumps(dict(phases=[0,.2,.5,.8,1],policies=dict(all_tiers=dict(assignments=assignments))),indent=2)+'\n')
    config=RunConfig(ffn=TIERS['base'],projection=TIERS['base'],attention_backend='bf16',
                     ffn_policy=str(policy.resolve()),ffn_policy_mode='all_tiers',strict_tuning=True)
    runner=CosmosRunner(a.model,config,tune=True,audio=case['task']=='av').prepare()
    runner.warmup(case)
    idle_guard();runner.table.save(a.output/'tuning.json')
    receipt=runner.receipt()
    observed={r['repair'] for r in receipt['observed_shapes'] if '.mlp_moe_gen.' in r['module']}
    if observed!={v.key for v in TIERS.values()}:raise AssertionError('Missing FFN tier')
    (a.output/'result.json').write_text(json.dumps(dict(complete=True,**receipt),indent=2)+'\n')


if __name__=='__main__':main()

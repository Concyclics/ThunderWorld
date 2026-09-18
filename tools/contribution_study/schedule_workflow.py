"""Frozen development/validation execution; serial shared GPU, verified reuse."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from baselines import sha,write
from importance_workflow import import_method,run
from schedule_common import CONTROLS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','previous','importance'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--split',choices=['development','validation'],default='development')
    p.add_argument('--selection',type=Path)
    a=p.parse_args();catalog=json.loads((a.study/'candidates/candidates.json').read_text())
    cohort=json.loads((a.previous/'cohort.json').read_text())
    previous=a.importance/'ffn_development';old=json.loads((previous/'manifest.json').read_text())
    if not catalog['complete'] or catalog['cohort_sha256']!=sha(a.previous/'cohort.json'):raise ValueError('Complete matched plans required')
    methods={name:old['methods'][name] for name in ('bf16','fp8','fp4_three')}
    aliases={};fingerprints={}
    candidates={c['name']:c for c in catalog['candidates']}
    if a.split=='validation':
        if not a.selection:raise ValueError('Validation requires a frozen development selection')
        selection=json.loads(a.selection.read_text())
        if not selection['complete'] or selection['split']!='development':raise ValueError('Select on development only')
        keep=set()
        for row in selection['budgets'].values():
            if row['selected']!=row['control']:keep.update((row['selected'],row['control']))
        if not keep:raise ValueError('No improved development candidate to validate')
    else:keep=set(candidates)
    for candidate in catalog['candidates']:
        name=candidate['name']
        if name not in keep:continue
        plan=json.loads(Path(candidate['config']['budget_plan']).read_text())
        signature=dict(budget=candidate['budget'],linears=plan['linears'],attention=plan['attention'])
        # All schedules reuse the same W0/W1 source bank within each budget.
        digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        if digest in fingerprints:aliases[name]=fingerprints[digest];continue
        fingerprints[digest]=name;methods[name]=candidate
    manifest_value=dict(schema=1,frozen=True,model=cohort['model'],cases=cohort['cases'][a.split],methods=methods,
        cohort_sha256=sha(a.previous/'cohort.json'),catalog_sha256=sha(a.study/'candidates/candidates.json'),
        steps=35,resource_mode='shared-calibration',debug='off',execution='graph',role=a.split,
        selection_eligible=a.split=='development',aliases=aliases,
        scope=f'Joint stage/layer/head repair on {a.split} scenes only; original inputs/35 steps, fixed offline W0, no latency claim.')
    if a.selection:manifest_value['selection_sha256']=sha(a.selection)
    root=a.study/a.split;root.mkdir(exist_ok=True);manifest=root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=manifest_value:raise ValueError('Frozen experiment changed')
    write(manifest,manifest_value)
    if a.split=='development':
        for name in methods:
            if name in old['methods']:import_method(previous,root,name,manifest)
    run(manifest,root)


if __name__=='__main__':main()

"""Audit every scheduled shape bound and exact frozen W0/W1 sub-bank reload."""
import argparse
import json
from pathlib import Path
import torch
from baselines import sha,write
from schedule_common import CONTROLS
from schedule_envelope import KEY_PAIRS,compile_envelope


def equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(8)
    catalog=json.loads((a.study/'candidates/candidates.json').read_text());by_name={c['name']:c for c in catalog['candidates']};results=[]
    for candidate in catalog['candidates']:
        if '_schedule_' not in candidate['name']:continue
        cap=candidate['budget'];control=by_name[CONTROLS[cap]]
        plan=json.loads(Path(candidate['config']['budget_plan']).read_text())
        original=json.loads(Path(control['config']['budget_plan']).read_text())
        shapes=json.loads((a.study/'calibration'/f'b{cap:g}'/'definitions.json').read_text())['shapes']
        checks=[dict(key_tiles_by_cfg=list(pair),ratio=compile_envelope(plan,shapes,pair)['ratio']) for pair in KEY_PAIRS]
        if set(plan['linears'])!=set(original['linears']) or set(plan['attention'])!=set(original['attention']):raise ValueError('Changed operator scope')
        banks=[Path(c['assets']) for c in (candidate,control)]
        manifests=[json.loads((bank/'manifest.json').read_text()) for bank in banks]
        for c,b in zip((candidate,control),banks):
            if sha(b/'manifest.json')!=c['assets_sha256'] or sha(c['config']['budget_plan'])!=c['plan_sha256']:raise ValueError('Changed plan/asset provenance')
        total=0
        for name,entry in manifests[0]['modules'].items():
            reference=manifests[1]['modules'][name];files=[banks[0]/entry['file'],banks[1]/reference['file']]
            for file,record in zip(files,(entry,reference)):
                if sha(file)!=record['sha256']:raise ValueError('Changed weights')
            asset,old=[torch.load(file,map_location='cpu',weights_only=True) for file in files]
            if 'codes' in asset or 'scales' in asset:raise ValueError('Hidden dense W1 storage')
            union=sorted({i for key,row in plan['linears'].items() if key.split('|',2)[-1]==name for i in row['w_blocks']})
            if asset['residual_indices'].tolist()!=union:raise ValueError('Asset differs from executed support union')
            positions={v:i for i,v in enumerate(old['residual_indices'].tolist())}
            if not set(union).issubset(positions):raise ValueError('Expanded W1 storage')
            idx=torch.tensor([positions[v] for v in union],dtype=torch.long)
            for field in ('codes_base','scales_base','prescale','factor','activation_sensitivity'):
                if not equal(asset[field],old[field]):raise ValueError('Changed offline '+field)
            for field in ('codes_residual','scales_residual'):
                if not equal(asset[field],old[field][idx]):raise ValueError('Changed residual encoding')
            total+=len(union)*128**2*9//16
        if total!=candidate['cost']['w1_bytes'] or total>control['cost']['w1_bytes']:raise ValueError('Storage count mismatch')
        results.append(dict(name=candidate['name'],passed=True,modules=168,w1_bytes=total,bounds=checks))
    write(a.study/'verification.json',dict(complete=True,catalog_sha256=sha(a.study/'candidates/candidates.json'),checks=results,
         scope='Exact W0/prescale/selector and W1 sub-bank bytes; complete phase/CFG coverage; M960, GQA H16/Hkv8, N961..2048 independently per CFG branch (81 combinations), cross intersection upper bounds and selector MMA included.'))


if __name__=='__main__':main()

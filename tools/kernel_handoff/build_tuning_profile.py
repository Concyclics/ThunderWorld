"""Export explicit GPU-specific launch settings from fully verified searches."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--attention',type=Path)
    p.add_argument('--ffn',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    profile={'schema_version':1,'gpu':None,'sources':{},'configs':{},'selection_notes':{},
             'selection_rule':'Prefer original attention layout/settings or unsplit FFN when the measured gain is below 3%.'}
    for kind,path in [('attention',a.attention),('ffn',a.ffn)]:
        if path is None:continue
        search=json.loads(path.read_text())
        assert search['complete'],f'{path} is incomplete'
        if profile['gpu'] is None:profile['gpu']=search['gpu']
        assert profile['gpu']==search['gpu']
        profile['sources'][kind]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for row in search['rows']:
            winner=next(c for c in row['candidates'] if c['key']==row['winner'])
            assert winner['passed']
            simpler=(next((c for c in row['candidates'] if c['key']=='m32_s2_w4_original'),None)
                     if kind=='attention' else next((c for c in row['candidates'] if c.get('split_k')==1),None))
            if simpler is not None and simpler['passed'] and winner['latency_ms']['median']>=.97*simpler['latency_ms']['median']:
                winner=simpler
                profile['selection_notes'][f"{kind}:{row['asset']}:{row['m']}:{row['preset']}"]='Below 3% improvement; prefer simpler execution'
            settings={key:winner[key] for key in ('tile_m','splits','num_warps','num_stages','transpose_v','split_k') if key in winner}
            if kind=='ffn':settings.update(block_k=row['block_k'],indexed_repair=not winner['interleave'])
            profile['configs'][f"{kind}:{row['asset']}:{row['m']}:{row['preset']}"]=settings
    if not profile['configs']:raise ValueError('No search inputs')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(profile,indent=2)+'\n')


if __name__=='__main__':main()

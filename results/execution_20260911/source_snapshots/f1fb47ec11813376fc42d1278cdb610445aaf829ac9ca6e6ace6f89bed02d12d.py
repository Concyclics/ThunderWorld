"""Validate complete coverage, numeric records, snapshots and cache boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[]
    for directory in a.inputs:
        r=json.loads((directory/'result.json').read_text())
        assert r['complete'],directory
        assert r['args']['steps']==35
        rows=r['rows'];cases=r['cases'];case_ids={c['id'] for c in cases}
        steps=35;layers=r['model_config']['num_hidden_layers'];heads=r['model_config']['num_attention_heads']
        expected={(m,c,s,i) for m in r['args']['modes'] for c in case_ids for s in r['args']['seeds'] for i in range(r['args']['rounds'])}
        got={(x['mode'],x['case'],x['seed'],x['repeat']) for x in rows}
        assert expected==got and len(got)==len(rows),(directory,len(expected),len(got))
        if 'group_policy' in r:
            p=r['group_policy'];assert p['model']==r['model'] and p['calibration_split']=='development'
            groups=p['group_ids_per_layer'];kv=r['model_config']['num_key_value_heads']
            assert set(groups)=={str(i) for i in range(layers)}
            assert sum(map(len,groups.values()))==layers*kv//4
            for ids in groups.values():assert len(set(ids))==len(ids) and all(0<=i<kv for i in ids)
        manifests=[r['source_sha256']]+[s['source_sha256'] for s in r.get('source_runs',[])]
        verified=set()
        for manifest in manifests:
            for sha in manifest.values():
                path=directory/'source_snapshots'/(sha+'.py')
                assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
                verified.add(sha)
        for row in rows:
            case=next(c for c in cases if c['id']==row['case'])
            branches=1 if case['task']=='policy' else 2
            assert row['calls']==steps*branches
            assert np.isfinite(row['wall_seconds']) and row['wall_seconds']>0
            assert np.isfinite(row['transformer_seconds']) and row['transformer_seconds']>0
            assert row['cache_max_age']<=1
            if 'ffn_calls' in row:
                expected_ffn=0 if row['mode']=='bf16' else layers*(2 if r['model_config']['hidden_act']=='relu2' else 3)*steps*branches
                assert row['ffn_calls']==expected_ffn
            stem=f'{row["mode"]}_{row["case"]}_seed{row["seed"]}_r{row["repeat"]}'
            data=np.load(directory/(stem+'_outputs.npz'))
            for key,value in data.items():
                assert np.isfinite(value).all()
                assert hashlib.sha256(value.tobytes()).hexdigest()==row['sha256'][key]
            if r['args']['sample']:
                z=np.load(directory/(stem+'_heads.npz'))
                assert z['mass'].shape==(steps,branches,layers,heads,6,6)
                assert z['features'].shape==(steps,branches,layers,heads,6,12)
                for b in range(branches):
                    roles=[i for i,n in enumerate(z['counts'][b]) if n and i!=0]
                    mass=z['mass'][:,b,:,: ,roles,:]
                    assert np.isfinite(mass).all()
                    assert mass.min()>=0 and mass.max()<=1+3e-5
                    assert np.allclose(mass.sum(-1),1,atol=3e-5)
                    assert np.isfinite(z['features'][1:,b,:,:,roles,:]).all()
                assert np.isfinite(z['timestep']).all()
            else:
                c=row['counts'];mode=row['mode']
                if mode.startswith('delta_reuse'):
                    assert c['base']==steps*branches*layers
                    assert c['full']+c['reused_delta']==steps*branches*layers
                elif mode.startswith('output_reuse') and mode!='output_reuse2_stable':
                    assert c['full']+c['reused_output']==steps*branches*layers
                elif mode=='output_reuse2_stable':
                    assert c['reused_output']>0
                    assert c['full']<=steps*branches*layers
                else:assert c['full']==steps*branches*layers
        checks.append(dict(input=str(directory),trajectories=len(rows),source_snapshots=len(verified),
                           complete_coverage=True,finite_and_hashes=True,cache_age_at_most_one=True))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

"""Assemble completed trajectories, explicitly retaining interrupted provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    reports=[json.loads((d/'result.json').read_text()) for d in a.inputs]
    first=reports[0]
    for r in reports:
        assert r['model']==first['model'] and r['model_config']==first['model_config']
        assert r['args']['steps']==first['args']['steps'] and r['args']['seeds']==first['args']['seeds']
        assert r['args']['sample'] and r['args']['samples_per_role']==first['args']['samples_per_role']
    cases={c['id']:c for r in reports for c in r['cases']}
    expected={(c,s) for c in cases for s in first['args']['seeds']}
    found={};sources=[]
    for d,r in zip(a.inputs,reports):
        sources.append(dict(path=str(d),complete=r['complete'],sha256=hashlib.sha256((d/'result.json').read_bytes()).hexdigest(),source_sha256=r['source_sha256'],audio_placement=r.get('audio_placement')))
        for row in r['rows']:
            key=row['case'],row['seed']
            if key in found:
                # A interrupted run may have finished one AV request while
                # termination was being delivered. Prefer the later complete
                # latent-saving run only when the generated video is identical.
                assert found[key][2]['sha256']['video']==row['sha256']['video']
            assert row['mode']=='bf16' and row['repeat']==0
            stem=f'bf16_{key[0]}_seed{key[1]}_r0'
            assert (d/(stem+'_heads.npz')).exists() and (d/(stem+'_outputs.npz')).exists()
            found[key]=(d,stem,row)
    assert set(found)==expected,(expected-set(found),set(found)-expected)
    a.output.mkdir();(a.output/'source_snapshots').mkdir()
    for d,r in zip(a.inputs,reports):
        for src in (d/'source_snapshots').glob('*.py'):
            dst=a.output/'source_snapshots'/src.name
            if not dst.exists():os.link(src,dst)
    rows=[]
    for d,stem,row in found.values():
        for src in d.glob(stem+'*'):os.link(src,a.output/src.name)
        rows.append({**row,'source_run':str(d)})
    result={**first,'complete':True,'cases':list(cases.values()),'rows':rows,
            'source_sha256':{},'source_runs':sources,
            'assembly_note':'Every expected model/case/seed trajectory is complete. Interrupted source runs are preserved and are not themselves marked complete. Sampling durations are not performance measurements.'}
    result.pop('error',None)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Assembled',len(rows),'complete trajectories')


if __name__=='__main__':main()

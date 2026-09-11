"""Archive the execution study after raw and paired validation succeed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    for name in ['analysis.json','validation.json']:
        assert json.loads((a.output/name).read_text())['passed'],name
    snapshots=a.output/'source_snapshots';snapshots.mkdir(exist_ok=True)
    inventory=[]
    names=[f'execution_{m}_{v}' for m in ['edge','nano'] for v in ['eager','graph','fused']]
    names += [f'graph_fusion_{m}_bench' for m in ['edge','nano']]
    names += [f'graph_fusion_{m}_chunk_tuning' for m in ['edge','nano']]
    for name in names:
        directory=root/'runs5090'/name
        report=json.loads((directory/'result.json').read_text());assert report['complete']
        dest=a.output/'records'/name;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(directory/'result.json',dest/'result.json')
        for source in sorted((directory/'source_snapshots').glob('*.py')):
            assert source.stem==sha(source);shutil.copy2(source,snapshots/source.name)
        for source in sorted(directory.iterdir()):
            if source.is_file():inventory.append(dict(path=str(source),bytes=source.stat().st_size,sha256=sha(source)))
    # Preserve the launch and calibration input identities used by local timing.
    for model in ['edge','nano']:
        dest=a.output/'inputs'/model;dest.mkdir(parents=True,exist_ok=True)
        for source in [root/f'runs5090/auto_budget_{model}_bench/benchmark.json',root/f'runs5090/auto_budget_{model}_calibration/policy_validated.json']:
            shutil.copy2(source,dest/source.name)
        benchmark=json.loads((dest/'benchmark.json').read_text())
        for name in sorted({x['payload'] for x in benchmark['rows']}):
            source=Path(name)
            if not source.is_absolute():source=root/source
            inventory.append(dict(path=str(source),bytes=source.stat().st_size,sha256=sha(source)))
    final={}
    sources=[*Path(__file__).parent.glob('*.py'),root/'tools/finding_resample/run.py',root/'tools/finding_resample/validate.py',root/'tools/cosmos_e2e/graph.py']
    for source in sorted(sources):
        digest=sha(source);shutil.copy2(source,snapshots/(digest+'.py'));final[str(source.relative_to(root))]=digest
    (a.output/'final_tools.json').write_text(json.dumps(final,indent=2)+'\n')
    (a.output/'local_raw_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    logs=a.output/'logs';logs.mkdir(exist_ok=True)
    for source in sorted(Path('/tmp').glob('thunderworld_execution_*')):
        if source.is_file() and source.suffix in ['.log','.sh']:shutil.copy2(source,logs/source.name)
    files=sorted(f for f in a.output.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
    (a.output/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(a.output)}\n' for f in files))
    print(f'Archived {len(files)} files; inventoried {len(inventory)} local raw files')


if __name__=='__main__':main()

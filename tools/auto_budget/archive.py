"""Archive small evidence and content-addressed source snapshots; index raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=True)
    snapshots=a.output/'source_snapshots';snapshots.mkdir(exist_ok=True);inventory=[]
    inputs=sorted((root/'runs5090').glob('auto_budget_*'))+sorted((root/'data_local').glob('auto_budget_*_capture'))
    for directory in inputs:
        if not directory.is_dir():continue
        target=a.output/'records'/directory.parent.name/directory.name;target.mkdir(parents=True,exist_ok=True)
        for source in sorted(directory.iterdir()):
            if source.is_file():
                inventory.append(dict(path=str(source),bytes=source.stat().st_size,sha256=sha(source)))
                if source.suffix in ['.json','.csv','.log','.ncu-rep']:shutil.copy2(source,target/source.name)
        for source in (directory/'source_snapshots').glob('*.py'):
            assert source.stem==sha(source);shutil.copy2(source,snapshots/source.name)
    final={}
    for folder in ['auto_budget','finding_resample']:
        for source in (root/'tools'/folder).glob('*.py'):
            digest=sha(source);shutil.copy2(source,snapshots/(digest+'.py'));final[str(source.relative_to(root))]=digest
    (a.output/'final_analysis_sources.json').write_text(json.dumps(final,indent=2)+'\n')
    (a.output/'local_raw_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    logs=a.output/'validation_logs';logs.mkdir(exist_ok=True)
    for source in (root/'data_local/e2e_inspection').glob('auto_budget*.log'):shutil.copy2(source,logs/source.name)
    files=sorted(f for f in a.output.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
    (a.output/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(a.output)}\n' for f in files))
    print('Archived',len(files),'files; inventoried',len(inventory),'local raw files')


if __name__=='__main__':main()

"""Archive reviewable evidence; inventory large local media without copying it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2];inventory=[]
    snapshots=a.output/'source_snapshots';snapshots.mkdir(exist_ok=True)
    for directory in a.inputs:
        report=json.loads((directory/'result.json').read_text());assert report['complete']
        target=a.output/'records'/directory.name;target.mkdir(parents=True,exist_ok=True)
        for filename in ['result.json','perceptual.json','audio_decode.json']:
            source=directory/filename
            if source.exists():shutil.copy2(source,target/filename)
        for source in (directory/'source_snapshots').glob('*.py'):
            assert source.stem==sha(source)
            if not (snapshots/source.name).exists():shutil.copy2(source,snapshots/source.name)
        for source in sorted(directory.iterdir()):
            if source.is_file():inventory.append(dict(path=str(source.resolve()),bytes=source.stat().st_size,sha256=sha(source)))
    # Preserve the final analysis tools as well as the sources actually executed.
    final={}
    for source in Path(__file__).parent.glob('*.py'):
        digest=sha(source);shutil.copy2(source,snapshots/(digest+'.py'));final[str(source.relative_to(root))]=digest
    (a.output/'final_analysis_sources.json').write_text(json.dumps(final,indent=2)+'\n')
    (a.output/'local_raw_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    logs=a.output/'validation_logs';logs.mkdir(exist_ok=True)
    for name in ['findings_tests.log','findings_memcheck.log','lpips_install.log','sound_download.log']:
        source=root/'data_local/e2e_inspection'/name
        if source.exists():shutil.copy2(source,logs/name)
    for name in ['data_local/finding_inputs/robot_download.json','data_local/perceptual_cache/manifest.json']:
        source=root/name
        if source.exists():shutil.copy2(source,logs/(source.parent.name+'_'+source.name))
    files=sorted(f for f in a.output.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
    (a.output/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(a.output)}\n' for f in files))
    print('Archived',len(files),'files; inventoried',len(inventory),'local raw files')


if __name__=='__main__':main()

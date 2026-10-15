"""Build source/recipe and optional panel archives; exclude models, binaries and caches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staging',type=Path,required=True,help='Contains recipes/ and data/ produced by export tools')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    a.output.mkdir(parents=True,exist_ok=False)
    core=a.output/'ThunderWorld-CUTLASS-5090';core.mkdir()
    def tree(source,target):
        for path in sorted(source.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or any(x.startswith('.') or x.endswith('.egg-info') for x in path.relative_to(source).parts):continue
            if path.suffix not in ('.py','.cu','.cuh','.json','.md','.toml','.txt','.pt','.csv','.gz'):continue
            dst=target/path.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    for relative in ('packages/fp4-repair','tools/joint_inference','experiments/cutlass_fused_repair'):
        tree(repo/relative,core/relative)
    tree(a.staging/'recipes',core/'recipes')
    for budget in ('i2v_b125','i2v_b150'):
        dst=core/'policies'/budget;dst.mkdir(parents=True)
        for name in ('config.json','plan.json'):shutil.copyfile(a.staging/'data'/budget/name,dst/name)
    for relative in ('data_local/e2e_inputs/robot_153.jpg','data_local/finding_inputs/robot_table.png'):
        dst=core/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/relative,dst)
    shutil.copyfile(repo/'experiments/cutlass_fused_repair/README_5090.md',core/'README.md')
    (core/'SOURCE.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)),scope='Working-tree file hashes in FILES.json are authoritative'),indent=2)+'\n')
    files={str(p.relative_to(core)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(core.rglob('*')) if p.is_file()}
    (core/'FILES.json').write_text(json.dumps(files,indent=2)+'\n')
    core_tar=a.output/'cutlass-5090-core.tar.gz'
    with tarfile.open(core_tar,'w:gz') as tf:tf.add(core,arcname=core.name)
    # Deduplicate repeated trajectory panels between budgets using staging-only
    # hardlinks; tar records these once. Original capture data is never changed.
    panels=a.output/'panels';panels.mkdir();seen={}
    for path in sorted((a.staging/'data').rglob('*')):
        if not path.is_file():continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest();dst=panels/path.relative_to(a.staging/'data')
        dst.parent.mkdir(parents=True,exist_ok=True)
        if digest in seen:dst.hardlink_to(seen[digest])
        else:shutil.copyfile(path,dst);seen[digest]=dst
    panel_tar=a.output/'cutlass-5090-panels.tar.gz'
    with tarfile.open(panel_tar,'w:gz') as tf:tf.add(panels,arcname='panels')
    result={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (core_tar,panel_tar)}
    (a.output/'archives.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

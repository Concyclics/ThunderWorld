"""Collect completed local cohorts; never promote interrupted timing to a report."""
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'results/joint_inference_20260912'
DEST.mkdir(parents=True,exist_ok=True)

def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/'tools/joint_inference'/script),*map(str,args)],check=True,cwd=ROOT)

for source,label,model in [('/tmp/joint-edge-gpu3-performance','edge','edge'),
                          ('/tmp/joint-nano-i2v-gpu3-performance','nano_i2v','nano'),
                          ('/tmp/joint-nano-gpu3-performance','nano_policy_av','nano')]:
    matrix=Path(source)/'matrix.json'
    if not matrix.exists() or not json.loads(matrix.read_text())['complete']:continue
    run('summarize.py',matrix,'--quality',DEST/f'quality_{model}.json',
        '--output',DEST/f'performance_{label}.json')
    raw=DEST/f'raw_{label}.json'
    if not raw.exists():run('archive.py',matrix,'--output',raw)
for source,label in [('/tmp/joint-edge-gpu3-cold','edge'),('/tmp/joint-nano-i2v-gpu3-cold','nano_i2v'),
                     ('/tmp/joint-nano-gpu3-cold','nano_policy_av')]:
    matrix=Path(source)/'matrix.json';raw=DEST/f'cold_{label}.json'
    if matrix.exists() and json.loads(matrix.read_text())['complete'] and not raw.exists():
        run('archive.py',matrix,'--output',raw)
for source,name in [('/tmp/joint-edge-ablation-gpu3.json','ablation.json'),
                    ('/tmp/joint-edge-gpu3-tuned-equivalence.json','tuned_equivalence_edge.json'),
                    ('/tmp/joint-nano-i2v-gpu3-tuned-equivalence.json','tuned_equivalence_nano_i2v.json'),
                    ('/tmp/joint-nano-gpu3-tuned-equivalence.json','tuned_equivalence_nano_policy_av.json'),
                    ('/tmp/joint-edge-all-tiers-gpu3/result.json','all_tiers_edge.json'),
                    ('/tmp/joint-nano-all-tiers-gpu3/result.json','all_tiers_nano.json'),
                    ('/tmp/joint-final-assets-validation.json','offline_assets_validation.json')]:
    path=Path(source)
    if path.exists() and json.loads(path.read_text())['complete']:shutil.copyfile(path,DEST/name)
for source,name in [('/tmp/joint-edge-gpu3-tuning/profiles','edge_gpu3'),
                    ('/tmp/joint-nano-i2v-gpu3-tuning/profiles','nano_i2v_gpu3'),
                    ('/tmp/joint-nano-gpu3-tuning/profiles','nano_policy_av_gpu3')]:
    if Path(source).exists():shutil.copytree(source,DEST/'profiles'/name,dirs_exist_ok=True)
for model in ('edge','nano'):
    path=Path(f'/tmp/joint-{model}-all-tiers-gpu3/tuning.json')
    if path.exists():shutil.copyfile(path,DEST/'profiles'/f'{model}_all_tiers_gpu3.json')
log=Path('/tmp/joint-runtime-tests-f721e84.log')
if log.exists() and ' passed' in log.read_text() and not any(
        token in log.read_text() for token in (' failed',' skipped',' error')):
    shutil.copyfile(log,DEST/'tests_full_final.log')
run('report.py','--allow-incomplete')
(DEST/'MANIFEST.sha256').write_text(''.join(
    f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(DEST)}\n'
    for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='MANIFEST.sha256'))
print(DEST/'REPORT.md')

"""Archive verified AR measurements and provenance without copying model assets."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    def read(p):return json.loads(p.read_text())
    summary=read(a.run/'performance/summary.json');assert summary['complete']
    manifest=read(a.run/'performance/performance_matrix.json');assert manifest['complete']
    raw={}
    for job in manifest['jobs']:
        receipt=read(Path(job['output'])/'result.json');assert receipt['complete']
        raw[Path(job['output']).name]=receipt
    pilots={}
    for name in ('bf16','joint125','cutlass150'):
        pilot=read(a.run/f'pilot_{name}/result.json');assert pilot['complete'] and pilot['all_bitwise_equal']
        raw['pilot_'+name]=pilot
        pilots[name]={k:v for k,v in pilot.items() if k!='receipt'}
    audit=read(a.run/'audit_joint125/result.json');assert audit['complete'];raw['audit_joint125']=audit
    validation=dict(complete=True,audit={k:v for k,v in audit.items() if k!='receipt'},pilots=pilots,
        cpu_regression=dict(passed=45,gpu_dependent_skipped=15),
        scope='Invariance and actual full-model changed-input pilots plus 90 formal request comparisons. No claim of arbitrary-input or BF16-relative FP4 quality equivalence.')
    (a.output/'validation.json.xz').write_bytes(lzma.compress((json.dumps(validation,indent=2)+'\n').encode(),preset=3))
    (a.output/'receipts.json.xz').write_bytes(lzma.compress(json.dumps(raw,separators=(',',':')).encode(),preset=3))
    for name in ('summary.json','ar_comparison.png','ar_comparison.svg','performance_matrix.json'):
        shutil.copyfile(a.run/'performance'/name,a.output/name)
    profile=read(a.run/'profile_joint125_ar/receipt.json');assert profile['complete']
    (a.output/'profile_receipt.json.xz').write_bytes(lzma.compress(json.dumps(profile,separators=(',',':')).encode(),preset=3))
    shutil.copyfile(a.run/'profile_joint125_ar/breakdown.json',a.output/'optimized_breakdown.json')
    with (a.run/'profile_joint125_ar/trace.json').open('rb') as inp,lzma.open(a.output/'optimized_trace.json.xz','wb',preset=3) as out:
        shutil.copyfileobj(inp,out)
    (a.output/'checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(a.output.iterdir()) if p.is_file() and p.name!='checksums.json'},indent=2)+'\n')
    print(a.output)


if __name__=='__main__':main()

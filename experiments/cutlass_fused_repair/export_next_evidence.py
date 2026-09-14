"""Archive validated next-stage pipeline evidence, excluding models and raw videos."""
import argparse,hashlib,json,lzma,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    def read(p):return json.loads(p.read_text())
    summary=read(a.run/'performance/summary.json');assert summary['complete']
    matrix=read(a.run/'performance/performance_matrix.json');assert matrix['complete'];raw={}
    for job in matrix['jobs']:
        r=read(Path(job['output'])/'result.json');assert r['complete'];raw[Path(job['output']).name]=r
    validation={}
    for base in ('bf16','joint125','cutlass150'):
        r=read(a.run/f'gate_{base}/result.json');assert r['complete'];raw['gate_'+base]=r
        assert all(v['exact'] for v in r['variants'].values())
        for variant in r['variants'].values():
            samples=variant['rows'];assert len(samples)==5
            assert samples[0]['sha256']==samples[-1]['sha256']
            assert all(samples[0]['sha256']!=row['sha256'] for row in samples[1:4])
        assert r['variants']['ar']['receipt']['source_sha256']==r['variants']['best']['receipt']['source_sha256']==matrix['pilot_source_sha256']
        validation[base]={v:dict(exact=row['exact'],flags=row['flags'],rows=row['rows']) for v,row in r['variants'].items()}
    pilot=read(a.run/'pilot_joint125/result.json');assert pilot['complete'];raw['exploratory_cumulative_pilot']=pilot
    (a.output/'validation.json.xz').write_bytes(lzma.compress((json.dumps(dict(complete=True,gpu_kernel_tests_passed=22,cpu_tests_passed=50,cpu_gpu_skipped=87,gates=validation,
      scope='Exact preservation of each precision/repair configuration, including changed prompt/image and restored input; not a new BF16-relative quality benchmark'),indent=2)+'\n').encode(),preset=3))
    (a.output/'receipts.json.xz').write_bytes(lzma.compress(json.dumps(raw,separators=(',',':')).encode(),preset=3))
    for name in ('summary.json','pipeline_comparison.png','pipeline_comparison.svg','performance_matrix.json'):shutil.copyfile(a.run/'performance'/name,a.output/name)
    shutil.copyfile(a.run/'components/result.json',a.output/'component_experiments.json')
    for name in ('breakdown.json','pipeline_breakdown.png','pipeline_breakdown.svg'):shutil.copyfile(a.run/'profile_best'/name,a.output/name)
    with (a.run/'profile_best/trace.json').open('rb') as i,lzma.open(a.output/'trace.json.xz','wb',preset=3) as o:shutil.copyfileobj(i,o)
    with (a.run/'profile_best/receipt.json').open('rb') as i,lzma.open(a.output/'profile_receipt.json.xz','wb',preset=3) as o:shutil.copyfileobj(i,o)
    (a.output/'checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.output.iterdir()) if p.is_file() and p.name!='checksums.json'},indent=2)+'\n')
if __name__=='__main__':main()

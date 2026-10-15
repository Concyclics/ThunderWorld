"""Finish and archive a running fixed-scene suite without selecting new policies."""
import argparse,gzip,json,shutil,subprocess,sys,time
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    progress=a.study/'metrics/progress.json'
    while not progress.exists() or not json.loads(progress.read_text()).get('complete'):
        suite=a.study/'generation/suite.json'
        if suite.exists() and json.loads(suite.read_text()).get('paused'):raise RuntimeError('Generation paused; retained progress')
        time.sleep(10)
    subprocess.run([sys.executable,str(Path(__file__).with_name('dataset_report.py')),str(a.study/'manifest.json'),
        '--metrics',str(a.study/'metrics'),'--generation',str(a.study/'generation'),'--output',str(a.study/'report')],check=True)
    a.output.mkdir(parents=True,exist_ok=True);evidence={}
    for label,path in [('manifest',a.study/'manifest.json'),('audit',a.study/'audit.json'),('quality',a.study/'report/quality.json'),
        ('costs',a.study/'report/costs.json'),('historical_psnr',a.study.parent/'contribution_psnr_backfill_20260917.json'),
        ('psnr_verification',a.study.parent/'contribution_psnr_verification_20260917.json')]:
        target=a.output/(label+'.json.gz')
        with target.open('wb') as handle:
            with gzip.GzipFile(filename='',mode='wb',fileobj=handle,mtime=0) as archive:archive.write(path.read_bytes())
        evidence[label]=dict(source=str(path),source_sha256=sha(path),archive=target.name,archive_sha256=sha(target))
    shutil.copyfile(a.study/'report/REPORT.md',a.output/'REPORT.md')
    write(a.output/'evidence.json',dict(complete=True,evidence=evidence,performance_eligible=False,
        implementation_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('dataset*.py')}))
    samples=[]
    for p in sorted((a.study/'generation').glob('*/*/result.json')):
        r=json.loads(p.read_text());samples.append(dict(receipt=str(p),sha256=sha(p),outputs=[dict(seed=x['seed'],path=str(p.parent/x['outputs']),sha256=x['outputs_sha256']) for x in r['rows']]))
    write(a.output/'samples.json',dict(complete=True,samples=samples))


if __name__=='__main__':main()

"""Keep the existing optimized 2.5x candidate as an explicit quality anchor."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from baselines import sha,write
from protocol import admit_run
from representation_trajectory import replace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('template','cohort','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--split',choices=('trajectory','validation','final'),required=True)
    p.add_argument('--frozen-selection',type=Path);a=p.parse_args()
    template=json.loads(a.template.read_text());command=list(next(r for r in template['runs'] if r['name']=='candidate')['command'])
    command[0]=sys.executable
    if '--idle-check' in command:command.remove('--idle-check')
    if '--reference' in command:
        i=command.index('--reference');del command[i:i+2]
    replace(command,'--warm-requests',['1'])
    # Canonical identity excludes only request seeds and destination.
    replace(command,'--seeds',['COHORT']);replace(command,'--output',['OUTPUT'])
    identity=dict(command=list(command),template_sha256=sha(a.template),files={str(Path(v).resolve()):sha(v)
        for v in command[1:] if Path(v).is_file()})
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cohort=json.loads(a.cohort.read_text());seeds=cohort['seeds'][a.split]
    admission=admit_run(a.cohort,a.split,seeds,'legacy_2.5',a.frozen_selection,digest)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    replace(command,'--seeds',list(map(str,seeds)));replace(command,'--output',[str(a.output)])
    with a.output.with_suffix('.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=a.output/'result.json';result=json.loads(receipt.read_text())
    if not result['complete']:raise ValueError('Incomplete legacy run')
    result.update(method='legacy_2.5',label='legacy_2.5',admission=admission,configuration=identity,
        configuration_sha256=digest,performance_eligible=False,
        contribution_scope='Existing optimized N128 2.5x anchor retained; no new quantization or lower-quality control substituted.')
    write(receipt,result);write(a.output/'command.json',dict(command=command))


if __name__=='__main__':main()

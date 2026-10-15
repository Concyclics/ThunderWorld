"""Run manifest-declared experiments serially; fail closed on incomplete runs."""
import argparse,json,subprocess,sys
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());root=Path(manifest['output'])
    root.mkdir(parents=True,exist_ok=True)
    receipt=root/'suite.json'
    state=dict(manifest_sha256=sha(a.manifest),complete=False,runs={})
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['manifest_sha256']!=state['manifest_sha256']:raise ValueError('Manifest changed during resume')
        state=old
    for run in manifest['runs']:
        name=run['name'];folder=root/name
        if name in state['runs']:
            if sha(folder/'result.json')!=state['runs'][name]:raise ValueError('Completed receipt changed')
            continue
        if folder.exists():raise RuntimeError(f'Incomplete run requires inspection: {folder}')
        cmd=[sys.executable,str(Path(__file__).with_name('external_infer.py')),
             *manifest['common_args'],*run['args'],'--output',str(folder)]
        write(root/(name+'.command.json'),dict(command=cmd))
        with (root/(name+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((folder/'result.json').read_text())
        if not result.get('complete'):raise ValueError('Incomplete child run')
        state['runs'][name]=sha(folder/'result.json');write(receipt,state)
    state['complete']=True;write(receipt,state)


if __name__=='__main__':main()

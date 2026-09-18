"""Serial shared-GPU multi-scene generation. Resume immutable completed requests."""
import argparse,json,subprocess,sys
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    study=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True)
    state=dict(manifest_sha256=sha(a.manifest),complete=False,methods={},performance_eligible=False)
    path=a.output/'suite.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['manifest_sha256']!=state['manifest_sha256']:raise ValueError('Changed manifest')
    write(path,state)
    try:
        for name in study['methods']:
            folder=a.output/name;receipt=folder/'progress.json'
            if receipt.exists() and json.loads(receipt.read_text()).get('complete'):
                if json.loads(receipt.read_text())['identity']['manifest_sha256']!=state['manifest_sha256']:raise ValueError('Changed completed run')
            else:
                command=[sys.executable,str(Path(__file__).with_name('dataset_infer.py')),str(a.manifest),'--method',name,'--output',str(folder)]
                with (a.output/(name+'.log')).open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            state['methods'][name]=sha(receipt);write(path,state)
        state['complete']=True;write(path,state)
    except Exception as error:
        state.update(paused=True,error=repr(error));write(path,state);raise


if __name__=='__main__':main()

"""CPU-only paired metrics; resume per scene and follow a serial generation suite."""
import argparse,hashlib,json,os,time
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path)
    p.add_argument('--generation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--follow',action='store_true')
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,numpy as np,lpips
    from thunderworld_fp4.cosmos.quality import compare
    torch.set_num_threads(8);torch.hub.set_dir('/tmp/thunderworld-lpips-cache')
    metric=lpips.LPIPS(net='alex',version='0.1').eval().cpu()
    study=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True)
    identity=dict(manifest_sha256=sha(a.manifest),source_sha256=sha(__file__),quality_sha256=sha(Path(__file__).parents[2]/'packages/fp4-repair/src/thunderworld_fp4/cosmos/quality.py'))
    methods=[name for name in study['methods'] if name!='bf16']
    expected=len(methods)*len(study['cases']);progress=a.output/'progress.json'
    if progress.exists() and json.loads(progress.read_text())['identity']!=identity:raise ValueError('Metric implementation changed during resume')
    def read(folder):
        path=folder/'result.json'
        if not path.exists():return None
        receipt=json.loads(path.read_text())
        if receipt['identity']['manifest_sha256']!=identity['manifest_sha256']:raise ValueError('Wrong generation manifest')
        return receipt if receipt['complete'] else None
    def load(folder,row):
        path=folder/row['outputs']
        if sha(path)!=row['outputs_sha256']:raise ValueError('Corrupt output archive')
        with np.load(path) as data:signals=dict(data)
        if {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in signals.items()}!=row['sha256']:raise ValueError('Signal hash mismatch')
        return signals
    while True:
        completed=0;new=0
        for entry in study['cases']:
            case_id=entry['case']['id'];bf=a.generation/'bf16'/case_id;reference=read(bf)
            if reference is None:continue
            for name in methods:
                target=a.output/name/(case_id+'.json');folder=a.generation/name/case_id;receipt=read(folder)
                if receipt is None:continue
                provenance=dict(identity=identity,reference_sha256=sha(bf/'result.json'),candidate_sha256=sha(folder/'result.json'))
                if target.exists():
                    old=json.loads(target.read_text())
                    if old['provenance']!=provenance or not old['complete']:raise ValueError('Changed metric inputs')
                    completed+=1;continue
                for key in ('case','image_sha256','seeds'):
                    if receipt['identity'][key]!=reference['identity'][key]:raise ValueError('Unmatched '+key)
                if receipt['model_identity']!=reference['model_identity'] or receipt['targets']!=reference['targets']:raise ValueError('Unmatched model/scope')
                if {r['seed'] for r in receipt['rows']}!=set(entry['seeds']) or {r['seed'] for r in reference['rows']}!=set(entry['seeds']):raise ValueError('Missing paired seeds')
                rows=[]
                for seed in entry['seeds']:
                    gold=next(r for r in reference['rows'] if r['seed']==seed);row=next(r for r in receipt['rows'] if r['seed']==seed)
                    if gold['steps']!=study['steps'] or row['steps']!=study['steps']:raise ValueError('Changed steps')
                    rows.append(dict(seed=seed,metrics=compare(load(folder,row),load(bf,gold),metric)))
                write(target,dict(complete=True,name=name,case_id=case_id,benchmark=entry['benchmark'],category=entry['category'],rows=rows,provenance=provenance))
                completed+=1;new+=1
                write(progress,dict(identity=identity,complete=False,completed=completed,expected=expected))
                print(json.dumps(dict(name=name,case=case_id)),flush=True)
        state=dict(identity=identity,complete=completed==expected,completed=completed,expected=expected)
        write(progress,state)
        if state['complete']:break
        suite=a.generation/'suite.json'
        if suite.exists():
            status=json.loads(suite.read_text())
            if status.get('paused'):raise RuntimeError('Generation paused: '+status.get('error',''))
            if status.get('complete'):raise RuntimeError('Generation complete but paired samples missing')
        if not a.follow:break
        if not new:time.sleep(10)


if __name__=='__main__':main()

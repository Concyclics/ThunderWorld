"""Exercise an interrupted two-seed FP4 request without changing source evidence."""
import argparse
import json
from pathlib import Path
import shutil

from baselines import sha, write
from dataset_infer import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--method',default='b2_joint')
    a=p.parse_args();manifest=a.study/'manifest.json';study=json.loads(manifest.read_text())
    original=a.study/'generation'/a.method;out=a.study/'resume_probe'/a.method
    receipt=out/'probe.json';identity=dict(manifest_sha256=sha(manifest),method=a.method,source_sha256=sha(__file__))
    if receipt.exists():
        prior=json.loads(receipt.read_text())
        if prior['identity']!=identity:raise ValueError('Changed resume probe')
        if prior['complete']:return
    out.mkdir(parents=True,exist_ok=True)
    source_receipts={e['case']['id']:json.loads((original/e['case']['id']/'result.json').read_text()) for e in study['cases']}
    first=study['cases'][0]['case']['id'];removed=source_receipts[first]['rows'][-1]['seed']
    baseline={}
    for cid,value in source_receipts.items():
        src=original/cid;dst=out/cid;dst.mkdir(exist_ok=True)
        baseline[cid]={row['seed']:row['sha256'] for row in value['rows']}
        for row in value['rows']:
            if cid==first and row['seed']==removed:continue
            if sha(src/row['outputs'])!=row['outputs_sha256']:raise ValueError('Changed original output')
            shutil.copyfile(src/row['outputs'],dst/row['outputs'])
        if (src/'compute.json').exists():shutil.copyfile(src/'compute.json',dst/'compute.json')
        if cid==first:
            value['rows']=[row for row in value['rows'] if row['seed']!=removed]
            value['complete']=False
        write(dst/'result.json',value)
    kept={str(f):f.stat().st_mtime_ns for f in out.glob('*/seed*.npz')}
    write(receipt,dict(identity=identity,complete=False,removed_case=first,removed_seed=removed))
    state=run(manifest,a.method,out,smoke=True)
    if not state['complete']:raise ValueError('Resume did not complete')
    for cid,expected in baseline.items():
        result=json.loads((out/cid/'result.json').read_text())
        if {r['seed']:r['sha256'] for r in result['rows']}!=expected:
            raise ValueError('Resumed signals differ from uninterrupted execution')
    if any(Path(f).stat().st_mtime_ns!=mtime for f,mtime in kept.items()):
        raise ValueError('Resume regenerated an already completed output')
    write(receipt,dict(identity=identity,complete=True,removed_case=first,removed_seed=removed,
        regenerated_signals_bitwise_equal=True,completed_outputs_untouched=True,
        performance_eligible=False))


if __name__=='__main__':main()

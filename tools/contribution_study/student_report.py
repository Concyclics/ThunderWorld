"""Compare student same-input BF16 errors to the selected teacher-panel proxy."""
import argparse,json,statistics
from pathlib import Path
from baselines import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','allocation','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.capture/'manifest.json').read_text());allocation=json.loads(a.allocation.read_text())
    if not manifest['complete'] or manifest.get('trajectory')!='student':raise ValueError('Complete student trajectory required')
    import torch
    torch.set_num_threads(8)
    rows=[]
    for row in manifest['rows']:
        path=a.capture/row['file']
        if sha(path)!=row['sha256']:raise ValueError('Student panel hash mismatch')
        panel=torch.load(path,weights_only=True,map_location='cpu')
        gold=panel['y'].float();pred=panel['candidate_y'].float()
        nmse=float((pred-gold).square().sum()/gold.square().sum().clamp_min(1e-20))
        key=f'{row["kind"]}|{row["branch"]}|{row["phase"]}|{row["name"]}'
        teacher=allocation['assignments'][key]['mean_nmse']
        rows.append(dict(kind=row['kind'],name=row['name'],seed=row['seed'],phase=row['phase'],branch=row['branch'],
            student_same_input_nmse=nmse,teacher_mean_nmse=teacher))
    groups=[]
    for kind in ('linear','attention'):
        for phase in range(4):
            subset=[r for r in rows if r['kind']==kind and r['phase']==phase]
            groups.append(dict(kind=kind,phase=phase,count=len(subset),
                student_mean_nmse=statistics.mean(r['student_same_input_nmse'] for r in subset),
                teacher_mean_nmse=statistics.mean(r['teacher_mean_nmse'] for r in subset)))
    write(a.output,dict(complete=True,capture_sha256=sha(a.capture/'manifest.json'),allocation_sha256=sha(a.allocation),
        rows=rows,groups=groups,performance_eligible=False,
        scope='Candidate current-input error against original BF16 operator, unweighted local NMSE. '
              'Different teacher/student inputs: this diagnoses distribution shift and is not a paired generation-quality gate. No further weight fitting performed.'))


if __name__=='__main__':main()

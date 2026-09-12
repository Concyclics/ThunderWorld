"""Merge one teacher cohort and one student trajectory cohort without copying tensors."""
import argparse,json,hashlib
from pathlib import Path
from capture_budget import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher','student','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();teacher=json.loads((a.teacher/'manifest.json').read_text());student=json.loads((a.student/'manifest.json').read_text())
    if not teacher['complete'] or not student['complete']:raise ValueError('Complete capture cohorts required')
    if student.get('calibration_round')!=1 or teacher.get('calibration_round',0)!=0:raise ValueError('Exactly one student recalibration round allowed')
    for key in ('model_identity','case','cohort_sha256'):
        if teacher[key]!=student[key]:raise ValueError(f'Teacher/student {key} mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    merged={**teacher,'calibration_round':1,'trajectory':'teacher+student','rows':[],
            'weights':{name:str((a.teacher/file).resolve()) for name,file in teacher['weights'].items()},
            'parents':{str(folder.resolve()):hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest() for folder in (a.teacher,a.student)}}
    for folder,manifest in ((a.teacher,teacher),(a.student,student)):
        merged['rows'].extend({**row,'file':str((folder/row['file']).resolve())} for row in manifest['rows'])
    keys=[(r['seed'],r['kind'],r['name'],r['phase'],r['branch']) for r in merged['rows']]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate panels in merged cohort')
    atomic_json(a.output/'manifest.json',merged)


if __name__=='__main__':main()

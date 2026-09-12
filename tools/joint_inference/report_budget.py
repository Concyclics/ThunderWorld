"""Inspect calibration progress and observed quality/compute/storage frontier, without latency claims."""
import argparse,json
from pathlib import Path
import numpy as np
from capture_budget import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path)
    p.add_argument('--quality',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=dict(performance_eligible=False,resource_mode='shared-calibration',complete=False)
    capture=a.experiment/'capture/manifest.json';base=a.experiment/'base_calibration/manifest.json'
    if capture.exists():
        c=json.loads(capture.read_text());report['capture']=dict(complete=c['complete'],panels=len(c['rows']),weights=len(c['weights']),cohort_sha256=c['cohort_sha256'])
    if base.exists():
        b=json.loads(base.read_text());rows=b['decisions']
        report['base_calibration']=dict(complete=b['complete'],modules=len(rows),
            average_original_objective=float(np.mean([r['baseline_loss'] for r in rows])),
            average_calibrated_objective=float(np.mean([r['loss'] for r in rows])),
            local_improved_modules=sum(r['loss']<r['baseline_loss'] for r in rows),
            scope='Native full-M local operator NMSE mean + 0.25 worst panel; not end-to-end quality')
    if a.quality:
        quality=json.loads(a.quality.read_text());report['quality']=quality
    else:report['quality_status']='Full cohort quality gate not yet completed'
    search=a.experiment/'repair_search/search.json'
    if search.exists():
        s=json.loads(search.read_text());report['repair_search']=dict(complete=s['complete'],linears=len(s['modules']),attention=len(s['attention']))
    report['scope']='Frontier within searched configurations only; latency Pareto frontier requires exclusive-GPU measurements'
    atomic_json(a.output/'report.json',report)
    lines=['# Shared GPU compute-constrained Edge I2V','',
           'These results are not eligible for performance ranking. No latency speedup is inferred from MMA counts.','']
    if 'capture' in report:lines.append(f"Captured {report['capture']['panels']} full-M panels across {report['capture']['weights']} generation projections.")
    if 'base_calibration' in report:
        b=report['base_calibration'];lines += ['',f"Local objective: {b['average_original_objective']:.8f} → {b['average_calibrated_objective']:.8f} across {b['modules']} modules.",
            'This is a teacher-input local objective; full videos must be evaluated separately.']
    lines+=['',report.get('quality_status','See report.json for paired quality results.'),'',report['scope']]
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

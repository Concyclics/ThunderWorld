"""Audit equal support/storage and summarize native outlier interaction results."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from baselines import sha, write
from paper_outlier import VARIANTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    state = json.loads((a.study/'progress.json').read_text())
    if not state['complete'] or not state['complete_model'] or len(state['modules']) != 168:
        raise ValueError('All-layer native calibration required')
    summary = {}
    for label in VARIANTS:
        rows = [r['results'][label] for r in state['modules'].values()]
        summary[label] = dict(modules=len(rows), removed_elements=sum(r['removed'] for r in rows),
            weighted_entire_w0_error=float(np.mean([r['weighted_entire_w0_error'] for r in rows])),
            weighted_deployed_w0_w1_error=float(np.mean([r['weighted_deployed_w0_w1_error'] for r in rows])),
            three_objective=float(np.mean([r['three']['objective'] for r in rows])),
            four_objective=float(np.mean([r['four']['objective'] for r in rows])),
            three_improved_modules=sum(r['results'][label]['three']['objective'] < r['results']['rtn']['three']['objective'] for r in state['modules'].values()),
            four_improved_modules=sum(r['results'][label]['four']['objective'] < r['results']['rtn']['four']['objective'] for r in state['modules'].values()),
            tensor_bytes=sum(r['exports'][label]['tensor_bytes'] for r in state['modules'].values()))
    if len({r['tensor_bytes'] for r in summary.values()}) != 1:
        raise ValueError('Fixed-support experiment changed tensor storage')
    for row in state['modules'].values():
        for label in VARIANTS:
            if not set(row['results'][label]['mandatory_w_blocks']) <= set(row['support']):
                raise ValueError('Outlier not covered by mandatory W1')
            item = row['exports'][label]
            if sha(a.study/item['file']) != item['sha256']:
                raise ValueError('Changed sparse asset')
    a.output.mkdir(parents=True, exist_ok=True)
    result = dict(complete=True, summary=summary, source_sha256=sha(__file__),
        calibration_sha256=sha(a.study/'progress.json'), performance_eligible=False,
        scope='Native operator calibration only, equal sparse tensor payload and fixed prescale/A selection/W1 support. '
              'MMA is matched within each cross setting; four-term costs are not assigned the original three-term request cap. '
              'Selection and evaluation use calibration panels; no generalization claim. Whole-video acceptance is pending.')
    write(a.output/'outlier.json', result)
    (a.output/'outlier_raw.json.gz').write_bytes(gzip.compress(json.dumps(state).encode(), mtime=0))
    lines = ['# Fixed-support outlier x cross', '', result['scope'], '',
        '| Representation | Three objective | Four objective | Improved modules (3/4) | Removed elements | Tensor MiB |',
        '|---|---:|---:|---:|---:|---:|']
    for label, row in summary.items():
        lines.append(f"| {label} | {row['three_objective']:.9g} | {row['four_objective']:.9g} | {row['three_improved_modules']}/{row['four_improved_modules']} | {row['removed_elements']} | {row['tensor_bytes']/2**20:.3f} |")
    (a.output/'OUTLIER.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()

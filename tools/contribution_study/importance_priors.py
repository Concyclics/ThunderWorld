"""Reduce all 16-scene x 4-phase x 2-CFG full-input RMS summaries to scale priors."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from baselines import sha, write
from thunderworld_fp4.cosmos.budget import PHASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cohort = json.loads((a.previous / 'cohort.json').read_text())
    calibration = json.loads((a.previous / 'calibration/manifest.json').read_text())
    if not calibration['complete'] or calibration['identity']['cohort_sha256'] != sha(a.previous / 'cohort.json'):
        raise ValueError('Unmatched calibration')
    moments = {}
    for entry in cohort['cases']['calibration']:
        path = a.previous / 'calibration' / (entry['case']['id'] + '.json')
        if sha(path) != calibration['cases'][entry['case']['id']]:
            raise ValueError('Changed calibration')
        data = json.loads(path.read_text())
        dataset_weight = .5 / (9 if entry['benchmark'] == 'rbench' else 7)
        for name, rows in data['linears'].items():
            moments.setdefault(name, np.zeros(rows[0]['shape'][2]))
            for row in rows:
                calls = PHASES[row['phase'] + 1] - PHASES[row['phase']]
                moments[name] += np.square(row['rms']) * dataset_weight * calls / 70
    a.output.mkdir(parents=True, exist_ok=True)
    torch.save({name: torch.from_numpy(value).float().sqrt() for name, value in moments.items()}, a.output / 'rms.pt')
    write(a.output / 'manifest.json', dict(complete=True, rms_sha256=sha(a.output / 'rms.pt'),
          cohort_sha256=sha(a.previous / 'cohort.json'), calibration_sha256=sha(a.previous / 'calibration/manifest.json'),
          modules=len(moments), scope='Equal dataset/category weighting, actual phase call counts, both CFG branches. No row clipping.'))


if __name__ == '__main__':
    main()

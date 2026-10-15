"""Within-operator interventions on unmodified, full 35-step BF16 trajectories."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from structure_repair import BOUNDS, COVERAGES, ROLES, candidates, digest, mask_cost, mask_summary, write

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/finding_resample'))
import run as runner
from runtime import Observer
from thunderworld_fp4 import RepairConfig, fp4_attention
from thunderworld_fp4.cosmos.resources import select_shared_device, shared_worker


class RepairObserver(Observer):
    prior = None
    selected_layers = ()
    selected_steps = (0, 7, 18, 28)
    records = None
    options = None
    output = None
    ordinal = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.ordinal = type(self).ordinal
        type(self).ordinal += 1
        self.mask_cache = {}

    def observe(self, q, k, v, output, kwargs, step, branch, layer, timestep):
        if step not in self.selected_steps or layer not in self.selected_layers or branch != 0:
            return
        labels, _, ar = self.layout(kwargs, branch)
        qr, kr = labels[ar:].cpu().numpy(), labels.cpu().numpy()
        if len(qr) != q.shape[1] or len(kr) != k.shape[1]:
            raise AssertionError('Incomplete token layout')
        m, n, h, hk = q.shape[1], k.shape[1], q.shape[2], k.shape[2]
        ph = next(i for i, (lo, hi) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:])) if lo <= step < hi)
        cfg = RepairConfig(coverage=0, terms=3, query_tile=32, splits=2, transpose_v=True, sparse_pack=True)
        gold = output.float()
        roles = sorted(set(qr.tolist()))
        indices = [torch.as_tensor(np.where(qr == role)[0], device=q.device) for role in roles]
        denominators = torch.stack([gold.square().sum()] + [gold[:, idx].square().sum() for idx in indices]).clamp_min(1e-20)
        frozen_inputs = tuple(x.clone() for x in (q, k, v, output))
        names, estimates = [], []
        for name, fraction in [('base', 0.), ('full_three', 1.)]:
            mask = np.full((h, (m + 127) // 128, (n + 127) // 128), bool(fraction))
            names.append((name, fraction, mask, mask_cost(mask, m, n, hk)))
        for fraction in COVERAGES:
            group = candidates(self.prior, ph, layer, qr, kr, fraction, self.options.random_repeats)
            costs = [mask_cost(mask, m, n, hk) for mask in group.values()]
            if len({cost['total_mma'] for cost in costs}) != 1:
                raise AssertionError('Native padded MMA differs between selectors')
            for (name, mask), cost in zip(group.items(), costs):
                names.append((name, fraction, mask, cost))
        for name, fraction, mask, cost in names:
            actual = torch.from_numpy(mask).to(q.device)
            pred = fp4_attention(q, k, v, cfg, repair_mask=actual)
            error = (pred.float() - gold).square()
            estimates.append(torch.stack([error.sum()] + [error[:, idx].sum() for idx in indices]) / denominators)
        values = torch.stack(estimates).cpu().numpy()
        if not np.isfinite(values).all():
            raise AssertionError('Nonfinite full attention output error')
        if not all(torch.equal(a, b) for a, b in zip(frozen_inputs, (q, k, v, output))):
            raise AssertionError('Observer modified the BF16 trajectory input/output')
        for (name, fraction, mask, cost), error in zip(names, values):
            self.records.append(dict(step=step, phase=ph, branch=branch, layer=layer,
                method=name, coverage=fraction, nmse=float(error[0]), relative_l2=float(np.sqrt(error[0])),
                role_nmse={ROLES[r]: float(e) for r, e in zip(roles, error[1:])},
                shape=[m, n, h, hk, 128], ar_tokens=ar, role_counts=np.bincount(kr, minlength=6).tolist(),
                mma=cost['mma_by_dtype'], total_mma=cost['total_mma'], base_mma=cost['base_mma'],
                mma_ratio=cost['total_mma']/cost['base_mma'], packing=cost['packing'],
                **mask_summary(mask, qr, kr, hk)))
        write(self.output / f'progress_request_{self.ordinal:03d}.json', dict(complete=False, panels=len(self.records)//len(names), last_step=step, last_layer=layer))

    def save(self, path):
        expected = len(self.selected_steps) * len(self.selected_layers)
        actual = {(r['step'], r['layer']) for r in self.records}
        if len(actual) != expected:
            raise AssertionError(('Missing intervention panels', len(actual), expected))
        payload = dict(complete=True, performance_eligible=False, native_fp4=True,
            steps=list(self.selected_steps), layers=list(self.selected_layers), random_repeats=self.options.random_repeats,
            scope='Local full-attention outputs on unchanged BF16 trajectories; branch 0; all base blocks retained; no full-generation quality claim',
            rows=self.records)
        write(path.with_suffix('.repair.json'), payload)
        np.savez_compressed(path, counts=self.counts, roles=ROLES, rows_json=np.asarray(json.dumps(self.records)))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--structure-prior', type=Path, required=True)
    parser.add_argument('--probe-layers', type=int, nargs='+', required=True)
    parser.add_argument('--random-repeats', type=int, default=8)
    a, remaining = parser.parse_known_args()
    if '--sample' not in remaining or a.random_repeats < 2:
        raise ValueError('Explicit sampling and at least two random masks required')
    if '--steps' in remaining and int(remaining[remaining.index('--steps')+1]) != 35:
        raise ValueError('Frozen four-phase experiment requires 35 steps')
    out = Path(remaining[remaining.index('--output')+1])
    with np.load(a.structure_prior, allow_pickle=False) as z:
        RepairObserver.prior = z['density']
    RepairObserver.selected_layers = tuple(a.probe_layers)
    RepairObserver.options, RepairObserver.output = a, out
    runner.Observer = RepairObserver
    sys.argv = [sys.argv[0], *remaining]
    model = remaining[remaining.index('--model')+1]
    select_shared_device(49152 if model == 'nano' else 12288)
    with shared_worker():
        runner.main()
    result = json.loads((out / 'result.json').read_text())
    result['structure_repair'] = dict(prior_sha256=digest(a.structure_prior),
        source_sha256={p.name:digest(p) for p in [Path(__file__), Path(__file__).with_name('structure_repair.py')]},
        same_padded_mma=True, output_and_inputs_unchanged=True, performance_eligible=False)
    write(out / 'result.json', result)


if __name__ == '__main__':
    main()

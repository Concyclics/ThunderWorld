"""Replay frozen BF16 trajectories with current-mass and structure interventions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from attention_mass_oracle import attention_tile_mass, mass_capture, oracle_mask, selector_cost
from structure_repair import BOUNDS, COVERAGES, ROLES, candidates, digest, mask_cost, mask_summary, write

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/finding_resample'))
import run as runner
from runtime import Observer
from thunderworld_fp4 import RepairConfig, fp4_attention
from thunderworld_fp4.cosmos.resources import select_shared_device, shared_worker


class MassOracleObserver(Observer):
    prior = None
    selected_layers = ()
    selected_steps = (0, 7, 18, 28)
    reference = None
    output = None
    ordinal = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self.mass_records = [], []
        self.ordinal = type(self).ordinal
        type(self).ordinal += 1

    def observe(self, q, k, v, output, kwargs, step, branch, layer, timestep):
        if step not in self.selected_steps or layer not in self.selected_layers or branch != 0:
            return
        labels, _, ar = self.layout(kwargs, branch)
        qr, kr = labels[ar:].cpu().numpy(), labels.cpu().numpy()
        if len(qr) != q.shape[1] or len(kr) != k.shape[1]:
            raise AssertionError('Incomplete token layout')
        m, n, h, hk, d = q.shape[1], k.shape[1], q.shape[2], k.shape[2], q.shape[3]
        phase = next(i for i, (lo, hi) in enumerate(zip(BOUNDS[:-1], BOUNDS[1:])) if lo <= step < hi)
        frozen = tuple(x.clone() for x in (q, k, v, output))
        mass = attention_tile_mass(q, k).cpu().numpy()
        cfg = RepairConfig(coverage=0, terms=3, query_tile=32, splits=2, transpose_v=True, sparse_pack=True)
        gold = output.float()
        roles = sorted(set(qr.tolist()))
        indices = [torch.as_tensor(np.where(qr == role)[0], device=q.device) for role in roles]
        denom = torch.stack([gold.square().sum()] + [gold[:, idx].square().sum() for idx in indices]).clamp_min(1e-20)
        entries = []
        for name, fraction in [('base', 0.), ('full_three', 1.)]:
            mask = np.full(mass.shape, bool(fraction))
            entries.append((name, fraction, mask, mask_cost(mask, m, n, hk)))
        for fraction in COVERAGES:
            group = candidates(self.prior, phase, layer, qr, kr, fraction, 8)
            oracle = oracle_mask(mass, fraction, 44071 + 1009 * layer + 7919 * phase)
            group['mass_oracle'] = oracle
            costs = {name: mask_cost(mask, m, n, hk) for name, mask in group.items()}
            if len({c['total_mma'] for c in costs.values()}) != 1:
                raise AssertionError('Matched padded FP4 repair budget failed')
            capture = {name: mass_capture(mass, mask, m) for name, mask in group.items()}
            if max(capture.values()) > capture['mass_oracle'] + 1e-7:
                raise AssertionError('Top-mass selector does not maximize captured mass')
            self.mass_records.append(dict(step=step, layer=layer, coverage=fraction,
                captured_mass=capture, mask_overlap=float((oracle & group['structure']).sum() / oracle.sum()),
                mass=mass.tolist(), oracle_selector=selector_cost(m, n, h, d)))
            entries.extend((name, fraction, group[name], costs[name]) for name in ('structure', 'mass_oracle'))
        estimates = []
        for name, fraction, mask, cost in entries:
            pred = fp4_attention(q, k, v, cfg, repair_mask=torch.from_numpy(mask).to(q.device))
            err = (pred.float() - gold).square()
            estimates.append(torch.stack([err.sum()] + [err[:, idx].sum() for idx in indices]) / denom)
        errors = torch.stack(estimates).cpu().numpy()
        if not np.isfinite(errors).all():
            raise AssertionError('Nonfinite output error')
        if not all(torch.equal(a, b) for a, b in zip(frozen, (q, k, v, output))):
            raise AssertionError('Modified the BF16 trajectory')
        for (name, fraction, mask, cost), err in zip(entries, errors):
            self.records.append(dict(step=step, phase=phase, branch=branch, layer=layer,
                method=name, coverage=fraction, nmse=float(err[0]), relative_l2=float(np.sqrt(err[0])),
                role_nmse={ROLES[r]: float(e) for r, e in zip(roles, err[1:])},
                shape=[m, n, h, hk, d], ar_tokens=ar, role_counts=np.bincount(kr, minlength=6).tolist(),
                mma=cost['mma_by_dtype'], total_mma=cost['total_mma'], base_mma=cost['base_mma'],
                mma_ratio=cost['total_mma'] / cost['base_mma'], packing=cost['packing'],
                captured_mass=mass_capture(mass, mask, m), **mask_summary(mask, qr, kr, hk)))
        write(self.output / f'progress_request_{self.ordinal:03d}.json',
              dict(complete=False, panels=len(self.records) // len(entries), last_step=step, last_layer=layer))

    def save(self, path):
        expected = len(self.selected_steps) * len(self.selected_layers)
        if len({(r['step'], r['layer']) for r in self.records}) != expected:
            raise AssertionError('Missing intervention panels')
        previous = json.loads((self.reference / path.with_suffix('.repair.json').name).read_text())
        key = lambda r: (r['step'], r['layer'], r['method'], r['coverage'])
        lookup = {key(r): r for r in previous['rows']}
        checked, mismatches = 0, []
        fields = ('nmse', 'role_nmse', 'shape', 'mask_sha256', 'total_mma')
        for row in self.records:
            if row['method'] == 'mass_oracle':
                continue
            old = lookup[key(row)]
            checked += 1
            for field in fields:
                if old[field] != row[field]:
                    mismatches.append(dict(key=key(row), field=field, old=old[field], new=row[field]))
        payload = dict(complete=not mismatches, native_fp4=True, performance_eligible=False,
                       previous_repair_sha256=digest(self.reference / path.with_suffix('.repair.json').name),
                       replay_audit=dict(exact_matches=checked - len({tuple(x['key']) for x in mismatches}), checked=checked, mismatches=mismatches),
                       rows=self.records, mass_diagnostics=self.mass_records)
        write(path.with_suffix('.oracle.json'), payload)
        if mismatches:
            raise AssertionError('Replay differs from frozen study; comparison cannot be silently merged')
        np.savez_compressed(path, counts=self.counts, roles=ROLES, rows_json=np.asarray(json.dumps(self.records)))


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--structure-prior', type=Path, required=True)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--probe-layers', type=int, nargs='+', required=True)
    a, remaining = p.parse_known_args()
    if '--sample' not in remaining or remaining[remaining.index('--steps') + 1] != '35':
        raise ValueError('Frozen protocol requires full 35-step BF16 observation')
    out = Path(remaining[remaining.index('--output') + 1])
    with np.load(a.structure_prior, allow_pickle=False) as z:
        MassOracleObserver.prior = z['density']
    MassOracleObserver.selected_layers = tuple(a.probe_layers)
    MassOracleObserver.reference, MassOracleObserver.output = a.reference_dir, out
    runner.Observer = MassOracleObserver
    sys.argv = [sys.argv[0], *remaining]
    admission = select_shared_device(12288)
    with shared_worker():
        runner.main()
    result = json.loads((out / 'result.json').read_text())
    reference = json.loads((a.reference_dir / 'result.json').read_text())
    index = {(r['case'], r['seed']): r for r in reference['rows']}
    for key in ('model_identity', 'source_sha256', 'dependency_sources', 'cases', 'case_manifest_sha256'):
        if result[key] != reference[key]:
            raise AssertionError('Replay source/input mismatch: ' + key)
    for row in result['rows']:
        if row['sha256'] != index[row['case'], row['seed']]['sha256']:
            raise AssertionError('Full generated output/trajectory hash mismatch')
    result['attention_mass_oracle'] = dict(exact_replay=True, request_hashes_matched=len(result['rows']),
        prior_sha256=digest(a.structure_prior), admission=admission, performance_eligible=False,
        reference_result_sha256=digest(a.reference_dir / 'result.json'))
    write(out / 'result.json', result)


if __name__ == '__main__':
    main()

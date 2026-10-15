"""Byte-level real-group audit and historical native-control reproducibility."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from baselines import sha, write


def nibbles(code):
    return torch.stack((code & 15, code >> 4), -1).reshape(-1, 16)


def audit(root, previous):
    progress = json.loads((root/'progress.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    historical = json.loads(previous.read_text())['modules']
    if not progress['complete']:
        raise ValueError('Finish the frozen cohort first')
    result = dict(groups_removed=0, nonmax_removed=0, same_base_scale_vs_scale_only=0,
                  same_scale_retained_codes_equal=0, different_base_scale_vs_scale_only=0,
                  all_native_control_panels=0, exact_native_control_panels=0,
                  native_control_max_abs_difference=0., accepted_mandatory_blocks=0,
                  source_sha256=sha(__file__), historical_sha256=sha(previous),
                  boundary_groups={}, boundary_elements=0, boundary_max_distance=0.)
    for name, op in progress['modules'].items():
        folder = root/'operators'/name
        methods = torch.load(folder/'patches.pt', weights_only=True)['methods']
        j, s = methods['joint'], methods['scale_only']
        active = j['removed'] >= 0
        same = (j['s0'].view(torch.uint8) == s['s0'].view(torch.uint8)).flatten()
        keep = torch.ones((len(active), 16), dtype=torch.bool)
        rows = torch.arange(len(active))
        keep[rows[active], j['removed'][active]] = False
        eq = ((nibbles(j['c0']) == nibbles(s['c0'])) | ~keep).all(-1)
        bad = (active & same & ~eq).nonzero().flatten()
        if len(bad):
            # A retained original is not silently re-encoded. At near-ties its
            # historical preparation can differ by one code from recalculation.
            asset_spec = protocol['modules'][name]
            raw = torch.load(Path(protocol['spec']['assets'])/asset_spec['asset']['file'], weights_only=True)
            weight = torch.load(Path(protocol['capture'])/asset_spec['weight']['file'], weights_only=True)
            groups = torch.load(folder/'patches.pt', weights_only=True)['groups']
            k = weight.shape[1]
            z = (weight.float()/raw['prescale']).bfloat16().float()/raw['factor']
            zg = z.reshape(weight.shape[0], k//16, 16)[groups['nr'], groups['kg']]
            mismatches = (nibbles(j['c0']) != nibbles(s['c0'])) & keep & (active & same)[:, None]
            br, bc = mismatches.nonzero(as_tuple=True)
            jc, sc = nibbles(j['c0'])[br, bc], nibbles(s['c0'])[br, bc]
            levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
            mid = (levels[jc.long() & 7]+levels[sc.long() & 7])/2
            distance = ((zg[br, bc]/j['s0'].float()[br, 0]).abs()-mid).abs()
            if not torch.equal(s['c0'][bad], methods['rtn']['c0'][bad]):
                raise ValueError('Unexpected difference between freshly encoded candidates')
            if not ((jc.long()-sc.long()).abs() == 1).all() or float(distance.max()) > 2e-6:
                raise ValueError('Unexplained retained-element mismatch beyond a rounding boundary')
            result['boundary_groups'][name] = bad.tolist()
            result['boundary_elements'] += len(br)
            result['boundary_max_distance'] = max(result['boundary_max_distance'], float(distance.max()))
        result['groups_removed'] += int(active.sum())
        result['same_base_scale_vs_scale_only'] += int((active & same).sum())
        result['same_scale_retained_codes_equal'] += int((active & same & eq).sum())
        result['different_base_scale_vs_scale_only'] += int((active & ~same).sum())
        with np.load(folder/'groups.npz') as arr:
            result['nonmax_removed'] += int((active.numpy() & ~arr['joint__removed_is_max']).sum())
        selected = op['chosen']['joint']
        result['accepted_mandatory_blocks'] += op['metrics'][selected].get('mandatory_blocks', 0)
        actual = op['raw_native']['rtn']['three']['panel_nmse']
        expected = historical[name]['results']['original']['panel_nmse']
        if len(actual) != len(expected):
            raise ValueError('Historical panel count mismatch')
        diffs = [abs(x-y) for x, y in zip(actual, expected)]
        result['all_native_control_panels'] += len(diffs)
        result['exact_native_control_panels'] += sum(d == 0 for d in diffs)
        result['native_control_max_abs_difference'] = max(result['native_control_max_abs_difference'], max(diffs))
    result['scope'] = 'Same-scale inlier exceptions are explicitly identified original-asset rounding boundaries; they require a separate sensitivity check.'
    write(root/'report/encoding_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    a = p.parse_args()
    audit(a.root, a.previous)

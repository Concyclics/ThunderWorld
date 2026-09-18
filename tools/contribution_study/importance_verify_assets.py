"""Verify frozen tile-count matching, W1 storage and exact offline reloads on CPU."""
import argparse
import json
from pathlib import Path
import torch
from baselines import sha, write
from thunderworld_fp4.cosmos.calibration import decode


def byte_equal(a, b):
    return a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    catalog = json.loads((a.study / 'candidates/candidates.json').read_text())
    decisions_path = a.study / 'joint_calibration/decisions.json'
    decisions = json.loads(decisions_path.read_text())
    controls = {c['budget']: c for c in catalog['candidates'] if c['name'] in ('b1.5_protected_n', 'b2_protected_n', 'b2.5_ffn_aware')}
    checks = []
    for candidate in catalog['candidates']:
        if candidate['name'] in {c['name'] for c in controls.values()}:
            continue
        plan = json.loads(Path(candidate['config']['budget_plan']).read_text())
        control = controls[candidate['budget']]
        old = json.loads(Path(control['config']['budget_plan']).read_text())
        if plan['attention'] != old['attention'] or plan['design']['calibration_sha256'] != sha(decisions_path):
            raise ValueError('Changed attention or calibration provenance')
        bank = Path(candidate['assets'])
        manifest = json.loads((bank / 'manifest.json').read_text())
        old_bank = Path(control['assets'])
        old_manifest = json.loads((old_bank / 'manifest.json').read_text())
        if sha(bank / 'manifest.json') != candidate['assets_sha256'] or sha(candidate['config']['budget_plan']) != candidate['plan_sha256']:
            raise ValueError('Changed deployment asset/plan')
        kept, full, unchanged = 0, 0, 0
        variant = plan['design']['variant']
        for name, entry in manifest['modules'].items():
            file = bank / entry['file']
            if sha(file) != entry['sha256']:
                raise ValueError('Changed packed weight')
            asset = torch.load(file, map_location='cpu', weights_only=True)
            if 'codes' in asset or 'scales' in asset:
                raise ValueError('Unexpected hidden complete residual plane')
            supports = []
            for key, row in plan['linears'].items():
                if key.split('|', 2)[-1] != name:
                    continue
                before = old['linears'][key]
                if row['a_blocks'] != before['a_blocks'] or len(row['w_blocks']) != len(before['w_blocks']) or row['cross'] != before['cross']:
                    raise ValueError('Changed matched repair counts')
                supports.append(row['w_blocks'])
            if len(supports) != 8 or any(v != supports[0] for v in supports):
                raise ValueError('W1 support changes across phase/CFG')
            if asset['residual_indices'].tolist() != supports[0]:
                raise ValueError('Stored W1 differs from exact execution union')
            item = decisions['modules'][name]
            relative = f'b{candidate["budget"]:g}/{item["index"]:03d}_{variant}.pt'
            source = a.study / 'joint_calibration' / relative
            if sha(source) != item['files'][relative]:
                raise ValueError('Changed selected calibration asset')
            selected = torch.load(source, map_location='cpu', weights_only=True)['asset']
            for key, value in selected.items():
                if isinstance(value, torch.Tensor) and not byte_equal(value, asset[key]):
                    raise ValueError(f'Offline reload changed {key}')
            before = torch.load(old_bank / old_manifest['modules'][name]['file'], map_location='cpu', weights_only=True)
            if variant == 'importance_rtn':
                for key in ('codes_base', 'scales_base', 'factor', 'prescale'):
                    if not byte_equal(asset[key], before[key]):
                        raise ValueError('Importance-only control changed base encoding')
            d = asset['prescale']
            if not ((d >= 1 / 16) & (d <= 16)).all():
                raise ValueError('Prescale outside permitted range')
            decision = item['decisions'][str(candidate['budget'])]
            fallback = variant == 'joint' and decision['selected']['kind'] == 'unchanged_control'
            if fallback:
                unchanged += 1
                if not byte_equal(asset['activation_sensitivity'], before['activation_sensitivity']):
                    raise ValueError('Changed fallback selector')
            else:
                represented = decode(asset['codes_base'], asset['scales_base'], asset['factor'])
                expected = represented.square().sum(0)
                torch.testing.assert_close(asset['activation_sensitivity'], expected, rtol=2e-6, atol=1e-8)
            if decision['selected_loss'] > decision['baseline_loss']:
                raise ValueError('Discarded feasible local control')
            kept += len(supports[0])
            full += asset['score'].numel()
        if kept * 128**2 * 9 // 16 != candidate['cost']['w1_bytes']:
            raise ValueError('Incorrect storage count')
        checks.append(dict(name=candidate['name'], modules=len(manifest['modules']), w1_blocks=kept,
                           w1_density=kept/full, unchanged_local_controls=unchanged, passed=True))
        print(json.dumps(checks[-1]), flush=True)
    write(a.output, dict(complete=True, checks=checks, catalog_sha256=sha(a.study / 'candidates/candidates.json'),
                        scope='CPU asset/support/reload and selector norm verification; runtime full-input Graph and codec tests reported separately.'))


if __name__ == '__main__':
    main()

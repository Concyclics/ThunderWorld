"""CPU audit of prior search coverage; no new generation or quality claim."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from bundle import ROOT
from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.workspace
    import torch
    from thunderworld_fp4.cosmos.calibration import _encode, decode
    torch.set_num_threads(4)
    frozen_path = root/'fp4_baseline_sweep_20260918/manifest.json'
    frozen = json.loads(frozen_path.read_text())
    result = dict(complete=False, source_sha256=sha(__file__), performance_eligible=False,
        new_generation=False, frozen_manifest_sha256=sha(frozen_path), outlier={}, schedule={},
        scope='Search-space and asset-identity audit. Local objectives have module-specific denominators; '
              'their means describe this search, not final quality or attainable improvement.')
    tensor_keys = ('codes_base', 'scales_base', 'codes_residual', 'scales_residual',
                   'factor', 'prescale', 'residual_indices')
    for budget, method in [('2', 'b2_joint'), ('2.5', 'b2.5_ffn_guarded')]:
        folder = root/'paper_evidence_20260918'/('outlier_b'+budget)
        state_path = folder/'progress.json'; state = json.loads(state_path.read_text())
        assert state['complete'] and state['identity']['method'] == method
        bank = Path(frozen['methods'][method]['assets'])
        assert sha(bank/'manifest.json') == frozen['methods'][method]['assets_sha256']
        manifest = json.loads((bank/'manifest.json').read_text())
        assert set(manifest['modules']) == set(state['modules'])
        rows = list(state['modules'].values())
        baseline_mean = statistics.mean(r['results']['rtn']['three']['objective'] for r in rows)
        variants = {}
        for label in ('rtn', 'four_six', 'magnitude', 'quant_error', 'output_joint'):
            values = [r['results'][label] for r in rows]
            mean = statistics.mean(v['three']['objective'] for v in values)
            variants[label] = dict(operators_with_removed_elements=sum(v['removed'] > 0 for v in values),
                removed_elements=sum(v['removed'] for v in values),
                mandatory_blocks=sum(len(v['mandatory_w_blocks']) for v in values),
                mean_operator_objective=mean, relative_mean_objective_reduction_vs_rtn=1-mean/baseline_mean)
        changed = []
        scale_metadata = Counter()
        for name, entry in manifest['modules'].items():
            old_path = bank/entry['file']
            new_entry = state['modules'][name]['exports']['rtn']
            new_path = folder/new_entry['file']
            assert sha(old_path) == entry['sha256'] and sha(new_path) == new_entry['sha256']
            old = torch.load(old_path, map_location='cpu', weights_only=True)
            new = torch.load(new_path, map_location='cpu', weights_only=True)
            keys = [k for k in tensor_keys if not torch.equal(
                old[k].contiguous().reshape(-1).view(torch.uint8),
                new[k].contiguous().reshape(-1).view(torch.uint8))]
            scale_metadata[str(old.get('weight_scale'))] += 1
            if keys:
                changed.append(dict(module=name, changed_tensors=keys,
                    original_scale=old.get('weight_scale'), rtn_scale=new.get('weight_scale')))
        result['outlier'][budget] = dict(progress_sha256=sha(state_path), operators=len(rows),
            operators_with_w1=sum(bool(r['support']) for r in rows),
            stored_w1_blocks=sum(len(r['support']) for r in rows), variants=variants,
            original_scale_metadata=dict(scale_metadata), rtn_changed_vs_frozen=changed)
        print(budget, variants['output_joint'], 'RTN changed operators', len(changed), flush=True)
    for budget in ('1.5', '2', '2.5'):
        folder = root/'joint_schedule_20260917/calibration'/('b'+budget)
        manifest_path = folder/'manifest.json'; m = json.loads(manifest_path.read_text())
        assert m['complete']
        steps = Counter(); groups = []
        for case, digest in m['cases'].items():
            path = folder/(case+'.json'); assert sha(path) == digest
            sample = json.loads(path.read_text()); assert sample['complete']
            groups.append(len(sample['groups']))
            steps.update(r['step'] for r in sample['groups'].values())
        result['schedule'][budget] = dict(manifest_sha256=sha(manifest_path), cases=len(groups),
            groups_per_case=sorted(set(groups)), observation_steps=dict(sorted(steps.items())))
    generator = torch.Generator().manual_seed(20260919)
    z = torch.randn(2048, 16, generator=generator).bfloat16().float()
    c, s = _encode(z, 1.); decoded = decode(c, s, 1.)
    error = (z-decoded).abs().masked_fill(z.abs() == z.abs().amax(-1, keepdim=True), -1.)
    positions = error.argmax(-1, keepdim=True)
    mask = torch.zeros_like(z, dtype=torch.bool).scatter_(1, positions, True)
    changed = z.masked_fill(mask, 0.)
    cc, ss = _encode(changed, 1.); changed_decoded = decode(cc, ss, 1.)
    equal_scale = torch.equal(s.view(torch.uint8), ss.view(torch.uint8))
    equal_retained = torch.equal(decoded[~mask], changed_decoded[~mask])
    assert equal_scale and equal_retained
    result['nonmaximum_removal_check'] = dict(groups=len(z), group_size=16, seed=20260919,
        scale_multiplier=1., scales_bitwise_unchanged=equal_scale,
        retained_decoded_elements_bitwise_unchanged=equal_retained,
        scope='Actual E4M3/E2M1 encoder, BF16 inputs, fixed outer factor. Removing a nonmaximum '
              'does not improve other base codes while the group amax scale rule remains unchanged. '
              'This does not characterize subsequent residual errors or generated video quality.')
    result['complete'] = True
    write(a.output, result)


if __name__ == '__main__':
    main()

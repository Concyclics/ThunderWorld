"""Audit exact source copies and unchanged execution/storage for whole-FFN choices."""
import argparse
import collections
import json
from pathlib import Path

from baselines import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    catalog = json.loads((a.study / 'ffn_guarded/candidates.json').read_text())
    decisions_path = a.study / 'ffn_guarded/decisions.json'
    decisions = json.loads(decisions_path.read_text())
    if not catalog['complete'] or not decisions['complete']:
        raise ValueError('Incomplete calibration')
    candidates = {c['name']: c for c in catalog['candidates']}
    controls = {1.5: 'b1.5_protected_n', 2.: 'b2_protected_n', 2.5: 'b2.5_ffn_aware'}
    checks = []
    for cap, control_name in controls.items():
        selected = candidates[f'b{cap:g}_ffn_guarded']
        sources = [candidates[control_name], candidates[f'b{cap:g}_joint']]
        configs = [selected] + sources
        plans, manifests = [], []
        for candidate in configs:
            plan = Path(candidate['config']['budget_plan'])
            bank = Path(candidate['assets'])
            if sha(plan) != candidate['plan_sha256'] or sha(bank / 'manifest.json') != candidate['assets_sha256']:
                raise ValueError('Changed source provenance')
            plans.append(json.loads(plan.read_text()))
            manifests.append(json.loads((bank / 'manifest.json').read_text()))
        if plans[0]['design']['ffn_calibration_sha256'] != sha(decisions_path):
            raise ValueError('Changed selected FFN decisions')
        if any(plan['attention'] != plans[1]['attention'] for plan in plans):
            raise ValueError('Attention repair changed')
        modules = list(manifests[1]['modules'])
        if any(list(manifest['modules']) != modules for manifest in manifests):
            raise ValueError('Changed module scope')
        copied = collections.Counter()
        improvements = []
        choices = collections.Counter()
        for layer in range(28):
            decision = decisions['modules'][f'{cap:g}|{layer}']
            u, d = decision['selected']
            if decision['selected_loss'] != decision['candidates'][f'{u}{d}']['loss']:
                raise ValueError('Selected FFN loss mismatch')
            if decision['selected_loss'] > min(v['loss'] for v in decision['candidates'].values()):
                raise ValueError('Selected FFN does not minimize calibration objective')
            improvements.append(1 - decision['selected_loss'] / decision['control_loss'])
            choices[f'{u}{d}'] += 1
        for name in modules:
            variant = 0
            if '.mlp_moe_gen.' in name:
                layer = int(name.split('.')[1])
                variant = decisions['modules'][f'{cap:g}|{layer}']['selected'][0 if name.endswith('up_proj') else 1]
            entry = manifests[0]['modules'][name]
            source_entry = manifests[variant + 1]['modules'][name]
            if entry['source_variant'] != variant or entry['sha256'] != source_entry['sha256']:
                raise ValueError('Not an exact selected asset copy')
            for candidate, record in ((selected, entry), (sources[variant], source_entry)):
                if sha(Path(candidate['assets']) / record['file']) != record['sha256']:
                    raise ValueError('Changed weight bytes')
            rows = []
            for key, row in plans[0]['linears'].items():
                if key.split('|', 2)[-1] != name:
                    continue
                old = plans[1]['linears'][key]
                if row != plans[variant + 1]['linears'][key]:
                    raise ValueError('Wrong source repair support')
                if row['a_blocks'] != old['a_blocks'] or len(row['w_blocks']) != len(old['w_blocks']) or row['cross'] != old['cross']:
                    raise ValueError('Changed A/W/cross budget')
                rows.append(row)
            if len(rows) != 8 or any(row != rows[0] for row in rows):
                raise ValueError('Changed phase/CFG support union')
            copied['joint' if variant else 'original'] += 1
        for field in ('w1_bytes', 'tensor_bytes'):
            if any(candidate['cost'][field] != selected['cost'][field] for candidate in sources):
                raise ValueError('Changed matched storage')
        ledgers = [json.loads((Path(c['config']['budget_plan']).parent / 'compute.json').read_text()) for c in configs]
        if any(ledger['total_mma'] != ledgers[0]['total_mma'] or ledger['base_mma'] != ledgers[0]['base_mma'] for ledger in ledgers):
            raise ValueError('Changed matched MMA')
        checks.append(dict(name=selected['name'], passed=True, modules=len(modules), copied=dict(copied),
                           ffn_choices=dict(choices), mean_full_ffn_objective_improvement=sum(improvements)/len(improvements),
                           tensor_bytes=selected['cost']['tensor_bytes'], w1_bytes=selected['cost']['w1_bytes']))
    write(a.output, dict(complete=True, checks=checks, catalog_sha256=sha(a.study / 'ffn_guarded/candidates.json'),
                         decisions_sha256=sha(decisions_path), source_sha256=sha(__file__),
                         scope='504 assets are exact copies of independently audited original/joint assets. Same A/W quotas, phase union, attention, tensor storage and issued MMA.'))


if __name__ == '__main__':
    main()

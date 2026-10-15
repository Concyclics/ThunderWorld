"""Native local + paired video evidence with exact support/storage accounting."""
import argparse
import gzip
import json
from pathlib import Path

from baselines import sha, write

FIELDS = ('lpips', 'ssim_mean', 'relative_l2', 'psnr_db')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(4)
    progress = json.loads((a.study/'all_layers/progress.json').read_text())
    root = a.study/'development'; manifest = json.loads((root/'manifest.json').read_text())
    mp = json.loads((root/'metrics/progress.json').read_text())
    if not progress['complete_model'] or not progress['complete'] or not mp['complete'] or mp['identity']['manifest_sha256'] != sha(root/'manifest.json'):
        raise ValueError('Complete native calibration and paired videos required')
    protocol = json.loads((a.study/'protocol.json').read_text())
    control = protocol['weight']['control']; source = Path(manifest['methods'][control]['assets'])
    source_manifest = json.loads((source/'manifest.json').read_text())
    local_rows = [r for r in progress['modules'].values() if not r['untouched']]
    local = {}
    for label in ('original', 'scale_only', 'removal_only', 'joint'):
        local[label] = dict(operators=len(local_rows), mean_objective=float(np.mean([r['results'][label]['objective'] for r in local_rows])),
            mean_four_diagnostic=float(np.mean([r['results'][label]['four_diagnostic']['objective'] for r in local_rows])))
        local[label]['relative_reduction'] = 1-local[label]['mean_objective']/local['original']['mean_objective']
    storage = {}
    for name, spec in manifest['methods'].items():
        if name in ('bf16', 'fp8'):
            continue
        bank = Path(spec['assets']); catalog = json.loads((bank/'manifest.json').read_text())
        if sha(bank/'manifest.json') != spec['assets_sha256']:
            raise ValueError('Changed assets')
        totals = dict(tensor_bytes=0, file_bytes=0, w1_bytes=0, w1_blocks=0, mandatory_blocks=0, changed_operators=0)
        for module, entry in catalog['modules'].items():
            file = bank/entry['file']; old_entry = source_manifest['modules'][module]; oldfile = source/old_entry['file']
            if sha(file) != entry['sha256'] or sha(oldfile) != old_entry['sha256']:
                raise ValueError('Changed asset payload')
            raw = torch.load(file, map_location='cpu', weights_only=True); old = torch.load(oldfile, map_location='cpu', weights_only=True)
            if 'codes' in raw or 'scales' in raw:
                raise ValueError('Hidden dense residual copy')
            for key in ('prescale', 'factor', 'activation_sensitivity', 'residual_indices', 'score', 'bias'):
                left, right = raw.get(key), old.get(key)
                if isinstance(left, torch.Tensor):
                    if not torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)):
                        raise ValueError('Changed fixed tensor: '+key)
                elif left != right:
                    raise ValueError('Changed fixed metadata: '+key)
            for key in ('activation_scale', 'selector'):
                if raw.get(key) != old.get(key):
                    raise ValueError('Changed activation semantics')
            ids = raw['residual_indices'].tolist()
            if not set(raw.get('mandatory_w_blocks', ())) <= set(ids):
                raise ValueError('Missing mandatory W1')
            nb, kb = raw['score'].shape
            active = torch.zeros(nb*kb, dtype=torch.bool); active[ids] = True
            for key in ('codes_base', 'scales_base'):
                width = raw[key].shape[-1]//kb
                def tiles(v):
                    return v.view(torch.uint8).reshape(nb, 128, kb, width).permute(0, 2, 1, 3).reshape(nb*kb, 128, width)
                if not torch.equal(tiles(raw[key])[~active], tiles(old[key])[~active]):
                    raise ValueError('Changed W0 outside paid W1 support')
            totals['tensor_bytes'] += sum(v.numel()*v.element_size() for v in raw.values() if isinstance(v, torch.Tensor))
            totals['file_bytes'] += file.stat().st_size
            totals['w1_bytes'] += raw['codes_residual'].numel()+raw['scales_residual'].numel()
            totals['w1_blocks'] += len(ids)
            totals['mandatory_blocks'] += len(raw.get('mandatory_w_blocks', ()))
            totals['changed_operators'] += sha(file) != sha(oldfile)
        storage[name] = totals
    if len({r['tensor_bytes'] for r in storage.values()}) != 1 or len({r['w1_bytes'] for r in storage.values()}) != 1:
        raise ValueError('Storage was not matched')
    costs = {}; summaries = {}; paired = {}; raw_metrics = []; by_seed = {}
    for name in manifest['methods']:
        if name == 'bf16':
            continue
        for dataset in ('rbench', 'paibench'):
            values = {}
            for entry in manifest['cases']:
                if entry['benchmark'] != dataset:
                    continue
                case = entry['case']['id']; path = root/'metrics'/name/(case+'.json')
                metric = json.loads(path.read_text())
                if not metric['complete'] or metric['provenance']['identity'] != mp['identity']:
                    raise ValueError('Changed metrics')
                for side, method in (('candidate', name), ('reference', 'bf16')):
                    receipt = root/'generation'/method/case/'result.json'
                    if metric['provenance'][side+'_sha256'] != sha(receipt):
                        raise ValueError('Changed generation')
                raw_metrics.append(metric)
                for row in metric['rows']:
                    values[case, row['seed']] = row['metrics']['video']
            by_seed[dataset, name] = values
            summaries.setdefault(dataset, {})[name] = {field: float(np.mean([r[field] for r in values.values()])) for field in FIELDS}
        if name != 'fp8':
            ratios = []
            for entry in manifest['cases']:
                case = entry['case']['id']; folder = root/'generation'/name/case
                old = json.loads((root/'generation'/control/case/'compute.json').read_text())
                new = json.loads((folder/'compute.json').read_text()); receipt = json.loads((folder/'result.json').read_text())
                for field in ('base_mma', 'total_mma', 'mma_by_dtype', 'non_mma_work', 'packing_traffic', 'w1_support_union_bytes', 'w1_support_union_blocks'):
                    if new[field] != old[field]:
                        raise ValueError('Unmatched request work: '+field)
                if not new['feasible'] or new['ratio'] > protocol['budget']:
                    raise ValueError('Over-budget request')
                if any(r['compute_sha256'] != sha(folder/'compute.json') or r['debug'] != 'off' or r['performance_eligible'] for r in receipt['rows']):
                    raise ValueError('Changed cost/measurement conditions')
                ratios.append(new['ratio'])
            costs[name] = dict(request_bound_range=[min(ratios), max(ratios)], exact_work_matched=True)
    candidates = [name for name in manifest['methods'] if '_group_' in name]
    gates = {}
    for name in candidates:
        gates[name] = {}
        for reference in (control, control+'_group_scale_only', 'fp8'):
            if reference == name:
                continue
            for dataset in ('rbench', 'paibench'):
                new, old = by_seed[dataset, name], by_seed[dataset, reference]
                if new.keys() != old.keys():
                    raise ValueError('Unpaired scenes/seeds')
                rows = {}
                for field in FIELDS:
                    diffs = {key: new[key][field]-old[key][field] for key in new}
                    scenes = sorted({key[0] for key in diffs})
                    scene_values = np.asarray([np.mean([v for (case, _), v in diffs.items() if case == scene]) for scene in scenes])
                    rng = np.random.default_rng(20260919)
                    boot = scene_values[rng.integers(len(scenes), size=(20000, len(scenes)))].mean(1)
                    lower = field in ('lpips', 'relative_l2')
                    worst = (max if lower else min)(diffs, key=diffs.get)
                    rows[field] = dict(mean_delta=float(scene_values.mean()), descriptive_scene_ci95=np.quantile(boot, [.025, .975]).tolist(),
                        worst_case=worst[0], worst_seed=worst[1], worst_delta=diffs[worst],
                        degraded_seeds=sum(v>0 if lower else v<0 for v in diffs.values()), videos=len(diffs))
                paired[f'{dataset}|{name}|{reference}'] = rows
                if reference == control:
                    gates[name][dataset] = dict(mean_guard=rows['lpips']['mean_delta']<0 and rows['ssim_mean']['mean_delta']>=0 and rows['relative_l2']['mean_delta']<=0)
    result = dict(complete=True, protocol=protocol, manifest_sha256=sha(root/'manifest.json'), source_sha256=sha(__file__),
        local=local, storage=storage, costs=costs, summaries=summaries, pairs=paired, gates=gates,
        raw_metrics=raw_metrics, performance_eligible=False, statistical_quality_qualified=False,
        scope='Exploratory development verification only. Operator means include 63 W1-supported modules, not a model quality metric. '
              'Paired CIs resample scenes with two seeds averaged; only 9 RBench and 7 PAI scenes. No held-out/final or speed claim.')
    a.output.mkdir(parents=True, exist_ok=True); write(a.output/'quality.json', result)
    for name, path in (('calibration', a.study/'all_layers/progress.json'), ('manifest', root/'manifest.json'), ('protocol', a.study/'protocol.json')):
        (a.output/(name+'.json.gz')).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    print(json.dumps(dict(summaries=summaries, gates=gates, storage=storage, costs=costs), indent=2))


if __name__ == '__main__':
    main()

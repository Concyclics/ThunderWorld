"""Paired development metrics for a complete bounded selector experiment."""
import argparse
import hashlib
import json
from pathlib import Path

from bundle import ROOT
from baselines import sha, write


def load_signals(path, row):
    import numpy as np
    if sha(path) != row['outputs_sha256']:
        raise ValueError('Output archive changed')
    with np.load(path) as archive:
        signals = dict(archive)
    if {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in signals.items()} != row['sha256']:
        raise ValueError('Signal bytes changed')
    return signals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--reference-study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    import lpips
    from thunderworld_fp4.cosmos.quality import compare
    torch.set_num_threads(8)
    torch.hub.set_dir('/tmp/thunderworld-lpips-cache')
    candidate_file = a.candidate/'result.json'
    candidate = json.loads(candidate_file.read_text())
    if not candidate['complete'] or candidate['identity']['operation'] != 'quality':
        raise ValueError('Completed full-generation quality experiment required')
    manifest = json.loads((a.reference_study/'manifest.json').read_text())
    method = candidate['identity']['method']
    metric = lpips.LPIPS(net='alex', version='0.1').eval().cpu()
    identity = dict(candidate_sha256=sha(candidate_file), source_sha256=sha(__file__),
                    quality_sha256=sha(ROOT/'packages/fp4-repair/src/thunderworld_fp4/cosmos/quality.py'),
                    reference_manifest_sha256=sha(a.reference_study/'manifest.json'))
    result = dict(identity=identity, complete=False, scenes={}, performance_eligible=False,
                  scope='Exploratory paired development comparison; seeds averaged within scene. No held-out or final-quality claim.')
    if a.output.exists():
        previous = json.loads(a.output.read_text())
        if previous['identity'] != identity:
            raise ValueError('Changed metric inputs')
        if previous['complete']:
            return
        result = previous
    fields = ('lpips', 'ssim_mean', 'relative_l2', 'psnr_db')
    for entry in manifest['cases']:
        case = entry['case']['id']
        if case not in candidate['cases']:
            raise ValueError('Missing frozen development scene')
        if case in result['scenes']:
            continue
        rows = candidate['cases'][case]['rows']
        if {r['seed'] for r in rows} != set(entry['seeds']) or len(rows) != len(entry['seeds']):
            raise ValueError('Missing or repeated seeds')
        reference_folder = a.reference_study/'generation/bf16'/case
        receipt = json.loads((reference_folder/'result.json').read_text())
        if receipt['model_identity'] != candidate['prepared']['model_identity']:
            raise ValueError('Model revision changed')
        old = json.loads((a.reference_study/'metrics'/method/(case+'.json')).read_text())
        if old['provenance']['identity']['quality_sha256'] != identity['quality_sha256']:
            raise ValueError('Metric definition changed')
        if old['provenance']['reference_sha256'] != sha(reference_folder/'result.json'):
            raise ValueError('Reference receipt changed')
        paired = []
        for row in rows:
            seed = row['seed']
            gold = next(r for r in receipt['rows'] if r['seed'] == seed)
            if row['steps'] != 35 or gold['steps'] != 35:
                raise ValueError('Changed generation steps')
            values = compare(load_signals(a.candidate/row['outputs'], row),
                             load_signals(reference_folder/gold['outputs'], gold), metric)['video']
            baseline = next(r['metrics']['video'] for r in old['rows'] if r['seed'] == seed)
            paired.append(dict(seed=seed, candidate=values, control=baseline,
                               delta={f: values[f]-baseline[f] for f in fields}))
        result['scenes'][case] = dict(benchmark=entry['benchmark'], category=entry['category'], rows=paired,
            means={side: {f: float(np.mean([r[side][f] for r in paired])) for f in fields}
                   for side in ('candidate', 'control', 'delta')})
        write(a.output, result)
        print(json.dumps(dict(case=case, completed=len(result['scenes']))), flush=True)
    result['summary'] = {}
    for benchmark in ('rbench', 'paibench'):
        scenes = [s for s in result['scenes'].values() if s['benchmark'] == benchmark]
        deltas = np.array([s['means']['delta']['lpips'] for s in scenes])
        rng = np.random.default_rng(181918)
        boot = deltas[rng.integers(0, len(deltas), (20000, len(deltas)))].mean(1)
        means = {side: {f: float(np.mean([s['means'][side][f] for s in scenes])) for f in fields}
                 for side in ('candidate', 'control', 'delta')}
        means['lpips_paired_scene_ci95'] = np.quantile(boot, [.025, .975]).tolist()
        means['mean_guard'] = means['delta']['lpips'] < 0 and means['delta']['ssim_mean'] >= 0 and means['delta']['relative_l2'] <= 0
        means['worst_seed_lpips_delta'] = max(r['delta']['lpips'] for s in scenes for r in s['rows'])
        means['degraded_seeds'] = sum(r['delta']['lpips'] > 0 for s in scenes for r in s['rows'])
        means['scenes'] = len(scenes)
        result['summary'][benchmark] = means
    result['complete'] = True
    result['recommend_candidate'] = all(s['mean_guard'] for s in result['summary'].values())
    write(a.output, result)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()

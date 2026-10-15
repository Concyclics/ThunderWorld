"""Bind cross-family arithmetic to previously observed matching request shapes.

This is a static issued-tile work ledger at the declared safe launch, not a new
hardware-counter measurement and not a latency estimate. Existing budgeted rows
retain their actual warmup ledger; BF16/FP8/AWQ are not assigned fake FP4 ratios.
"""
import argparse
import collections
import json
from pathlib import Path
from baselines import sha, write


def compile_anchor(config, geometry):
    from thunderworld_fp4.cosmos.settings import RunConfig
    from thunderworld_fp4.cosmos.budget import linear_cost, attention_cost
    cfg = RunConfig.from_dict(config)
    if cfg.precision != 'fp4' or cfg.attention_backend != 'fp4' or cfg.budget_plan:
        raise ValueError('Only full, fixed native anchors are derived here')
    totals = collections.Counter()
    families = collections.Counter()
    packing = collections.Counter()
    non_mma = collections.Counter()
    weights = {}
    for row in geometry['rows']:
        if row['kind'] == 'linear':
            m, n, k = row['shape']
            spec = cfg.ffn if '.mlp_moe_gen.' in row['name'] else cfg.projection
            na, nw = spec.counts(n, k)
            if nw not in (0, n//128*(k//128)):
                raise ValueError('A partial W support needs its actual deployment ledger')
            cost = linear_cost(m, n, k, bm=128, a_blocks=na, w_blocks=list(range(nw)), cross=spec.cross)
            family = 'ffn' if '.mlp_moe_gen.' in row['name'] else 'projection'
            families[family] += cost['mma_by_dtype']['fp4']['cross']*row['calls']
            weights[row['name']] = dict(base_codes_scales=n*k*9//16, residual_codes_scales=nw*128*128*9//16)
        else:
            m, n, h, hk, _ = row['shape']
            ac = cfg.attention.default
            if cfg.attention.rules:
                raise ValueError('Unexpected nonuniform anchor policy')
            cost = attention_cost(m, n, h, hk, bm=ac.query_tile or (16 if m <= 128 else 32),
                qk_counts=ac.head_counts(h, n), pv_counts=ac.pv_counts(h, n),
                qk_terms=ac.terms, pv_terms=ac.pv_terms or ac.terms)
            for family in ('qk', 'pv'):
                families[family] += cost['mma_by_dtype']['fp4'][family+'_cross']*row['calls']
        calls = row['calls']
        totals['base_mma'] += cost['base_mma']*calls
        totals['total_mma'] += cost['total_mma']*calls
        for dtype, terms in cost['mma_by_dtype'].items():
            totals[dtype] += sum(terms.values())*calls
        for key, value in cost['packing'].items():
            if isinstance(value, int) and 'allocation' not in key:
                packing[key] += value*calls
        for key, value in cost['non_mma'].items():
            if isinstance(value, int):
                non_mma[key] += value*calls
    if totals['base_mma'] != geometry['base_mma']:
        raise ValueError('Changed C0 denominator')
    return dict(**totals, ratio=totals['total_mma']/totals['base_mma'], cross_by_family=dict(families),
        packing_traffic_model=dict(packing), non_mma_work_model=dict(non_mma),
        generated_weight_planes={key: sum(row[key] for row in weights.values()) for key in ('base_codes_scales', 'residual_codes_scales')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.study/'manifest.json').read_text())
    old = json.loads((a.geometry/'manifest.json').read_text())
    if manifest['cases'] != old['cases'] or manifest['model'] != old['model']:
        raise ValueError('Different shape inventory cohort/model')
    rows = []
    for entry in manifest['cases']:
        case = entry['case']['id']
        folder = a.geometry/'generation/b2_joint'/case
        geometry = json.loads((folder/'compute.json').read_text())
        receipt = json.loads((folder/'result.json').read_text())
        if not receipt['complete'] or receipt['identity']['case'] != entry['case'] or any(r['compute_sha256'] != sha(folder/'compute.json') for r in receipt['rows']):
            raise ValueError('Unverified warmup shape source')
        for method, spec in manifest['methods'].items():
            if spec['config']['precision'] != 'fp4' or spec['config']['budget_plan']:
                continue
            cost = compile_anchor(spec['config'], geometry)
            rows.append(dict(method=method, case=case, benchmark=entry['benchmark'], cost=cost,
                geometry_sha256=sha(folder/'compute.json')))
    result = dict(complete=True, rows=rows, manifest_sha256=sha(a.study/'manifest.json'),
        source_sha256=sha(__file__), performance_eligible=False,
        scope='Derived issued-tile work at safe launches (linear M128, attention M32), using matched request warmup geometry. '
              'No tuning table is passed to these anchor runs. Packing figures are traffic models; weight planes exclude indices/prescale and AR weights. '
              'No claim of observed Tensor Core saturation, request speed or total model VRAM.')
    a.output.mkdir(parents=True, exist_ok=True)
    write(a.output/'costs.json', result)
    lines = ['# Cross-family compute and storage', '', result['scope'], '',
        '| Method | MMA/C0 range | W0+W1 planes MiB |', '|---|---:|---:|']
    for method in dict.fromkeys(r['method'] for r in rows):
        costs = [r['cost'] for r in rows if r['method'] == method]
        ratios = [r['ratio'] for r in costs]
        sizes = {sum(r['generated_weight_planes'].values()) for r in costs}
        if len(sizes) != 1:
            raise ValueError('Weight planes changed across requests')
        lines.append(f'| {method} | {min(ratios):.6f}–{max(ratios):.6f} | {next(iter(sizes))/2**20:.3f} |')
    (a.output/'COSTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()

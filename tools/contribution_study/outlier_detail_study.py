"""Prepare/run a paired outlier-encoding experiment on a frozen 2x W1 bank.

No new kernel, online selector or runtime policy. GPU times are never performance
results. Historical teacher panels are split for this experiment only; neither
split is an independent final test of assets previously fitted on these scenes.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from baselines import sha, write

FIT = (0, 2, 5, 7, 9, 11, 12, 14)
REVIEW = tuple(i for i in range(16) if i not in FIT)
HERE = Path(__file__).resolve().parent


def sources():
    return {n: sha(HERE/n) for n in ('outlier_detail.py', 'outlier_detail_study.py',
                                    'joint_group_verify.py', 'importance_joint_calibrate.py')}


def prepare(a):
    catalog = json.loads(a.methods.read_text())
    spec = catalog['methods'][a.method]
    bank = Path(spec['assets'])
    assets = json.loads((bank/'manifest.json').read_text())
    capture = json.loads((a.capture/'manifest.json').read_text())
    cohort = json.loads(a.cohort.read_text())
    plan = json.loads(Path(spec['config']['budget_plan']).read_text())
    if not assets['complete'] or not capture['complete'] or assets['model_identity'] != capture['model_identity']:
        raise ValueError('Complete model-matched teacher panels/assets required')
    if sha(bank/'manifest.json') != spec['assets_sha256'] or sha(spec['config']['budget_plan']) != spec['plan_sha256']:
        raise ValueError('Changed fixed control')
    if capture['identity']['cohort_sha256'] != sha(a.cohort) or catalog['model'] != cohort['model']:
        raise ValueError('Mismatched cohort/model')
    modules = {}
    for name in assets['modules']:
        settings = [r for key, r in plan['linears'].items() if key.split('|', 2)[-1] == name]
        if len(settings) != 8 or any(r != settings[0] for r in settings) or settings[0]['cross']:
            raise ValueError('Expected stage-invariant three-term reference policy')
        ids = sorted(settings[0]['w_blocks'])
        if not ids:
            continue
        key = lambda block: hashlib.sha256(f'20260924|{name}|{block}'.encode()).hexdigest()
        sampled = sorted(sorted(ids, key=key)[:4])
        rows = sorted((r for r in capture['rows'] if r['name'] == name), key=lambda r: r['scene_index'])
        if [r['scene_index'] for r in rows] != list(range(16)):
            raise ValueError('Sixteen complete-M panels required')
        for indices in (FIT, REVIEW):
            if {(rows[i]['phase'], rows[i]['branch']) for i in indices} != {(p, b) for p in range(4) for b in range(2)}:
                raise ValueError('Fit/review must each cover all phases and CFG branches')
        modules[name] = dict(support=ids, sampled_blocks=sampled, a_blocks=settings[0]['a_blocks'],
                             panel_rows=rows, asset=assets['modules'][name], weight=capture['weights'][name])
    protocol = dict(schema=1, method=a.method, spec=spec, sources=sources(),
                    methods_sha256=sha(a.methods), assets_sha256=sha(bank/'manifest.json'),
                    capture=str(a.capture.resolve()), capture_sha256=sha(a.capture/'manifest.json'),
                    cohort_sha256=sha(a.cohort), model_identity=assets['model_identity'],
                    fit=list(FIT), review=list(REVIEW), modules=modules,
                    cases=[dict(scene_index=i, benchmark=c['benchmark'], category=c['category'],
                                source_id=c['source_id'], seeds=c['seeds']) for i, c in enumerate(cohort['cases']['calibration'])],
                    search=dict(base_scales=[.75, .875, 1., 1.5], residual_scales=[.875, 1., 1.125],
                                positions='none + all 16', blocks_per_operator=4, block_sampling='sha256 seed20260924',
                                objective='actual-A K16 covariance proposal; native fit mean(dataset means)+0.25*worst panel',
                                native_gate='scale:min(rtn,proposal); max:min(rtn,scale,proposal); joint:min(rtn,scale,max,proposal)',
                                selection_arithmetic='three-term only; four-term is a separate diagnostic'),
                    scope='Edge I2V existing paid attention projection W1 blocks; FFN and QK/PV unchanged. '
                          'All original assets used these historical development scenes. Review is held out from this new search only.',
                    performance_eligible=False, independent_final_test=False)
    path = a.output/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Refuse to overwrite a different frozen experiment')
    else:
        write(path, protocol)
    print(json.dumps(dict(protocol=str(path), modules=len(modules), blocks=sum(len(m['sampled_blocks']) for m in modules.values()),
                          groups=1024*sum(len(m['sampled_blocks']) for m in modules.values()))), flush=True)


def checked_load(path, expected):
    import torch
    if sha(path) != expected:
        raise ValueError(f'Changed input: {path.name}')
    return torch.load(path, map_location='cpu', weights_only=True)


def tensor_bytes(value):
    import torch
    return sum(v.numel()*v.element_size() for v in value.values() if isinstance(v, torch.Tensor))


def run(a):
    protocol = json.loads((a.output/'protocol.json').read_text())
    if protocol['sources'] != sources():
        raise ValueError('Source changed since protocol freeze')
    path = a.output/'progress.json'
    state = dict(complete=False, protocol_sha256=sha(a.output/'protocol.json'), modules={}, performance_eligible=False)
    if path.exists():
        state = json.loads(path.read_text())
        if state['protocol_sha256'] != sha(a.output/'protocol.json'):
            raise ValueError('Changed protocol')
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        state['resource'] = select_shared_device(12288)
        state.pop('paused', None)
        write(path, state)
        import numpy as np
        import torch
        from outlier_detail import METHODS, search, group_indices, patch_asset, diagnostics
        from joint_group_verify import activation_covariance
        from importance_joint_calibrate import objective
        from thunderworld_fp4.cosmos.assets import compact, expand
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec, RunConfig
        from thunderworld_fp4.cosmos.budget import linear_cost
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        benchmarks = [c['benchmark'] for c in protocol['cases']]
        fit, review = protocol['fit'], protocol['review']
        spec = protocol['spec']
        bank = Path(spec['assets']); capture = Path(protocol['capture'])
        done = 0
        try:
            with torch.inference_mode():
                for ordinal, (name, item) in enumerate(protocol['modules'].items()):
                    if name in state['modules']:
                        for artifact in state['modules'][name]['artifacts']:
                            if sha(a.output/artifact['file']) != artifact['sha256']:
                                raise ValueError('Changed completed artifact')
                        continue
                    print(f'[{ordinal+1}/{len(protocol["modules"])}] {name}', flush=True)
                    ids, na = item['support'], item['a_blocks']
                    raw = checked_load(bank/item['asset']['file'], item['asset']['sha256'])
                    if sorted(raw['residual_indices'].tolist()) != ids:
                        raise ValueError('Fixed support mismatch')
                    old = expand(raw, ids)
                    if old.get('mandatory_w_blocks') or tuple(old.get('weight_scale', (1., 1.))) != (1., 1.):
                        raise ValueError('Expected unsplit RTN-residual control')
                    w = checked_load(capture/item['weight']['file'], item['weight']['sha256']).cuda()
                    n, k = w.shape
                    d, g = old['prescale'].cuda(), old['factor'].cuda()
                    z = (w.float()/d).bfloat16().float()/g
                    panels = []
                    for row in item['panel_rows']:
                        panel = checked_load(capture/row['file'], row['sha256'])
                        panels.append((panel['x'].cuda(), panel['y'].cuda().float()))
                    m = len(panels[0][0])
                    if any(x.shape != (m, k) for x, y in panels):
                        raise ValueError('Unexpected panel shape')
                    def evaluate(asset, cross=False):
                        ls = LinearSpec(0, 0, bool(cross and na), activation_blocks=na, weight_support=tuple(ids))
                        controller = SimpleNamespace(warming=False, config=RunConfig.from_dict(spec['config']), linear_spec=lambda *_: ls)
                        op = RepairLinear(SimpleNamespace(in_features=k, out_features=n), controller, name, ls, asset)
                        op.prepare_shape(m, ls)
                        errors = [float((op(x).float()-y).square().sum()/y.square().sum().clamp_min(1e-20)) for x, y in panels]
                        if not torch.isfinite(torch.tensor(errors)).all():
                            raise ValueError('Nonfinite native output')
                        return dict(panel_nmse=errors,
                                    fit_objective=objective([errors[i] for i in fit], [benchmarks[i] for i in fit]),
                                    review_objective=objective([errors[i] for i in review], [benchmarks[i] for i in review]),
                                    fit_mean=sum(errors[i] for i in fit)/len(fit),
                                    review_mean=sum(errors[i] for i in review)/len(review))
                    native = {'rtn': dict(three=evaluate(old), four=evaluate(old, True))}
                    cov = activation_covariance([panels[i] for i in fit], old, na, [benchmarks[i] for i in fit])
                    blocks = torch.tensor(item['sampled_blocks'], device=w.device)
                    nr, kg = group_indices(blocks, k)
                    cz = z.reshape(n, k//16, 16)[nr, kg]
                    codes, scales = old['codes'].cuda(), old['scales'].cuda()
                    base = {f'c{i}': codes[i].reshape(n, k//16, 8)[nr, kg] for i in range(2)}
                    base.update({f's{i}': scales[i].view(torch.uint8)[nr, kg, None].view(scales.dtype) for i in range(2)})
                    base.update(removed=torch.full((len(cz),), -1, device=w.device, dtype=torch.long),
                                base_scale=torch.ones(len(cz), device=w.device), residual_scale=torch.ones(len(cz), device=w.device))
                    proposed = search(cz, base['c0'], base['s0'], base['c1'], base['s1'], cov[kg])
                    all_proposals = dict(rtn=base, **proposed)
                    folder = a.output/'operators'/name
                    folder.mkdir(parents=True, exist_ok=True)
                    patches = {label: {key: value.cpu() for key, value in proposal.items()} for label, proposal in all_proposals.items()}
                    patch_path = folder/'patches.pt'
                    torch.save(dict(groups=dict(nr=nr.cpu(), kg=kg.cpu(), blocks=blocks.cpu()), methods=patches), patch_path)
                    reload = torch.load(patch_path, weights_only=True)
                    for label, patch in patches.items():
                        for key, value in patch.items():
                            if not torch.equal(value.view(torch.uint8), reload['methods'][label][key].view(torch.uint8)):
                                raise ValueError('Patch reload changed encoded representation')
                    arrays = {}
                    metrics = {}
                    for label, proposal in all_proposals.items():
                        values = diagnostics(cz, proposal, base, proposed['scale_only'])
                        values = {key: value.cpu().numpy() for key, value in values.items()}
                        arrays.update({f'{label}__{key}': value for key, value in values.items()})
                        active = values['removed'] >= 0
                        energy = float(values['energy'].astype(np.float64).sum())
                        nmse = lambda key: float(values[key].astype(np.float64).sum()/max(energy, 1e-20))
                        metrics[label] = dict(groups=len(cz), removed_groups=int(active.sum()),
                            removed_nonmax=int((active & ~values['removed_is_max']).sum()),
                            base_nmse=nmse('base_sse'), dual_nmse=nmse('dual_sse'),
                            inlier_sse=float(values['inlier_sse'].astype(np.float64).sum()),
                            inlier_rtn_sse=float(values['inlier_rtn_sse'].astype(np.float64).sum()),
                            inlier_scale_sse=float(values['inlier_scale_sse'].astype(np.float64).sum()))
                        if label == 'rtn':
                            continue
                        changed = patch_asset(old, reload['methods'][label], reload['groups']['nr'], reload['groups']['kg'], reload['groups']['blocks'])
                        packed = compact(changed, ids)
                        if tensor_bytes(packed) != tensor_bytes(raw) or not torch.equal(packed['residual_indices'], raw['residual_indices']):
                            raise ValueError('Unmatched support/storage')
                        restored = expand(packed, ids)
                        for key in ('codes', 'scales'):
                            if not torch.equal(restored[key].view(torch.uint8), changed[key].view(torch.uint8)):
                                raise ValueError('Compact reload changed represented tensor')
                        metrics[label]['mandatory_blocks'] = len(changed['mandatory_w_blocks'])
                        native[label] = dict(three=evaluate(restored), four=evaluate(restored, True))
                        del changed, packed, restored
                    np.savez_compressed(folder/'groups.npz', **arrays)
                    # Gate on FIT only. Review values never influence accepted policy.
                    chosen = {'rtn': 'rtn', 'max_fixed': 'max_fixed'}
                    parents = dict(scale_only=['rtn'], max_scale=['rtn', 'scale_only'],
                                   joint=['rtn', 'scale_only', 'max_scale'], removal_only=['rtn'])
                    for label in ('scale_only', 'max_scale', 'joint', 'removal_only'):
                        candidates = list(dict.fromkeys([chosen[p] for p in parents[label]]+[label]))
                        chosen[label] = min(candidates, key=lambda x: native[x]['three']['fit_objective'])
                    entry = dict(name=name, shape=[m, n, k], sampled_blocks=item['sampled_blocks'],
                        support_blocks=len(ids), a_blocks=na, metrics=metrics, raw_native=native, chosen=chosen,
                        accepted_native={label: native[selected] for label, selected in chosen.items()},
                        same_tensor_bytes=tensor_bytes(raw),
                        cost_three=linear_cost(m, n, k, a_blocks=na, w_blocks=ids, cross=False),
                        cost_four=linear_cost(m, n, k, a_blocks=na, w_blocks=ids, cross=True),
                        cost_scope='All labels execute identical work per arithmetic setting; four-term is not the 2x request policy.',
                        artifacts=[dict(file=str(p.relative_to(a.output)), sha256=sha(p)) for p in (patch_path, folder/'groups.npz')])
                    write(folder/'result.json', entry)
                    entry['artifacts'].append(dict(file=str((folder/'result.json').relative_to(a.output)), sha256=sha(folder/'result.json')))
                    state['modules'][name] = entry
                    write(path, state)
                    print(json.dumps(dict(done=len(state['modules']), total=len(protocol['modules']), name=name,
                                          accepted_joint=chosen['joint'],
                                          review_rtn=native['rtn']['three']['review_mean'],
                                          review_joint=entry['accepted_native']['joint']['three']['review_mean'])), flush=True)
                    del w, d, g, z, panels, panel, cov, cz, nr, kg, blocks, codes, scales, base, proposed, all_proposals, proposal
                    gc.collect()
                    torch.cuda.empty_cache()
                    done += 1
                    if a.max_modules and done >= a.max_modules:
                        break
            state['complete'] = len(state['modules']) == len(protocol['modules'])
            write(path, state)
        except (RuntimeError, MemoryError) as exc:
            state['paused'] = str(exc)
            write(path, state)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--methods', type=Path, required=True)
    prep.add_argument('--method', default='b2_joint')
    prep.add_argument('--capture', type=Path, required=True)
    prep.add_argument('--cohort', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--output', type=Path, required=True)
    runner.add_argument('--max-modules', type=int, default=0)
    a = p.parse_args()
    (prepare if a.operation == 'prepare' else run)(a)


if __name__ == '__main__':
    main()

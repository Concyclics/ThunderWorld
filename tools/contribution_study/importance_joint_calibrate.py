"""Native fixed-budget W0/W1/activation-importance search, without training.

All candidates retain per-operator A quotas and W tile counts. W support positions,
channel compensation, actual FP4 base/residual scales and A sensitivity may change.
The unchanged RTN plan competes in every operator search. Full trajectory decides.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from baselines import sha, write


def objective(errors, benchmarks):
    means = [sum(e for e, b in zip(errors, benchmarks) if b == dataset) / benchmarks.count(dataset)
             for dataset in sorted(set(benchmarks))]
    return sum(means) / len(means) + .25 * max(errors)


def top_weight_tiles(weight, codes, scales, factor, prescale, rms, count):
    """Diagonal-input contribution proxy, not a claim of exact marginal output gain."""
    import torch
    from thunderworld_fp4.cosmos.calibration import decode
    n, k = weight.shape
    if type(count) is not int or not 0 <= count <= (n // 128) * (k // 128):
        raise ValueError('Invalid W tile quota')
    represented = (weight.float() / prescale).bfloat16().float()
    w0 = decode(codes[0], scales[0], factor)
    wr = decode(codes[1], scales[1], factor)
    error = represented - w0
    channel_moment = (rms * prescale).square()
    gain = ((error.square() - (error - wr).square()) * channel_moment[None, :])
    gain = gain.reshape(n // 128, 128, k // 128, 128).sum((1, 3))
    ids = torch.argsort(gain.flatten(), descending=True, stable=True)[:count].sort().values.cpu().tolist()
    sensitivity = w0.square().sum(0)
    return ids, gain, sensitivity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--controls', type=Path, help='Explicit protected-plan candidates; preserves historical defaults when omitted')
    p.add_argument('--budgets', type=float, nargs='+', default=[1.5, 2., 2.5])
    a = p.parse_args()
    capture_path = a.study / 'capture/manifest.json'
    while True:
        capture = json.loads(capture_path.read_text()) if capture_path.exists() else {}
        if capture.get('paused'):
            raise RuntimeError('Teacher capture paused')
        if capture.get('complete'):
            break
        time.sleep(5)
    cohort = json.loads((a.previous / 'cohort.json').read_text())
    priors_manifest = json.loads((a.study / 'priors/manifest.json').read_text())
    if priors_manifest['cohort_sha256'] != sha(a.previous / 'cohort.json') or priors_manifest['rms_sha256'] != sha(a.study / 'priors/rms.pt'):
        raise ValueError('Changed scale priors')
    if a.controls:
        old = json.loads(a.controls.read_text())['candidates']
        names = {f'b{cap:g}_protected_n' for cap in a.budgets}
    else:
        old = json.loads((a.previous / 'candidates/candidates.json').read_text())['candidates']
        old += json.loads((a.previous / 'ffn_followup/candidates/candidates.json').read_text())['candidates']
        names = {'b1.5_protected_n', 'b2_protected_n', 'b2.5_ffn_aware'}
    controls = {c['budget']: c for c in old if c['name'] in names and c['budget'] in a.budgets}
    if set(controls) != set(a.budgets):
        raise ValueError('Missing frozen controls')
    source_path = a.previous / 'calibration/rtn_assets'
    source = json.loads((source_path / 'manifest.json').read_text())
    if source['model_identity'] != capture['model_identity']:
        raise ValueError('Different teacher weights')
    root = a.study / 'joint_calibration'
    root.mkdir(exist_ok=True)
    provenance = dict(capture_sha256=sha(capture_path), priors_sha256=sha(a.study / 'priors/manifest.json'),
                      source_sha256=sha(__file__), assets_sha256=sha(source_path / 'manifest.json'),
                      control_plans={str(cap): c['plan_sha256'] for cap, c in controls.items()})
    state_path = root / 'progress.json'
    state = dict(complete=False, provenance=provenance, modules={}, performance_eligible=False)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['provenance'] != provenance:
            raise ValueError('Changed calibration resume')
        if state['complete']:
            return
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        state['resource'] = select_shared_device()
        write(state_path, state)
        import torch
        from thunderworld_fp4._calibrated import pack
        from thunderworld_fp4.cosmos.assets import expand, compact
        from thunderworld_fp4.cosmos.calibration import prescale_candidates
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec, RunConfig
        from thunderworld_fp4.cosmos.budget import RepairPlan, compile_request
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        if not state['modules']:
            import pytest
            test = Path(__file__).resolve().parents[2] / 'packages/fp4-repair/tests/test_compute_cuda.py'
            if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
                raise RuntimeError('Native SM120 verification requires a usable GPU')
            if pytest.main(['-q', str(test), '-k', 'sparse_activation_codes_current_residual_and_graph or zero_full_attention_and_native_weight_codec']) != 0:
                raise RuntimeError('Native codec/current-input preflight failed')
            write(root / 'gpu_preflight.json', dict(complete=True, tests=3,
                  scope='M129/M416 current residual masks, selected-code byte equivalence, eager/Graph current input, shared W1 bank; independent E4M3/E2M1 weight codec, zero attention.'))
        priors = torch.load(a.study / 'priors/rms.pt', map_location='cpu', weights_only=True)
        plans = {cap: json.loads(Path(c['config']['budget_plan']).read_text()) for cap, c in controls.items()}
        tables = {cap: json.loads((Path(c['config']['budget_plan']).parent / 'compute.json').read_text()) for cap, c in controls.items()}
        benchmarks = [entry['benchmark'] for entry in cohort['cases']['calibration']]
        try:
            with torch.inference_mode():
                for index, (name, entry) in enumerate(source['modules'].items()):
                    if name in state['modules']:
                        for file, digest in state['modules'][name]['files'].items():
                            if sha(root / file) != digest:
                                raise ValueError('Changed completed calibration asset')
                        continue
                    original_file = source_path / entry['file']
                    if sha(original_file) != entry['sha256']:
                        raise ValueError('Changed RTN asset')
                    original = expand(torch.load(original_file, map_location='cpu', weights_only=True), 1.)
                    rows = sorted([r for r in capture['rows'] if r['name'] == name], key=lambda r: r['scene_index'])
                    if len(rows) != 16 or [r['scene_index'] for r in rows] != list(range(16)):
                        raise ValueError('Missing full-M scene panels')
                    panels = []
                    for row in rows:
                        file = a.study / 'capture' / row['file']
                        if sha(file) != row['sha256']:
                            raise ValueError('Changed full-M input')
                        panel = torch.load(file, map_location='cpu', weights_only=True)
                        panels.append(dict(x=panel['x'].cuda(), y=panel['y'].cuda().float()))
                    wfile = a.study / 'capture' / capture['weights'][name]['file']
                    if sha(wfile) != capture['weights'][name]['sha256']:
                        raise ValueError('Changed BF16 weight')
                    w = torch.load(wfile, map_location='cpu', weights_only=True).cuda()
                    n, k = w.shape
                    m = panels[0]['x'].shape[0]
                    if any(tuple(p['x'].shape) != (m, k) or tuple(p['y'].shape) != (m, n) for p in panels):
                        raise ValueError('Mixed shapes require a separate calibration')
                    rms = priors[name].cuda()
                    denominators = [p['y'].square().sum().clamp_min(1e-20) for p in panels]
                    decisions, files = {}, {}
                    for cap, plan in plans.items():
                        settings = [r for key, r in plan['linears'].items() if key.split('|', 2)[-1] == name]
                        if len(settings) != 8 or any(r != settings[0] for r in settings):
                            raise ValueError('This coordinate search needs static per-operator quotas')
                        setting = settings[0]
                        na, nw = setting['a_blocks'], len(setting['w_blocks'])
                        tested = []
                        best = [float('inf'), None, None, None]

                        def evaluate(asset, ids, meta):
                            spec = LinearSpec(0, 0, activation_blocks=na, weight_support=tuple(ids))
                            controller = SimpleNamespace(warming=False, config=RunConfig(resource_mode='shared-calibration'), linear_spec=lambda *_: spec)
                            op = RepairLinear(SimpleNamespace(in_features=k, out_features=n), controller, name, spec, asset)
                            op.prepare_shape(m, spec)
                            errors = []
                            for panel, denominator in zip(panels, denominators):
                                pred = op(panel['x']).float()
                                errors.append(float((pred - panel['y']).square().sum() / denominator))
                            loss = objective(errors, benchmarks)
                            if not all(torch.isfinite(torch.tensor(errors))):
                                raise ValueError('Nonfinite calibration output')
                            tested.append(dict(**meta, loss=loss, panel_nmse=errors))
                            if loss < best[0]:
                                best[:] = [loss, asset, ids, meta]
                            del op

                        def encode(d, base, residual):
                            scaled = (w.float() / d).bfloat16().contiguous()
                            codes, scales, factor, _ = pack(scaled, torch.ones_like(d), True, True, w, base=base, res=residual)
                            ids, gain, sensitivity = top_weight_tiles(w, codes, scales, factor, d, rms, nw)
                            asset = dict(codes=codes.cpu(), scales=scales.cpu(), factor=factor.cpu(), prescale=d.cpu(),
                                         score=gain.cpu(), bias=original.get('bias'), selector='residual_sensitivity',
                                         activation_sensitivity=sensitivity.cpu(), activation_scale=(1., 1.),
                                         weight_scale=(base, residual), identity_prescale=bool((d == 1).all()))
                            return asset, ids

                        evaluate(original, setting['w_blocks'], dict(kind='unchanged_control', prescale='identity', base=1., residual=1.))
                        baseline_loss = tested[-1]['loss']
                        codes, scales, factor = (original[key].cuda() for key in ('codes', 'scales', 'factor'))
                        ids, gain, sensitivity = top_weight_tiles(w, codes, scales, factor, torch.ones_like(rms), rms, nw)
                        importance = dict(original, score=gain.cpu(), activation_sensitivity=sensitivity.cpu())
                        evaluate(importance, ids, dict(kind='importance_rtn', prescale='identity', base=1., residual=1.))
                        budget_root = root / f'b{cap:g}'
                        budget_root.mkdir(exist_ok=True)
                        for tag, asset, support in [('importance_rtn', importance, ids)]:
                            file = budget_root / f'{index:03d}_{tag}.pt'
                            torch.save(dict(asset=compact(asset, support), support=support), file)
                            files[str(file.relative_to(root))] = sha(file)
                        seen = set()
                        for label, d in prescale_candidates(rms, w):
                            if label == 'legacy_orientation':
                                continue
                            digest = hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest()
                            if digest in seen:
                                continue
                            seen.add(digest)
                            for base in (.875, 1.):
                                if label == 'identity' and base == 1.:
                                    continue
                                asset, support = encode(d, base, 1.)
                                evaluate(asset, support, dict(kind='joint_search', prescale=label, base=base, residual=1.))
                        d = best[1]['prescale'].cuda()
                        base = best[1].get('weight_scale', (1., 1.))[0]
                        fixed_meta = best[3]
                        for residual in (.875, 1.125):
                            asset, support = encode(d, base, residual)
                            evaluate(asset, support, dict(kind='joint_residual_coordinate', prescale=fixed_meta['prescale'], base=base, residual=residual))
                        if best[0] > baseline_loss:
                            raise AssertionError('Unchanged control must remain feasible')
                        file = budget_root / f'{index:03d}_joint.pt'
                        torch.save(dict(asset=compact(best[1], best[2]), support=best[2]), file)
                        files[str(file.relative_to(root))] = sha(file)
                        decisions[str(cap)] = dict(a_blocks=na, w_blocks=nw, baseline_loss=baseline_loss,
                                                   selected_loss=best[0], selected=best[3], candidates=tested)
                    state['modules'][name] = dict(index=index, decisions=decisions, files=files)
                    write(state_path, state)
                    print(json.dumps(dict(completed_modules=len(state['modules']), total=len(source['modules']), module=name,
                                          improvements={cap: 1 - d['selected_loss'] / d['baseline_loss'] for cap, d in decisions.items()})), flush=True)
                    del panels, w, original, best, rms
            write(root / 'decisions.json', dict(complete=True, provenance=provenance, modules=state['modules'],
                  scope='Full native deployed-linear objective, equal dataset means plus 0.25 worst panel. No whole-FFN/trajectory guarantee.'))
            candidates = list(controls.values())
            for cap, original_plan in plans.items():
                for variant in ('importance_rtn', 'joint'):
                    folder = a.study / 'candidates' / f'b{cap:g}_{variant}'
                    bank = folder / 'assets'
                    bank.mkdir(parents=True, exist_ok=True)
                    plan = copy.deepcopy(original_plan)
                    plan['design'] = dict(source_plan_sha256=controls[cap]['plan_sha256'], w1_static_across_phase_cfg=True,
                                          w1_axis='N128 K128 tiles', activation_importance='current Ar with per-module original sensitivity or represented W0 channel norm; see decisions',
                                          calibration_sha256=sha(root / 'decisions.json'), variant=variant)
                    manifest = dict(complete=False, precision='fp4', model_identity=source['model_identity'], modules={}, cohort_sha256=sha(a.previous / 'cohort.json'))
                    total_tensor = total_w1 = 0
                    for name, result in state['modules'].items():
                        file = root / f'b{cap:g}' / f'{result["index"]:03d}_{variant}.pt'
                        data = torch.load(file, weights_only=True, map_location='cpu')
                        for key, row in plan['linears'].items():
                            if key.split('|', 2)[-1] == name:
                                row['w_blocks'] = data['support']
                        target = bank / f'{result["index"]:03d}.pt'
                        torch.save(data['asset'], target)
                        manifest['modules'][name] = dict(file=target.name, sha256=sha(target))
                        total_tensor += sum(v.numel() * v.element_size() for v in data['asset'].values() if isinstance(v, torch.Tensor))
                        total_w1 += len(data['support']) * 128 * 128 * 9 // 16
                    RepairPlan(plan)
                    rows = copy.deepcopy(tables[cap]['rows'])
                    for row in rows:
                        row.pop('calls', None)
                        if row['kind'] == 'linear':
                            row['w_blocks'] = plan['linears'][f'{row["branch"]}|{row["phase"]}|{row["name"]}']['w_blocks']
                    ledger = compile_request(rows, cap, expected_modules=list(source['modules']),
                                             expected_layers={r['name'] for r in rows if r['kind'] == 'attention'})
                    if ledger['total_mma'] != tables[cap]['total_mma']:
                        raise AssertionError('Offline coordinate search changed MMA count')
                    write(folder / 'plan.json', plan)
                    write(folder / 'compute.json', ledger)
                    manifest['complete'] = True
                    write(bank / 'manifest.json', manifest)
                    config = copy.deepcopy(controls[cap]['config'])
                    config['budget_plan'] = str((folder / 'plan.json').resolve())
                    cost = dict(tensor_bytes=total_tensor, w1_bytes=total_w1,
                                file_bytes=sum((bank / e['file']).stat().st_size for e in manifest['modules'].values()))
                    c = dict(name=f'b{cap:g}_{variant}', budget=cap, config=config, assets=str(bank.resolve()),
                             assets_sha256=sha(bank / 'manifest.json'), plan_sha256=sha(folder / 'plan.json'), cost=cost, bound_ratio=ledger['ratio'])
                    write(folder / 'candidate.json', c)
                    candidates.append(c)
            write(a.study / 'candidates/candidates.json', dict(complete=True, candidates=candidates, provenance=provenance,
                  scope='Identical per-operator tile counts and MMA. RTN control, activation-weighted tile/sensitivity control, native support-conditioned offline coordinate search. No final quality claim.'))
            state['complete'] = True
            state.pop('paused', None)
            state.pop('error', None)
            write(state_path, state)
        except Exception as error:
            state.update(paused=True, error=repr(error))
            write(state_path, state)
            raise


if __name__ == '__main__':
    main()

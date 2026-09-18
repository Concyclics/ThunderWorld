"""One bounded followup: keep attention projection assets, select whole-FFN improvements."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from baselines import sha, write
from importance_joint_calibrate import objective


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    a = p.parse_args()
    while True:
        status = json.loads(a.after.read_text()) if a.after.exists() else {}
        if status.get('paused'):
            raise RuntimeError('Previous GPU work paused')
        if status.get('complete'):
            break
        time.sleep(5)
    catalog = json.loads((a.study / 'candidates/candidates.json').read_text())
    by_name = {c['name']: c for c in catalog['candidates']}
    controls = {1.5: 'b1.5_protected_n', 2.: 'b2_protected_n', 2.5: 'b2.5_ffn_aware'}
    capture = json.loads((a.study / 'capture/manifest.json').read_text())
    cohort = json.loads((a.previous / 'cohort.json').read_text())
    benchmarks = [entry['benchmark'] for entry in cohort['cases']['calibration']]
    root = a.study / 'ffn_guarded'
    root.mkdir(exist_ok=True)
    state = dict(complete=False, modules={}, performance_eligible=False,
                 provenance=dict(source_sha256=sha(__file__), catalog_sha256=sha(a.study / 'candidates/candidates.json'),
                                 capture_sha256=sha(a.study / 'capture/manifest.json')))
    path = root / 'progress.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['provenance'] != state['provenance']:
            raise ValueError('Changed guarded calibration resume')
        state = old
        if state['complete']:
            return
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        state['resource'] = select_shared_device()
        write(path, state)
        import torch
        from thunderworld_fp4.cosmos.assets import expand
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec, RunConfig
        from thunderworld_fp4.cosmos.budget import RepairPlan, compile_request
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False

        def make(name, manifest, bank, plan, m):
            entry = manifest['modules'][name]
            if sha(bank / entry['file']) != entry['sha256']:
                raise ValueError('Changed source asset')
            raw = torch.load(bank / entry['file'], weights_only=True, map_location='cpu')
            row = plan['linears'][f'0|0|{name}']
            spec = LinearSpec(0, 0, activation_blocks=row['a_blocks'], weight_support=tuple(row['w_blocks']))
            asset = expand(raw, row['w_blocks'])
            n, k = asset['codes'].shape[1], asset['codes'].shape[2] * 2
            ctl = SimpleNamespace(warming=False, config=RunConfig(resource_mode='shared-calibration'), linear_spec=lambda *_: spec)
            op = RepairLinear(SimpleNamespace(in_features=k, out_features=n), ctl, name, spec, asset)
            op.prepare_shape(m, spec)
            return op

        try:
            with torch.inference_mode():
                for cap, original_name in controls.items():
                    original, joint = by_name[original_name], by_name[f'b{cap:g}_joint']
                    bank0, bank1 = Path(original['assets']), Path(joint['assets'])
                    asset0, asset1 = (json.loads((b / 'manifest.json').read_text()) for b in (bank0, bank1))
                    plan0, plan1 = (json.loads(Path(c['config']['budget_plan']).read_text()) for c in (original, joint))
                    for layer in range(28):
                        key = f'{cap:g}|{layer}'
                        if key in state['modules']:
                            continue
                        name = f'layers.{layer}.mlp_moe_gen'
                        entries = {side: sorted([r for r in capture['rows'] if r['name'] == name + '.' + side + '_proj'], key=lambda r: r['scene_index']) for side in ('up', 'down')}
                        if any(len(v) != 16 for v in entries.values()):
                            raise ValueError('Incomplete paired FFN inputs')
                        data = []
                        for up, down in zip(entries['up'], entries['down']):
                            if (up['scene_index'], up['phase'], up['branch']) != (down['scene_index'], down['phase'], down['branch']):
                                raise ValueError('FFN inputs are from different teacher calls')
                            pair = []
                            for row in (up, down):
                                file = a.study / 'capture' / row['file']
                                if sha(file) != row['sha256']:
                                    raise ValueError('Changed teacher panel')
                                pair.append(torch.load(file, weights_only=True, map_location='cpu'))
                            data.append((pair[0]['x'].cuda(), pair[1]['y'].cuda().float()))
                        m = len(data[0][0])
                        ups = [make(name + '.up_proj', manifest, bank, plan, m) for manifest, bank, plan in ((asset0, bank0, plan0), (asset1, bank1, plan1))]
                        downs = [make(name + '.down_proj', manifest, bank, plan, m) for manifest, bank, plan in ((asset0, bank0, plan0), (asset1, bank1, plan1))]
                        errors = {(u, d): [] for u in (0, 1) for d in (0, 1)}
                        for x, gold in data:
                            denom = gold.square().sum().clamp_min(1e-20)
                            hidden = [torch.relu(up(x)).square() for up in ups]
                            for (u, d), values in errors.items():
                                pred = downs[d](hidden[u]).float()
                                values.append(float((pred - gold).square().sum() / denom))
                        losses = {k: objective(v, benchmarks) for k, v in errors.items()}
                        best = min(losses, key=lambda k: (losses[k], sum(k), k))
                        state['modules'][key] = dict(name=name, selected=list(best),
                            control_loss=losses[0, 0], selected_loss=losses[best],
                            candidates={f'{u}{d}': dict(loss=losses[u, d], panel_nmse=values) for (u, d), values in errors.items()})
                        write(path, state)
                        print(json.dumps(dict(budget=cap, layer=layer, selected=best, improvement=1-losses[best]/losses[0, 0])), flush=True)
                        del data, ups, downs, hidden
            write(root / 'decisions.json', dict(complete=True, provenance=state['provenance'], modules=state['modules'],
                  scope='Single calibration-only full-FFN coordinate choice among original/new up/down assets; attention projection assets frozen. No whole-trajectory guarantee.'))
            candidates = copy.deepcopy(catalog['candidates'])
            for cap, original_name in controls.items():
                original, joint = by_name[original_name], by_name[f'b{cap:g}_joint']
                banks = [Path(c['assets']) for c in (original, joint)]
                manifests = [json.loads((b / 'manifest.json').read_text()) for b in banks]
                plans = [json.loads(Path(c['config']['budget_plan']).read_text()) for c in (original, joint)]
                plan = copy.deepcopy(plans[0])
                plan['design'] = dict(ffn_calibration_sha256=sha(root / 'decisions.json'), w1_static_across_phase_cfg=True,
                                      projection_assets='unchanged control', ffn_assets='full-FFN up/down coordinate selection', w1_axis='N128 K128 tiles')
                folder = root / f'b{cap:g}_ffn_guarded'
                bank = folder / 'assets'
                bank.mkdir(parents=True, exist_ok=True)
                manifest = dict(complete=False, precision='fp4', model_identity=manifests[0]['model_identity'], modules={})
                tensors = w1 = 0
                for index, name in enumerate(manifests[0]['modules']):
                    selected = 0
                    if '.mlp_moe_gen.' in name:
                        layer = int(name.split('.')[1])
                        side = 0 if name.endswith('up_proj') else 1
                        selected = state['modules'][f'{cap:g}|{layer}']['selected'][side]
                    source = banks[selected] / manifests[selected]['modules'][name]['file']
                    target = bank / f'{index:03d}.pt'
                    shutil.copyfile(source, target)
                    manifest['modules'][name] = dict(file=target.name, sha256=sha(target), source_variant=selected)
                    asset = torch.load(target, weights_only=True, map_location='cpu')
                    tensors += sum(v.numel()*v.element_size() for v in asset.values() if isinstance(v, torch.Tensor))
                    w1 += len(asset['residual_indices']) * 128**2 * 9 // 16
                    for key in plan['linears']:
                        if key.split('|', 2)[-1] == name:
                            plan['linears'][key] = plans[selected]['linears'][key]
                RepairPlan(plan)
                table = json.loads((Path(original['config']['budget_plan']).parent / 'compute.json').read_text())
                rows = copy.deepcopy(table['rows'])
                for row in rows:
                    row.pop('calls', None)
                    if row['kind'] == 'linear':
                        row['w_blocks'] = plan['linears'][f'{row["branch"]}|{row["phase"]}|{row["name"]}']['w_blocks']
                ledger = compile_request(rows, cap, expected_modules=list(manifest['modules']), expected_layers={r['name'] for r in rows if r['kind']=='attention'})
                if ledger['total_mma'] != table['total_mma']:
                    raise ValueError('Guarded choice changed MMA')
                write(folder / 'plan.json', plan)
                write(folder / 'compute.json', ledger)
                manifest['complete'] = True
                write(bank / 'manifest.json', manifest)
                config = dict(original['config'], budget_plan=str((folder / 'plan.json').resolve()))
                candidate = dict(name=f'b{cap:g}_ffn_guarded', budget=cap, config=config, assets=str(bank.resolve()),
                                 assets_sha256=sha(bank / 'manifest.json'), plan_sha256=sha(folder / 'plan.json'),
                                 cost=dict(tensor_bytes=tensors, w1_bytes=w1, file_bytes=sum((bank/e['file']).stat().st_size for e in manifest['modules'].values())), bound_ratio=ledger['ratio'])
                write(folder / 'candidate.json', candidate)
                candidates.append(candidate)
            write(root / 'candidates.json', dict(complete=True, candidates=candidates, scope='At most four candidates per budget: original, importance RTN, linear-objective joint, and one full-FFN guarded followup.'))
            state['complete'] = True
            write(path, state)
        except Exception as error:
            state.update(paused=True, error=repr(error))
            write(path, state)
            raise


if __name__ == '__main__':
    main()

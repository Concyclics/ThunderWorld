"""Bounded, fixed-bank W0/W1 representation verification (no online changes).

Proposal loss uses full K16 activation covariance and the *actual* current-input
partial A representation. For three terms, e = eta W + (A0+Ar)(W-W0-Wr)
+ Ar Wr, eta = X-A0-Ar. Thus the omitted cross term is not assumed harmless.
Groups choose independently; inter-group correlations and BF16 output rounding
are checked by the existing native operator on all sixteen complete-M panels.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from baselines import sha, write

LABELS = ('scale_only', 'removal_only', 'joint')
BASE_SCALES = (.75, .875, 1., 1.5)
RES_SCALES = (.875, 1., 1.125)


def group_loss(z, w0, wr, covariance, cross=False):
    import torch
    # covariance = E[[eta, A0+Ar, Ar]^T [eta, A0+Ar, Ar]].
    v = torch.cat((z, z-w0-wr, torch.zeros_like(wr) if cross else wr), -1)
    return torch.einsum('gd,gde,ge->g', v, covariance, v)


def proposals(z, c0, s0, c1, s1, covariance):
    """One removed position per group, only inside the already-paid W1 bank.

    Positions: unchanged, largest magnitude, and the two largest actual W0
    errors, independently for each group. This is a bounded proposal search,
    not an exhaustive output-response oracle over all sixteen positions.
    """
    import torch
    from thunderworld_fp4.cosmos.calibration import _encode, decode
    z = z.reshape(-1, 16)
    original0, original1 = decode(c0, s0, 1.), decode(c1, s1, 1.)
    initial = group_loss(z, original0, original1, covariance)
    best = {label: dict(loss=initial.clone(), c0=c0.clone(), s0=s0.clone(),
                       c1=c1.clone(), s1=s1.clone(), removed=torch.full_like(initial, -1, dtype=torch.long))
            for label in LABELS}
    errors = (z-original0).abs()
    indices = [None, z.abs().argmax(-1), errors.argmax(-1), errors.argsort(dim=-1, descending=True, stable=True)[:, 1]]
    rows = torch.arange(len(z), device=z.device)
    for index in indices:
        base_input = z.clone()
        if index is not None:
            base_input[rows, index] = 0.
        for bs in BASE_SCALES:
            nc0, ns0 = _encode(base_input, bs)
            w0 = decode(nc0, ns0, 1.)
            for rs in RES_SCALES:
                nc1, ns1 = _encode(z-w0, rs)
                wr = decode(nc1, ns1, 1.)
                loss = group_loss(z, w0, wr, covariance)
                labels = ['joint']
                if index is None:
                    labels.append('scale_only')
                if bs == 1. and rs == 1.:
                    labels.append('removal_only')
                for label in labels:
                    state = best[label]
                    take = loss < state['loss']
                    state['loss'] = torch.where(take, loss, state['loss'])
                    for key, value in (('c0', nc0), ('s0', ns0), ('c1', nc1), ('s1', ns1)):
                        # float8 where/index kernels are not supported on every device.
                        old = state[key].view(torch.uint8)
                        old[take] = value.view(torch.uint8)[take]
                    moved = torch.full_like(state['removed'], -1) if index is None else index
                    state['removed'] = torch.where(take, moved, state['removed'])
    return best


def activation_covariance(panels, asset, na, benchmarks):
    import torch
    from thunderworld_fp4._sparse_pack import pack_sparse
    from thunderworld_fp4._calibrated import pack
    from thunderworld_fp4.cosmos.calibration import decode
    from thunderworld_fp4.cosmos.linears import select_tiles
    import triton
    d = asset['prescale'].cuda()
    sensitivity = asset.get('activation_sensitivity', torch.ones_like(d)).cuda()
    k = len(d)
    covariance = torch.zeros(k//16, 48, 48, device=d.device)
    for (x, y), benchmark in zip(panels, benchmarks):
        m = len(x)
        mask = torch.full(((m+127)//128, k//128), na == k//128, device=d.device, dtype=torch.bool)
        base, res = asset.get('activation_scale', (1., 1.))
        if 0 < na < k//128 and asset.get('selector') == 'residual_sensitivity':
            c, s, f, _ = pack_sparse(x, d, sensitivity, mask, na, base=base, res=res)
        else:
            if 0 < na < k//128:
                select_tiles[(len(mask),)](x, d, mask, m, k, triton.next_power_of_2(k//128), na, num_warps=4)
            c, s, f, _ = pack(x, d, na > 0, na == k//128, mask, base=base, res=res)
        a0 = decode(c[0], s[0], f)[:m]
        if na:
            # Sparse pack deliberately leaves unselected residual bytes unwritten.
            # Never read those bytes as float8 (they can decode to NaN).
            active = mask.repeat_interleave(128, 0)[:c.shape[1]]
            cc = c[1].clone().masked_fill(~active.repeat_interleave(64, 1), 0)
            ss = s[1].view(torch.uint8).clone().masked_fill(~active.repeat_interleave(8, 1), 0).view(s.dtype)
            ar = decode(cc, ss, f)[:m]
        else:
            ar = torch.zeros_like(a0)
        t = (x.float()*d).bfloat16().float()
        features = torch.cat([(t-a0-ar).reshape(m, -1, 16), (a0+ar).reshape(m, -1, 16), ar.reshape(m, -1, 16)], -1)
        weight = 1./(len(set(benchmarks))*benchmarks.count(benchmark))
        covariance += torch.einsum('mgd,mge->gde', features, features)*(weight/y.square().sum().clamp_min(1e-20))
    return covariance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('methods', 'capture', 'cohort', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--method', default='b2_joint')
    p.add_argument('--modules', nargs='+')
    p.add_argument('--chunk-blocks', type=int, default=16)
    a = p.parse_args()
    if a.chunk_blocks < 1:
        raise ValueError('Positive offline chunk size required')
    catalog = json.loads(a.methods.read_text()); spec = catalog['methods'][a.method]
    bank = Path(spec['assets']); assets = json.loads((bank/'manifest.json').read_text())
    capture = json.loads((a.capture/'manifest.json').read_text()); cohort = json.loads(a.cohort.read_text())
    plan = json.loads(Path(spec['config']['budget_plan']).read_text())
    if not assets['complete'] or not capture['complete'] or assets['model_identity'] != capture['model_identity']:
        raise ValueError('Complete model-matched assets and full-M panels required')
    if sha(bank/'manifest.json') != spec['assets_sha256'] or sha(spec['config']['budget_plan']) != spec['plan_sha256']:
        raise ValueError('Changed fixed control')
    if capture['identity']['cohort_sha256'] != sha(a.cohort) or catalog['model'] != cohort['model']:
        raise ValueError('Mismatched cohort/model')
    names = a.modules or list(assets['modules'])
    if len(set(names)) != len(names) or not set(names) <= set(assets['modules']):
        raise ValueError('Invalid module list')
    identity = dict(source_sha256=sha(__file__), methods_sha256=sha(a.methods),
        assets_sha256=spec['assets_sha256'], capture_sha256=sha(a.capture/'manifest.json'),
        cohort_sha256=sha(a.cohort), modules=names, chunk_blocks=a.chunk_blocks)
    a.output.mkdir(parents=True, exist_ok=True); path = a.output/'progress.json'
    state = dict(complete=False, identity=identity, modules={}, performance_eligible=False,
        scope='Fixed prescale, A selector, W1 support, three-term arithmetic and runtime; K16 covariance is only a proposal proxy. '
              'All sixteen complete-M panels select with native NMSE; complete generation is a separate gate.')
    if path.exists():
        state = json.loads(path.read_text())
        if state['identity'] != identity:
            raise ValueError('Changed resume')
        if state['complete']:
            return
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        state['resource'] = select_shared_device(); write(path, state)
        import torch
        from importance_joint_calibrate import objective
        from thunderworld_fp4.cosmos.assets import compact, expand
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec, RunConfig
        torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = False
        benchmarks = [e['benchmark'] for e in cohort['cases']['calibration']]
        try:
            with torch.inference_mode():
                for name in names:
                    if name in state['modules']:
                        for item in state['modules'][name]['exports'].values():
                            if sha(a.output/item['file']) != item['sha256']:
                                raise ValueError('Changed completed asset')
                        continue
                    entry = assets['modules'][name]; file = bank/entry['file']
                    if sha(file) != entry['sha256']:
                        raise ValueError('Changed source asset')
                    raw = torch.load(file, map_location='cpu', weights_only=True)
                    ids = sorted(raw['residual_indices'].tolist()); old = expand(raw, ids)
                    settings = [r for key, r in plan['linears'].items() if key.split('|', 2)[-1] == name]
                    if len(settings) != 8 or any(r != settings[0] for r in settings) or sorted(settings[0]['w_blocks']) != ids or settings[0]['cross']:
                        raise ValueError('Requires fixed, always-executed W bank and three-term control')
                    if old.get('mandatory_w_blocks') or old.get('weight_scale', (1., 1.)) != (1., 1.):
                        raise ValueError('This removal-only ablation requires the original (1,1), unsplit asset')
                    na = settings[0]['a_blocks']; results = {}; chosen = {}; proposal_metadata = {}
                    selected = {label: old for label in LABELS}
                    if ids:
                        we = capture['weights'][name]; wf = a.capture/we['file']
                        if sha(wf) != we['sha256']:
                            raise ValueError('Changed BF16 source weight')
                        w = torch.load(wf, weights_only=True).cuda(); n, k = w.shape
                        d = old['prescale'].cuda(); g = old['factor'].cuda()
                        z = (w.float()/d).bfloat16().float()/g
                        rows = sorted((r for r in capture['rows'] if r['name'] == name), key=lambda r: r['scene_index'])
                        if [r['scene_index'] for r in rows] != list(range(16)):
                            raise ValueError('Sixteen complete-M panels required')
                        panels = []
                        for row in rows:
                            pf = a.capture/row['file']
                            if sha(pf) != row['sha256']:
                                raise ValueError('Changed panel')
                            panel = torch.load(pf, weights_only=True)
                            panels.append((panel['x'].cuda(), panel['y'].cuda().float()))
                        m = len(panels[0][0])
                        if any(x.shape != (m, k) for x, _ in panels):
                            raise ValueError('Mixed shapes outside this experiment')
                        def evaluate(asset, cross=False):
                            ls = LinearSpec(0, 0, bool(cross and na), activation_blocks=na, weight_support=tuple(ids))
                            controller = SimpleNamespace(warming=False, config=RunConfig.from_dict(spec['config']), linear_spec=lambda *_: ls)
                            op = RepairLinear(SimpleNamespace(in_features=k, out_features=n), controller, name, ls, asset)
                            op.prepare_shape(m, ls)
                            errors = [float((op(x).float()-y).square().sum()/y.square().sum().clamp_min(1e-20)) for x, y in panels]
                            if not torch.isfinite(torch.tensor(errors)).all():
                                raise ValueError('Nonfinite native output')
                            return dict(objective=objective(errors, benchmarks), panel_nmse=errors)
                        results['original'] = evaluate(old)
                        cov = activation_covariance(panels, old, na, benchmarks)
                        codes, scales = old['codes'].cuda(), old['scales'].cuda()
                        new = {label: dict(old, codes=codes.clone(), scales=scales.clone()) for label in LABELS}
                        removed = {label: 0 for label in LABELS}; mandatory = {label: set() for label in LABELS}
                        for start in range(0, len(ids), a.chunk_blocks):
                            chunk = torch.tensor(ids[start:start+a.chunk_blocks], device=w.device)
                            ni = (chunk//(k//128))[:, None]*128+torch.arange(128, device=w.device)
                            ki = (chunk%(k//128))[:, None]*8+torch.arange(8, device=w.device)
                            nr = ni[:, :, None].expand(-1, -1, 8).reshape(-1)
                            kg = ki[:, None, :].expand(-1, 128, -1).reshape(-1)
                            cz = z.reshape(n, k//16, 16)[nr, kg]
                            cc = [codes[j].reshape(n, k//16, 8)[nr, kg] for j in range(2)]
                            ss = [scales[j].view(torch.uint8)[nr, kg, None].view(scales.dtype) for j in range(2)]
                            proposals_by_label = proposals(cz, cc[0], ss[0], cc[1], ss[1], cov[kg])
                            for label, proposal in proposals_by_label.items():
                                for plane in range(2):
                                    new[label]['codes'][plane].reshape(n, k//16, 8)[nr, kg] = proposal['c'+str(plane)]
                                    new[label]['scales'][plane].view(torch.uint8)[nr, kg] = proposal['s'+str(plane)].view(torch.uint8).flatten()
                                moved = proposal['removed'].reshape(-1, 128, 8) >= 0
                                removed[label] += int(moved.sum())
                                mandatory[label].update(chunk[moved.flatten(1).any(1)].cpu().tolist())
                        for label in LABELS:
                            asset = new[label]
                            asset.update(codes=asset['codes'].cpu(), scales=asset['scales'].cpu(),
                                mandatory_w_blocks=tuple(sorted(mandatory[label])), offline_encoder='joint_k16_covariance_'+label)
                            result = evaluate(asset)
                            proposal_metadata[label] = dict(native=result, removed_elements=removed[label], mandatory_blocks=len(mandatory[label]))
                            # The unchanged original is a candidate, never silently re-encoded.
                            if result['objective'] < results['original']['objective']:
                                selected[label] = asset; results[label] = result; chosen[label] = label
                            else:
                                results[label] = results['original']; chosen[label] = 'original'
                        # Joint must also compete against the accepted scale-only alternative.
                        if results['scale_only']['objective'] < results['joint']['objective']:
                            selected['joint'] = selected['scale_only']; results['joint'] = results['scale_only']; chosen['joint'] = chosen['scale_only']
                        for label in ('original', *LABELS):
                            asset = old if label == 'original' else selected[label]
                            results[label]['four_diagnostic'] = evaluate(asset, True)
                        del cov, panels, w, z, codes, scales, new
                    exports = {}
                    for label in LABELS:
                        output = a.output/'assets'/label/(name+'.pt'); output.parent.mkdir(parents=True, exist_ok=True)
                        if selected[label] is old:
                            shutil.copyfile(file, output)
                        else:
                            payload = compact(selected[label], ids)
                            if sum(v.numel()*v.element_size() for v in payload.values() if isinstance(v, torch.Tensor)) != sum(v.numel()*v.element_size() for v in raw.values() if isinstance(v, torch.Tensor)):
                                raise ValueError('Tensor storage changed at fixed support')
                            torch.save(payload, output)
                            loaded = torch.load(output, weights_only=True)
                            for key, value in payload.items():
                                if isinstance(value, torch.Tensor) and not torch.equal(value.reshape(-1).view(torch.uint8), loaded[key].reshape(-1).view(torch.uint8)):
                                    raise ValueError('Reload changed bytes')
                        exports[label] = dict(file=str(output.relative_to(a.output)), sha256=sha(output))
                    state['modules'][name] = dict(results=results, chosen=chosen, proposals=proposal_metadata,
                        exports=exports, support_blocks=len(ids), a_blocks=na, untouched=not ids)
                    write(path, state)
                    print(json.dumps(dict(completed=len(state['modules']), total=len(names), module=name,
                        objectives={key: value['objective'] for key, value in results.items()}, chosen=chosen)), flush=True)
                full = set(names) == set(assets['modules'])
                candidates = []
                for label in LABELS:
                    exported = a.output/'assets'/label
                    manifest = copy.deepcopy(assets)
                    manifest.update(complete=full, modules={name: dict(file=name+'.pt', sha256=state['modules'][name]['exports'][label]['sha256']) for name in names},
                        representation_verification=dict(identity=identity, variant=label, full_generation_qualified=False))
                    write(exported/'manifest.json', manifest)
                    candidate = copy.deepcopy(spec)
                    candidate.update(name=a.method+'_group_'+label, assets=str(exported.resolve()), assets_sha256=sha(exported/'manifest.json'), origin=a.method)
                    candidates.append(candidate)
                state.update(complete=True, complete_model=full); state.pop('paused', None); state.pop('error', None); write(path, state)
                write(a.output/'candidates.json', dict(complete=full, candidates=candidates, identity=identity, performance_eligible=False))
        except Exception as error:
            state.update(paused=True, error=repr(error)); write(path, state); raise


if __name__ == '__main__':
    main()

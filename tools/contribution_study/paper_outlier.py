"""Native, fixed-support outlier x cross experiments; export real sparse assets.

The bounded output search moves the same tested K16 position in each eligible
group of one pre-reserved N128/K128 block per operator. It is NOT the exhaustive
per-group output-response oracle. Every complete proposed mask is reevaluated
through the native kernel. Full generation remains a separate acceptance gate.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from baselines import sha, write

VARIANTS = ('rtn', 'four_six', 'magnitude', 'quant_error', 'output_joint')


def effective_spec(a_blocks, w_blocks, cross):
    from thunderworld_fp4.cosmos.settings import LinearSpec
    # An empty side makes the cross intersection empty, not an illegal request
    # for a fourth product. Keep this operator in the full inventory.
    return LinearSpec(0, 0, bool(cross and a_blocks and w_blocks),
                      activation_blocks=a_blocks, weight_support=tuple(w_blocks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('methods', 'capture', 'cohort', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--method', default='b2_joint')
    p.add_argument('--modules', nargs='+')
    a = p.parse_args()
    methods = json.loads(a.methods.read_text())
    spec = methods['methods'][a.method]
    bank = Path(spec['assets'])
    assets = json.loads((bank/'manifest.json').read_text())
    capture = json.loads((a.capture/'manifest.json').read_text())
    cohort = json.loads(a.cohort.read_text())
    plan = json.loads(Path(spec['config']['budget_plan']).read_text())
    if not assets['complete'] or not capture['complete'] or assets['model_identity'] != capture['model_identity']:
        raise ValueError('Complete matching full-M capture and assets required')
    if sha(bank/'manifest.json') != spec['assets_sha256'] or sha(spec['config']['budget_plan']) != spec['plan_sha256']:
        raise ValueError('Changed fixed support/control')
    if capture['identity']['cohort_sha256'] != sha(a.cohort) or methods['model'] != cohort['model']:
        raise ValueError('Mismatched calibration cohort')
    names = a.modules or list(assets['modules'])
    if len(set(names)) != len(names) or not set(names) <= set(assets['modules']):
        raise ValueError('Invalid module inventory')
    identity = dict(method=a.method, methods_sha256=sha(a.methods), capture_sha256=sha(a.capture/'manifest.json'),
        cohort_sha256=sha(a.cohort), source_sha256=sha(__file__), modules=names)
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output/'progress.json'
    state = dict(complete=False, identity=identity, modules={}, performance_eligible=False,
        scope='Native local calibration, fixed A quota/selector and fixed W1 support. One pre-reserved block per operator, '
              'thresholds 2 and 3, at most one removed element per K16 group. Joint position proposals are bounded, '
              'not an exhaustive element oracle. Three/four are evaluated on identical assets; four costs are separate '
              'and may exceed the parent request cap. No claim of whole-trajectory improvement.')
    if path.exists():
        state = json.loads(path.read_text())
        if state['identity'] != identity:
            raise ValueError('Changed calibration resume')
        if state['complete']:
            return
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        state['resource'] = select_shared_device()
        import torch
        from representation import encode, mask_from_indices
        from importance_joint_calibrate import objective
        from thunderworld_fp4._calibrated import pack
        from thunderworld_fp4.cosmos.calibration import decode
        from thunderworld_fp4.cosmos.assets import expand, compact
        from thunderworld_fp4.cosmos.linears import RepairLinear
        from thunderworld_fp4.cosmos.settings import LinearSpec, RunConfig
        from thunderworld_fp4.cosmos.budget import linear_cost
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        benchmarks = [entry['benchmark'] for entry in cohort['cases']['calibration']]
        try:
            with torch.inference_mode():
                for name in names:
                    if name in state['modules']:
                        for variant, item in state['modules'][name]['exports'].items():
                            if sha(a.output/item['file']) != item['sha256']:
                                raise ValueError('Changed completed export')
                        continue
                    entry = assets['modules'][name]
                    if sha(bank/entry['file']) != entry['sha256']:
                        raise ValueError('Changed source weight asset')
                    raw = torch.load(bank/entry['file'], map_location='cpu', weights_only=True)
                    ids = sorted(raw['residual_indices'].tolist())
                    old = expand(raw, ids)
                    settings = [row for key, row in plan['linears'].items() if key.split('|', 2)[-1] == name]
                    if len(settings) != 8 or any(row != settings[0] for row in settings) or sorted(settings[0]['w_blocks']) != ids:
                        raise ValueError('Require one fixed W1 support, always executed across all phases/CFG')
                    if old.get('mandatory_w_blocks'):
                        raise ValueError('Cannot undo an existing mandatory outlier asset')
                    na = settings[0]['a_blocks']
                    weight_entry = capture['weights'][name]
                    wf = a.capture/weight_entry['file']
                    if sha(wf) != weight_entry['sha256']:
                        raise ValueError('Changed BF16 source weight')
                    w = torch.load(wf, weights_only=True).cuda()
                    n, k = w.shape
                    d = old['prescale'].cuda()
                    represented = (w.float()/d).bfloat16().contiguous()
                    c, s, g, _ = pack(represented, torch.ones_like(d), True, True, w, base=1., res=1.)
                    base_asset = dict(old, codes=c.cpu(), scales=s.cpu(), factor=g.cpu(),
                        weight_scale=(1., 1.), mandatory_w_blocks=())
                    panels = []
                    rows = sorted((r for r in capture['rows'] if r['name'] == name), key=lambda r: r['scene_index'])
                    if [r['scene_index'] for r in rows] != list(range(16)):
                        raise ValueError('All sixteen complete-M calibration panels required')
                    for row in rows:
                        file = a.capture/row['file']
                        if sha(file) != row['sha256']:
                            raise ValueError('Changed current input panel')
                        panel = torch.load(file, weights_only=True)
                        panels.append((panel['x'].cuda(), panel['y'].cuda().float()))
                    m = len(panels[0][0])
                    if any(len(x) != m for x, _ in panels):
                        raise ValueError('Mixed M is outside this controlled comparison')
                    losses = []

                    def evaluate(asset, cross=False):
                        ls = effective_spec(na, ids, cross)
                        controller = SimpleNamespace(warming=False, config=RunConfig(resource_mode='shared-calibration'), linear_spec=lambda *_: ls)
                        op = RepairLinear(SimpleNamespace(in_features=k, out_features=n), controller, name, ls, asset)
                        op.prepare_shape(m, ls)
                        errors = [float((op(x).float()-y).square().sum()/y.square().sum().clamp_min(1e-20)) for x, y in panels]
                        if not torch.isfinite(torch.tensor(errors)).all():
                            raise ValueError('Nonfinite native output')
                        return dict(objective=objective(errors, benchmarks), panel_nmse=errors)

                    baseline = evaluate(base_asset)
                    selected = {label: base_asset for label in VARIANTS}
                    removed = {label: 0 for label in VARIANTS}
                    metadata = {label: {} for label in VARIANTS}
                    z = represented.float()/g
                    # FourOverSix is a distinct all-group scale-choice control.
                    c0, s0 = encode(z, 'four_six')
                    c1, s1 = encode(z-decode(c0, s0, 1.), 'four_six')
                    selected['four_six'] = dict(base_asset, codes=torch.stack((c0, c1)).cpu(),
                        scales=torch.stack((s0, s1)).cpu(), offline_encoder='four_six')
                    if ids:
                        support = max(ids, key=lambda i: (float(old['score'].flatten()[i]), -i))
                        nb, kb = divmod(support, k//128)
                        ns, ks = slice(nb*128, (nb+1)*128), slice(kb*128, (kb+1)*128)
                        local = z[ns, ks]
                        rms = torch.stack([(x.float()*d).square().mean(0) for x, _ in panels]).mean(0).sqrt()
                        contribution = represented[ns, ks].float().abs().reshape(128, 8, 16)*rms[ks].reshape(1, 8, 16)
                        old0 = decode(c[0], s[0], g)[ns, ks]

                        def candidate(mask):
                            newc, news = c.clone(), s.clone()
                            lc0, ls0 = encode(local.masked_fill(mask, 0.))
                            lc1, ls1 = encode(local-decode(lc0, ls0, 1.))
                            for j, (lc, ls) in enumerate(((lc0, ls0), (lc1, ls1))):
                                newc[j, ns, kb*64:(kb+1)*64] = lc
                                news[j, ns, kb*8:(kb+1)*8] = ls
                            return dict(base_asset, codes=newc.cpu(), scales=news.cpu(),
                                mandatory_w_blocks=(support,) if mask.any() else ())

                        best = {label: float('inf') for label in ('magnitude', 'quant_error')}
                        best['output_joint'] = baseline['objective']
                        for threshold in (2., 3.):
                            active = contribution.amax(-1) > threshold*contribution.square().mean(-1).sqrt()
                            candidates = [('magnitude', represented[ns, ks].abs().reshape(128, 8, 16).argmax(-1)),
                                ('quant_error', (represented[ns, ks].float()-old0).abs().reshape(128, 8, 16).argmax(-1))]
                            candidates += [('output_joint', torch.full((128, 8), j, device=w.device, dtype=torch.long)) for j in range(16)]
                            for label, index in candidates:
                                mask = mask_from_indices(index, active, (128, 128))
                                asset = candidate(mask)
                                trial = evaluate(asset)
                                meta = dict(method=label, threshold=threshold,
                                    position=int(index[0, 0]) if label == 'output_joint' else None,
                                    removed_elements=int(mask.sum()), objective=trial['objective'])
                                losses.append(meta)
                                if trial['objective'] < best[label]:
                                    best[label] = trial['objective']
                                    selected[label], removed[label], metadata[label] = asset, int(mask.sum()), meta
                    results, exports = {}, {}
                    for label, asset in selected.items():
                        r0 = decode(asset['codes'][0].cuda(), asset['scales'][0].cuda(), asset['factor'].cuda())
                        r1 = decode(asset['codes'][1].cuda(), asset['scales'][1].cuda(), asset['factor'].cuda())
                        support_mask = torch.zeros((n//128, k//128), device=w.device, dtype=torch.bool)
                        support_mask.flatten()[ids] = True
                        deployed = r0+r1*support_mask.repeat_interleave(128, 0).repeat_interleave(128, 1)
                        moment = torch.stack([(x.float()*d).square().mean(0) for x, _ in panels]).mean(0)
                        denom = (represented.float().square()*moment).sum().clamp_min(1e-20)
                        results[label] = dict(three=evaluate(asset), four=evaluate(asset, True), removed=removed[label],
                            weighted_entire_w0_error=float(((represented.float()-r0).square()*moment).sum()/denom),
                            weighted_deployed_w0_w1_error=float(((represented.float()-deployed).square()*moment).sum()/denom),
                            mandatory_w_blocks=list(asset.get('mandatory_w_blocks', ())), selection=metadata[label])
                        file = a.output/'assets'/label/(name+'.pt')
                        file.parent.mkdir(parents=True, exist_ok=True)
                        compacted = compact(asset, ids)
                        torch.save(compacted, file)
                        loaded = torch.load(file, weights_only=True)
                        for key, value in compacted.items():
                            if isinstance(value, torch.Tensor) and not torch.equal(value.reshape(-1).view(torch.uint8), loaded[key].reshape(-1).view(torch.uint8)):
                                raise ValueError('Offline asset reload changed bytes')
                        exports[label] = dict(file=str(file.relative_to(a.output)), sha256=sha(file),
                            tensor_bytes=sum(x.numel()*x.element_size() for x in compacted.values() if isinstance(x, torch.Tensor)))
                    state['modules'][name] = dict(results=results, exports=exports, proposals=losses,
                        shape=[m, n, k], support=ids, a_blocks=na,
                        costs={label: linear_cost(m, n, k, a_blocks=na, w_blocks=ids, cross=cross)
                               for label, cross in (('three', False), ('four', True))})
                    write(path, state)
                    print(json.dumps(dict(completed=len(state['modules']), total=len(names), module=name,
                        rtn=results['rtn']['three']['objective'], joint=results['output_joint']['three']['objective'])), flush=True)
                complete_model = set(names) == set(assets['modules'])
                for label in VARIANTS:
                    out = copy.deepcopy(assets)
                    out.update(complete=complete_model, modules={name: dict(file=name+'.pt', sha256=state['modules'][name]['exports'][label]['sha256']) for name in names},
                        paper_representation=dict(identity=identity, variant=label, full_generation_qualified=False))
                    write(a.output/'assets'/label/'manifest.json', out)
                if complete_model:
                    candidates = []
                    for label in VARIANTS:
                        candidate = copy.deepcopy(spec)
                        exported = a.output/'assets'/label
                        candidate.update(name=a.method+'_outlier_'+label, assets=str(exported.resolve()),
                            assets_sha256=sha(exported/'manifest.json'), origin=a.method,
                            scope='Fixed support, prescale and A selection; native three-term local calibration only. Four-term interaction is not assigned the original budget cap.')
                        candidates.append(candidate)
                    write(a.output/'candidates.json', dict(complete=True, candidates=candidates, identity=identity,
                        full_generation_qualified=False, performance_eligible=False))
                state.update(complete=True, complete_model=complete_model)
                state.pop('paused', None)
                state.pop('error', None)
                write(path, state)
        except Exception as error:
            state.update(paused=True, error=repr(error))
            write(path, state)
            raise


if __name__ == '__main__':
    main()

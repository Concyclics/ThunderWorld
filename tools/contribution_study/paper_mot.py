"""Current-student MoT repair interventions on complete scene x phase x CFG panels.

AR/DM are key roles: all queries are denoising tokens. Pure AR tiles and the
mixed boundary are kept distinct. No attention base connection is removed.
"""
import argparse
import json
import math
from pathlib import Path

from baselines import sha, write
from dataset_infer import clear_case


def role_tiles(m, n):
    ar = n - m
    if ar <= 0:
        raise ValueError('Expected a prepended AR prefix')
    nt = math.ceil(n / 128)
    pure_ar = [i for i in range(nt) if min((i+1)*128, n) <= ar]
    pure_dm = [i for i in range(nt) if i*128 >= ar]
    mixed = [i for i in range(nt) if i not in pure_ar and i not in pure_dm]
    return dict(ar_tokens=ar, pure_ar=pure_ar, pure_dm=pure_dm, mixed=mixed)


def current_v_residual_energy(q, k, v):
    """Decode actual native base bytes; raw V energy is a different feature."""
    import torch
    from thunderworld_fp4._kernels import pack_inputs
    from thunderworld_fp4.cosmos.calibration import decode
    packed, _, padded = pack_inputs(q, k, v, False, False, 3, True, False)
    hk, n = v.shape[2], v.shape[1]
    nt = padded//128
    codes = packed[6][0].reshape(hk, nt, 64, 128).transpose(-1, -2).reshape(-1, 64)
    scales = packed[7][0].view(torch.uint8).reshape(hk, nt, 8, 128).transpose(-1, -2).reshape(-1, 8).view(torch.float8_e4m3fn)
    base = decode(codes, scales, packed[8][:, None]).reshape(hk, nt, 128, 128)
    raw = torch.zeros((hk, padded, 128), device=v.device, dtype=torch.float32)
    raw[:, :n] = v[0].transpose(0, 1).float()
    raw = raw.reshape(hk, nt, 128, 128).transpose(-1, -2)
    return (raw-base).square().sum((-1, -2)).repeat_interleave(q.shape[2]//hk, 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--methods', type=Path, required=True)
    p.add_argument('--method', default='b2_joint')
    p.add_argument('--split', choices=('calibration', 'development'), default='calibration')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='One calibration scene per dataset; never a full study')
    a = p.parse_args()
    cohort = json.loads(a.cohort.read_text())
    methods = json.loads(a.methods.read_text())
    spec = methods['methods'][a.method]
    if not cohort['frozen'] or cohort['model'] != methods['model']:
        raise ValueError('Frozen matching development cohort required')
    if sha(Path(spec['assets'])/'manifest.json') != spec['assets_sha256'] or sha(spec['config']['budget_plan']) != spec['plan_sha256']:
        raise ValueError('Changed frozen student assets/plan')
    entries = cohort['cases'][a.split]
    if a.smoke:
        entries = [next(e for e in entries if e['benchmark'] == b) for b in ('rbench', 'paibench')]
    identity = dict(cohort_sha256=sha(a.cohort), methods_sha256=sha(a.methods),
        source_sha256=sha(__file__), method=a.method, split=a.split, smoke=a.smoke)
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output/'manifest.json'
    state = dict(complete=False, identity=identity, cases={}, performance_eligible=False,
        scope='Full current-student attention interventions; same scenes cross all four phases and both CFG branches. '
              'Full M and actual amax. Diagnostic dense FP32 feature QK is separately counted and is not a deployable selector. '
              'Local NMSE and correlations cannot establish final-video or latency improvement.')
    if path.exists():
        state = json.loads(path.read_text())
        if state['identity'] != identity:
            raise ValueError('Changed resume identity')
        if state['complete']:
            return
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    with shared_worker():
        state['resource'] = select_shared_device()
        import torch
        from thunderworld_fp4 import fp4_attention, RepairConfig
        from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
        from thunderworld_fp4.cosmos.settings import phase
        from thunderworld_fp4.cosmos.budget import attention_cost
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        cfg = RunConfig.from_dict(dict(spec['config'], execution='eager', debug='off'))
        runner = CosmosRunner(cohort['model'], cfg, assets=spec['assets'], ar_cache=True, first_frame=True).prepare()
        active = [False]
        current = {}
        seen = set()
        preceding = {}
        common = dict(query_tile=32, transpose_v=True, sparse_pack=True)

        def callback(layer, q, k, v, actual):
            if not active[0] or runner.engine.warming:
                return
            e = runner.engine
            ph = phase(e.step, e.steps)
            key = f'{e.branch}|{ph}|{layer}'
            if key in seen:
                if key in preceding:
                    step, old = preceding.pop(key)
                    if e.step != step+1:
                        raise ValueError('Expected adjacent denoising calls')
                    current['panels'][key]['adjacent_step_change'] = {
                        name: float((x.float()-y.float()).square().sum().sqrt()/y.float().square().sum().sqrt().clamp_min(1e-20))
                        for name, x, y in zip(('q', 'k', 'v'), (q, k, v), old)}
                return
            seen.add(key)
            m, n, h, hk = q.shape[1], k.shape[1], q.shape[2], k.shape[2]
            mt, nt = math.ceil(m/128), math.ceil(n/128)
            roles = role_tiles(m, n)
            full = (nt,)*h
            zero = (0,)*h
            gold = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=True).transpose(1, 2).contiguous().float()
            denom = gold.square().sum().clamp_min(1e-20)
            errors, costs = {}, {}

            def observe(label, config, mask=None):
                pred = fp4_attention(q, k, v, config, repair_mask=mask)
                errors[label] = ((pred.float()-gold).square().sum((0, 1, 3))/denom).cpu().tolist()
                masks = None if mask is None else dict(qk=mask.cpu().tolist(), pv=mask.cpu().tolist())
                costs[label] = attention_cost(m, n, h, hk, qk_counts=config.head_counts(h, n),
                    pv_counts=config.pv_counts(h, n), qk_terms=config.terms,
                    pv_terms=config.pv_terms or config.terms, explicit_masks=masks)
                return pred

            three = RepairConfig(coverage=1., terms=3, **common)
            predicted = observe('three', three)
            if not torch.equal(predicted, actual):
                raise ValueError('Local anchor differs from current deployed student attention')
            observe('four', RepairConfig(coverage=1., terms=4, **common))
            observe('qk_four', RepairConfig(coverage=1., terms=4, pv_terms=3, **common))
            observe('pv_four', RepairConfig(coverage=1., terms=3, pv_terms=4, **common))
            observe('qk_zero', RepairConfig(coverage=0., head_quotas=zero, pv_head_quotas=full, **common))
            observe('pv_zero', RepairConfig(coverage=0., head_quotas=full, pv_head_quotas=zero, **common))
            observe('uniform_native_half', RepairConfig(coverage=.5, terms=3, **common))
            count = math.ceil(nt/2)
            # Native selector ENERGY is raw V energy, not quantization residual.
            # This baseline explicitly decodes the current represented base.
            energy = current_v_residual_energy(q, k, v)
            order = torch.argsort(energy, dim=-1, descending=True, stable=True)[:, :count]
            selected = torch.zeros((h, nt), device=q.device, dtype=torch.bool).scatter_(1, order, True)
            mask = selected[:, None, :].expand(h, mt, nt).contiguous()
            observe('current_v_residual_half', three, mask)
            roles_to_drop = dict(drop_ar_pure=roles['pure_ar'],
                drop_dm_matched=roles['pure_dm'][:len(roles['pure_ar'])], drop_mixed_boundary=roles['mixed'])
            if len(roles_to_drop['drop_dm_matched']) != len(roles['pure_ar']):
                raise ValueError('Not enough pure DM tiles for equal-cost role control')
            for label, indices in roles_to_drop.items():
                mask = torch.ones((h, mt, nt), device=q.device, dtype=torch.bool)
                mask[:, :, indices] = False
                observe(label, three, mask)
            if costs['drop_ar_pure']['total_mma'] != costs['drop_dm_matched']['total_mma']:
                raise AssertionError('AR/DM role intervention is not matched in issued MMA')
            # Features are diagnostic only; their dense matrix cost is explicit.
            prob = (torch.einsum('bmhd,bnhd->bhmn', q.float(), k.repeat_interleave(h//hk, 2).float())/math.sqrt(128)).softmax(-1)
            ar = roles['ar_tokens']
            mass = prob[..., :ar].sum(-1).mean((0, 2))
            current['panels'][key] = dict(layer=layer, phase=ph, branch=e.branch, step=e.step,
                shape=[m, n, h, hk], roles=roles, errors=errors, costs=costs,
                ar_mass=mass.cpu().tolist(), ar_density=(mass/ar).cpu().tolist(),
                dm_density=((1-mass)/m).cpu().tolist(),
                v_head_rms=v.float().square().mean((0, 1, 3)).sqrt().repeat_interleave(h//hk).cpu().tolist(),
                q_head_energy=q.float().square().mean((0, 1, 3)).cpu().tolist(),
                v_residual_energy=energy.sum(-1).cpu().tolist(),
                extra_diagnostic_work=dict(feature_qk_fp32_flops=2*m*n*h*128,
                    residual_feature_extra_base_pack_and_decode=True, deployment_selector_qualified=False),
                baseline_bitwise_equal=True)
            preceding[key] = (e.step, tuple(x.detach().clone() for x in (q, k, v)))

        try:
            runner.engine.capture_attention = callback
            for entry in entries:
                case = entry['case']['id']
                file = a.output/(case+'.json')
                if case in state['cases']:
                    if sha(file) != state['cases'][case]:
                        raise ValueError('Changed completed case')
                    continue
                if sha(entry['case']['image']) != entry['image_sha256']:
                    raise ValueError('Changed scene image')
                active[0] = False
                clear_case(runner)
                runner.warmup(entry['case'], 35)
                current = dict(complete=False, entry=entry, seed=entry['seeds'][0], panels={})
                seen.clear()
                preceding.clear()
                active[0] = True
                with torch.inference_mode():
                    runner.infer(entry['case'], entry['seeds'][0], 35)
                active[0] = False
                expected = {f'{b}|{p}|{l}' for b in range(2) for p in range(4) for l in range(28)}
                if set(current['panels']) != expected or any('adjacent_step_change' not in row for row in current['panels'].values()):
                    raise ValueError('Missing scene x phase x CFG or adjacent-step evidence')
                current['complete'] = True
                write(file, current)
                state['cases'][case] = sha(file)
                write(path, state)
                print(json.dumps(dict(complete_cases=len(state['cases']), total=len(entries))), flush=True)
            state['complete'] = True
            state.pop('paused', None)
            state.pop('error', None)
            write(path, state)
        except Exception as error:
            state.update(paused=True, error=repr(error))
            write(path, state)
            raise
        finally:
            active[0] = False
            runner.engine.capture_attention = None
            preceding.clear()
            runner.engine.clear()


if __name__ == '__main__':
    main()

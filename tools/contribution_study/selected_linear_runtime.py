"""Frozen selected linear policies with the ablation's identical base attention.

This is a composition of existing qualified kernels, not a new quantizer. The
selected production assets/linear schedules change; common BF16 operators and
the one-plane expansion attention used by the controls do not.
"""
import json
from pathlib import Path
import types

from baselines import sha

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def source_identity():
    files = [HERE / n for n in (
        'selected_linear_runtime.py', 'selected_linear_study.py', 'selected_linear_assets.py',
        'baseline_runtime.py', 'dataset_infer.py', 'expansion_kernels.py')]
    files += sorted((REPO / 'packages/fp4-repair/src/thunderworld_fp4').rglob('*.py'))
    return {str(p.relative_to(REPO)): sha(p) for p in files}


def verify(study, name):
    import baseline_runtime
    lock = json.loads(Path(study['selected_implementation_lock']).read_text())
    if not lock['complete'] or lock['sources'] != source_identity():
        raise ValueError('Changed frozen selected-policy implementation')
    baseline_runtime._selected_original_verify(study, name)
    if name == 'bf16':
        return
    spec = study['methods'][name]
    if spec['attention_repair'] is not False or spec.get('runtime_options'):
        raise ValueError('Only selected linear policy may change in this comparison')
    if spec['config']['fuse_ffn'] or spec['config']['share_pack']:
        raise ValueError('Common nonlinear/packing execution must remain fixed')
    plan = json.loads(Path(spec['config']['budget_plan']).read_text())
    origin = json.loads(Path(spec['selected_source_plan']).read_text())
    if sha(spec['selected_source_plan']) != spec['selected_source_plan_sha256']:
        raise ValueError('Changed original selected plan')
    if spec['repair_family'] == 'bilateral':
        if plan['linears'] != origin['linears']:
            raise ValueError('Selected linear supports/quotas changed')
    else:
        from selected_linear_study import side_plan
        bank = json.loads((Path(spec['assets'])/'manifest.json').read_text())
        expected = side_plan(origin, bank, spec['repair_family'])
        if plan['linears'] != expected['linears']:
            raise ValueError('Single-side endpoint is not full coverage')
        if not all(r['selected_base_bitwise_equal'] and r['selected_residual_bitwise_equal']
                   for r in bank['modules'].values()):
            raise ValueError('Single-side assets differ from selected representation')
    if any(row.get('cross', False) for row in plan['linears'].values()):
        raise ValueError('Unexpected cross correction')
    for row in plan['attention'].values():
        if row != dict(qk_counts=[0]*16, pv_counts=[0]*16, qk_terms=2, pv_terms=2):
            raise ValueError('Attention correction leaked into selected-policy ablation')


def compile_budget(engine):
    from thunderworld_fp4.cosmos.engine import JointEngine
    ledger = JointEngine.compile_budget(engine)
    rows = ledger['rows']
    linear = [r for r in rows if r['kind'] == 'linear']
    attention = [r for r in rows if r['kind'] == 'attention']
    lb = sum(r['base_mma']*r['calls'] for r in linear)
    lt = sum(r['total_mma']*r['calls'] for r in linear)
    if not lb or lt > engine.config.compute_budget*lb:
        raise ValueError('Selected policy exceeds linear MMA cap')
    if not attention or any(r['total_mma'] != r['base_mma'] for r in attention):
        raise ValueError('Expected uncorrected dense FP4 attention')
    if any(r['launch']['bm'] != 128 for r in linear):
        raise ValueError('Changed linear execution tile relative to controls')
    for r in attention:
        r['execution_backend'] = 'frozen_expansion_one_plane'
        r['packing_implementation'] = 'one_plane_no_residuals'
    ledger.update(linear_base_mma=lb, linear_total_mma=lt,
        linear_ratio=lt/lb, linear_budget=engine.config.compute_budget,
        attention_ratio=1., selector_mma=0, cross_mma=0,
        performance_eligible=False,
        scope='Generation projection/FFN and QK/PV issued MMA, all padding and 35 steps; common BF16 AR/nonlinear operators outside C0',
        comparison='Selected linear policy and offline representation versus fixed-quota controls; not a selection-only or same-bank ablation')
    return ledger


def make_runner(study, name):
    import baseline_runtime
    verify(study, name)
    runner, adapter = baseline_runtime._selected_original_make(study, name)
    if name == 'bf16':
        return runner, adapter
    from expansion_kernels import attention
    from thunderworld_fp4.cosmos.engine import phase

    def attend(engine, layer, q, k, v):
        cfg = engine.attention_config(layer, engine.step, engine.branch)
        key = (layer, tuple(q.shape), tuple(k.shape), cfg, engine.branch)
        if engine.warming:
            if any(cfg.head_counts(q.shape[2], k.shape[1])) or any(cfg.pv_counts(q.shape[2], k.shape[1])):
                raise ValueError('Attention repairs must remain disabled')
            engine.attention_prepared[key] = (cfg, None)
            engine.attention_shapes[engine.task, engine.branch, phase(engine.step, engine.steps), layer] = dict(
                q=list(q.shape), k=list(k.shape), attention_repair=False)
        elif key not in engine.attention_prepared:
            raise RuntimeError('Unwarmed base attention')
        return attention(q, k, v, 1, 1, 1)

    runner.engine.attend = types.MethodType(attend, runner.engine)
    runner.engine.compile_budget = types.MethodType(compile_budget, runner.engine)
    spec = study['methods'][name]
    adapter = dict(method='selected_linear_fixed_base_attention',
        origin=spec['origin'], attention_repair=False,
        attention_expansion=dict(weight_planes=1, activation_planes=1, mask=1, terms=1),
        source_sha256=source_identity(), assets_sha256=spec['assets_sha256'],
        plan_sha256=spec['plan_sha256'], linear_budget=spec['budget'],
        repair_family=spec['repair_family'],
        linear_policy_unchanged=spec['repair_family']=='bilateral', offline_w0_unchanged=True,
        common_non_mma_path=True, current_input_activation_selection=True,
        performance_eligible=False,
        limitation='Production linear quantization/prescale differs from the expansion bank; report complete selected-linear policy, not isolated bilateral support benefit')
    return runner, adapter


def install():
    import baseline_runtime
    baseline_runtime._selected_original_verify = baseline_runtime.verify
    baseline_runtime._selected_original_make = baseline_runtime.make_runner
    baseline_runtime.verify, baseline_runtime.make_runner = verify, make_runner

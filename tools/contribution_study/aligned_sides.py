"""Align one-sided linear repair and joint policies under FP4 ACT2 attention."""
import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

from baselines import sha, write
from importance_workflow import import_method, run
from schedule_envelope import KEY_PAIRS, compile_envelope


NEW_METHODS = ('single_a_act2', 'single_w_act2', 'joint2_act2', 'joint1.5_act2')


def act2_plan(source, shapes, side=None):
    value = copy.deepcopy(source)
    if side not in (None, 'a', 'w'):
        raise ValueError('Unknown linear repair side')
    if side:
        value['budget'] = 2.
        for key in value['linears']:
            name = key.split('|', 2)[-1]
            _, n, k = shapes['linears'][name]
            value['linears'][key] = dict(a_blocks=k // 128 if side == 'a' else 0,
                w_blocks=list(range(n // 128 * (k // 128))) if side == 'w' else [], cross=False)
    value['attention'] = {key: dict(full_attention=True, qk_terms=2, pv_terms=2)
                          for key in value['attention']}
    value['design'] = dict(attention='full ACT2: Q0K0 + Q1K0; P0V0 + P1V0',
        linear_repair=side or 'unchanged frozen joint policy', trainable=False,
        scope='One-sided refers to generated projection/FFN weights and activations; K/V are activations, not model weights.')
    return value


def prepare(args):
    from thunderworld_fp4.cosmos import RunConfig
    from thunderworld_fp4 import RepairConfig, RepairPolicy
    old = json.loads((args.joint / 'development/manifest.json').read_text())
    side = json.loads((args.importance / 'two_term/manifest.json').read_text())
    if old['model'] != side['model'] or old['cases'] != side['cases']:
        raise ValueError('Matched model, scenes, inputs and seeds required')
    args.output.mkdir(parents=True, exist_ok=False)
    methods = {name: old['methods'][name] for name in
               ('bf16', 'fp8', 'fp4_three', 'b1.5_protected_n', 'b2_joint')}
    methods.update({name: side['methods'][name] for name in ('linear_a_only', 'linear_w_only')})
    sources = {name: str((args.joint / 'development').resolve()) for name in methods if name in old['methods']}
    sources.update({name: str((args.importance / 'two_term').resolve()) for name in ('linear_a_only', 'linear_w_only')})
    checks = {}
    cfg_attn = RepairPolicy(RepairConfig(coverage=1., terms=2, query_tile=32,
                                       transpose_v=True, sparse_pack=True))
    for name, origin, repair_side in (
        ('single_a_act2', 'linear_a_only', 'a'), ('single_w_act2', 'linear_w_only', 'w'),
        ('joint2_act2', 'b2_joint', None), ('joint1.5_act2', 'b1.5_protected_n', None)):
        cap = 1.5 if origin == 'b1.5_protected_n' else 2.
        definitions = json.loads((args.joint / 'calibration' / f'b{cap:g}' / 'definitions.json').read_text())
        template = old['methods']['b1.5_protected_n' if cap == 1.5 else 'b2_joint']
        plan = act2_plan(json.loads(Path(template['config']['budget_plan']).read_text()),
                         definitions['shapes'], repair_side)
        folder = args.output / 'policies' / name
        folder.mkdir(parents=True)
        write(folder / 'plan.json', plan)
        cfg = copy.deepcopy(methods[origin]['config'])
        cfg.update(attention_backend='fp4', compute_budget=cap,
                   budget_plan=str((folder / 'plan.json').resolve()), attention=asdict(cfg_attn))
        RunConfig.from_dict(cfg)
        spec = dict(config=cfg, budget=cap, plan_sha256=sha(folder / 'plan.json'),
                    origin=origin, attention_control='full ACT2',
                    scope='Changed attention only; linear numerical representation and repair policy retained.')
        if repair_side:
            spec['scope'] = 'Original RTN two-term linear arithmetic, now a compact explicit support plan; full ACT2 attention. Exactly 2x whole-request MMA.'
        else:
            for key in ('assets', 'assets_sha256', 'cost'):
                spec[key] = methods[origin][key]
        ratios = []
        for pair in KEY_PAIRS:
            ledger = compile_envelope(plan, definitions['shapes'], pair)
            if repair_side and ledger['total_mma'] != 2 * ledger['base_mma']:
                raise AssertionError('Single-sided baseline must be exactly 2x, including tails')
            ratios.append(dict(key_tiles_by_cfg=list(pair), ratio=ledger['ratio']))
        checks[name] = ratios
        methods[name] = spec
    manifest = dict(schema=1, frozen=True, model=old['model'], cases=old['cases'], methods=methods,
        steps=35, debug='off', execution='graph', resource_mode='shared-calibration',
        role='development', selection_eligible=False, parent_manifest_sha256=sha(args.joint / 'development/manifest.json'),
        side_manifest_sha256=sha(args.importance / 'two_term/manifest.json'), imports=sources,
        primary_comparisons=[dict(candidate=c, reference=r) for c in
            ('b2_joint', 'b1.5_protected_n', 'joint2_act2', 'joint1.5_act2') for r in ('single_a_act2', 'single_w_act2')],
        scope='Fixed-method alignment on 16 development scenes, two seeds each. No search, no validation/final use, no shared-GPU speed claim.')
    write(args.output / 'manifest.json', manifest)
    write(args.output / 'budget_checks.json', dict(complete=True, checks=checks,
        shape_scope='M960, 28 layers, 168 generated linears, GQA H16/Hkv8, independent CFG key lengths 8..16 tiles',
        performance_eligible=False))
    print(json.dumps(dict(complete=True, new_methods=list(NEW_METHODS), shape_checks=sum(map(len, checks.values())))))


def execute(args):
    root = args.output
    manifest = root / 'manifest.json'
    value = json.loads(manifest.read_text())
    probe = json.loads((root / 'probe.json').read_text())
    if not probe['complete']:
        raise ValueError('Numerical endpoint qualification required')
    for name, source in value['imports'].items():
        import_method(Path(source), root, name, manifest)
    run(manifest, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--joint', type=Path)
    parser.add_argument('--importance', type=Path)
    args = parser.parse_args()
    (prepare if args.operation == 'prepare' else execute)(args)


if __name__ == '__main__':
    main()

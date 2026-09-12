"""Offline residual-support storage. Expansion is preparation-only on CPU."""
import math
import torch
from .settings import TIERS


def required_weight_fraction(config, name, policy=None):
    if config.budget_plan:
        from .budget import RepairPlan
        plan=RepairPlan.load(config.budget_plan)
        rows=[r['w_blocks'] for key,r in plan.linears.items() if key.split('|',2)[-1]==name]
        if not rows:raise ValueError(f'Missing budget support: {name}')
        return sorted(set(i for ids in rows for i in ids))
    default = config.ffn if '.mlp_moe_gen.' in name else config.projection
    if policy is None or '.mlp_moe_gen.' not in name:
        return default.weight
    parts = name.split('.')
    suffix = f"|{parts[1]}.{parts[-1]}"
    assignments = policy['policies'][config.ffn_policy_mode]['assignments']
    values = [TIERS[tier].weight for key, tier in assignments.items() if key.endswith(suffix)]
    if not values:
        raise ValueError(f'Missing offline support policy for {name}')
    return max(values)


def compact(asset, fraction):
    score = asset['score'].flatten()
    explicit=isinstance(fraction,(list,tuple,set))
    count = len(fraction) if explicit else math.ceil(score.numel() * fraction)
    if explicit:
        from .budget import support
        support(fraction,score.numel())
    indices = torch.tensor(sorted(fraction),dtype=torch.long) if explicit else torch.argsort(score, descending=True, stable=True)[:count]
    if not set(asset.get('mandatory_w_blocks',())).issubset(indices.tolist()):
        raise ValueError('Cannot discard mandatory outlier support')
    result = {k: v for k, v in asset.items() if k not in ('codes', 'scales')}
    result.update(storage_schema=2 if explicit else 1, weight_support_fraction=count/score.numel(), residual_indices=indices)
    nb, kb = asset['score'].shape
    for name in ('codes', 'scales'):
        value = asset[name]
        width = value.shape[-1] // kb
        blocks = value[1].view(torch.uint8).reshape(nb, 128, kb, width).permute(0, 2, 1, 3)
        result[name + '_base'] = value[0].clone()
        result[name + '_residual'] = blocks.reshape(nb * kb, 128, width)[indices].contiguous()
    return result


def expand(asset, required):
    if 'storage_schema' not in asset:
        return asset
    explicit=isinstance(required,(list,tuple,set))
    insufficient=(not set(required).issubset(asset['residual_indices'].tolist())) if explicit else required>asset['weight_support_fraction']
    if asset['storage_schema'] not in (1,2) or insufficient or (asset['storage_schema']==2 and not explicit):
        raise ValueError('Offline W1 support is insufficient; prepare this policy again')
    result = {k: v for k, v in asset.items() if k not in (
        'storage_schema', 'weight_support_fraction', 'residual_indices',
        'codes_base', 'codes_residual', 'scales_base', 'scales_residual')}
    nb, kb = asset['score'].shape
    for name in ('codes', 'scales'):
        base = asset[name + '_base']
        width = base.shape[-1] // kb
        blocks = torch.zeros((nb * kb, 128, width), dtype=torch.uint8)
        blocks[asset['residual_indices']] = asset[name + '_residual']
        residual = blocks.reshape(nb, kb, 128, width).permute(0, 2, 1, 3).reshape(base.shape)
        result[name] = torch.stack((base.view(torch.uint8), residual)).view(base.dtype)
    return result

"""Offline residual-support storage. Expansion is preparation-only on CPU."""
import math
import torch
from .settings import TIERS


def required_weight_fraction(config, name, policy=None):
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
    count = math.ceil(score.numel() * fraction)
    indices = torch.argsort(score, descending=True, stable=True)[:count]
    result = {k: v for k, v in asset.items() if k not in ('codes', 'scales')}
    result.update(storage_schema=1, weight_support_fraction=fraction, residual_indices=indices)
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
    if asset['storage_schema'] != 1 or required > asset['weight_support_fraction']:
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

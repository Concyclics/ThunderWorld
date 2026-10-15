"""Independent FP4 mechanism reference; this is NOT a native inference benchmark.

Every residual is formed from the represented base, before residual quantization.
An outer factor is shared between both planes, matching ThunderWorld packing.
"""
import torch
from thunderworld_fp4.cosmos.calibration import _encode, decode


def encode(z, rule='rtn'):
    c6, s6 = _encode(z, 1.)
    if rule == 'rtn':
        return c6, s6
    if rule != 'four_six':
        raise ValueError(rule)
    # Mechanism reproduction of 4/6 block scaling, not upstream qualification.
    c4, s4 = _encode(z, 1.5)
    d6 = decode(c6, s6, 1.).reshape(*s6.shape, 16)
    d4 = decode(c4, s4, 1.).reshape(*s4.shape, 16)
    target = z.reshape(*s6.shape, 16)
    use4 = (d4-target).square().sum(-1) < (d6-target).square().sum(-1)
    codes = torch.where(use4.repeat_interleave(8, -1), c4, c6)
    scales = torch.where(use4, s4.float(), s6.float()).to(s6.dtype)
    return codes, scales


def planes(value, rule='rtn', remove=None):
    value = value.float()
    peak = value.abs().amax()
    factor = torch.where(peak > 0, peak / 2688., torch.ones_like(peak))
    z = value / factor
    c0, s0 = encode(z if remove is None else z.masked_fill(remove, 0.), rule)
    base = decode(c0, s0, 1.)
    c1, s1 = encode(z-base, rule)
    return dict(codes=torch.stack((c0,c1)), scales=torch.stack((s0,s1)), factor=factor,
                base=base*factor, residual=decode(c1,s1,1.)*factor)


def mask_from_indices(indices, active, shape):
    result = torch.zeros(shape, device=indices.device, dtype=torch.bool)
    result.reshape(*indices.shape,16).scatter_(-1, indices[...,None], active[...,None])
    return result


def removal_masks(w, rms, allowed, threshold=2.):
    """Same contribution threshold and allowed W1 blocks for every selector.

    Joint selection uses a diagonal activation-covariance surrogate, explicitly
    distinct from actual full-panel output error used to accept candidates.
    """
    n,k = w.shape
    if n % 128 or k % 128 or allowed.shape != (n//128,k//128):
        raise ValueError('Expected N128/K128 reserved support')
    groups = w.reshape(n,k//16,16).float()
    contribution = groups.abs()*rms.reshape(1,k//16,16)
    active = contribution.amax(-1) > threshold*contribution.square().mean(-1).sqrt()
    active &= allowed.repeat_interleave(128,0).repeat_interleave(8,1)
    base = planes(w)
    error = (w-base['base']).abs().reshape_as(groups)
    masks = {
        'magnitude': mask_from_indices(groups.abs().argmax(-1),active,w.shape),
        'quant_error': mask_from_indices(error.argmax(-1),active,w.shape),
    }
    best = ((w-base['base']-base['residual']).reshape_as(groups)*rms.reshape(1,k//16,16)).square().sum(-1)
    selected = torch.zeros_like(best,dtype=torch.long)
    accepted = torch.zeros_like(active)
    # Coordinate proposal at constant tensor amax. No input-row subsampling.
    for index in range(16):
        ids = torch.full_like(selected,index)
        p = planes(w,remove=mask_from_indices(ids,active,w.shape))
        loss = ((w-p['base']-p['residual']).reshape_as(groups)*rms.reshape(1,k//16,16)).square().sum(-1)
        improved = active & (loss < best)
        selected = torch.where(improved,ids,selected)
        accepted |= improved
        best = torch.minimum(loss,best)
    masks['joint_diagonal'] = mask_from_indices(selected,accepted,w.shape)
    return masks


def terms(a, w):
    """FP32 decoded reference with TF32 disabled by caller; includes cross once."""
    base = a['base'] @ w['base'].T
    ar = a['residual'] @ w['base'].T
    wr = a['base'] @ w['residual'].T
    cross = a['residual'] @ w['residual'].T
    return dict(base=base, act2=base+ar, weight2=base+wr,
                three=base+ar+wr, four=base+ar+wr+cross)


def storage_union(asset, supports):
    """Physically compact CPU W1; callers must not deploy via dense expand()."""
    from thunderworld_fp4.cosmos.assets import compact
    union = sorted(set(i for support in supports for i in support))
    mandatory = set(asset.get('mandatory_w_blocks', ()))
    if any(not mandatory.issubset(s) for s in map(set,supports)):
        raise ValueError('A phase disables mandatory W1 support')
    packed = compact(asset, union)
    sizes = {k:v.numel()*v.element_size() for k,v in packed.items() if isinstance(v,torch.Tensor)}
    return packed, dict(union=union, union_density=len(union)/asset['score'].numel(),
                        tensor_bytes=sizes, total_tensor_bytes=sum(sizes.values()),
                        runtime_qualified=False)

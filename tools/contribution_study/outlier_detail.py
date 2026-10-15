"""Fixed-support K16 outlier ablation; actual FP4/FP8 coding, no gradients.

Search sees only fit-panel covariance. All 16 possible removed positions compete
with no removal; residuals always reference the represented base and original W.
Native whole-operator fit checks and separate review panels live in the runner.
"""
import torch
from thunderworld_fp4.cosmos.calibration import _encode, decode

METHODS = ('scale_only', 'max_scale', 'joint', 'removal_only', 'max_fixed')
BASE_SCALES = (.75, .875, 1., 1.5)
RES_SCALES = (.875, 1., 1.125)


def losses(z, w0, w1, cov):
    v = torch.cat((z.expand_as(w0), z-w0-w1, w1), -1)
    return torch.einsum('pgd,gde,pge->pg', v, cov, v)


def search(z, c0, s0, c1, s1, cov):
    if z.ndim != 2 or z.shape[1] != 16 or cov.shape != (len(z), 48, 48):
        raise ValueError('Expected complete K16 groups and actual-input covariance')
    g = len(z)
    original0, original1 = decode(c0, s0, 1.), decode(c1, s1, 1.)
    initial = losses(z, original0[None], original1[None], cov)[0]
    result = {label: dict(loss=initial.clone(), c0=c0.clone(), s0=s0.clone(),
                         c1=c1.clone(), s1=s1.clone(),
                         removed=torch.full((g,), -1, device=z.device, dtype=torch.long),
                         base_scale=torch.ones(g, device=z.device),
                         residual_scale=torch.ones(g, device=z.device))
              for label in METHODS}
    inputs = z[None].expand(17, -1, -1).clone()
    for j in range(16):
        inputs[j+1, :, j] = 0
    rows = torch.arange(g, device=z.device)
    max_position = z.abs().argmax(-1)+1
    for bs in BASE_SCALES:
        nc0, ns0 = _encode(inputs.reshape(-1, 16), bs)
        w0 = decode(nc0, ns0, 1.).reshape(17, g, 16)
        for rs in RES_SCALES:
            nc1, ns1 = _encode((z[None]-w0).reshape(-1, 16), rs)
            w1 = decode(nc1, ns1, 1.).reshape(17, g, 16)
            score = losses(z, w0, w1, cov)
            choices = dict(scale_only=torch.zeros_like(rows), joint=score.argmin(0))
            choices['max_scale'] = torch.where(score[max_position, rows] < score[0], max_position, 0)
            if bs == 1. and rs == 1.:
                choices.update(removal_only=score.argmin(0), max_fixed=max_position)
            for label, positions in choices.items():
                state = result[label]
                value = score[positions, rows]
                take = torch.ones(g, device=z.device, dtype=torch.bool) if label == 'max_fixed' else value < state['loss']
                state['loss'][take] = value[take]
                for key, tensor in (('c0', nc0), ('s0', ns0), ('c1', nc1), ('s1', ns1)):
                    chosen = tensor.view(torch.uint8).reshape(17, g, -1)[positions, rows]
                    state[key].view(torch.uint8)[take] = chosen[take]
                state['removed'][take] = positions[take]-1
                state['base_scale'][take] = bs
                state['residual_scale'][take] = rs
    return result


def group_indices(blocks, k):
    nr = (blocks//(k//128))[:, None]*128+torch.arange(128, device=blocks.device)
    kg = (blocks%(k//128))[:, None]*8+torch.arange(8, device=blocks.device)
    return nr[:, :, None].expand(-1, -1, 8).reshape(-1), kg[:, None, :].expand(-1, 128, -1).reshape(-1)


def patch_asset(original, proposal, nr, kg, blocks):
    """Copy only sampled groups; W1 support and every other encoded byte stay fixed."""
    n, packed_k = original['codes'].shape[1:]
    k = packed_k*2
    codes = original['codes'].clone()
    scales = original['scales'].clone()
    for plane in range(2):
        codes[plane].reshape(n, k//16, 8)[nr, kg] = proposal['c'+str(plane)]
        scales[plane].view(torch.uint8)[nr, kg] = proposal['s'+str(plane)].view(torch.uint8).flatten()
    mandatory = blocks[(proposal['removed'] >= 0).reshape(-1, 1024).any(1)].tolist()
    return dict(original, codes=codes, scales=scales, mandatory_w_blocks=tuple(mandatory),
                offline_encoder='outlier_detail_k16')


def diagnostics(z, proposal, original, scale):
    """Per-group raw sums: aggregate by operator, not fictitious iid elements."""
    w0, w1 = (decode(proposal['c'+str(i)], proposal['s'+str(i)], 1.) for i in range(2))
    old0 = decode(original['c0'], original['s0'], 1.)
    scale0 = decode(scale['c0'], scale['s0'], 1.)
    mask = torch.ones_like(z, dtype=torch.bool)
    removed = proposal['removed']
    rows = torch.arange(len(z), device=z.device)
    active = removed >= 0
    mask[rows[active], removed[active]] = False
    ratio = z.abs().amax(-1)/z.square().mean(-1).sqrt().clamp_min(1e-20)
    return dict(energy=z.square().sum(-1), base_sse=(z-w0).square().sum(-1),
                dual_sse=(z-w0-w1).square().sum(-1),
                inlier_energy=(z.square()*mask).sum(-1),
                inlier_sse=((z-w0).square()*mask).sum(-1),
                inlier_rtn_sse=((z-old0).square()*mask).sum(-1),
                inlier_scale_sse=((z-scale0).square()*mask).sum(-1),
                outlier_ratio=ratio, removed=removed,
                removed_is_max=active & (removed == z.abs().argmax(-1)),
                base_scale_changed=(proposal['s0'].view(torch.uint8) != original['s0'].view(torch.uint8)).flatten(),
                base_multiplier=proposal['base_scale'], residual_multiplier=proposal['residual_scale'])

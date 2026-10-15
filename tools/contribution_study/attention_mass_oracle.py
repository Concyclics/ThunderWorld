"""Privileged current-input attention-mass selector, not a repair-error oracle."""
import math

import numpy as np

from structure_repair import ranked_mask


def attention_tile_mass(q, k, tile=128):
    """Full FP32 softmax, all real rows/keys, GQA, no causal mask or sampling.

    Returns [H, ceil(M/tile), ceil(N/tile)] mean per-query probability mass.
    Chunking only the query dimension does not alter the softmax domain.
    The additional FP32 QK/softmax is privileged analysis work, outside the
    matched FP4 repair-execution budget; it is not a deployable free selector.
    """
    import torch
    import torch.nn.functional as F

    if (q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1
            or q.shape[-1] != k.shape[-1] or q.device != k.device
            or q.shape[2] % k.shape[2] or min(*q.shape, *k.shape, tile) < 1):
        raise ValueError('Expected nonempty batch-one Q/K with compatible GQA heads')
    if q.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('Oracle requires full FP32 QK; disable TF32')
    m, n, h, d = q.shape[1], k.shape[1], q.shape[2], q.shape[3]
    nt = math.ceil(n / tile)
    key = k[0].transpose(0, 1).float().repeat_interleave(h // k.shape[2], 0)
    scores = []
    for start in range(0, m, tile):
        query = q[0, start:start + tile].transpose(0, 1).float()
        prob = (query @ key.transpose(1, 2) / math.sqrt(d)).softmax(-1)
        blocks = F.pad(prob, (0, nt * tile - n)).reshape(h, len(query[0]), nt, tile)
        scores.append(blocks.sum(-1).mean(1))
    return torch.stack(scores, 1)


def mass_capture(scores, mask, m, tile=128):
    """Fraction of all real-query probability mass covered (no padded queries)."""
    scores = np.asarray(scores, np.float64)
    mask = np.asarray(mask, bool)
    if scores.shape != mask.shape or scores.shape[1] != math.ceil(m / tile):
        raise ValueError('Score, mask or real-query count mismatch')
    sizes = np.minimum(tile, m - tile * np.arange(scores.shape[1]))
    return float((scores * mask * sizes[None, :, None]).sum() / (scores.shape[0] * m))


def oracle_mask(scores, coverage, seed):
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Finite nonnegative probability mass required')
    if not np.allclose(scores.sum(-1), 1., rtol=2e-6, atol=2e-6):
        raise ValueError('Each real head/query tile must include the full key softmax')
    return ranked_mask(scores, coverage, seed)


def selector_cost(m, n, h, d, tile=128):
    return dict(fp32_qk_flops=2 * m * n * h * d, softmax_real_elements=h * m * n,
                fp32_probability_tile_bytes=4 * h * min(tile, m) * n,
                emitted_mass_bytes=4 * h * math.ceil(m / tile) * math.ceil(n / tile),
                ranking_candidates_per_query_tile=h * math.ceil(n / tile),
                included_in_matched_fp4_budget=False,
                scope='Additional privileged FP32 QK, softmax, reduction, host transfer and sorting; no timing claim')


def paired_gap(structure, oracle, base, replicates=10000):
    """Pair seeds within fixed scenes; panels/heads are not independent samples."""
    a, b, c = [np.asarray(x, np.float64) for x in (structure, oracle, base)]
    if a.shape != b.shape or a.shape != c.shape or a.ndim != 2 or min(a.shape) < 1:
        raise ValueError('Expected equal nonempty [scene,seed] arrays')
    if not all(np.isfinite(x).all() for x in (a, b, c)) or (b <= 0).any():
        raise ValueError('Finite errors and positive oracle error required')
    rng = np.random.default_rng(924072)
    indices = rng.integers(0, a.shape[1], size=(replicates, a.shape[0], a.shape[1]))
    scene = np.arange(a.shape[0])[None, :, None]
    aa, bb = [x[scene, indices].mean((1, 2)) for x in (a, b)]
    gap = 100 * (aa / bb - 1)
    benefit = c.mean() - b.mean()
    return dict(structure_nmse=float(a.mean()), mass_oracle_nmse=float(b.mean()),
                nmse_gap_pct=float(100 * (a.mean() / b.mean() - 1)),
                nmse_gap_ci95_pct=np.quantile(gap, [.025, .975]).tolist(),
                difference_ci95=np.quantile(aa - bb, [.025, .975]).tolist(),
                difference_ci_familywise95=np.quantile(aa - bb, [.05 / 6, 1 - .05 / 6]).tolist(),
                oracle_benefit_recovered_pct=(float(100 * (c.mean() - a.mean()) / benefit) if benefit > 0 else None),
                structure_wins=int((a < b).sum()), requests=int(a.size),
                worst_request_gap_pct=float((100 * (a / b - 1)).max()),
                scope='Paired seed bootstrap, fixed scenes; familywise interval covers three budgets per cohort. Benefit recovery is not clipped at 100%.')

"""Matched-budget attention repair masks and a frozen descriptive role prior.

No BF16 output, test-set error, or current attention probability enters the
structure selector. Query/key roles are directional and mixed tiles stay whole.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROLES = ('ar', 'vision_clean', 'vision_noisy', 'sound', 'action', 'special')
BOUNDS = (0, 7, 18, 28, 35)
COVERAGES = (.25, .5, .75)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fit_prior(paths):
    """Only historical development observations, conditional CFG branch 0."""
    summaries = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            if tuple(z['roles']) != ROLES:
                raise ValueError('Role convention mismatch')
            mass = z['mass'][:, 0].astype(np.float64)
            counts = z['counts'][0].astype(np.float64)
            density = mass / np.where(counts > 0, counts, np.nan)
            phases = []
            for lo, hi in zip(BOUNDS[:-1], BOUNDS[1:]):
                x = density[lo:hi]
                count = np.isfinite(x).sum(0)
                phases.append(np.divide(np.nansum(x, 0), count,
                                        out=np.full_like(x[0], np.nan), where=count > 0))
            summaries.append(np.stack(phases))
    values = np.stack(summaries)
    count = np.isfinite(values).sum(0)
    return np.divide(np.nansum(values, 0), count,
                     out=np.zeros_like(values[0]), where=count > 0)


def role_counts(labels):
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer)
            or labels.min() < 0 or labels.max() >= len(ROLES)):
        raise ValueError('Nonempty integer role layout required')
    result = np.zeros((math.ceil(len(labels) / 128), len(ROLES)), np.int64)
    for row in range(len(result)):
        result[row] = np.bincount(labels[row * 128:(row + 1) * 128], minlength=len(ROLES))
    return result


def tile_scores(density, query_roles, key_roles):
    density = np.asarray(density, np.float64)
    if density.ndim != 3 or density.shape[1:] != (6, 6) or not np.isfinite(density).all() or (density < 0).any():
        raise ValueError('Expected nonnegative [heads,query_role,key_role] density')
    qc, kc = role_counts(query_roles), role_counts(key_roles)
    return np.einsum('qr,hrs,ks->hqk', qc / qc.sum(1, keepdims=True), density, kc)


def quota(shape, coverage):
    if len(shape) != 3 or min(shape) < 1 or not 0 <= coverage <= 1:
        raise ValueError('Invalid tile shape or coverage')
    return math.floor(shape[0] * shape[2] * coverage)


def ranked_mask(scores, coverage, seed=44071):
    """Same H*K tile quota in every M128 row, so padded MMA is exactly matched.

    A fixed random order only breaks exactly equal descriptive feature scores.
    It is independent of calibration errors, test inputs, and model RNG state.
    """
    scores = np.asarray(scores)
    count = quota(scores.shape, coverage)
    mask = np.zeros_like(scores, dtype=bool)
    rng = np.random.default_rng(seed)
    for row in range(scores.shape[1]):
        flat = scores[:, row].reshape(-1)
        order = np.lexsort((rng.random(len(flat)), -flat))
        chosen = np.zeros(len(flat), dtype=bool)
        chosen[order[:count]] = True
        mask[:, row] = chosen.reshape(scores.shape[0], scores.shape[2])
    return mask


def random_mask(shape, coverage, seed):
    return ranked_mask(np.zeros(shape), coverage, seed)


def window_mask(shape, coverage, m, n):
    """Bidirectional token-local window centered on the true global query index.

    The AR prefix shifts DM query positions. Each head gets floor/ceil(K*f)
    tiles, with the integer remainder rotated by M row. No base block is cut.
    """
    h, mt, nt = shape
    if mt != math.ceil(m / 128) or nt != math.ceil(n / 128) or n < m:
        raise ValueError('Shape/layout mismatch')
    count = quota(shape, coverage)
    mask = np.zeros(shape, bool)
    centers = (np.arange(nt) * 128 + np.minimum(np.arange(1, nt + 1) * 128, n) - 1) / 2
    for row in range(mt):
        center = n - m + (row * 128 + min((row + 1) * 128, m) - 1) / 2
        order = np.argsort(np.abs(centers - center), kind='stable')
        for head in range(h):
            take = count // h + ((head - row) % h < count % h)
            mask[head, row, order[:take]] = True
    return mask


def candidates(prior, phase, layer, query_roles, key_roles, coverage, random_repeats=8):
    scores = tile_scores(prior[phase, layer], query_roles, key_roles)
    h, mt, nt = scores.shape
    seed = 44071 + 1009 * layer + 7919 * phase
    result = {
        'structure': ranked_mask(scores, coverage, seed),
        'structure_no_head': ranked_mask(tile_scores(np.broadcast_to(prior[phase, layer].mean(0), (h, 6, 6)), query_roles, key_roles), coverage, seed),
        'structure_no_step': ranked_mask(tile_scores(prior[:, layer].mean(0), query_roles, key_roles), coverage, seed),
        'window': window_mask(scores.shape, coverage, len(query_roles), len(key_roles)),
    }
    for repeat in range(random_repeats):
        result[f'random_{repeat:02d}'] = random_mask((h, mt, nt), coverage, seed + 104729 * (repeat + 1))
    expected = quota(scores.shape, coverage)
    if not all(np.all(mask.sum((0, 2)) == expected) for mask in result.values()):
        raise AssertionError('M128 row budget mismatch')
    return result


def mask_cost(mask, m, n, hk, terms=3):
    from thunderworld_fp4.cosmos.budget import attention_cost
    return attention_cost(m, n, mask.shape[0], hk, bm=32, qk_terms=terms, pv_terms=terms,
                          selector=False, explicit_masks=dict(qk=mask.tolist(), pv=mask.tolist()))


def mask_summary(mask, query_roles, key_roles, hk):
    qc, kc = role_counts(query_roles), role_counts(key_roles)
    selected = np.einsum('hmn,mr,ns->rs', mask.astype(np.float64), qc, kc)
    total = mask.shape[0] * np.outer(qc.sum(0), kc.sum(0))
    fraction = np.divide(selected, total, out=np.zeros_like(selected), where=total > 0)
    union = mask.reshape(hk, mask.shape[0] // hk, *mask.shape[1:]).any((1, 2))
    return dict(selected_tiles=int(mask.sum()), total_tiles=int(mask.size),
                gqa_kv_union_density=float(union.mean()), relation_repair_fraction=fraction.tolist(),
                mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())

"""Explicit import/compilation helpers for calibrated head findings."""
import math
from .config import RepairConfig


def config_from_head_scores(scores, *, protected_count, coverage=0.5,
                            protected_coverage=0.75, terms=3):
    """Offline ranking; ties favor lower head index. Scores must be calibrated.

    This changes head quotas, rather than multiplying all within-head scores
    by a constant (which cannot alter a within-head top-k). It does not claim
    that projection sensitivity predicts attention repair benefit.
    """
    scores = tuple(scores)
    if not scores or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores):
        raise ValueError("scores must contain finite per-query-head values")
    if type(protected_count) is not int or not 0 <= protected_count <= len(scores):
        raise ValueError("protected_count must be between 0 and the number of heads")
    heads = sorted(range(len(scores)), key=lambda h: (-scores[h], h))[:protected_count]
    return RepairConfig(coverage=coverage, terms=terms,
                        head_budgets=tuple((h, protected_coverage) for h in heads))


def make_repair_mask(query_tokens, key_tokens, query_heads, head_blocks, *, device):
    """Build exact per-head repair support from {head: [(q_block, k_block), ...]}.

    Block indices refer to 128-token blocks. Unlisted blocks retain their FP4
    base products. This metadata helper runs outside CUDA Graph capture.
    """
    import torch
    for name, value in (("query_tokens", query_tokens), ("key_tokens", key_tokens), ("query_heads", query_heads)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    shape = (query_heads, math.ceil(query_tokens / 128), math.ceil(key_tokens / 128))
    mask = torch.zeros(shape, dtype=torch.bool)
    for head, blocks in head_blocks.items():
        if type(head) is not int or not 0 <= head < shape[0]:
            raise ValueError("head index outside the query-head dimension")
        for qb, kb in blocks:
            if type(qb) is not int or type(kb) is not int or not 0 <= qb < shape[1] or not 0 <= kb < shape[2]:
                raise ValueError("repair block index outside the current input shape")
            mask[head, qb, kb] = True
    return mask.to(device=device)


def policy_from_head_atlas(atlas, *, coverage=0.5, protected_coverage=1.0,
                           terms=3, field="compiled_q_fp8_heads"):
    """Explicit adapter for ThunderWorld's existing per-layer head atlas.

    The legacy field identifies protected query heads, not FP8 execution here.
    Using a projection atlas for attention is an experimental transfer, not a
    validated quality recommendation. The caller must match model/head ordering.
    """
    from .config import RepairPolicy, RepairRule
    default = RepairConfig(coverage=coverage, terms=terms)
    RepairConfig(coverage=protected_coverage)  # validate even an empty atlas
    if not isinstance(atlas, dict) or not isinstance(atlas.get("layers"), list):
        raise ValueError("atlas must contain a layers list")
    rules, seen = [], set()
    for row in atlas["layers"]:
        if not isinstance(row, dict) or "layer" not in row or field not in row:
            raise ValueError(f"each atlas layer requires layer and {field}")
        layer = row["layer"]
        if type(layer) is not int or layer < 0 or layer in seen:
            raise ValueError("atlas layer indices must be unique nonnegative integers")
        seen.add(layer)
        config = RepairConfig(coverage=coverage, terms=terms,
                              head_budgets=tuple((head, protected_coverage) for head in row[field]))
        rules.append(RepairRule(config=config, layers=(layer,)))
    return RepairPolicy(default=default, rules=tuple(rules))

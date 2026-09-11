"""Native FP4 block-repair attention; importing configuration needs no CUDA."""
from .config import (PRESETS, RepairConfig, RepairPolicy, RepairRule, preset,
                     LinearRepairConfig, InferenceConfig)
from .head_finding import config_from_head_scores, make_repair_mask, policy_from_head_atlas

__version__ = "0.1.0"
__all__ = ["PRESETS", "RepairConfig", "RepairPolicy", "RepairRule", "preset",
           "fp4_attention", "FP4Attention", "LinearRepairConfig", "InferenceConfig",
           "FP4Linear", "config_from_head_scores", "make_repair_mask", "policy_from_head_atlas"]


def fp4_attention(*args, **kwargs):
    from .runtime import fp4_attention as implementation
    return implementation(*args, **kwargs)


def __getattr__(name):
    if name == "FP4Attention":
        from .runtime import FP4Attention
        return FP4Attention
    if name == "FP4Linear":
        from .linear import FP4Linear
        return FP4Linear
    raise AttributeError(name)

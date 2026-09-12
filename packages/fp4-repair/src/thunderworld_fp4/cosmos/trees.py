"""Bounded whole-transformer graph replay with current-input copies and owned outputs.

Only used for resident inference. Clear before replacing or moving model weights.
Python shape/CFG/mode metadata is part of the key; tensor values are never cached.
"""
import torch


def signature(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
    if isinstance(value, dict):
        return ('dict', tuple((k, signature(v)) for k, v in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(signature(v) for v in value))
    if value is None or isinstance(value, (str, bool, int, float)):
        return (type(value).__name__, value)
    raise TypeError(f'Unsupported graph metadata: {type(value)}')


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    return value


def copy_tree(target, source):
    if isinstance(source, torch.Tensor):
        target.copy_(source)
    elif isinstance(source, dict):
        for k in source:
            copy_tree(target[k], source[k])
    elif isinstance(source, (tuple, list)):
        for x, y in zip(target, source):
            copy_tree(x, y)

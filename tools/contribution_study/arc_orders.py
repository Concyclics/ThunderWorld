"""Load compact, frozen ARC channel permutations without teacher activations."""
import json
from pathlib import Path

from baselines import sha


def load_orders(root, model_identity, targets):
    import torch
    from portable import tensor_hash
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('schema') != 'arc-channel-order-v1' or not manifest.get('complete')
            or manifest['model_identity'] != model_identity or manifest['panels_per_module'] != 16):
        raise ValueError('Incomplete or mismatched ARC channel orders')
    path = (root/manifest['file']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path) != manifest['sha256']:
        raise ValueError('Changed ARC channel-order file')
    orders = torch.load(path, weights_only=True, map_location='cpu')
    if set(orders) != set(targets) or set(orders) != set(manifest['tensor_hashes']):
        raise ValueError('ARC target coverage mismatch')
    for name, order in orders.items():
        if (order.dtype != torch.int64 or order.ndim != 1 or
                not torch.equal(torch.sort(order).values, torch.arange(order.numel()))):
            raise ValueError('Invalid ARC permutation: '+name)
        if tensor_hash(order) != manifest['tensor_hashes'][name]:
            raise ValueError('Changed ARC permutation: '+name)
    return orders

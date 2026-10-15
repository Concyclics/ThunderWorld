"""Exact live CUDA model storage by semantic owner; never infer peak components."""
from collections import defaultdict


def component(name):
    if '.mlp_moe_gen.' in name:
        return 'generation_ffn'
    if '.mlp.' in name:
        return 'ar_ffn'
    if '.self_attn.' in name and any('.' + s in name for s in
            ('add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out')):
        return 'generation_qkvo'
    if '.self_attn.' in name and any('.' + s in name for s in
            ('to_q', 'to_k', 'to_v', 'to_out')):
        return 'ar_qkvo'
    if '.embed_tokens.' in name:
        return 'embedding'
    if name.startswith('vae.'):
        return 'vae'
    return 'auxiliary'


def inventory(runner):
    import torch
    stores = {}

    def add(tensor, name, part):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            return
        s = tensor.untyped_storage()
        key = (str(tensor.device), s.data_ptr())
        if key in stores:
            stores[key]['aliases'].append(name)
            return
        stores[key] = dict(owner=name, component=component(name), part=part,
            bytes=s.nbytes(), dtype=str(tensor.dtype), shape=list(tensor.shape), aliases=[])

    native_parts = dict(wc='base_payload', ws='metadata', wg='metadata',
        d='metadata', wr='repair_payload', wrs='repair_metadata',
        wi='repair_metadata', wloc='repair_metadata', off='metadata',
        w0_scales='extra_scale_layout')
    for name, op in runner.pipe.transformer.named_modules():
        if not hasattr(op, 'ops'):
            continue
        for key, prepared in op.ops.items():
            for index, native in enumerate(getattr(prepared, 'tuning_variants', (prepared,))):
                for field, part in native_parts.items():
                    add(getattr(native, field, None),
                        f'transformer.{name}.native[{key},{index}].{field}', part)

    for root, module in runner.pipe.components.items():
        if not isinstance(module, torch.nn.Module):
            continue
        for name, value in list(module.named_parameters()) + list(module.named_buffers()):
            full = root + '.' + name
            leaf = name.rsplit('.', 1)[-1]
            part = 'metadata'
            if component(full) in ('generation_ffn', 'generation_qkvo', 'ar_ffn'):
                if leaf in ('weight', 'weight_codes', 'weight_q'):
                    part = 'base_payload'
                elif leaf == 'scale_w_sparse':
                    part = 'extra_scale_layout'
            else:
                part = 'unquantized'
            add(value, full, part)
    rows = list(stores.values())
    groups = defaultdict(int)
    for row in rows:
        groups[row['component'], row['part']] += row['bytes']
    return dict(total_bytes=sum(r['bytes'] for r in rows), storage_count=len(rows),
        rows=rows, groups=[dict(component=k[0], part=k[1], bytes=v) for k, v in sorted(groups.items())],
        scope='Deduplicated live CUDA model storage, excluding runtime/Graph/workspaces',
        no_device_copy=True)

"""Read checkpoint headers only; no Torch import, tensor load, or GPU allocation."""
import hashlib
import json
from pathlib import Path
import re
import struct
import sys


def category(name):
    if '.mlp_moe_gen.' in name:
        return 'generation_ffn'
    if re.search(r'\.self_attn\.(add_[qkv]_proj|to_add_out)\.', name):
        return 'generation_attention_projections'
    if '.mlp.' in name:
        return 'understanding_ffn'
    if re.search(r'\.self_attn\.to_[qkvo]', name):
        return 'understanding_attention_projections'
    if name.startswith('embed_tokens.'):
        return 'text_embedding'
    if name.startswith('lm_head.'):
        return 'language_output_head'
    if name.startswith('action_'):
        return 'action_heads'
    if name.startswith(('audio_', 'sound_')):
        return 'audio_heads'
    if name.startswith('time_'):
        return 'timestep_embedding'
    if name.startswith(('proj_in.', 'proj_out.')):
        return 'video_input_output_projections'
    if 'norm' in name:
        return 'norms'
    return 'other_transformer'


def main():
    result = {}
    for root_arg in sys.argv[1:]:
        root = Path(root_arg)
        components, buckets, sources, largest = {}, {}, {}, []
        seen = set()
        for file in sorted(root.rglob('*.safetensors')):
            with file.open('rb') as stream:
                header_bytes = stream.read(struct.unpack('<Q', stream.read(8))[0])
            header = json.loads(header_bytes)
            sources[str(file.relative_to(root))] = hashlib.sha256(header_bytes).hexdigest()
            component = str(file.relative_to(root)).split('/')[0]
            for name, item in header.items():
                if name == '__metadata__':
                    continue
                if (component, name) in seen:
                    raise ValueError('Duplicate checkpoint tensor')
                seen.add((component, name))
                size = item['data_offsets'][1] - item['data_offsets'][0]
                group = components.setdefault(component, dict(bytes=0, tensors=0, by_dtype={}))
                group['bytes'] += size
                group['tensors'] += 1
                group['by_dtype'][item['dtype']] = group['by_dtype'].get(item['dtype'], 0) + size
                if component == 'transformer':
                    key = category(name)
                    group = buckets.setdefault(key, dict(bytes=0, tensors=0, by_dtype={}))
                    group['bytes'] += size
                    group['tensors'] += 1
                    group['by_dtype'][item['dtype']] = group['by_dtype'].get(item['dtype'], 0) + size
                    largest.append(dict(name=name, bytes=size, dtype=item['dtype'], shape=item['shape'], category=key))
        if sum(v['bytes'] for v in buckets.values()) != components['transformer']['bytes']:
            raise ValueError('Incomplete transformer partition')
        result[root.name] = dict(model_directory=str(root), components=components,
            transformer_partition=buckets, largest_tensors=sorted(largest, key=lambda x: -x['bytes'])[:20],
            safetensors_header_sha256=sources,
            scope='File tensor bytes, not measured runtime residency; optional checkpoint components may not be loaded')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

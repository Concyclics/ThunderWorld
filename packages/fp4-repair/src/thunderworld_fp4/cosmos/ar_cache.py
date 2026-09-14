"""Request-local exact AR prefill for Cosmos3's text-only causal prefix.

DM reads cached per-layer AR K/V. Refill both CFG branches for every request,
including benchmark requests; stable buffers allow existing DM graphs to replay.
Only the runner's static-text I2V flow is supported. Weights must remain fixed.
"""
import time
import torch
from .trees import signature, clone_tree, copy_tree


def rotate(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@torch.inference_mode()
def prefill(model, kwargs, *, validate=True):
    """Use the reference AR operation order, including its separate DM-facing K norm."""
    und_len = kwargs['und_len']
    for name in ('vision_sequence_indexes', 'sound_sequence_indexes', 'action_sequence_indexes'):
        index = kwargs.get(name)
        if validate and index is not None and index.numel() and bool((index < und_len).any()):
            raise ValueError('AR cache requires a text-only understanding prefix')
    hidden = model.embed_tokens(kwargs['input_ids'])
    states = hidden.new_zeros((kwargs['sequence_length'], model.config.hidden_size))
    states[kwargs['text_indexes']] = hidden
    und = states[:und_len]
    pos = kwargs['position_ids']
    cos, sin = model.rotary_emb(position_ids=pos.unsqueeze(0) if pos.ndim == 1 else pos.unsqueeze(1),
        device=states.device, dtype=states.dtype)
    cos = cos.squeeze(0)[:und_len].unsqueeze(1)
    sin = sin.squeeze(0)[:und_len].unsqueeze(1)
    rows = []
    for layer in model.layers:
        attn = layer.self_attn
        norm = layer.input_layernorm(und)
        q = attn.to_q(norm).view(-1, attn.num_attention_heads, attn.head_dim)
        k = attn.to_k(norm).view(-1, attn.num_key_value_heads, attn.head_dim)
        v = attn.to_v(norm).view(-1, attn.num_key_value_heads, attn.head_dim)
        q = attn.norm_q(q); k = attn.norm_k(k)
        kg = attn.k_norm_und_for_gen(k) if attn.k_norm_und_for_gen is not None else k
        kg_before_rope = kg
        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        kg = kg * cos + rotate(kg) * sin
        dispatch = type(attn.processor).__call__.__globals__['dispatch_attention_fn']
        out = dispatch(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True,
            enable_gqa=True, backend=attn.processor._attention_backend,
            parallel_config=attn.processor._parallel_config)
        residual = und + attn.to_out(out.squeeze(0).flatten(-2, -1))
        und = residual + layer.mlp(layer.post_attention_layernorm(residual))
        rows.append(dict(k=kg, v=v, out=und, k_before_rope=kg_before_rope))
    return rows


class ARCache:
    def __init__(self, engine):
        self.rope = lambda x,c,s: x*c+rotate(x)*s
        self.engine = engine
        self.model = engine.model
        if type(self.model).__name__ != 'Cosmos3OmniTransformer':
            raise ValueError('Unsupported AR-cache model')
        if self.model._cp_shard_fn is not None or self.model._cp_gather_fn is not None:
            raise ValueError('AR cache supports single-device inference')
        self.slots = {}; self.graphs = {}; self.seen = set(); self.active = None
        self.capture_seconds = 0.
        self.prefills = 0; self.request_prefills = 0; self.calls = 0
        for i, layer in enumerate(self.model.layers):
            def forward(und_seq, gen_seq, rotary_emb, _layer=layer, _i=i):
                if self.active is None: raise RuntimeError('AR prefix was not prepared')
                cached = self.active[_i]; attn = _layer.self_attn
                norm = _layer.input_layernorm_moe_gen(gen_seq)
                q = attn.add_q_proj(norm).view(-1, attn.num_attention_heads, attn.head_dim)
                k = attn.add_k_proj(norm).view(-1, attn.num_key_value_heads, attn.head_dim)
                v = attn.add_v_proj(norm).view(-1, attn.num_key_value_heads, attn.head_dim)
                q = attn.norm_added_q(q); k = attn.norm_added_k(k)
                cos, sin = rotary_emb[2].unsqueeze(1), rotary_emb[3].unsqueeze(1)
                q = self.rope(q, cos, sin)
                k = self.rope(k, cos, sin)
                all_k = torch.cat((cached['k'], k), dim=0)
                all_v = torch.cat((cached['v'], v), dim=0)
                dispatch = type(attn.processor).__call__.__globals__['dispatch_attention_fn']
                out = dispatch(q.unsqueeze(0), all_k.unsqueeze(0), all_v.unsqueeze(0), is_causal=False,
                    enable_gqa=True, backend=attn.processor._attention_backend,
                    parallel_config=attn.processor._parallel_config)
                residual = gen_seq + attn.to_add_out(out.squeeze(0).flatten(-2, -1))
                gen_out = residual + _layer.mlp_moe_gen(_layer.post_attention_layernorm_moe_gen(residual))
                return cached['out'], gen_out
            layer.forward = forward

    def begin_request(self, task):
        if task != 'i2v': raise ValueError('AR cache currently supports static-text I2V only')
        self.seen.clear(); self.request_prefills = 0

    def prepare_call(self, kwargs):
        if torch.is_grad_enabled() or self.model.training:
            raise RuntimeError('AR cache requires inference/eval mode')
        branch = self.engine.call % self.engine.branches
        key = (branch, kwargs['und_len'], kwargs['sequence_length'],
            signature(kwargs['input_ids']), signature(kwargs['position_ids']),
            torch.cuda.current_stream().cuda_stream)
        if key not in self.seen:
            if key not in self.slots and not self.engine.warming:
                raise RuntimeError('Unwarmed AR prefix shape/stream')
            if key not in self.slots and len(self.slots) >= 16:
                raise RuntimeError('AR prefix capacity reached; use a fresh runner')
            # Validate modality boundaries outside graph capture. The runner keeps
            # text and position inputs static within each request/branch.
            prefix = {name: kwargs.get(name) for name in ('input_ids', 'text_indexes', 'position_ids',
                'und_len', 'sequence_length', 'vision_sequence_indexes', 'sound_sequence_indexes', 'action_sequence_indexes')}
            for name in ('vision_sequence_indexes', 'sound_sequence_indexes', 'action_sequence_indexes'):
                index = prefix[name]
                if index is not None and index.numel() and bool((index < prefix['und_len']).any()):
                    raise ValueError('AR cache requires a text-only understanding prefix')
            if key not in self.slots:
                static = clone_tree(prefix)
                for _ in range(2): prefill(self.model, static, validate=False)
                torch.cuda.synchronize(); graph = torch.cuda.CUDAGraph(); start = time.perf_counter()
                with torch.cuda.graph(graph): rows = prefill(self.model, static, validate=False)
                torch.cuda.synchronize(); self.capture_seconds += time.perf_counter() - start
                self.slots[key] = [{name: row[name] for name in ('k','v','out')} for row in rows]
                self.graphs[key] = (static, graph)
            static, graph = self.graphs[key]
            copy_tree(static, prefix); graph.replay()
            self.seen.add(key); self.prefills += 1; self.request_prefills += 1
        self.active = self.slots[key]; self.calls += 1

    def receipt(self):
        return dict(enabled=True, slots=len(self.slots), total_prefills=self.prefills,
            last_request_prefills=self.request_prefills, calls=self.calls,
            prefill_graphs=len(self.graphs), capture_seconds=self.capture_seconds,
            buffer_bytes=sum(t.numel()*t.element_size() for rows in self.slots.values() for row in rows for t in row.values()),
            scope='Fresh graphed AR prefill per request and CFG branch; stable K/V/output buffers; prefill and input copies included in request timing')

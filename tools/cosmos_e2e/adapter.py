"""Generator-only adapters; Edge ReLU² / Nano SiLU logic stays in Diffusers.

This pilot uses model-local RTN W4 and full ACT2, with identity prescale.
It deliberately does not transplant the Nano layer-0 calibration or head atlas.
"""
import importlib.util
import math
import statistics
from pathlib import Path
import sys

import torch
import triton.testing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/fp4-repair/src'))
sys.path.insert(0, str(ROOT / 'tools/kernel_handoff'))
from thunderworld_fp4.linear import _pack
from thunderworld_fp4.config import RepairConfig
from thunderworld_fp4.runtime import _forward as attention_forward
from frozen_baseline import load as load_frozen


def load_native():
    spec = importlib.util.spec_from_file_location('cosmos_e2e_native', ROOT / 'experiments/ffn_block_repair_20260907/native.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Controller:
    def __init__(self, allow_split_k=False):
        self.native = load_native()
        self.original_native, self.original_runtime, self.baseline_hashes = load_frozen()
        self.mode = 'ffn_optimized'
        self.launches = {}
        self.searches = []
        self.calls = {'ffn': 0, 'attention': 0}
        self.attention_shapes = set()
        self.tuning = True
        self.allow_split_k = allow_split_k


class PreparedLinear(torch.nn.Module):
    @torch.inference_mode()
    def __init__(self, linear, controller, name):
        super().__init__()
        if linear.bias is not None:
            raise ValueError('This adapter requires bias-free generator FFNs')
        self.controller, self.name = controller, name
        self.in_features, self.out_features = linear.in_features, linear.out_features
        w = linear.weight.detach().to('cuda').contiguous()
        n, k = w.shape
        d = torch.ones(k, device='cuda', dtype=torch.float32)
        c, s, g, _ = _pack(w, d, True, True, w)
        am = torch.ones((1, k), device='cuda', dtype=torch.bool)
        wm = torch.zeros((n, k), device='cuda', dtype=torch.bool)
        self.op = controller.native.Native(c, s, g, d, am, wm, bm=128, interleave=True, num_stages=2)
        self.old = object.__new__(controller.original_native.Native)
        self.old.__dict__.update(self.op.__dict__)
        self.old.bm = 32
        for key, value in self.op.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(key, value.cpu())
        self._bind()

    def _bind(self):
        for key, value in self.named_buffers(recurse=False):
            setattr(self.op, key, value)
            setattr(self.old, key, value)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self._bind()
        return self

    def _support(self, m):
        rows = math.ceil(m / 128)
        for op in (self.op, self.old):
            op.am = self.am.expand(rows, -1)
            op.ai = self.ai.expand(rows, -1)
            op.union = self.union.expand(rows, -1)

    def _tune(self, x, key):
        gold = self.old(x).float()
        records = []
        low_grid = math.ceil(x.shape[0]/128)*(self.out_features//128) < 340
        splits = (1, 2, 4, 8) if self.controller.allow_split_k and low_grid and self.in_features > self.out_features else (1,)
        for bm in (64, 128):
            for stages in (2, 3):
                for split in splits:
                    self.op.bm, self.op.num_stages, self.op.split_k = bm, stages, split
                    y = self.op(x).float()
                    error = float((y-gold).norm() / gold.norm().clamp_min(1e-12))
                    if not torch.isfinite(y).all() or error >= .004:
                        raise AssertionError((self.name, bm, stages, split, error))
                    samples = [triton.testing.do_bench_cudagraph(lambda: self.op(x), rep=20) for _ in range(3)]
                    records.append(dict(bm=bm, num_stages=stages, split_k=split, ms=statistics.median(samples),
                                        samples_ms=samples, relative_l2=error))
        winner = min(records, key=lambda r: r['ms'])
        self.controller.launches[key] = {k: winner[k] for k in ('bm', 'num_stages', 'split_k')}
        self.controller.searches.append(dict(shape=key, input_module=self.name, candidates=records))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features).contiguous()
        self._support(x.shape[0])
        c = self.controller
        c.calls['ffn'] += 1
        if c.mode.endswith('original'):
            result = self.old(x)
        else:
            key = f'{x.shape[0]}:{self.out_features}:{self.in_features}'
            if key not in c.launches:
                if not c.tuning:
                    raise RuntimeError(f'Unwarmed FFN shape {key}')
                self._tune(x, key)
            for k, v in c.launches[key].items():
                setattr(self.op, k, v)
            result = self.op(x)
        return result.reshape(*shape[:-1], self.out_features)


@torch.inference_mode()
def install_ffn(transformer, controller):
    """Called on a CPU-resident model between trials, never in a timed request."""
    replaced = []
    for layer_id, layer in enumerate(transformer.layers):
        mlp = layer.mlp_moe_gen
        for name, child in list(mlp.named_children()):
            if isinstance(child, torch.nn.Linear):
                path = f'layers.{layer_id}.mlp_moe_gen.{name}'
                setattr(mlp, name, PreparedLinear(child, controller, path))
                replaced.append(path)
        if layer_id % 8 == 0:
            print(f'Packed generator FFN layer {layer_id}', flush=True)
    return replaced


def install_attention_dispatch(controller):
    """Intercept only the exact unmasked generation dispatch; causal path unchanged."""
    from diffusers.models.transformers import transformer_cosmos3 as module
    original_dispatch = module.dispatch_attention_fn

    def dispatch(q, k, v, **kwargs):
        if not controller.mode.startswith('joint') or kwargs.get('is_causal', False):
            return original_dispatch(q, k, v, **kwargs)
        if kwargs.get('attn_mask') is not None or kwargs.get('parallel_config') is not None:
            raise ValueError('FP4 pilot only supports single-GPU, unmasked generation attention')
        controller.calls['attention'] += 1
        controller.attention_shapes.add((tuple(q.shape), tuple(k.shape)))
        optimized = controller.mode.endswith('optimized')
        config = RepairConfig(coverage=.75, terms=3, query_tile=32, num_warps=4,
                              splits=2, transpose_v=optimized)
        fn = attention_forward if optimized else controller.original_runtime._forward
        return fn(q, k, v, config)[0]

    module.dispatch_attention_fn = dispatch
    return original_dispatch

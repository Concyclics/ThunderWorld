"""Model-local RTN term ablations; deliberately no imported calibration policy."""
import math
import sys

import torch

from runtime import ROOT
sys.path.insert(0, str(ROOT / 'tools/cosmos_e2e'))
from adapter import load_native
from thunderworld_fp4.linear import _pack


class TermLinear(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        if source.bias is not None:
            raise ValueError('Bias-free generation FFN expected')
        self.in_features = source.in_features
        self.out_features = source.out_features
        w = source.weight.detach().to('cuda').contiguous()
        d = torch.ones(self.in_features, device='cuda')
        codes, scales, factor, _ = _pack(w, d, True, True, w)
        # Offline immutable two-plane source; only selected planes are deployed.
        self.codes, self.scales, self.factor = codes.cpu(), scales.cpu(), factor.cpu()
        self.native_cls = load_native().Native
        self.op = None
        self.configure('ffn_act2')

    def configure(self, mode):
        a = mode in ('ffn_act2', 'ffn_dual', 'ffn_four')
        w = mode in ('ffn_wgt2', 'ffn_dual', 'ffn_four')
        if mode not in ('ffn_base', 'ffn_act2', 'ffn_wgt2', 'ffn_dual', 'ffn_four'):
            raise ValueError(mode)
        self.op = None
        k, n = self.in_features, self.out_features
        am = torch.full((1, k), a, dtype=torch.bool, device='cuda')
        wm = torch.full((n, k), w, dtype=torch.bool, device='cuda')
        self.op = self.native_cls(self.codes.cuda(), self.scales.cuda(), self.factor.cuda(),
                                  torch.ones(k, device='cuda'), am, wm, cross=mode == 'ffn_four',
                                  bm=128, interleave=True, num_stages=2, split_k=1)
        self.am, self.ai, self.union = self.op.am, self.op.ai, self.op.union

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        rows = math.ceil(flat.shape[0] / 128)
        self.op.am = self.am.expand(rows, -1)
        self.op.ai = self.ai.expand(rows, -1)
        self.op.union = self.union.expand(rows, -1)
        return self.op(flat).reshape(*shape[:-1], self.out_features)


def install(transformer):
    modules = []
    for i, layer in enumerate(transformer.layers):
        for name, child in list(layer.mlp_moe_gen.named_children()):
            if isinstance(child, torch.nn.Linear):
                replacement = TermLinear(child)
                setattr(layer.mlp_moe_gen, name, replacement)
                modules.append(replacement)
        if i % 8 == 0:
            print('Prepared FFN layer', i, flush=True)
    return modules

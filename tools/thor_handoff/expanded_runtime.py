"""I2V-only deployment experiment; copied into an isolated runtime by expand_build.py.

AR uses native FP4 MMA. Embeddings decode only requested rows. Codec/auxiliary
weights use explicitly labelled W4A16 (FP32 timestep) decode-per-call paths.
There is no retained BF16 weight cache, and this is not an all-W4A4 claim.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ._quantize import quantize
from .cosmos.linears import RepairLinear, FP8Linear
from .cosmos.settings import LinearSpec
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d


@dataclass(frozen=True)
class Expansion:
    trim: bool = True
    ar: str = 'bf16'
    ar_scope: str = 'all'
    embedding_planes: int = 0
    auxiliary_planes: int = 0
    vae_planes: int = 0
    asset_dir: str | None = None

    def __post_init__(self):
        if type(self.trim) is not bool or self.ar not in ('bf16', 'fp8', 'fp4_base', 'fp4_three', 'fp4_four'):
            raise ValueError('Invalid I2V expansion profile')
        if self.ar_scope not in ('all', 'ffn'):
            raise ValueError('AR scope must be all or ffn')
        if any(type(v) is not int or v not in (0, 1, 2) for v in
               (self.embedding_planes, self.auxiliary_planes, self.vae_planes)):
            raise ValueError('Packed weight planes must be 0, 1 or 2')


@triton.jit
def _encode_rows(X, C0, S0, C1, S1, G, N: tl.constexpr, K: tl.constexpr,
                 KP: tl.constexpr, B: tl.constexpr, TWO: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)[None, :]
    c, s, cr, sr, factor = quantize(x, 1, B, TWO)
    cc = tl.arange(0, B // 2)
    sc = tl.arange(0, B // 16)
    tl.store(C0 + row * (KP // 2) + cc, tl.reshape(c, (B // 2,)), cc < KP // 2)
    tl.store(S0 + row * (KP // 16) + sc, tl.reshape(s, (B // 16,)), sc < KP // 16)
    if TWO:
        tl.store(C1 + row * (KP // 2) + cc, tl.reshape(cr, (B // 2,)), cc < KP // 2)
        tl.store(S1 + row * (KP // 16) + sc, tl.reshape(sr, (B // 16,)), sc < KP // 16)
    tl.store(G + row, tl.sum(factor, 0))


@triton.jit
def _value(c):
    a = c & 7
    v = tl.where(a < 4, a * .5, tl.where(a < 6, a - 2., 2. * a - 8.))
    return tl.where(c >= 8, -v, v)


@triton.jit
def _decode_rows(C0, S0, C1, S1, G, IDS, Y, N: tl.constexpr, K: tl.constexpr,
                 KP: tl.constexpr, M: tl.constexpr, TWO: tl.constexpr,
                 GATHER: tl.constexpr, B: tl.constexpr):
    outrow = tl.program_id(0)
    row = tl.load(IDS + outrow) if GATHER else outrow
    tl.device_assert((row >= 0) & (row < N), 'embedding index out of bounds')
    valid = (row >= 0) & (row < N)
    j = tl.program_id(1) * B + tl.arange(0, B)
    packed = tl.load(C0 + row * (KP // 2) + j // 2, (j < K) & valid, 0).to(tl.int32)
    code = (packed >> ((j % 2) * 4)) & 15
    scale = tl.load(S0 + row * (KP // 16) + j // 16, (j < K) & valid, 0.).to(tl.float32)
    value = _value(code) * scale
    if TWO:
        packed1 = tl.load(C1 + row * (KP // 2) + j // 2, (j < K) & valid, 0).to(tl.int32)
        code1 = (packed1 >> ((j % 2) * 4)) & 15
        scale1 = tl.load(S1 + row * (KP // 16) + j // 16, (j < K) & valid, 0.).to(tl.float32)
        value = value + _value(code1) * scale1
    factor = tl.load(G + row, valid, 0)
    tl.store(Y + outrow * K + j, value * factor, j < K)


class PackedRows(torch.nn.Module):
    """Offline E2M1 + E4M3 K16 encoding, per-row outer FP32 scale."""
    def __init__(self, weight, planes, asset=None):
        super().__init__()
        self.original_shape = tuple(weight.shape)
        self.n, self.k = weight.shape[0], weight.numel() // weight.shape[0]
        self.kp = triton.cdiv(self.k, 16) * 16
        self.planes = planes
        self.output_dtype = weight.dtype
        if asset is None:
            # Bound preparation memory even for the complete vocabulary table.
            outputs = [[] for _ in range(5)]
            flat = weight.detach().reshape(self.n, self.k)
            for lo in range(0, self.n, 1024):
                x = flat[lo:lo + 1024].cuda().contiguous()
                n = x.shape[0]
                c0 = torch.empty((n, self.kp // 2), device='cuda', dtype=torch.uint8)
                s0 = torch.empty((n, self.kp // 16), device='cuda', dtype=torch.float8_e4m3fn)
                c1 = torch.empty_like(c0) if planes == 2 else c0
                s1 = torch.empty_like(s0) if planes == 2 else s0
                g = torch.empty(n, device='cuda', dtype=torch.float32)
                _encode_rows[(n,)](x, c0, s0, c1, s1, g, n, self.k, self.kp,
                                    triton.next_power_of_2(self.kp), planes == 2,
                                    enable_fp_fusion=False)
                values = [c0, s0, c1 if planes == 2 else None, s1 if planes == 2 else None, g]
                for dst, value in zip(outputs, values):
                    if value is not None: dst.append(value.cpu())
            asset = {name: torch.cat(rows) if rows else None
                     for name, rows in zip(('c0', 's0', 'c1', 's1', 'factor'), outputs)}
        for name, value in asset.items(): self.register_buffer(name, value)

    def export(self):
        return {n: None if getattr(self, n) is None else getattr(self, n).cpu()
                for n in ('c0', 's0', 'c1', 's1', 'factor')}

    def decode(self, ids=None):
        m = self.n if ids is None else ids.numel()
        if ids is not None and (ids.dtype not in (torch.int64, torch.int32) or ids.device != self.c0.device):
            raise ValueError('Embedding indices must be integral on the weight device')
        y = torch.empty((m, self.k), device=self.c0.device, dtype=self.output_dtype)
        if m:
            _decode_rows[(m, triton.cdiv(self.k, 512))](
                self.c0, self.s0, self.c1 if self.planes == 2 else self.c0,
                self.s1 if self.planes == 2 else self.s0, self.factor,
                self.factor if ids is None else ids.contiguous(), y, self.n, self.k, self.kp,
                m, self.planes == 2, ids is not None, 512, enable_fp_fusion=False)
        return y


class PackedEmbedding(torch.nn.Module):
    def __init__(self, source, planes, asset=None):
        super().__init__()
        if source.max_norm is not None: raise ValueError('In-place embedding normalization unsupported')
        self.num_embeddings, self.embedding_dim = source.num_embeddings, source.embedding_dim
        self.rows = PackedRows(source.weight, planes, asset)

    def forward(self, ids):
        return self.rows.decode(ids).reshape(*ids.shape, self.embedding_dim)


class PackedLinear(torch.nn.Module):
    """Explicit weight-only fallback for small/FP32 auxiliary projections."""
    def __init__(self, source, planes, asset=None):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.rows = PackedRows(source.weight, planes, asset)
        # Cosmos queries next(time_embedder.parameters()).dtype. Keep the tiny
        # bias as a non-trainable parameter so its FP32 contract survives.
        self.register_parameter('bias', None if source.bias is None else
                                torch.nn.Parameter(source.bias.detach().clone(), requires_grad=False))

    def forward(self, x):
        return F.linear(x, self.rows.decode(), self.bias)


class PackedCausalConv3d(torch.nn.Module):
    """Storage experiment: FP4 weights, unchanged BF16 cuDNN convolution.

    Only the weight of the currently executing convolution is decoded. Codec
    Graph capture is prohibited so separate captures cannot retain every decoded
    weight in graph-private pools. This path makes no FP4-MMA speed claim.
    """
    def __init__(self, source, planes, asset=None):
        super().__init__()
        if type(source).__name__ != 'CosmosCausalConv3d':
            raise TypeError('Only the validated Cosmos causal padding contract is supported')
        self.rows = PackedRows(source.weight, planes, asset)
        self.register_buffer('bias', None if source.bias is None else source.bias.detach().clone())
        for name in ('stride', 'padding', 'dilation', 'groups', 'temporal_pad', 'spatial_pad', 'pad_mode'):
            setattr(self, name, getattr(source, name))

    def forward(self, x):
        first = x[:, :, :1].repeat(1, 1, self.temporal_pad, 1, 1)
        x = torch.cat((first, x), dim=2)
        x = F.pad(x, (*self.spatial_pad, 0, 0), mode=self.pad_mode, value=0.)
        weight = self.rows.decode().reshape(self.rows.original_shape)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class _PackedConv:
    def _init_packed(self, source, planes, asset):
        torch.nn.Module.__init__(self)
        self.rows = PackedRows(source.weight, planes, asset)
        self.register_buffer('bias', None if source.bias is None else source.bias.detach().clone())
        for name in ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
                     'dilation', 'groups', 'padding_mode', '_reversed_padding_repeated_twice', '_padding'):
            if hasattr(source, name): setattr(self, name, getattr(source, name))

    @property
    def weight(self):
        return self.rows.decode().reshape(self.rows.original_shape)


class PackedWanConv3d(_PackedConv, WanCausalConv3d):
    # Inherit the exact original padding/cache forward and preserve isinstance:
    # AutoencoderKLWan.clear_cache() counts WanCausalConv3d modules every request.
    def __init__(self, source, planes, asset=None):
        self._init_packed(source, planes, asset)


class PackedConv2d(_PackedConv, torch.nn.Conv2d):
    def __init__(self, source, planes, asset=None):
        self._init_packed(source, planes, asset)


def packed_conv(source, planes, asset=None):
    if type(source) is WanCausalConv3d: return PackedWanConv3d(source, planes, asset)
    if type(source) is torch.nn.Conv2d: return PackedConv2d(source, planes, asset)
    return PackedCausalConv3d(source, planes, asset)


class ARRepairLinear(RepairLinear):
    """Separate AR ledger; do not silently add AR work to the frozen DM budget."""
    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.in_features).contiguous()
        key = (len(flat), self.default.key)
        if key not in self.ops:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('AR shape must be prepared before Graph capture')
            if getattr(self, 'frozen', False): raise RuntimeError('New AR shape needs a fresh runner')
            self.prepare_shape(len(flat), self.default)
            # Expose native storage to model-state accounting without copies.
            for field in ('wc', 'ws', 'wg', 'd', 'wr', 'wrs', 'wi', 'off', 'wloc', 'w0_scales'):
                value = getattr(self.bound, field, None)
                if isinstance(value, torch.Tensor):
                    self.register_buffer(f'prepared_{len(self.ops)}_{field}', value, persistent=False)
        else: self.bound = self.ops[key]
        return self._compute(flat).reshape(*shape[:-1], self.out_features)


class RemovedI2VModule(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError('This I2V-only deployment removed an unused modality/language head')


def ar_targets(model, scope='all'):
    if scope not in ('all', 'ffn'):
        raise ValueError('AR scope must be all or ffn')
    names = []
    for i, layer in enumerate(model.layers):
        for name, child in layer.mlp.named_children():
            if isinstance(child, torch.nn.Linear): names.append(f'layers.{i}.mlp.{name}')
        if scope == 'ffn': continue
        for name in ('to_q', 'to_k', 'to_v', 'to_out'):
            path = f'layers.{i}.self_attn.{name}'
            if not isinstance(model.get_submodule(path), torch.nn.Linear):
                raise TypeError('Unexpected AR projection: ' + path)
            names.append(path)
    return names


def tensor_bytes(module):
    seen = {}
    for t in (*module.parameters(), *module.buffers()):
        storage = t.untyped_storage()
        seen[(str(t.device), storage.data_ptr())] = storage.nbytes()
    return sum(seen.values())


def install(runner):
    profile = Expansion(**(runner.expansion or {'trim': False}))
    if runner.audio: raise ValueError('Expanded deployment currently requires I2V without sound')
    if profile.vae_planes and runner.codec_graph_enabled:
        raise ValueError('Packed codec requires eager decode to bound transient weight storage')
    if profile.embedding_planes and runner.static_rotary_enabled:
        raise ValueError('StaticRotary assumes a dense embedding weight; disable it for the packed-row experiment')
    pipe = runner.pipe
    model = pipe.transformer
    report = dict(profile=asdict(profile), removed=[], quantized=[], protected=[],
                  ar_compute_scope='Additional AR prefill work, excluded from frozen generation C0',
                  codec_execution='W4A16 decode-per-convolution, BF16 cuDNN; no FP4 MMA claim')
    if profile.trim:
        for name in ('lm_head', 'action_proj_in', 'action_proj_out', 'audio_proj_in', 'audio_proj_out'):
            old = getattr(model, name, None)
            if old is not None:
                report['removed'].append(dict(module=name, original_bytes=tensor_bytes(old)))
                setattr(model, name, RemovedI2VModule())
        for name in ('action_modality_embed', 'audio_modality_embed'):
            old = getattr(model, name, None)
            if old is not None:
                report['removed'].append(dict(module=name, original_bytes=old.numel() * old.element_size()))
                model.register_parameter(name, None)

    assets = Path(profile.asset_dir) if profile.asset_dir else None
    if assets: assets.mkdir(parents=True, exist_ok=True)
    def replace(parent, name, kind, factory):
        source = parent.get_submodule(name)
        original = tensor_bytes(source)
        digest = hashlib.sha256(source.weight.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        key = hashlib.sha256((name + kind + digest).encode()).hexdigest()
        path = assets / (key + '.pt') if assets else None
        asset = torch.load(path, weights_only=True, map_location='cpu') if path and path.exists() else None
        op = factory(source, asset)
        if path and asset is None:
            value = op.rows.export() if hasattr(op, 'rows') else op.export()
            tmp = path.with_suffix('.tmp')
            torch.save(value, tmp); tmp.replace(path)
        parent.set_submodule(name, op)
        report['quantized'].append(dict(module=('vae.' if parent is pipe.vae else 'transformer.') + name,
            kind=kind, original_bytes=original, source_weight_sha256=digest,
            asset_file=str(path) if path else None))

    if profile.ar != 'bf16':
        specs = {'fp4_base': LinearSpec(0, 0), 'fp4_three': LinearSpec(1, 1), 'fp4_four': LinearSpec(1, 1, True)}
        controller = SimpleNamespace(config=runner.config, budget_plan=None)
        for name in ar_targets(model, profile.ar_scope):
            if profile.ar == 'fp8': factory = lambda source, a: FP8Linear(source, a)
            else: factory = lambda source, a, name=name: ARRepairLinear(source, controller, name, specs[profile.ar], a)
            replace(model, name, profile.ar, factory)
    if profile.embedding_planes:
        replace(model, 'embed_tokens', f'row_fp4_{profile.embedding_planes}',
                lambda s, a: PackedEmbedding(s, profile.embedding_planes, a))
    if profile.auxiliary_planes:
        for name in ('proj_in', 'proj_out', 'time_embedder.linear_1', 'time_embedder.linear_2'):
            replace(model, name, f'weight_only_fp4_{profile.auxiliary_planes}',
                    lambda s, a: PackedLinear(s, profile.auxiliary_planes, a))
    if profile.vae_planes:
        names = [name for name, module in pipe.vae.named_modules() if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d))]
        for name in names:
            replace(pipe.vae, name, f'weight_only_fp4_{profile.vae_planes}',
                    lambda s, a: packed_conv(s, profile.vae_planes, a))
    report['protected_scan_stage'] = 'before_generation_replacement'
    report['protected'] = [dict(module=component + '.' + name, shape=list(p.shape), dtype=str(p.dtype),
                               bytes=p.numel()*p.element_size())
        for component, obj in (('transformer', model), ('vae', pipe.vae))
        for name, p in obj.named_parameters()]
    return report


def audit_protected(runner):
    runner.expansion_report['protected'] = [dict(module=component + '.' + name,
        shape=list(p.shape), dtype=str(p.dtype), bytes=p.numel()*p.element_size())
        for component, obj in (('transformer', runner.pipe.transformer), ('vae', runner.pipe.vae))
        for name, p in obj.named_parameters()]
    runner.expansion_report['protected_scan_stage'] = 'after_generation_replacement'


def finalize(runner):
    """Release CPU staging after all request templates/CFG branches are warmed."""
    freed = 0
    rows = []
    for name, op in runner.pipe.transformer.named_modules():
        if not isinstance(op, ARRepairLinear): continue
        freed += sum(v.numel()*v.element_size() for v in op.asset.values() if isinstance(v, torch.Tensor))
        op.asset = {k:v for k,v in op.asset.items() if k in ('selector', 'activation_scale', 'identity_prescale')}
        op.frozen = True
        rows.extend(dict(module=name, m=m, n=op.out_features, k=op.in_features,
                         terms=4 if op.default.cross else 1 + int(op.default.activation > 0) + int(op.default.weight > 0),
                         padded_m=triton.cdiv(m, 128)*128, prepared_spec=key)
                    for m, key in op.ops)
    runner.expansion_report.update(released_ar_host_bytes=freed, ar_prepared_shapes=rows)
    return freed

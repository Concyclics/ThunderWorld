"""Explicit research-only attention observation and temporal ablations.

Caches are isolated by request, CFG branch and layer. No production defaults
are changed. Repair delta means repaired attention output minus base output,
including the effect of its changed softmax; it is not a separable PV tile.
"""
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'packages/fp4-repair/src'))
from thunderworld_fp4.config import RepairConfig
from thunderworld_fp4.runtime import _forward as fp4

ROLES = ('ar', 'vision_clean', 'vision_noisy', 'sound', 'action', 'special')
FEATURES = ('ar_mass', 'entropy', 'top1_mass', 'output_norm', 'q_cos_prev',
            'output_cos_prev', 'output_rel_l2_prev', 'support_jaccard_prev',
            'previous_support_mass', 'current_support_mass', 'dominant_stable_prev',
            'role_js_prev')


def config(coverage=.5, terms=3):
    return RepairConfig(coverage=coverage, terms=terms, query_tile=32,
                        num_warps=4, splits=2, transpose_v=True)


def cosine(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1, eps=1e-12)


class Observer:
    def __init__(self, layers, heads, steps, branches, seed=771, samples_per_role=32):
        self.layers, self.heads = layers, heads
        self.steps, self.branches = steps, branches
        self.seed, self.samples = seed, samples_per_role
        self.mass = np.full((steps, branches, layers, heads, len(ROLES), len(ROLES)), np.nan, np.float32)
        self.features = np.full((steps, branches, layers, heads, len(ROLES), len(FEATURES)), np.nan, np.float32)
        self.counts = np.zeros((branches, len(ROLES)), np.int32)
        self.sigma = np.full(steps, np.nan, np.float32)
        self.previous = {}
        self.metadata = {}

    def layout(self, kwargs, branch):
        if branch in self.metadata:
            return self.metadata[branch]
        n, ar = int(kwargs['sequence_length']), int(kwargs['und_len'])
        labels = torch.full((n,), ROLES.index('special'), device='cuda', dtype=torch.long)
        labels[:ar] = 0
        for name, role in [('vision', 'vision_clean'), ('sound', 'sound'), ('action', 'action')]:
            indices = kwargs.get(name + '_sequence_indexes')
            if indices is not None:
                labels[indices] = ROLES.index(role)
        indices = kwargs['vision_mse_loss_indexes']
        labels[indices] = ROLES.index('vision_noisy')
        # Text/special markers in generation remain a separate role.
        g = torch.Generator(device='cpu').manual_seed(self.seed + branch)
        picks = {}
        for role in range(len(ROLES)):
            all_idx = torch.where(labels[ar:] == role)[0].cpu()
            self.counts[branch, role] = int((labels == role).sum())
            if all_idx.numel():
                idx = all_idx[torch.randperm(len(all_idx), generator=g)[:self.samples]].sort().values
                picks[role] = idx.to('cuda')
        self.metadata[branch] = (labels, picks, ar)
        return labels, picks, ar

    def observe(self, q, k, v, output, kwargs, step, branch, layer, timestep):
        labels, picks, ar = self.layout(kwargs, branch)
        assert k.shape[1] == len(labels) and q.shape[1] == len(labels) - ar
        self.sigma[step] = timestep
        h, hk = q.shape[2], k.shape[2]
        key = k[0].transpose(0, 1).float().repeat_interleave(h // hk, 0)
        roles = F.one_hot(labels, len(ROLES)).float()
        nt = math.ceil(k.shape[1] / 128)
        for role, idx in picks.items():
            qq = q[0, idx].transpose(0, 1).float()
            oo = output[0, idx].transpose(0, 1).float()
            prob = (qq @ key.transpose(1, 2) / math.sqrt(q.shape[-1])).softmax(-1)
            relation = (prob @ roles).mean(1)
            tile_mass = F.pad(prob, (0, nt * 128 - k.shape[1])).reshape(h, len(idx), nt, 128).sum(-1).mean(1)
            support = torch.zeros_like(tile_mass, dtype=torch.bool)
            support.scatter_(1, torch.argsort(tile_mass, dim=1, descending=True, stable=True)[:, :math.ceil(nt / 2)], True)
            norm = oo.flatten(1).norm(dim=1)
            values = torch.full((h, len(FEATURES)), float('nan'), device='cuda')
            values[:, 0] = relation[:, 0]
            values[:, 1] = -(prob * prob.clamp_min(1e-30).log()).sum(-1).mean(-1)
            values[:, 2] = prob.max(-1).values.mean(-1)
            values[:, 3] = norm
            values[:, 9] = (tile_mass * support).sum(-1)
            pk = (branch, layer, role)
            if pk in self.previous:
                pq, po, ps, pr = [x.to('cuda') for x in self.previous[pk]]
                values[:, 4] = cosine(qq, pq)
                values[:, 5] = cosine(oo, po)
                values[:, 6] = (oo - po).flatten(1).norm(dim=1) / norm.clamp_min(1e-12)
                values[:, 7] = (support & ps).sum(-1) / (support | ps).sum(-1).clamp_min(1)
                values[:, 8] = (tile_mass * ps).sum(-1)
                values[:, 10] = (relation.argmax(-1) == pr.argmax(-1)).float()
                mean = (relation + pr) / 2
                values[:, 11] = .5 * ((relation * (relation.clamp_min(1e-30) / mean.clamp_min(1e-30)).log()).sum(-1)
                                     + (pr * (pr.clamp_min(1e-30) / mean.clamp_min(1e-30)).log()).sum(-1))
            self.previous[pk] = [x.detach().cpu() for x in (qq, oo, support, relation)]
            self.mass[step, branch, layer, :, role] = relation.cpu().numpy()
            self.features[step, branch, layer, :, role] = values.cpu().numpy()

    def save(self, path):
        np.savez_compressed(path, mass=self.mass, features=self.features, counts=self.counts,
                            timestep=self.sigma, roles=ROLES, feature_names=FEATURES,
                            query_sampling_seed=self.seed, samples_per_role=self.samples)


class AttentionStudy:
    def __init__(self, transformer):
        from diffusers.models.transformers import transformer_cosmos3 as module
        self.original = module.dispatch_attention_fn
        self.transformer = transformer
        self.mode = 'bf16'
        self.observer = None
        self.handles = []
        self.handles.append(transformer.register_forward_pre_hook(self.pre, with_kwargs=True))
        self.handles.append(transformer.register_forward_hook(self.post))
        for i, layer in enumerate(transformer.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self.layer_hook(i)))
        module.dispatch_attention_fn = self.dispatch
        self.reset(35, 2)

    def layer_hook(self, index):
        def hook(module, args):
            self.layer = index
        return hook

    def reset(self, steps, branches):
        self.steps, self.branches = steps, branches
        self.call = 0
        self.cache = {}
        self.cache_steps = {}
        self.counts = dict(full=0, base=0, reused_output=0, reused_delta=0, reused_mask=0)
        self.events = []
        self.ages = []
        self.timestep_values = []

    def pre(self, module, args, kwargs):
        self.step, self.branch = divmod(self.call, self.branches)
        self.kwargs = kwargs
        t = kwargs.get('vision_timesteps')
        if t is None or not t.numel():
            t = kwargs.get('action_timesteps')
        self.timestep = float(t[0]) if t is not None and t.numel() else float('nan')
        self.timestep_values.append(self.timestep)
        e = torch.cuda.Event(enable_timing=True); e.record()
        self.events.append([e, None])

    def post(self, module, args, output):
        e = torch.cuda.Event(enable_timing=True); e.record()
        self.events[-1][1] = e
        self.call += 1

    def dispatch(self, q, k, v, **kwargs):
        if kwargs.get('is_causal', False):
            return self.original(q, k, v, **kwargs)
        if kwargs.get('attn_mask') is not None or kwargs.get('parallel_config') is not None:
            raise ValueError('Research backend requires unmasked noncausal single-GPU generation')
        mode = self.mode
        key = (self.branch, self.layer, tuple(q.shape), tuple(k.shape))
        if mode in ('bf16', 'observe') or mode.startswith('ffn_'):
            out = self.original(q, k, v, **kwargs)
            self.counts['full'] += 1
        elif mode.startswith('output_reuse'):
            late = mode.endswith('_late')
            active = not late or self.step >= self.steps // 2
            refresh = self.step % 2 == 0 or key not in self.cache or not active
            if refresh:
                out = self.original(q, k, v, **kwargs)
                self.cache[key] = out.detach().clone()
                self.cache_steps[key] = self.step
                self.counts['full'] += 1
            else:
                out = self.cache[key].clone()
                self.counts['reused_output'] += 1
                self.ages.append(self.step - self.cache_steps[key])
        elif mode.startswith('delta_reuse'):
            late = mode.endswith('_late')
            refresh = self.step % 2 == 0 or key not in self.cache or (late and self.step < self.steps // 2)
            base = fp4(q, k, v, config(0))[0]
            self.counts['base'] += 1
            if refresh:
                out = fp4(q, k, v, config(.5))[0]
                self.cache[key] = out.float() - base.float()
                self.cache_steps[key] = self.step
                self.counts['full'] += 1
            else:
                out = (base.float() + self.cache[key]).to(q.dtype)
                self.counts['reused_delta'] += 1
                self.ages.append(self.step - self.cache_steps[key])
        elif mode.startswith('mask_reuse'):
            refresh = self.step % 2 == 0 or key not in self.cache
            if refresh:
                out, mask, _ = fp4(q, k, v, config(.5), return_mask=True)
                self.cache[key] = mask.detach().clone()
                self.cache_steps[key] = self.step
            else:
                out = fp4(q, k, v, config(.5), repair_mask=self.cache[key])[0]
                self.counts['reused_mask'] += 1
                self.ages.append(self.step - self.cache_steps[key])
            self.counts['full'] += 1
        else:
            cfgs = {'attn_base': (0, 3), 'attn_act2': (1, 2), 'attn_dual25': (.25, 3),
                    'attn_dual50': (.5, 3), 'attn_dual75': (.75, 3),
                    'attn_dual100': (1, 3), 'attn_four100': (1, 4)}
            if mode not in cfgs:
                raise ValueError(mode)
            out = fp4(q, k, v, config(*cfgs[mode]))[0]
            self.counts['full'] += 1
        if self.observer is not None:
            self.observer.observe(q, k, v, out, self.kwargs, self.step, self.branch, self.layer, self.timestep)
        return out

"""Exact offline AR prefixes and one-device Super DM workers.

This deployment adapter leaves the frozen numerical implementation unchanged.
It never substitutes an old denoising output: only the text-only BF16 AR branch,
which is independent of the video noise, seed, denoising step and DM quantizer,
is persisted. Every request validates its actual prefix against the cache key.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

from baselines import sha, write
from super_runtime import is_ar

HERE = Path(__file__).resolve().parent
PREFIX_FIELDS = ('input_ids', 'text_indexes', 'position_ids', 'und_len',
                 'sequence_length', 'vision_sequence_indexes',
                 'sound_sequence_indexes', 'action_sequence_indexes')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prefix_cpu(kwargs):
    import torch
    result = {}
    for name in PREFIX_FIELDS:
        value = kwargs.get(name)
        result[name] = value.detach().contiguous().cpu() if isinstance(value, torch.Tensor) else value
    for name in ('vision_sequence_indexes', 'sound_sequence_indexes', 'action_sequence_indexes'):
        value = result[name]
        if value is not None and value.numel() and bool((value < result['und_len']).any()):
            raise ValueError('Offline AR requires a text-only understanding prefix')
    return result


def describe_prefix(prefix):
    import torch
    result = {}
    for name in PREFIX_FIELDS:
        value = prefix[name]
        result[name] = (dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            if isinstance(value, torch.Tensor) else value)
    return result


class PrefixBank:
    def __init__(self, root, model_identity):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['model_identity'] != model_identity:
            raise ValueError('AR cache/model identity mismatch')
        self.manifest_sha256 = sha(self.root / 'manifest.json')
        self.loaded = {}

    def lookup(self, kwargs):
        prefix = prefix_cpu(kwargs)
        descriptor = describe_prefix(prefix)
        key = digest(descriptor)
        entry = self.manifest['prefixes'].get(key)
        if entry is None:
            raise ValueError('Unprepared AR input: ' + key)
        if key not in self.loaded:
            import torch
            path = self.root / entry['file']
            if not path.resolve().is_relative_to(self.root.resolve()) or sha(path) != entry['sha256']:
                raise ValueError('Corrupt or unsafe AR cache asset')
            data = torch.load(path, map_location='cpu', weights_only=True)
            if data['descriptor'] != descriptor or data['model_identity'] != self.manifest['model_identity']:
                raise ValueError('Changed AR cache input identity')
            if len(data['rows']) != self.manifest['layers']:
                raise ValueError('Incomplete AR cache layers')
            # Only the current case is resident on the CPU. The unconditional
            # prefix may be shared across cases, but no unbounded tensor cache.
            self.loaded[key] = data['rows']
        return key, self.loaded[key]


def single_device():
    from thunderworld_fp4.cosmos.resources import allowed_devices
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if raw not in tuple(map(str, allowed_devices())):
        raise ValueError('One explicitly authorized GPU required')
    return int(raw)


def place_partition(pipe, partition):
    """Discard the unused branch after the unchanged official CPU load."""
    import torch
    if partition not in ('ar', 'dm'):
        raise ValueError('Unknown model partition')
    kept = dropped = 0
    for name, module in pipe.transformer.named_modules():
        # Direct modality embeddings are tiny and determine the pipeline's
        # execution-device discovery before any transformer forward is called.
        keep = (not is_ar(name)) if partition == 'dm' else (is_ar(name) or name in ('', 'embed_tokens', 'rotary_emb'))
        for key, value in list(module.named_parameters(recurse=False)):
            size = value.numel() * value.element_size()
            if keep:
                value.data = value.data.to('cuda:0')
                kept += size
            else:
                module._parameters[key] = torch.nn.Parameter(torch.empty_like(value, device='meta'), requires_grad=False)
                dropped += size
        for key, value in list(module.named_buffers(recurse=False)):
            module._buffers[key] = value.to('cuda:0') if keep else torch.empty_like(value, device='meta')
    for name, component in pipe.components.items():
        if name != 'transformer' and isinstance(component, torch.nn.Module):
            component.to('cuda:0')
    pipe._super_cached_placement = dict(partition=partition, physical_gpu=single_device(),
        resident_transformer_bytes=kept, discarded_transformer_bytes=dropped,
        unused_weights_device='meta', arithmetic_changed=False)
    return pipe


@contextmanager
def placement(partition, owner=None):
    from diffusers import Cosmos3OmniPipeline
    original = Cosmos3OmniPipeline.to
    def placed(pipe, *args, **kwargs):
        if (owner is None or pipe is owner.pipe) and args == ('cuda',) and not kwargs:
            return place_partition(pipe, partition)
        return original(pipe, *args, **kwargs)
    Cosmos3OmniPipeline.to = placed
    try:
        yield
    finally:
        Cosmos3OmniPipeline.to = original


def cache_class(base, bank):
    import torch
    from thunderworld_fp4.cosmos.trees import signature
    class OfflineARCache(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.branch_slots = {}
            self.request_keys = {}
            self.bound_keys = {}
            self.cache_load_seconds = 0.

        def begin_request(self, task):
            super().begin_request(task)
            self.request_keys.clear()

        def prepare_call(self, kwargs):
            if torch.is_grad_enabled() or self.model.training:
                raise RuntimeError('AR cache requires inference/eval mode')
            branch = self.engine.call % self.engine.branches
            # No host readback in the 35-step loop. Validate/load once per CFG
            # branch at request entry, matching the original AR prefill scope.
            if branch not in self.request_keys:
                start = time.perf_counter()
                key, rows = bank.lookup(kwargs)
                slot = (branch, kwargs['und_len'], kwargs['sequence_length'],
                    signature(kwargs['input_ids']), signature(kwargs['position_ids']),
                    torch.cuda.current_stream().cuda_stream)
                if slot not in self.slots:
                    if not self.engine.warming:
                        raise RuntimeError('Unwarmed offline AR input')
                    self.slots[slot] = [{n: row[n].to('cuda:0') for n in ('k', 'v', 'out')} for row in rows]
                    self.bound_keys[slot] = key
                elif self.bound_keys.get(slot) != key:
                    # Existing DM graphs retain these exact buffer addresses.
                    for target, row in zip(self.slots[slot], rows):
                        for name, value in target.items():
                            value.copy_(row[name])
                    self.bound_keys[slot] = key
                self.branch_slots[branch] = slot
                self.request_keys[branch] = key
                self.cache_load_seconds += time.perf_counter() - start
                # Bound CPU tensors to this request's conditional/unconditional pair.
                bank.loaded = {k: v for k, v in bank.loaded.items() if k in self.request_keys.values()}
            self.active = self.slots[self.branch_slots[branch]]
            self.calls += 1

        def receipt(self):
            value = super().receipt()
            value.update(mode='offline_exact_BF16_AR_prefix', scope='Content-addressed text-only AR outputs; actual prefix validated once per request/CFG. Current noisy DM inputs are always recomputed.',
                cache_manifest_sha256=bank.manifest_sha256, current_prefix_keys=self.request_keys,
                cache_load_seconds=self.cache_load_seconds, offline_ar_compute_in_request_seconds=False,
                performance_eligible=False)
            return value
    return OfflineARCache


def activate(bank_root, deployment=None):
    import torch
    import thunderworld_fp4.cosmos as package
    from thunderworld_fp4.cosmos import ar_cache as ar_module, resources
    from thunderworld_fp4.cosmos.runner import CosmosRunner as OriginalRunner
    single_device()
    original_admit = resources.select_shared_device
    resources.select_shared_device = lambda minimum_mib=88*1024: original_admit(minimum_mib)
    original_cache = ar_module.ARCache

    class CachedRunner(OriginalRunner):
        @torch.inference_mode()
        def prepare(self, export=None):
            if not self.ar_cache_enabled or self.audio or self.config.resource_mode != 'shared-calibration':
                raise ValueError('Cached Super supports AR-cached I2V quality only')
            self.prefix_bank = PrefixBank(bank_root, self.identity)
            ar_module.ARCache = cache_class(original_cache, self.prefix_bank)
            try:
                with placement('dm', self):
                    return super().prepare(export)
            finally:
                ar_module.ARCache = original_cache

        def receipt(self):
            value = super().receipt()
            value.update(super_cached_placement=self.pipe._super_cached_placement,
                super_cached_runtime_sha256=sha(__file__), deployment_manifest=deployment,
                ar_cache_manifest_sha256=self.prefix_bank.manifest_sha256)
            return value

        @torch.inference_mode()
        def infer(self, *args, **kwargs):
            signals, row = super().infer(*args, **kwargs)
            row.update(timing_scope='cached_AR_DM_request_including_cache_lookup_excluding_offline_prefill',
                super_cached_runtime_sha256=sha(__file__), deployment_manifest=deployment,
                ar_cache_manifest_sha256=self.prefix_bank.manifest_sha256,
                ar_prefix_keys=dict(self.ar_cache.request_keys), physical_gpu=single_device(),
                performance_eligible=False)
            return signals, row
    package.CosmosRunner = CachedRunner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--deployment', type=Path)
    p.add_argument('script', type=Path)
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.script.resolve().is_relative_to(HERE):
        raise ValueError('Expected a contribution-study tool')
    deployment = None if a.deployment is None else dict(path=str(a.deployment.resolve()), sha256=sha(a.deployment))
    activate(a.bank, deployment)
    sys.argv = [str(a.script), *a.args]
    runpy.run_path(str(a.script), run_name='__main__')


if __name__ == '__main__':
    main()

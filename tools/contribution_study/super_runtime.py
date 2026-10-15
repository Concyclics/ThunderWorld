"""Isolated, exact two-device placement for Super quality experiments.

The 64B MoT checkpoint has two ~32B branches. Keep the generation branch on
logical CUDA 0 and the unchanged BF16 understanding branch on logical CUDA 1.
Only fresh per-request, per-CFG AR cache tensors cross devices. No weights,
arithmetic, sampling settings or quantization scopes are substituted.

activate() is process-local and deliberately leaves the frozen Edge/Nano source
files untouched. Run existing study tools through this module's `run` command.
"""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
from types import SimpleNamespace

from baselines import sha

HERE = Path(__file__).resolve().parent
AR_LAYER = re.compile(r"layers\.\d+\.(?:input_layernorm|post_attention_layernorm|mlp)(?:\.|$)")
AR_ATTN = re.compile(r"layers\.\d+\.self_attn\.(?:to_q|to_k|to_v|to_out|norm_q|norm_k|k_norm_und_for_gen)(?:\.|$)")


def is_ar(name):
    return bool(AR_LAYER.match(name) or AR_ATTN.match(name))


def physical_pair():
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not re.fullmatch(r'\d+,\d+', raw):
        raise ValueError('Super requires two explicit CUDA_VISIBLE_DEVICES, generation first')
    pair = tuple(map(int, raw.split(',')))
    allowed = tuple(map(int, os.environ.get('TW_SHARED_CALIBRATION_GPUS', '0,2').split(',')))
    if pair[0] == pair[1] or not set(pair) <= set(allowed):
        raise ValueError('Super device pair is not authorized')
    return pair


def admit(minimum_mib=None):
    from thunderworld_fp4.cosmos.resources import free_memory
    pair = physical_pair()
    free = free_memory()
    # Includes space for BF16 teacher + temporary native repair operators.
    thresholds = (88 * 1024, 72 * 1024) if minimum_mib is None else (minimum_mib,) * 2
    for device, required in zip(pair, thresholds):
        if free.get(device, 0) < required:
            raise RuntimeError(f'PAUSED: GPU{device} has {free.get(device, 0)} MiB; requires {required}')
    return dict(devices=list(pair), logical_generation=0, logical_understanding=1,
                free_mib={str(d): free[d] for d in pair}, minimum_mib=list(thresholds),
                resource_mode='shared-calibration', performance_eligible=False)


@contextmanager
def pair_worker(lock_path=None):
    pair = physical_pair()
    path = Path(lock_path or f'/tmp/thunderworld-shared-calibration-{os.getuid()}.lock')
    with ExitStack() as stack:
        for file, flag in [(path, fcntl.LOCK_SH), *[(path.with_name(path.name + f'.gpu{d}'), fcntl.LOCK_EX) for d in sorted(pair)]]:
            handle = stack.enter_context(file.open('a+'))
            try:
                fcntl.flock(handle, flag | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('PAUSED: another worker owns the Super device pair') from exc
        yield


def place_pipeline(pipe):
    import torch
    model = pipe.transformer
    counts = {0: 0, 1: 0}
    # Move direct parameters/buffers exactly once; never materialize the whole
    # 128 GB checkpoint on either GPU, even transiently.
    for name, module in model.named_modules():
        device = 1 if is_ar(name) else 0
        for value in module.parameters(recurse=False):
            value.data = value.data.to(f'cuda:{device}')
            counts[device] += value.numel() * value.element_size()
        for key, value in module.named_buffers(recurse=False):
            module._buffers[key] = value.to(f'cuda:{device}')
    for name, component in pipe.components.items():
        if name != 'transformer' and isinstance(component, torch.nn.Module):
            component.to('cuda:0')
    # The main forward still embeds its text prefix. A small, exact duplicate
    # embedding avoids remote lookups in the separate AR prefill graph.
    proxy = SimpleNamespace(config=model.config, layers=model.layers,
        embed_tokens=copy.deepcopy(model.embed_tokens).to('cuda:1'),
        rotary_emb=copy.deepcopy(model.rotary_emb).to('cuda:1'))
    model._super_ar_proxy = proxy
    pipe._super_placement = dict(kind='BF16_AR_on_second_device_DM_on_first',
        physical_devices=list(physical_pair()), transformer_bytes=counts,
        duplicate_embedding_bytes=sum(p.numel()*p.element_size() for p in proxy.embed_tokens.parameters()),
        arithmetic_changed=False, performance_eligible=False)
    for device in (0, 1):
        torch.cuda.synchronize(device)
    return pipe


def activate():
    """Install only within this single-request experiment subprocess."""
    import torch
    import thunderworld_fp4.cosmos as package
    from thunderworld_fp4.cosmos import resources
    from thunderworld_fp4.cosmos import ar_cache as ar_module
    from thunderworld_fp4.cosmos.runner import CosmosRunner as OriginalRunner
    from thunderworld_fp4.cosmos.trees import signature, clone_tree, copy_tree
    if getattr(package.CosmosRunner, '_super_two_device', False):
        return
    physical_pair()
    resources.shared_worker = pair_worker
    resources.select_shared_device = admit
    original_cache = ar_module.ARCache
    from thunderworld_fp4.cosmos import linears as linear_module
    original_linear = linear_module.RepairLinear

    class CalibrationRepairLinear(original_linear):
        def forward(self, *args, **kwargs):
            # The teacher intervention tools use a SimpleNamespace controller
            # and prepare every call. Do not retain two FP4 planes for *all*
            # 32B teacher linears alongside their BF16 weights. Online deployed
            # JointEngine instances retain their usual resident packed banks.
            ephemeral = type(self.controller) is SimpleNamespace and self.controller.warming
            try:
                return super().forward(*args, **kwargs)
            finally:
                if ephemeral:
                    self.ops.clear()
                    self.bound = None
                    self.weight_bank = None

    linear_module.RepairLinear = CalibrationRepairLinear

    class TwoDeviceARCache(original_cache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # torch.cuda.graph's implicit side stream is process-global and
            # may have been created on CUDA0. Always bind the AR stream to 1.
            self.ar_capture_stream = torch.cuda.Stream(device=1)

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
                    raise RuntimeError('AR prefix capacity reached')
                prefix = {name: kwargs.get(name) for name in ('input_ids', 'text_indexes',
                    'position_ids', 'und_len', 'sequence_length', 'vision_sequence_indexes',
                    'sound_sequence_indexes', 'action_sequence_indexes')}
                for name in ('vision_sequence_indexes', 'sound_sequence_indexes', 'action_sequence_indexes'):
                    index = prefix[name]
                    if index is not None and index.numel() and bool((index < prefix['und_len']).any()):
                        raise ValueError('AR cache requires a text-only understanding prefix')
                with torch.cuda.device(1):
                    remote = {name: value.to('cuda:1') if isinstance(value, torch.Tensor) else value
                              for name, value in prefix.items()}
                    if key not in self.slots:
                        static = clone_tree(remote)
                        for _ in range(2):
                            ar_module.prefill(self.model._super_ar_proxy, static, validate=False)
                        torch.cuda.synchronize(1)
                        graph = torch.cuda.CUDAGraph()
                        start = time.perf_counter()
                        with torch.cuda.graph(graph, stream=self.ar_capture_stream):
                            rows = ar_module.prefill(self.model._super_ar_proxy, static, validate=False)
                        torch.cuda.synchronize(1)
                        self.capture_seconds += time.perf_counter() - start
                        self.graphs[key] = (static, graph, rows)
                    static, graph, rows = self.graphs[key]
                    copy_tree(static, remote)
                    graph.replay()
                    # Once per CFG/request, before DM graph execution. This is
                    # not a per-step diagnostic sync or host timestep readback.
                    torch.cuda.synchronize(1)
                if key not in self.slots:
                    self.slots[key] = [{n: row[n].to('cuda:0') for n in ('k', 'v', 'out')} for row in rows]
                else:
                    for target, row in zip(self.slots[key], rows):
                        for name, value in target.items():
                            value.copy_(row[name])
                self.seen.add(key)
                self.prefills += 1
                self.request_prefills += 1
            self.active = self.slots[key]
            self.calls += 1

        def receipt(self):
            value = super().receipt()
            value.update(placement='AR CUDA1, DM CUDA0; fresh cache copied per request/CFG',
                remote_cache_bytes=sum(t.numel()*t.element_size() for _, _, rows in self.graphs.values()
                    for row in rows for t in row.values()), performance_eligible=False)
            return value

    class SuperRunner(OriginalRunner):
        _super_two_device = True

        @torch.inference_mode()
        def prepare(self, export=None):
            from diffusers import Cosmos3OmniPipeline
            if not self.ar_cache_enabled or self.audio or self.config.resource_mode != 'shared-calibration':
                raise ValueError('Super placement currently supports AR-cached I2V quality only')
            original_to = Cosmos3OmniPipeline.to

            def placed(pipe, *args, **kwargs):
                if pipe is self.pipe and args == ('cuda',) and not kwargs:
                    return place_pipeline(pipe)
                return original_to(pipe, *args, **kwargs)

            Cosmos3OmniPipeline.to = placed
            try:
                return super().prepare(export)
            finally:
                Cosmos3OmniPipeline.to = original_to

        def receipt(self):
            result = super().receipt()
            result['super_placement'] = self.pipe._super_placement
            result['super_runtime_sha256'] = sha(__file__)
            result['memory_by_device'] = {str(i): dict(allocated=torch.cuda.memory_allocated(i),
                reserved=torch.cuda.memory_reserved(i), peak_allocated=torch.cuda.max_memory_allocated(i)) for i in (0, 1)}
            return result

    ar_module.ARCache = TwoDeviceARCache
    package.CosmosRunner = SuperRunner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('run',))
    p.add_argument('script', type=Path)
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.script.resolve().is_relative_to(HERE):
        raise ValueError('Expected an existing contribution-study tool')
    activate()
    sys.argv = [str(a.script), *a.args]
    runpy.run_path(str(a.script), run_name='__main__')


if __name__ == '__main__':
    main()

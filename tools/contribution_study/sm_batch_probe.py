"""Process-local SM partitions and independent-input operator groups.

This is a shared-GPU diagnostic, never a formal speed ranking. The frozen
quality backend is retained. Each FP4 slot has its own current-input amax and
repair mask. BF16, per-token FP8, and W4A16 additionally get a concatenated-M
baseline; restricting those baselines to small independent GEMMs is misleading.
No global MPS settings, production runner state, or quantization assets change.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

from baselines import sha, write


def driver_value(result):
    from cuda.bindings import driver
    if result[0] != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(str(result))
    return result[1]


def stream_partition(stream):
    from cuda.bindings import driver
    green = driver_value(driver.cuStreamGetGreenCtx(stream.cuda_stream))
    if int(green) == 0:
        return None
    resource = driver_value(driver.cuGreenCtxGetDevResource(
        green, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    return int(resource.sm.smCount)


def checked_load(folder, entry):
    import torch
    path = folder / entry['file']
    if sha(path) != entry['sha256']:
        raise ValueError('Changed input asset: ' + str(path))
    return torch.load(path, map_location='cpu', weights_only=True)


def make_method(method, weight, asset, awq, plan, name, m, branch, phase):
    import torch
    from thunderworld_fp4.cosmos.linears import FP8Linear, RepairLinear
    from thunderworld_fp4.cosmos.settings import LinearSpec
    n, k = weight.shape
    source = torch.nn.Linear(k, n, bias=False, device='meta', dtype=torch.bfloat16)
    source.weight = torch.nn.Parameter(weight.cuda(), requires_grad=False)
    if method == 'bf16':
        return source
    if method == 'fp8':
        return FP8Linear(source)
    if method == 'awq':
        from awq_marlin import make_linear
        return make_linear(awq)
    spec = plan.linear(plan.value['task'], branch, phase, name) if method == 'repair' else LinearSpec(0, 0)
    controller = SimpleNamespace(budget_plan=plan if method == 'repair' else None,
        config=SimpleNamespace(kernel_variant='optimized', linear_backend='triton'))
    if method == 'repair':
        from thunderworld_fp4.cosmos.assets import expand
        union = sorted({i for key, row in plan.linears.items()
                        if key.split('|', 2)[-1] == name for i in row['w_blocks']})
        asset = expand(asset, union)
    layer = RepairLinear(source, controller, name, spec, asset if method == 'repair' else None)
    layer.prepare_shape(m, spec)
    return layer


def independent_slot(layer, method):
    """Share only read-only tensors; clone mutable selectors / Marlin workspace."""
    if method in ('bf16', 'fp8'):
        return layer
    slot = copy.copy(layer)
    slot._buffers = layer._buffers.copy()
    if method == 'awq':
        slot.workspace = layer.workspace.clone()
    else:
        slot.bound = copy.copy(layer.bound)
        slot.bound.am = slot.bound.union = layer.bound.am.clone()
        slot.ops = {}  # This frozen operator probe never selects another spec.
    return slot


def delta(actual, expected):
    import torch
    diff = actual.float() - expected.float()
    return dict(bitwise_equal=torch.equal(actual, expected),
                max_abs=float(diff.abs().max()),
                relative_l2=float(diff.norm() / expected.float().norm().clamp_min(1e-20)))


def group_probe(method, layer, panel, slots, parent, worker_streams, expected_sms, rounds, replays):
    import torch
    # Changes are deterministic and intentionally differ between slots. This is
    # an operator stress input, not a new quality dataset or server workload.
    batch = torch.stack([panel.cuda() + i * .015625 for i in range(slots)])
    inputs = list(batch.unbind(0))
    layers = [independent_slot(layer, method) for _ in inputs]
    streams = worker_streams[:slots]
    if any(stream_partition(s) != expected_sms for s in streams):
        raise RuntimeError('A worker stream escaped the selected SM partition')
    if method in ('fp4', 'repair'):
        assert len({s.bound.am.data_ptr() for s in layers}) == slots
        assert len({s.bound.wc.data_ptr() for s in layers}) == 1
    if method == 'awq':
        assert len({s.workspace.data_ptr() for s in layers}) == slots
        assert len({s.weight.data_ptr() for s in layers}) == 1

    def call(op, x):
        return op._compute(x) if method in ('fp4', 'repair') else op(x)

    def serial():
        return [call(op, x) for op, x in zip(layers, inputs)]

    def concurrent():
        current = torch.cuda.current_stream()
        outputs = []
        for op, x, stream in zip(layers, inputs, streams):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                x.record_stream(stream)
                outputs.append(call(op, x))
        for output, stream in zip(outputs, streams):
            current.wait_stream(stream)
            output.record_stream(current)
        return outputs

    def concatenated():
        # FP8 uses per-token A scales; AWQ keeps activations in BF16. Unlike
        # current FP4 global-amax packing, neither merges quantization domains.
        return list(call(layer, batch.flatten(0, 1)).reshape(slots, panel.shape[0], -1).unbind(0))

    functions = dict(serial=serial, concurrent=concurrent)
    if method in ('bf16', 'fp8', 'awq'):
        functions['concatenated'] = concatenated
    graphs, outputs = {}, {}
    with torch.cuda.stream(parent):
        for name, fn in functions.items():
            for _ in range(3):
                fn()
            parent.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=parent):
                output = fn()
            graphs[name], outputs[name] = graph, output
        checks = []
        for shift in (0., .03125, -.0625):
            batch.add_(shift)
            reference = [t.clone() for t in serial()]
            for name, graph in graphs.items():
                graph.replay()
                parent.synchronize()
                if len({t.data_ptr() for t in outputs[name]}) != slots:
                    raise AssertionError('Output slots alias')
                differences = [delta(x, y) for x, y in zip(outputs[name], reference)]
                if name != 'concatenated' and not all(d['bitwise_equal'] for d in differences):
                    raise AssertionError(f'{method}/{name}: current-input outputs changed')
                checks.append(dict(mode=name, shift=shift, differences=differences))
        # Events surround the whole graph group; no per-kernel timing hooks.
        measured = {name: [] for name in graphs}
        rng = random.Random(20260918)
        for _ in range(rounds):
            order = list(graphs)
            rng.shuffle(order)
            for name in order:
                for _ in range(10):
                    graphs[name].replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record(parent)
                for _ in range(replays):
                    graphs[name].replay()
                end.record(parent)
                end.synchronize()
                measured[name].append(start.elapsed_time(end) / replays)
        medians = {name: statistics.median(values) for name, values in measured.items()}
        return dict(slots=slots, samples_ms=measured, group_ms=medians,
            ops_per_second={name: slots * 1000 / ms for name, ms in medians.items()},
            concurrent_over_serial=medians['serial'] / medians['concurrent'],
            checks=checks, performance_eligible=False,
            serial_output_sha256=[hashlib.sha256(t.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                                  for t in reference],
            allocation_note='Diagnostic: all captured execution-mode graph pools coexist; not deployment peak memory')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--awq-assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sms', type=int, nargs='+', default=[0, 48, 96, 144], help='0: unrestricted primary context')
    p.add_argument('--slots', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--modules', nargs='+', default=['layers.0.self_attn.add_q_proj',
        'layers.0.self_attn.add_k_proj', 'layers.0.mlp_moe_gen.down_proj'])
    p.add_argument('--methods', nargs='+', choices=['bf16', 'fp8', 'awq', 'fp4', 'repair'],
                   default=['bf16', 'fp8', 'awq', 'fp4', 'repair'])
    p.add_argument('--scene', type=int, default=0)
    p.add_argument('--phase', type=int, choices=range(4), default=0)
    p.add_argument('--branch', type=int, choices=range(2), default=0)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--replays', type=int, default=20)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if any(n < 0 for n in a.sms) or any(n < 1 or n > 8 for n in a.slots) or min(a.rounds, a.replays) < 1:
        p.error('Invalid SM count, slots, or repetition count')
    if any(k.startswith('CUDA_MPS_') for k in os.environ):
        raise RuntimeError('Do not mix MPS resource limits with Green Context diagnostics')
    capture = json.loads((a.capture / 'manifest.json').read_text())
    assets = json.loads((a.assets / 'manifest.json').read_text())
    awq = json.loads((a.awq_assets / 'manifest.json').read_text())
    from thunderworld_fp4.cosmos.budget import RepairPlan
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device
    plan = RepairPlan.load(a.plan)
    if not capture['complete'] or not assets['complete'] or not awq['complete']:
        raise ValueError('Complete, frozen input assets required')
    if not (capture['model_identity'] == assets['model_identity'] == awq['model_identity'] == plan.value['model_identity']):
        raise ValueError('Model identities differ')
    result = dict(complete=False, performance_eligible=False, source_sha256=sha(__file__), rows=[],
        declared_request_budget=plan.value['budget'],
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        inputs_sha256={k: sha(v) for k, v in dict(capture=a.capture/'manifest.json',
            assets=a.assets/'manifest.json', awq=a.awq_assets/'manifest.json', plan=a.plan).items()},
        scope='Shared-GPU operator diagnostics with synthetic per-slot perturbations of a real full-M panel; not E2E, quality qualification, best-baseline ranking, or a concurrency-safe Cosmos service',
        repair_backend='Frozen quality-path Triton; no silent substitution with a different CUTLASS accumulation order',
        limitations=['Fixed legal launches, no tuning on shared load',
            'Only selected layer/phase/CFG panels; request budget does not equal each operator budget',
            'ARCQuant host-alpha and SharQ native fusion need separate batch/Graph qualification',
            'Green Context partitions SMs, not proportional DRAM bandwidth or a smaller GPU emulator',
            'Concatenated BF16/FP8/AWQ can change accumulation; non-bitwise results require full trajectory qualification'])
    with shared_worker():
        result['resource'] = select_shared_device()
        import torch
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            torch.empty(1, device='cuda')  # Establish the primary context first.
            props = torch.cuda.get_device_properties(0)
            result['device'] = dict(name=props.name, physical_sms=props.multi_processor_count,
                torch=torch.__version__, cuda=torch.version.cuda)
            write(a.output, result)
            try:
                for requested in a.sms:
                    if requested > props.multi_processor_count:
                        raise ValueError('Requested SM count exceeds device')
                    green = torch.cuda.GreenContext.create(requested, 0) if requested else None
                    if green:
                        green.set_context()
                    owned_streams, workers = [], []
                    try:
                        parent = green.Stream() if green else torch.cuda.Stream()
                        allocated = stream_partition(parent)
                        if requested and allocated != requested:
                            raise ValueError(f'Requested {requested} SMs but allocated {allocated}')
                        if green:
                            # torch.cuda.Stream() uses a primary-context pool in
                            # this Torch version even after set_context(). Create
                            # workers explicitly in the GC and verify each one.
                            from cuda.bindings import driver
                            handle = driver_value(driver.cuStreamGetGreenCtx(parent.cuda_stream))
                            for _ in range(max(a.slots)):
                                owned_streams.append(driver_value(driver.cuGreenCtxStreamCreate(
                                    handle, int(driver.CUstream_flags.CU_STREAM_NON_BLOCKING), 0)))
                            workers = [torch.cuda.ExternalStream(int(s), device=0) for s in owned_streams]
                        else:
                            workers = [torch.cuda.Stream() for _ in range(max(a.slots))]
                        with torch.cuda.stream(parent):
                            for name in a.modules:
                                row = next(r for r in capture['rows'] if r['name'] == name and
                                    r['phase'] == a.phase and r['branch'] == a.branch and r['scene_index'] == a.scene)
                                panel = checked_load(a.capture, row)['x']
                                weight = checked_load(a.capture, capture['weights'][name])
                                asset = checked_load(a.assets, assets['modules'][name])
                                packed_awq = checked_load(a.awq_assets, awq['modules'][name])
                                methods = list(a.methods)
                                random.Random(20260918 + requested).shuffle(methods)
                                for method in methods:
                                    layer = make_method(method, weight, asset, packed_awq, plan,
                                                        name, len(panel), a.branch, a.phase)
                                    for count in a.slots:
                                        record = group_probe(method, layer, panel, count, parent,
                                                             workers, allocated, a.rounds, a.replays)
                                        record.update(method=method, module=name, shape=row['shape'],
                                            requested_sms=requested, allocated_sms=allocated or props.multi_processor_count,
                                            context='green' if green else 'primary',
                                            frozen_spec=json.loads(layer.default.key) if method in ('fp4', 'repair') else None)
                                        if method in ('fp4', 'repair'):
                                            from thunderworld_fp4.cosmos.budget import linear_cost
                                            na, _ = layer.default.counts(*weight.shape)
                                            record['per_slot_compute'] = linear_cost(len(panel), *weight.shape,
                                                a_blocks=na, w_blocks=layer.default.weight_support or (),
                                                cross=layer.default.cross)
                                        prior = next((r for r in result['rows'] if
                                            (r['method'], r['module'], r['slots']) == (method, name, count)), None)
                                        record['same_output_as_first_partition'] = (
                                            record['serial_output_sha256'] == prior['serial_output_sha256']) if prior else None
                                        if method in ('fp4', 'repair') and record['same_output_as_first_partition'] is False:
                                            raise AssertionError('SM partition changed frozen FP4 output')
                                        result['rows'].append(record)
                                        write(a.output, result)
                                        print(json.dumps({k: record[k] for k in ('method', 'module', 'slots', 'allocated_sms', 'group_ms')}), flush=True)
                                    del layer
                                    gc.collect()
                                    torch.cuda.empty_cache()
                        parent.synchronize()
                    finally:
                        if owned_streams:
                            for stream in workers:
                                stream.synchronize()
                            gc.collect()
                            torch.cuda.empty_cache()
                            for handle in owned_streams:
                                status = driver.cuStreamDestroy(handle)[0]
                                if status != driver.CUresult.CUDA_SUCCESS:
                                    raise RuntimeError(str(status))
                        if green:
                            green.pop_context()
                result['complete'] = True
                write(a.output, result)
            except Exception as error:
                result.update(paused=True, error=repr(error))
                write(a.output, result)
                raise


if __name__ == '__main__':
    main()

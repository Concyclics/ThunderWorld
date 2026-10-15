"""Complete decoded-video throughput of independent, non-reentrant runners.

Cosmos' upstream pipeline accepts only one sample (a prompt list is truncated).
Here batch means concurrently admitted complete requests, not a merged-M GEMM.
Each slot owns its model, RNG, scheduler, AR/DM graphs, selectors and outputs.
Default-stream-only official ARC/SharQ kernels stay on that stream. The worker
does not port/tune them, and does not mistake their host overhead for MMA cost.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager
from functools import wraps
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import time

from baselines import sha, write


def install_threadsafe_sdpa():
    """Protect Torch's process-wide backend flags from concurrent VAE dispatch.

    The selected algorithms and arguments stay unchanged. Lock only host
    dispatch/context changes, never GPU completion or whole inference requests.
    Install once in this isolated worker before any runner is prepared.
    """
    import torch
    lock = threading.RLock()
    original_context = torch.nn.attention.sdpa_kernel
    original_dispatch = torch.nn.functional.scaled_dot_product_attention

    @contextmanager
    def context(*args, **kwargs):
        with lock:
            with original_context(*args, **kwargs) as value:
                yield value

    @wraps(original_dispatch)
    def dispatch(*args, **kwargs):
        with lock:
            return original_dispatch(*args, **kwargs)

    torch.nn.attention.sdpa_kernel = context
    torch.nn.functional.scaled_dot_product_attention = dispatch


def hashes(signals):
    return {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in signals.items()}


def complete_request(slot, case, seed, gate, shared):
    import numpy as np
    import torch
    runner, stream = slot['runner'], slot['stream']
    with torch.inference_mode(), torch.cuda.device(0), torch.cuda.stream(stream):
        gate.wait(timeout=60)
        # Same body and output conversion as CosmosRunner.infer, with global
        # synchronize/reset moved to the whole-batch boundary. Never invoke
        # mutable runner state from two concurrent requests on the same slot.
        runner._context(case, 35, 'waveform')
        output = runner.pipe(**runner._args(case, seed, 35))
        signals = {'video': np.asarray(output.video)}
        if output.action is not None:
            signals['action'] = output.action[0].float().cpu().numpy()
        if runner.sound_latent is not None:
            signals['sound_latent'] = runner.sound_latent.float().cpu().numpy()
        if output.sound is not None:
            signals['waveform'] = output.sound.float().cpu().numpy()
        stream.synchronize()
        elapsed = time.perf_counter() - shared['start']
        if runner.engine.call != 35 * runner.engine.branches:
            raise AssertionError('Incorrect complete-request step/CFG count')
        if runner.diagnostics.mode != 'off' or runner.diagnostics.events:
            raise AssertionError('Diagnostic events leaked into throughput run')
        if len(runner.engine.cache) != slot['captures']:
            raise AssertionError('New Graph capture occurred during inference')
        return signals, elapsed


def group(pool, slots, case, seeds):
    import torch
    from thunderworld_fp4.cosmos.cli import idle_guard
    idle_guard()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gate = threading.Barrier(len(slots) + 1)
    shared = {}
    futures = [pool.submit(complete_request, slot, case, seed, gate, shared)
               for slot, seed in zip(slots, seeds)]
    shared['start'] = time.perf_counter()
    gate.wait(timeout=60)
    try:
        outputs = [future.result(timeout=120) for future in futures]
    except FutureTimeout:
        # A stuck CUDA launch cannot be cleaned up by synchronizing or by the
        # ThreadPoolExecutor's blocking shutdown. OS exit releases only this
        # experiment context; the suite retains the last completed measurement.
        print('A complete request exceeded the 120-second concurrency watchdog', flush=True)
        os._exit(124)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - shared['start']
    idle_guard()
    return outputs, dict(group_seconds=seconds, videos_per_second=len(slots)/seconds,
        frames_per_second=len(slots)*case['frames']/seconds,
        completion_seconds=[duration for _, duration in outputs],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), performance_eligible=False)


def check_outputs(outputs, seeds, expected):
    import numpy as np
    for index, ((signals, _), seed) in enumerate(zip(outputs, seeds)):
        if not all(np.isfinite(value).all() for value in signals.values()):
            raise AssertionError('Nonfinite complete video')
        if hashes(signals) != expected[str(seed)]:
            raise AssertionError(f'Concurrent complete output differs for slot {index}, seed {seed}: '
                                 f'{hashes(signals)} != {expected[str(seed)]}')
    # The CPU output of each request must remain independent after all workers
    # complete and after subsequent requests reuse the graph's device buffers.
    for i, (a, _) in enumerate(outputs):
        for b, _ in outputs[i+1:]:
            if any(np.shares_memory(a[key], b[key]) for key in a):
                raise AssertionError('Concurrent output storage aliases')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--waves', type=int, default=3)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--reserve-mib', type=int, default=12*1024,
                   help='Free-memory reserve before adding a replica; use 2048 for the 24 GiB 5090 handoff. Original PRO 6000 default unchanged.')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.batches != sorted(set(a.batches)) or any(b not in (1, 2, 4, 8) for b in a.batches):
        p.error('Use increasing distinct batches from 1,2,4,8')
    if min(a.rounds, a.waves) < 1:
        p.error('Positive repetition counts required')
    if a.reserve_mib < 2048:
        p.error('Keep at least 2048 MiB reserved')
    if not a.pilot and (a.rounds != 5 or a.waves != 3):
        p.error('Frozen complete sweep requires five rounds, three waves')
    manifest_path = a.study/'manifest.json'
    study = json.loads(manifest_path.read_text())
    if a.method not in study['batch_protocol']['methods']:
        raise ValueError('Method is not in the frozen throughput cohort')
    entry = next(e for e in study['cases'] if e['case']['id'] == a.case)
    case = entry['case']
    if case['task'] != 'i2v' or sha(case['image']) != entry['image_sha256']:
        raise ValueError('Only unchanged I2V inputs are qualified')
    quality_path = a.study/'generation'/a.method/a.case/'result.json'
    quality = json.loads(quality_path.read_text())
    if not quality['complete'] or quality['identity']['manifest_sha256'] != sha(manifest_path):
        raise ValueError('Matching complete quality outputs required')
    result = dict(complete=False, performance_eligible=False, pilot=a.pilot, rows=[],
        method=a.method, case=case, manifest_sha256=sha(manifest_path),
        quality_receipt_sha256=sha(quality_path), source_sha256=sha(__file__),
        runtime_source_sha256=sha(Path(__file__).with_name('baseline_runtime.py')),
        batches=a.batches, rounds=a.rounds, waves=a.waves, reserve_mib=a.reserve_mib,
        batch_kind=study['batch_protocol']['kind'],
        timing_scope='complete decoded-video request, including image read, new noise, AR prefill, 35 steps, VAE decode, CPU output conversion; exclude disk serialization and post-timing hashes',
        memory_policy='Independent resident runner/model per slot; weights are replicated. Not a merged-batch or shared-weight serving-memory upper bound.',
        concurrent_quality_policy='Speed-only as requested: batch > 1 has no numerical output qualification. Single-request quality must not be attributed to concurrent outputs.',
        limitations=['Fixed frozen launches, no baseline-specific or shared-load tuning',
            'Current implementation throughput, not a hardware or algorithmic maximum',
            'ARC/SharQ pinned kernels use the default stream and eager host dispatch',
            'Idle GPU checked at each group boundary; this diagnostic is not the strict-tuning formal leaderboard'])
    from thunderworld_fp4.cosmos.resources import shared_worker, select_shared_device, free_memory
    with shared_worker():
        result['resource'] = select_shared_device()
        from thunderworld_fp4.cosmos.cli import idle_guard
        idle_guard()
        import torch
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        install_threadsafe_sdpa()
        result['attention_dispatch_policy'] = 'Process-local reentrant lock around SDPA backend context and host dispatch; unchanged kernels, no GPU synchronization.'
        from baseline_runtime import make_runner, default_stream_only
        single_stream = default_stream_only(study, a.method)
        result['stream_policy'] = 'official_default_stream' if single_stream else 'one_current_stream_per_request'
        result['graph_capture_policy'] = 'Separate capture stream per resident runner; no cross-runner library-workspace reuse through Torch default capture stream.'
        slots, expected = [], {}
        # Three fixed seeds are reused across rounds, with rotation across slots.
        seeds = [study['batch_protocol']['seed_base'] + i for i in range(3)]
        result['timing_seeds'] = seeds
        a.output.parent.mkdir(parents=True, exist_ok=True)
        write(a.output, result)
        flat_count = 0
        previous_throughput = None
        per_slot_reserve = 12 * 1024
        try:
            for batch in a.batches:
                before = time.perf_counter()
                while len(slots) < batch:
                    available = free_memory()[result['resource']['device']]
                    if available < a.reserve_mib + per_slot_reserve:
                        result.update(complete=True, stop_reason='memory_admission_before_next_slot',
                                      requested_batch=batch, free_mib=available)
                        write(a.output, result)
                        return
                    idle_guard()
                    memory_before = torch.cuda.memory_allocated()
                    runner, adapter = make_runner(study, a.method)
                    stream = torch.cuda.default_stream() if single_stream else torch.cuda.Stream()
                    capture_stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    # All captures are sequential and outside measurement. The
                    # upstream runner does not pass an explicit capture stream;
                    # isolate the default for this preparation only, then restore
                    # it. AR and DM graphs retain independent library workspaces.
                    previous_capture_stream = torch.cuda.graph.default_capture_stream
                    torch.cuda.graph.default_capture_stream = capture_stream
                    try:
                        with torch.cuda.stream(stream):
                            runner.warmup(case, 35)
                            runner.infer(case, 6099, 35)
                    finally:
                        torch.cuda.graph.default_capture_stream = previous_capture_stream
                    slots.append(dict(runner=runner, stream=stream, capture_stream=capture_stream,
                                      captures=len(runner.engine.cache)))
                    per_slot_reserve = max(per_slot_reserve,
                        int((torch.cuda.memory_allocated()-memory_before)/2**20) + 2048)
                    if len(slots) == 1:
                        result.update(adapter=adapter, runtime=runner.receipt())
                        # Reproduce the frozen quality receipt before extending
                        # to timing seeds. Stream changes cannot substitute new quality.
                        with torch.cuda.stream(stream):
                            for row in quality['rows']:
                                signals, _ = runner.infer(case, row['seed'], 35)
                                if hashes(signals) != row['sha256']:
                                    raise AssertionError('Throughput path differs from frozen full quality output')
                            for seed in seeds:
                                signals, _ = runner.infer(case, seed, 35)
                                expected[str(seed)] = hashes(signals)
                        result['serial_reference_sha256'] = expected
                    print(json.dumps(dict(method=a.method, case=a.case, prepared_slots=len(slots),
                                          requested_batch=batch)), flush=True)
                prep_seconds = time.perf_counter()-before
                # The thread pool and all runner state live across warmup and
                # measurements. No model load or Graph capture occurs in group().
                with ThreadPoolExecutor(max_workers=batch) as pool:
                    initial_seeds = [seeds[i % len(seeds)] for i in range(batch)]
                    retained, _ = group(pool, slots[:batch], case, initial_seeds)
                    if batch == 1:
                        check_outputs(retained, initial_seeds, expected)
                    changed_seeds = [seeds[(i+1) % len(seeds)] for i in range(batch)]
                    changed, _ = group(pool, slots[:batch], case, changed_seeds)
                    if batch == 1:
                        check_outputs(changed, changed_seeds, expected)
                        check_outputs(retained, initial_seeds, expected)
                    del retained, changed
                    measurements = []
                    for round_id in range(a.rounds):
                        for wave in range(a.waves):
                            group_seeds = [seeds[(i+wave) % len(seeds)] for i in range(batch)]
                            outputs, row = group(pool, slots[:batch], case, group_seeds)
                            if batch == 1:
                                check_outputs(outputs, group_seeds, expected)
                            row.update(round=round_id, wave=wave, seeds=group_seeds,
                                       numerical_checked=batch == 1,
                                       full_output_bitwise_equal=True if batch == 1 else None)
                            measurements.append(row)
                            del outputs
                            # Persist after each complete group for safe interruption.
                            result['active'] = dict(batch=batch, measurements=measurements)
                            write(a.output, result)
                    latency = [v for r in measurements for v in r['completion_seconds']]
                    import numpy as np
                    summary = dict(batch=batch, measurements=measurements,
                        median_videos_per_second=statistics.median(r['videos_per_second'] for r in measurements),
                        median_group_seconds=statistics.median(r['group_seconds'] for r in measurements),
                        request_p50_seconds=float(np.percentile(latency, 50)),
                        request_p90_seconds=float(np.percentile(latency, 90)),
                        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in measurements),
                        peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in measurements),
                        additional_prepare_seconds=prep_seconds, numerical_checked=batch == 1,
                        bitwise_equal=True if batch == 1 else None, performance_eligible=False)
                    result['rows'].append(summary)
                    result.pop('active', None)
                    write(a.output, result)
                    print(json.dumps(dict(method=a.method, case=a.case, batch=batch,
                        videos_per_second=summary['median_videos_per_second'],
                        group_seconds=summary['median_group_seconds'])), flush=True)
                    if previous_throughput is not None:
                        flat_count = flat_count+1 if summary['median_videos_per_second'] < previous_throughput*1.05 else 0
                    previous_throughput = summary['median_videos_per_second']
                    if flat_count >= 2:
                        result['stop_reason'] = 'observed_two_doubling_plateau_below_5_percent'
                        break
            result.update(complete=True, stop_reason=result.get('stop_reason', 'maximum_predeclared_batch_reached'))
            write(a.output, result)
        except Exception as error:
            result.update(paused=True, error=repr(error))
            write(a.output, result)
            raise
        finally:
            slots.clear()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

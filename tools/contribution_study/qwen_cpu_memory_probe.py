"""CPU-only reproduction of the judge's native video decode workload.

No model weights, GPU context, scoring requests, or benchmark result writes.
Every sampled RGB array must match a full PyAV decode of the frozen source.
Run in a bounded systemd user service; retain samples even on interruption.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

from qwen_resource_session import cgroup_path, resource_snapshot
from qwen_task_judge import load_jobs, verify_manifest
from qwen_task_protocol import now, sha, write


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU probe requires CUDA_VISIBLE_DEVICES to be empty')
    group = cgroup_path()
    cap = (group / 'memory.max').read_text().strip()
    if cap == 'max' or int(cap) > 32 * 1024**3:
        raise RuntimeError('CPU probe requires a memory cgroup capped at 32 GiB or less')
    args.output.mkdir(parents=True, exist_ok=False)
    def stage(name, **extra):
        value = dict(utc=now(), stage=name, **extra)
        write(args.output / 'stage.json', value)
        print(json.dumps(value), flush=True)

    stage('verify_manifest')
    _, manifest_hash = verify_manifest(args.root)
    stage('import_video_libraries')
    import av
    import cv2
    import numpy as np
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY
    loader = VIDEO_LOADER_REGISTRY.load('opencv')
    stage('prepare_independent_references')
    indices = list(range(0, 61, 4))
    grouped = defaultdict(list)
    for job in load_jobs(args.root):
        if job['benchmark'] == 'paibench':
            grouped[(job['model'], job['method'])].append(job)
    rng = random.Random(9232026)
    jobs = [job for key in sorted(grouped) for job in rng.sample(grouped[key], 4)]
    gold = {}
    for job in jobs:
        stage('reference_decode', job_id=job['id'], references_done=len(gold))
        path = Path(job['source_video'])
        if sha(path) != job['source_video_sha256']:
            raise ValueError('Frozen source video changed')
        selected = []
        full = hashlib.sha256()
        frames = 0
        with av.open(str(path)) as source:
            stream = source.streams.video[0]
            stream.thread_count = 1
            for index, frame in enumerate(source.decode(stream)):
                rgb = frame.to_ndarray(format='rgb24')
                full.update(rgb.tobytes())
                if index in indices:
                    selected.append(rgb)
                frames += 1
        if frames != 61 or full.hexdigest() != job['source_rgb_sha256']:
            raise ValueError('Independent RGB reference does not match the frozen input')
        gold[job['id']] = hashlib.sha256(np.stack(selected).tobytes()).hexdigest()
    del selected, rgb
    # Do not introduce an extra open/release sequence: the production path is
    # load_bytes. A stand-alone release probe crashes this OpenCV build and is
    # not evidence of a failure in the production workload.
    actual_threads = None
    settings = dict(utc=now(), manifest_sha256=manifest_hash, judge_inference=False,
        gpu_allowed=False, workers=args.workers, repeats=args.repeats, unique_videos=len(jobs),
        ffmpeg_threads_environment=os.environ.get('OPENCV_FFMPEG_THREADS', 'unset'),
        actual_ffmpeg_threads=actual_threads, expected_frame_indices=indices,
        cgroup=str(group), source_code_sha256=sha(Path(__file__)),
        native_loader_code_sha256=sha(Path(__import__('vllm.multimodal.video', fromlist=['']).__file__)),
        sources=[dict(id=j['id'], path=j['source_video'], sha256=j['source_video_sha256'],
                      selected_rgb_sha256=gold[j['id']]) for j in jobs])
    write(args.output / 'settings.json', settings)
    stage('native_decode_stress')

    def decode(job):
        frames, metadata = loader.load_bytes(Path(job['source_video']).read_bytes(), num_frames=16)
        if (frames.shape != (16, 192, 320, 3) or metadata['frames_indices'] != indices
                or metadata['fps'] != 24 or metadata['total_num_frames'] != 61
                or hashlib.sha256(frames.tobytes()).hexdigest() != gold[job['id']]):
            raise ValueError('Native decode differs from the independent frozen RGB reference')
        return 1

    started = time.monotonic()
    observations = []
    with (args.output / 'samples.jsonl').open('a') as log:
        def sample(completed):
            observation = resource_snapshot(group)
            observation.update(completed_decodes=completed, elapsed_seconds=time.monotonic()-started,
                               open_fds=len(list(Path('/proc/self/fd').iterdir())))
            observations.append(observation)
            log.write(json.dumps(observation) + '\n')
            log.flush()
            write(args.output / 'progress.json', observation)
        sample(0)
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for _ in range(args.repeats):
                completed += sum(executor.map(decode, jobs))
                sample(completed)
        gc.collect()
        sample(completed)
    warm = observations[min(8, len(observations)-1)]
    final = observations[-1]
    result = dict(complete=True, completed_decodes=completed, elapsed_seconds=time.monotonic()-started,
        all_rgb_checks_passed=True, actual_ffmpeg_threads=actual_threads,
        baseline_pss_bytes=observations[0]['totals']['pss_bytes'],
        post_warmup_pss_bytes=warm['totals']['pss_bytes'],
        final_pss_bytes=final['totals']['pss_bytes'],
        pss_growth_after_warmup_bytes=final['totals']['pss_bytes']-warm['totals']['pss_bytes'],
        peak_sampled_pss_bytes=max(x['totals']['pss_bytes'] for x in observations),
        peak_sampled_threads=max(x['totals']['threads'] for x in observations),
        kernel_cgroup_peak_bytes=final['cgroup_peak_bytes'],
        memory_events=final['cgroup_events'],
        limitations='CPU native decoding only; does not reproduce vLLM inference, processor caches, or historic host conditions')
    write(args.output / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.repeats <= 256:
        parser.error('Use bounded worker and repeat counts')
    run(args)

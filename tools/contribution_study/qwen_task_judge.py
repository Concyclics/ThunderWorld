"""Concurrent, resumable Qwen3-VL/vLLM task judging of the full video cohort."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import time

import httpx

from qwen_task_protocol import (MODEL_ID, MODEL_REVISION, SAMPLE_SEED, build_inputs,
    digest, frame_indices, now, parse_response, protocol, rbench_indices, read,
    scene_queries, sha, upstream, write)

HERE = Path(__file__).resolve()


class JudgeClient:
    def __init__(self, endpoint, concurrency=32):
        self.endpoint = endpoint
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(base_url=endpoint + '/', timeout=600,
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency))

    async def close(self):
        await self.client.aclose()

    async def request(self, messages, *, max_tokens=1024, schema=None, choice=None):
        body = dict(model='qwen3-vl-8b-fp8-judge', messages=messages, temperature=0,
                    top_p=1, seed=SAMPLE_SEED, max_tokens=max_tokens)
        if schema:
            body['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'benchmark_score', 'schema': schema}}
        if choice:
            body['structured_outputs'] = {'choice': choice}
        async with self.semaphore:
            started = time.time()
            r = await self.client.post('chat/completions', json=body)
            r.raise_for_status()
            result = r.json()
        return dict(raw=result, endpoint=self.endpoint, elapsed_seconds=time.time() - started,
                    request_sha256=digest(body), utc=now())


async def get_clients(root):
    state = read(root / 'servers.json')
    clients = [JudgeClient(r['endpoint']) for r in state['replicas']]
    for client in clients:
        r = await client.client.get('models')
        r.raise_for_status()
        if 'qwen3-vl-8b-fp8-judge' not in [m['id'] for m in r.json()['data']]:
            raise RuntimeError('Wrong model served on judge endpoint')
    return clients


def messages(query, media=None):
    content = []
    if media:
        typ = 'image_url' if media['kind'] == 'grid' else 'video_url'
        content.append({'type': typ, typ: {'url': Path(media['path']).as_uri()}})
    content.append({'type': 'text', 'text': query['prompt']})
    result = []
    if query.get('system'):
        result.append({'role': 'system', 'content': query['system']})
    result.append({'role': 'user', 'content': content})
    return result


async def prepare(a):
    if (a.root / 'manifest.json').exists():
        verify_manifest(a.root)
        print('Existing frozen manifest verified; use a new root for a different protocol', flush=True)
        return
    inputs, funcs = build_inputs(a.root, a.metrics_root, a.cosmos_root)
    clients = await get_clients(a.root)
    chains = {}
    chain_sem = asyncio.Semaphore(8)

    async def make_chain(i, cid, scene):
        prompt = funcs['visual_reasoning']['create_question_chain'](scene['prompt'])
        path = a.root / 'chains' / (cid + '.json')
        identity = digest([prompt, MODEL_ID, MODEL_REVISION, SAMPLE_SEED])
        if path.exists():
            saved = read(path)
            if saved['identity'] != identity:
                raise ValueError('Frozen chain provenance mismatch')
            chains[cid] = saved['questions']
            return
        attempts = []
        async with chain_sem:
            for trial in range(3):
                response = await clients[i % len(clients)].request(
                    [{'role': 'user', 'content': prompt}], max_tokens=1024)
                attempts.append(response)
                output = response['raw']['choices'][0]
                text = output['message']['content'].strip()
                count = len(re.findall(r'^\s*\d+[.)]\s+.+', text, re.M))
                if output['finish_reason'] == 'stop' and 3 <= count <= 5:
                    write(path, dict(identity=identity, questions=text, prompt=prompt,
                                     attempts=attempts, source='upstream create_question_chain; Qwen text-only'))
                    chains[cid] = text
                    return
        write(path.with_suffix('.failure.json'), dict(attempts=attempts))
        raise ValueError('Invalid visual reasoning question chain: ' + cid)

    try:
        await asyncio.gather(*(make_chain(i, cid, s) for i, (cid, s) in
            enumerate(inputs['scenes'].items()) if s['category'] == 'visual_reasoning'))
    finally:
        await asyncio.gather(*(c.close() for c in clients))
    scenes = inputs['scenes']
    for cid, scene in scenes.items():
        scene['queries'] = scene_queries(scene, funcs, chains.get(cid))
        for q in scene['queries']:
            q['sha256'] = digest(q)
    write(a.root / 'scenes.json', scenes)
    jobs = sorted(inputs['jobs'], key=lambda j: j['id'])
    with (a.root / 'jobs.jsonl').open('w') as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + '\n')
    query_count = sum(len(scenes[j['case_id']]['queries']) for j in jobs)
    code_paths = [HERE, HERE.with_name('qwen_task_protocol.py')]
    manifest = dict(frozen=True, created_utc=now(), protocol=protocol(),
        inputs=dict(scenes_sha256=sha(a.root / 'scenes.json'), jobs_sha256=sha(a.root / 'jobs.jsonl'),
                    upstream=inputs['source_hashes'], metric_manifest_sha256=inputs['metric_manifest_sha256']),
        code_sha256={str(p): sha(p) for p in code_paths},
        chain_sha256={cid: sha(a.root / 'chains' / (cid + '.json')) for cid in chains},
        counts=dict(videos=len(jobs), scenes=len(scenes), judge_requests=query_count,
            binary_requests=sum(q['kind'] == 'binary' for j in jobs for q in scenes[j['case_id']]['queries']),
            empty_annotation_scenes=len(inputs['empty_official_vqa'])),
        dependencies={n: importlib.metadata.version(n) for n in ['vllm', 'torch', 'transformers', 'httpx', 'av']},
        server_configuration=read(a.root / 'servers.json'),
        flashinfer_runtime='isolated flashinfer-python 0.6.13 + flashinfer-cubin 0.6.13 (vLLM pinned requirements)',
        model_weights=read(a.root / 'model_weights_verified.json'))
    write(a.root / 'manifest.json', manifest)
    print(json.dumps(manifest['counts']), flush=True)


def verify_manifest(root):
    m = read(root / 'manifest.json')
    if not m['frozen'] or m['protocol'] != protocol():
        raise ValueError('Changed judge protocol requires a new root')
    for name in ('scenes', 'jobs'):
        p = root / (name + ('.jsonl' if name == 'jobs' else '.json'))
        if sha(p) != m['inputs'][name + '_sha256']:
            raise ValueError('Frozen input file changed')
    for p, expected in m['code_sha256'].items():
        if sha(p) != expected:
            raise ValueError('Frozen scoring implementation changed: ' + p)
    return m, sha(root / 'manifest.json')


def load_jobs(root):
    return [json.loads(line) for line in (root / 'jobs.jsonl').read_text().splitlines()]


def stop_servers(root):
    state = read(root / 'servers.json')
    control = Path(state.get('control_root', str(root)))
    (control / 'STOP_SERVERS').write_text(now() + '\n')


async def reuse(a):
    """Import completed answers only when every judging input is provably identical."""
    if a.from_root is None:
        raise ValueError('reuse requires --from-root')
    old_root = a.from_root.resolve()
    old_manifest = read(old_root / 'manifest.json')
    new_manifest, new_hash = verify_manifest(a.root)
    old_hash = sha(old_root / 'manifest.json')
    old_protocol = dict(old_manifest['protocol'])
    new_protocol = dict(new_manifest['protocol'])
    old_protocol.pop('methods')
    new_protocol.pop('methods')
    if old_protocol != new_protocol:
        raise ValueError('Cannot reuse answers across different judge protocols')
    snapshots = read(old_root / 'frozen_code_receipt.json')
    for path, expected in old_manifest['code_sha256'].items():
        if snapshots[path]['sha256'] != expected or sha(snapshots[path]['copy']) != expected:
            raise ValueError('Old scoring-code provenance not preserved')
    for name in ('scenes', 'jobs'):
        path = old_root / (name + ('.jsonl' if name == 'jobs' else '.json'))
        if sha(path) != old_manifest['inputs'][name + '_sha256']:
            raise ValueError('Old frozen input changed')
    old_scenes, new_scenes = read(old_root / 'scenes.json'), read(a.root / 'scenes.json')
    if old_scenes != new_scenes:
        raise ValueError('Scene questions/annotations differ; reuse is forbidden')
    old_jobs = {j['id']: j for j in load_jobs(old_root)}
    new_jobs = {j['id']: j for j in load_jobs(a.root)}
    imported = []
    for path in sorted((old_root / 'results').glob('*.json')):
        result = read(path)
        old_job = old_jobs[result['job']['id']]
        new_job = new_jobs.get(old_job['id'])
        if new_job is None:
            continue
        if old_job != new_job or result['job_sha256'] != digest(old_job) or result['manifest_sha256'] != old_hash or not result['complete']:
            raise ValueError('Reused video identity mismatch')
        if sha(result['media']['path']) != result['media']['sha256']:
            raise ValueError('Reused judge media changed')
        for answer, query in zip(result['answers'], new_scenes[new_job['case_id']]['queries'], strict=True):
            if answer['identity'] != digest([old_hash, old_job, query['sha256'], result['media'], '']):
                raise ValueError('Reused response identity mismatch')
            out = answer['response']['raw']['choices'][0]
            if parse_response(query, out['message']['content'], out['finish_reason']) != answer['parsed']:
                raise ValueError('Reused response parsing changed')
        target = a.root / 'results' / path.name
        if target.exists():
            saved = read(target)
            if saved['manifest_sha256'] != new_hash or saved['job_sha256'] != digest(new_job):
                raise ValueError('New result identity mismatch')
            continue
        proof = dict(source=str(path), source_sha256=sha(path), source_manifest_sha256=old_hash,
                     identical_queries=True, identical_media=True, imported_utc=now())
        result['manifest_sha256'] = new_hash
        result['reused_from'] = proof
        write(target, result)
        imported.append(proof)
    receipt = a.root / ('reuse_audit_' + old_hash[:16] + '.json')
    if not receipt.exists():
        write(receipt, dict(complete=True, manifest_sha256=new_hash, imported=len(imported),
            source_root=str(old_root), source_manifest_sha256=old_hash, proofs=imported))
    print(json.dumps(dict(imported=len(imported), source=str(old_root))), flush=True)


def make_media(root, job):
    source = Path(job['source_video'])
    if sha(source) != job['source_video_sha256']:
        raise ValueError('Source lossless video hash mismatch: ' + job['id'])
    directory = root / 'media'
    directory.mkdir(exist_ok=True)
    if job['benchmark'] == 'paibench':
        target = directory / (job['id'] + '.mkv')
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError('Wrong media symlink')
        elif target.exists():
            raise ValueError('Unexpected non-symlink media file')
        else:
            target.symlink_to(source)
        return dict(kind='video', path=str(target), sha256=job['source_video_sha256'],
                    frame_indices=frame_indices(61, 16), fps=24, total_num_frames=61)
    import av
    import numpy as np
    from PIL import Image
    target = directory / (job['id'] + '.png')
    receipt = target.with_suffix('.json')
    if receipt.exists():
        saved = read(receipt)
        if saved['source_sha256'] != job['source_video_sha256'] or sha(target) != saved['sha256']:
            raise ValueError('Cached grid media changed')
        return saved
    indices = rbench_indices()
    frames = []
    rgb_hash = __import__('hashlib').sha256()
    count = 0
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        if float(stream.average_rate) != 24:
            raise ValueError('Unexpected source fps')
        for i, frame in enumerate(container.decode(stream)):
            rgb = frame.to_ndarray(format='rgb24')
            if rgb.shape != (192, 320, 3):
                raise ValueError('Unexpected source shape')
            rgb_hash.update(rgb.tobytes())
            if i in indices:
                frames.append(rgb)
            count += 1
    if count != 61 or len(frames) != 6:
        raise ValueError('Incomplete frame decode')
    if rgb_hash.hexdigest() != job['source_rgb_sha256']:
        raise ValueError('Decoded RGB differs from verified lossless source')
    grid = np.concatenate([np.concatenate(frames[i:i+2], axis=1) for i in (0, 2, 4)], axis=0)
    temporary = target.with_suffix('.tmp.png')
    Image.fromarray(grid).save(temporary)
    temporary.replace(target)
    data = dict(kind='grid', path=str(target), sha256=sha(target),
                source_sha256=job['source_video_sha256'], frame_indices=indices,
                fps=24, total_num_frames=count, width=640, height=576)
    write(receipt, data)
    return data


async def answer(root, client, job, query, media, manifest_hash, *, suffix=''):
    path = root / 'responses' / job['id'] / (query['sha256'][:20] + suffix + '.json')
    identity = digest([manifest_hash, job, query['sha256'], media, suffix])
    if path.exists():
        saved = read(path)
        if saved['identity'] != identity:
            raise ValueError('Response resume identity mismatch')
        parsed = parse_response(query, saved['response']['raw']['choices'][0]['message']['content'],
                                 saved['response']['raw']['choices'][0]['finish_reason'])
        if parsed != saved['parsed']:
            raise ValueError('Response resume parser mismatch')
        return saved
    attempts = []
    for trial in range(3):
        try:
            response = await client.request(messages(query, media),
                max_tokens=query['max_tokens'] * (2 if trial else 1),
                schema=query.get('schema'), choice=['YES', 'NO'] if query['kind'] == 'binary' else None)
            attempts.append(response)
            out = response['raw']['choices'][0]
            parsed = parse_response(query, out['message']['content'], out['finish_reason'])
            result = dict(identity=identity, query_id=query['id'], query_sha256=query['sha256'],
                response=response, parsed=parsed, attempts=attempts, complete=True)
            write(path, result)
            return result
        except Exception as exc:
            attempts.append(dict(error=repr(exc), utc=now()))
            write(path.with_suffix('.attempts.json'), dict(identity=identity, attempts=attempts))
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (400, 401, 403, 404, 413, 422):
                raise RuntimeError(f'Judge request rejected: {exc.response.text[:2000]}') from exc
            await asyncio.sleep(1 + trial)
    raise RuntimeError(f'Judge response failed after three attempts: {job["id"]} {query["id"]}')


def summarize_video(scene, answers):
    if scene['benchmark'] == 'rbench':
        p = answers[0]['parsed']
        return dict(task_score=p['score'], task_score_normalized=p['normalized_score'],
                    aspects=p['aspects'], official_domain_accuracy=None)
    binary = [a['parsed']['correct'] for a, q in zip(answers, scene['queries']) if q['kind'] == 'binary']
    return dict(official_domain_accuracy=sum(binary) / len(binary) if binary else None,
                binary_questions=len(binary), supplemental_adherence=answers[0]['parsed'].get('score') if not binary else None)


def pilot_jobs(jobs, scenes):
    chosen = {}
    for job in sorted(jobs, key=lambda j: (j['case_id'], j['seed'])):
        key = (job['model'], job['method'], job['benchmark'], job['category'])
        chosen.setdefault(key, job)
    for job in jobs:
        if job['benchmark'] == 'paibench' and not scenes[job['case_id']]['annotation']:
            chosen.setdefault((job['model'], job['method'], 'empty_annotation'), job)
    return list(chosen.values())


async def score(a):
    manifest, manifest_hash = verify_manifest(a.root)
    scenes = read(a.root / 'scenes.json')
    jobs = load_jobs(a.root)
    if a.pilot:
        jobs = pilot_jobs(jobs, scenes)
    elif not (a.root / 'pilot_complete.json').exists():
        raise RuntimeError('Run the category-covering pilot before the full cohort')
    random.Random(SAMPLE_SEED).shuffle(jobs)
    clients = await get_clients(a.root)
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    stats = dict(started_utc=now(), started_unix=time.time(), pid=os.getpid(),
        manifest_sha256=manifest_hash, pilot=a.pilot, total_videos=len(jobs),
        completed=0, resumed=0, failed=0, completed_requests=0, by_group={}, active_ids=[])
    stop = asyncio.Event()
    active = set()
    errors = []
    group_counts = Counter()
    progress_path = a.root / ('pilot_progress.json' if a.pilot else 'progress.json')
    (a.root / 'results').mkdir(exist_ok=True)

    def update():
        stats['updated_utc'] = now()
        stats['elapsed_seconds'] = time.time() - stats['started_unix']
        stats['by_group'] = dict(group_counts)
        stats['active_ids'] = sorted(active)
        stats['remaining'] = len(jobs) - stats['completed'] - stats['failed']
        write(progress_path, stats)

    async def worker(client):
        while not stop.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            active.add(job['id'])
            try:
                path = a.root / 'results' / (job['id'] + '.json')
                if path.exists():
                    result = read(path)
                    if result['manifest_sha256'] != manifest_hash or result['job_sha256'] != digest(job) or not result['complete']:
                        raise ValueError('Video result resume identity mismatch')
                    stats['resumed'] += 1
                else:
                    media = await asyncio.to_thread(make_media, a.root, job)
                    scene = scenes[job['case_id']]
                    answers = await asyncio.gather(*(answer(a.root, client, job, q, media, manifest_hash)
                        for q in scene['queries']))
                    result = dict(complete=True, utc=now(), manifest_sha256=manifest_hash,
                        job_sha256=digest(job), job=job, media=media, answers=answers,
                        metrics=summarize_video(scene, answers))
                    write(path, result)
                stats['completed'] += 1
                stats['completed_requests'] += len(scenes[job['case_id']]['queries'])
                group_counts['/'.join([job['model'], job['benchmark'], job['method']])] += 1
            except Exception as exc:
                stats['failed'] += 1
                failure = dict(job=job, error=repr(exc), utc=now())
                errors.append(failure)
                write(a.root / 'failures' / (job['id'] + '.json'), failure)
                # Do not turn a broken service into thousands of failed scores.
                stop.set()
            finally:
                active.discard(job['id'])
                queue.task_done()
                if stats['completed'] % 25 == 0 or stats['failed']:
                    update()

    async def heartbeat():
        while not stop.is_set():
            update()
            try:
                await asyncio.wait_for(stop.wait(), timeout=20)
            except asyncio.TimeoutError:
                pass

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*(worker(client) for client in clients for _ in range(a.video_workers)))
    finally:
        stop.set()
        await heartbeat_task
        update()
        await asyncio.gather(*(c.close() for c in clients))
    if errors:
        raise RuntimeError(f'{len(errors)} video(s) failed; paused, successful responses retained: {errors[0]["error"]}')
    if stats['completed'] != len(jobs):
        raise RuntimeError('Incomplete queue cannot be marked complete')
    write(a.root / ('pilot_complete.json' if a.pilot else 'scoring_complete.json'), stats)
    print(json.dumps(stats), flush=True)
    if not a.pilot:
        from qwen_task_report import report
        report(a.root)


async def qualify(a):
    """Ensure both replicas consume actual image/video inputs; controls are not benchmark scores."""
    import numpy as np
    from PIL import Image
    import av
    manifest, mh = verify_manifest(a.root)
    jobs = load_jobs(a.root)
    clients = await get_clients(a.root)
    responses = []
    black_png = a.root / 'media/ingestion_black.png'
    Image.fromarray(np.zeros((576, 640, 3), dtype=np.uint8)).save(black_png)
    black_video = a.root / 'media/ingestion_black.mkv'
    with av.open(str(black_video), 'w') as c:
        s = c.add_stream('ffv1', rate=24)
        s.width, s.height, s.pix_fmt = 320, 192, 'bgr0'
        for _ in range(61):
            for packet in s.encode(av.VideoFrame.from_ndarray(np.zeros((192, 320, 3), np.uint8), format='rgb24')):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    try:
        for benchmark in ('rbench', 'paibench'):
            job = next(j for j in jobs if j['benchmark'] == benchmark)
            actual = await asyncio.to_thread(make_media, a.root, job)
            # Use vLLM's actual native loader to audit indices, metadata and RGB.
            if benchmark == 'paibench':
                from vllm.multimodal.video import VIDEO_LOADER_REGISTRY
                frames, metadata = VIDEO_LOADER_REGISTRY.load('opencv').load_bytes(
                    Path(job['source_video']).read_bytes(), num_frames=16)
                if metadata['frames_indices'] != frame_indices(61, 16) or frames.shape != (16, 192, 320, 3) or metadata['fps'] != 24:
                    raise ValueError('Native vLLM video sampler differs from frozen protocol')
                write(a.root / 'video_sampling_qualification.json', dict(metadata=metadata,
                    shape=list(frames.shape), passed=True, model=MODEL_ID))
            for i, client in enumerate(clients):
                for black in (False, True):
                    media = actual if not black else dict(kind=actual['kind'],
                        path=str(black_png if benchmark == 'rbench' else black_video))
                    query = dict(kind='binary', expected='NO' if black else 'YES',
                        prompt='Are there any visible people, animals, robots, objects, or scenery in these frames? Answer only YES or NO.',
                        system='Answer the question using only the visual input.')
                    response = await client.request(messages(query, media), max_tokens=16, choice=['YES', 'NO'])
                    out = response['raw']['choices'][0]
                    parsed = parse_response(query, out['message']['content'], out['finish_reason'])
                    responses.append(dict(benchmark=benchmark, replica=i, black=black,
                                          expected=query['expected'], response=response, parsed=parsed))
                    if not parsed['correct']:
                        raise ValueError('Visual ingestion control failed; do not launch full scoring')
        write(a.root / 'ingestion_qualification.json', dict(passed=True, manifest_sha256=mh, responses=responses))
    except Exception as exc:
        write(a.root / 'ingestion_qualification.json', dict(passed=False, error=repr(exc), responses=responses))
        raise
    finally:
        await asyncio.gather(*(c.close() for c in clients))


async def repeat(a):
    """Measure judge repeatability under another batch order, never select the better answer."""
    from qwen_task_report import consistency
    _, mh = verify_manifest(a.root)
    scenes = read(a.root / 'scenes.json')
    jobs = pilot_jobs(load_jobs(a.root), scenes)
    # Fixed by identity, before any score comparison; one per model/category.
    selected = {}
    for job in sorted(jobs, key=lambda j: j['id']):
        selected.setdefault((job['model'], job['benchmark'], job['category']), job)
    clients = await get_clients(a.root)
    results = []

    async def run_one(i, job):
        original = read(a.root / 'results' / (job['id'] + '.json'))
        scene = scenes[job['case_id']]
        media = original['media']
        ans = await asyncio.gather(*(answer(a.root, clients[i % len(clients)], job,
            q, media, mh, suffix='.repeat') for q in scene['queries']))
        metrics = summarize_video(scene, ans)
        comparison = consistency(dict(answers=ans), original, scene)
        name = 'task_score' if job['benchmark'] == 'rbench' else 'official_domain_accuracy'
        ov, rv = original['metrics'].get(name), metrics.get(name)
        results.append(dict(job=job, metrics=metrics, original_metrics=original['metrics'],
            consistency=comparison, absolute_score_change=abs(ov-rv) if ov is not None and rv is not None else None))
    try:
        await asyncio.gather(*(run_one(i, job) for i, job in enumerate(selected.values())))
        write(a.root / 'judge_repeatability.json', dict(complete=True, manifest_sha256=mh,
            videos=len(results), results=results, note='Diagnostic repeated judgements are excluded from primary scores.'))
    finally:
        await asyncio.gather(*(c.close() for c in clients))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['prepare', 'reuse', 'qualify', 'score', 'repeat', 'report'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--metrics-root', type=Path, default=Path('/home/yufeng/thunderworld/judge_full_metrics_20260923'))
    p.add_argument('--cosmos-root', type=Path, default=Path('/home/yufeng/thunderworld/cosmos'))
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--video-workers', type=int, default=12, help='Concurrent videos per replica; requests bounded to 32')
    p.add_argument('--shutdown-servers', action='store_true')
    p.add_argument('--from-root', type=Path)
    a = p.parse_args()
    a.root = a.root.resolve()
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root / 'client.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if a.operation == 'report':
                from qwen_task_report import report
                report(a.root)
            else:
                with ThreadPoolExecutor(max_workers=12) as executor:
                    async def execute():
                        asyncio.get_running_loop().set_default_executor(executor)
                        if a.operation == 'score':
                            qualification = read(a.root / 'ingestion_qualification.json')
                            if not qualification['passed'] or qualification['manifest_sha256'] != sha(a.root / 'manifest.json'):
                                raise RuntimeError('Visual ingestion qualification required')
                        await {'prepare': prepare, 'reuse': reuse, 'qualify': qualify, 'score': score, 'repeat': repeat}[a.operation](a)
                    asyncio.run(execute())
        finally:
            if a.shutdown_servers:
                stop_servers(a.root)


if __name__ == '__main__':
    main()

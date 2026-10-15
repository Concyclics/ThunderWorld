"""Operational recovery of a frozen judge run, without changing its questions.

Invalid model answers are isolated and retried after other videos. They never
become zero scores or completed videos. All numerical/protocol functions remain
the hash-verified originals; this module only changes queue/failure handling.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import random
import time

import httpx

from qwen_task_judge import (answer, get_clients, load_jobs, make_media, qualify,
                             stop_servers, summarize_video, verify_manifest)
from qwen_task_protocol import SAMPLE_SEED, digest, now, parse_response, read, sha, write

HERE = Path(__file__).resolve()


def validate_saved(result, job, scene, manifest_hash):
    if (not result['complete'] or result['manifest_sha256'] != manifest_hash or
            result['job_sha256'] != digest(job) or result['job'] != job):
        raise ValueError('Saved result identity mismatch')
    for a, q in zip(result['answers'], scene['queries'], strict=True):
        if a['query_id'] != q['id'] or a['query_sha256'] != q['sha256']:
            raise ValueError('Saved question identity mismatch')
        choice = a['response']['raw']['choices'][0]
        if parse_response(q, choice['message']['content'], choice['finish_reason']) != a['parsed']:
            raise ValueError('Saved response parsing changed')
    if summarize_video(scene, result['answers']) != result['metrics']:
        raise ValueError('Saved video aggregation changed')


def archive_attempts(root, job_id, tag):
    for path in (root / 'responses' / job_id).glob('*.attempts.json'):
        dest = root / 'retry_history' / job_id / (tag + '_' + path.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise ValueError('Refusing to overwrite retry history')
        path.rename(dest)


async def await_service(root):
    started = time.monotonic()
    while True:
        state = read(root / 'servers.json')
        if state['status'] == 'ready':
            try:
                clients = await get_clients(root)
                await asyncio.gather(*(c.close() for c in clients))
                return
            except httpx.HTTPError:
                pass
        if not Path(f'/proc/{state["pid"]}').exists():
            raise RuntimeError('Judge service exited during startup')
        if time.monotonic() - started > 1800:
            raise TimeoutError('Judge service startup timed out')
        await asyncio.sleep(3)


async def run(a):
    _, mh = verify_manifest(a.root)
    scenes = read(a.root / 'scenes.json')
    jobs = load_jobs(a.root)
    completed, pending = {}, []
    run_id = str(time.time_ns())
    runner_hash = sha(HERE)
    for job in jobs:
        path = a.root / 'results' / (job['id'] + '.json')
        if path.exists():
            result = read(path)
            validate_saved(result, job, scenes[job['case_id']], mh)
            completed[job['id']] = job
        else:
            pending.append(job)
    initial = len(completed)
    started = time.time()
    stats = dict(started_utc=now(), started_unix=started, pid=os.getpid(), manifest_sha256=mh,
        total_videos=len(jobs), completed=initial, resumed=initial, failed=0,
        operation='recovery', runner_sha256=runner_hash, run_id=run_id,
        retry_policy='up to 3 queue passes; original 3-attempt request policy within each pass; first valid response only',
        performance_eligible=False)
    active, unresolved, fatal = set(), {}, []
    groups = Counter('/'.join([j['model'], j['benchmark'], j['method']]) for j in completed.values())
    completed_requests = sum(len(scenes[j['case_id']]['queries']) for j in completed.values())
    stop = asyncio.Event()

    def update(phase):
        stats.update(updated_utc=now(), elapsed_seconds=time.time()-started,
            completed=len(completed), failed=len(unresolved), remaining=len(jobs)-len(completed),
            completed_requests=completed_requests, active_ids=sorted(active), by_group=dict(groups), phase=phase)
        new = len(completed)-initial
        stats['new_videos_per_minute'] = new / max(stats['elapsed_seconds'], 1) * 60
        stats['eta_seconds'] = stats['remaining'] / (new / stats['elapsed_seconds']) if new else None
        write(a.root / 'progress.json', stats)
        write(a.root / 'campaign.json', dict(stage=phase, utc=now(), pid=os.getpid(),
            complete=phase=='complete', operation='recovery', run_id=run_id))

    update('waiting_for_services')
    await await_service(a.root)
    # Recheck both current replicas, including any newly available GPU.
    q = a.root / 'ingestion_qualification.json'
    if q.exists():
        write(a.root / 'history' / ('ingestion_qualification_' + run_id + '.json'), read(q))
    update('qualifying_replicas')
    await qualify(a)
    clients = await get_clients(a.root)

    async def worker(client, queue, pass_index):
        nonlocal completed_requests
        while not stop.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            active.add(job['id'])
            try:
                await asyncio.to_thread(archive_attempts, a.root, job['id'], f'{run_id}_pass{pass_index}')
                media = await asyncio.to_thread(make_media, a.root, job)
                scene = scenes[job['case_id']]
                # Await every in-flight question even if another question fails.
                # Valid question responses are independently checkpointed.
                ans = await asyncio.gather(*(answer(a.root, client, job, q, media, mh)
                    for q in scene['queries']), return_exceptions=True)
                errors = [x for x in ans if isinstance(x, BaseException)]
                if errors:
                    raise errors[0]
                result = dict(complete=True, utc=now(), manifest_sha256=mh,
                    job_sha256=digest(job), job=job, media=media, answers=ans,
                    metrics=summarize_video(scene, ans),
                    queue_runner=dict(run_id=run_id, sha256=runner_hash, pass_index=pass_index))
                write(a.root / 'results' / (job['id'] + '.json'), result)
                completed[job['id']] = job
                unresolved.pop(job['id'], None)
                completed_requests += len(ans)
                groups['/'.join([job['model'], job['benchmark'], job['method']])] += 1
                failure_path = a.root / 'failures' / (job['id'] + '.json')
                if failure_path.exists():
                    dest = a.root / 'resolved_failures' / (job['id'] + '_' + run_id + '.json')
                    dest.parent.mkdir(exist_ok=True)
                    failure_path.rename(dest)
            except Exception as exc:
                failure = dict(job=job, error=repr(exc), utc=now(), pass_index=pass_index,
                    run_id=run_id, unresolved=True, assigned_score=None)
                unresolved[job['id']] = job
                write(a.root / 'failures' / (job['id'] + '.json'), failure)
                # Parsing/truncation failures are local. Identity/IO errors and
                # service failure remain fatal to prevent mass bad requests.
                isolated = isinstance(exc, RuntimeError) and str(exc).startswith('Judge response failed after three attempts:')
                if isolated:
                    try:
                        r = await client.client.get('models', timeout=5)
                        r.raise_for_status()
                    except httpx.HTTPError:
                        isolated = False
                if not isolated or len(unresolved) >= 20:
                    fatal.append(failure)
                    stop.set()
            finally:
                active.discard(job['id'])
                queue.task_done()

    async def heartbeat():
        while not stop.is_set():
            update('full_recovery')
            try:
                await asyncio.wait_for(stop.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        for pass_index in range(1, a.passes+1):
            random.Random(SAMPLE_SEED + pass_index).shuffle(pending)
            # Revisit the previously failing item while other work continues.
            pending.sort(key=lambda j: not (a.root/'failures'/(j['id']+'.json')).exists())
            queue = asyncio.Queue()
            for job in pending:
                queue.put_nowait(job)
            offset = (pass_index - 1) % len(clients)
            ordered_clients = clients[offset:] + clients[:offset]
            await asyncio.gather(*(worker(c, queue, pass_index) for c in ordered_clients for _ in range(a.video_workers)))
            pending = [j for j in jobs if j['id'] not in completed]
            if not pending or fatal:
                break
    finally:
        stop.set()
        await heartbeat_task
        await asyncio.gather(*(c.close() for c in clients))
    if fatal:
        update('paused_service_or_integrity_error')
        raise RuntimeError(f'Queue paused: {fatal[0]["error"]}')
    if pending:
        update('incomplete_invalid_answers')
        write(a.root / 'unresolved_judge_answers.json', dict(complete=False, jobs=pending, manifest_sha256=mh))
        from qwen_task_report import report
        report(a.root)  # Explicitly partial; no completion marker.
        raise RuntimeError(f'{len(pending)} invalid judge answers remain; no fabricated scores')
    update('reporting')
    write(a.root / 'scoring_complete.json', stats)
    from qwen_task_report import report
    result = report(a.root)
    if not result['complete']:
        raise RuntimeError('Queue/report coverage mismatch')
    update('complete')
    write(a.root / 'recovery.json', dict(complete=True, utc=now(), run_id=run_id, results=result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--passes', type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--video-workers', type=int, default=12)
    a = p.parse_args()
    a.root = a.root.resolve()
    with (a.root / 'client.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with ThreadPoolExecutor(max_workers=12) as executor:
                async def execute():
                    asyncio.get_running_loop().set_default_executor(executor)
                    await run(a)
                asyncio.run(execute())
        except BaseException as exc:
            write(a.root / 'campaign_failure.json', dict(utc=now(), operation='recovery', error=repr(exc)))
            raise
        finally:
            stop_servers(a.root)


if __name__ == '__main__':
    main()

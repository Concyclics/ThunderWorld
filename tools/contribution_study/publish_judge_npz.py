"""Checksum, hardlink and publish the original benchmark NPZs with portable indices."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from baselines import sha, write


def now():
    return datetime.now(timezone.utc).isoformat()


def component(value):
    if not value or Path(value).name != value or value in ('.', '..'):
        raise ValueError('Unsafe output path component')
    return value


def prepare(source, output):
    manifest = source/'manifest.json'
    index = source/'judge_index.jsonl'
    job = json.loads(manifest.read_text())
    expected = sum(m['expected_videos'] for m in job['models'].values())
    if not json.loads((source/'pool_status.json').read_text())['complete']:
        raise ValueError('Only complete benchmark snapshots may be published')
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    if len(rows) != expected or len({r['sample_id'] for r in rows}) != expected:
        raise ValueError('Incomplete or duplicate benchmark index')
    identity = dict(source_manifest_sha256=sha(manifest), source_index_sha256=sha(index),
                    expected_npz=expected)
    plan_path = output/'staging_identity.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != identity:
        raise ValueError('Changed source snapshot; use a new staging folder')
    write(plan_path, identity)
    payload = output/'payload'
    public, tasks, totals = [], [], Counter()
    for r in rows:
        metadata = json.loads(Path(r['metadata']).read_text())
        if not metadata['complete'] or metadata['sample_id'] != r['sample_id']:
            raise ValueError('Incomplete or mismatched video metadata')
        src = Path(r['raw_outputs']).resolve()
        digest = metadata['provenance']['npz_sha256']
        if metadata['request']['outputs_sha256'] != digest or src != Path(metadata['raw_outputs']).resolve():
            raise ValueError('NPZ provenance mismatch')
        model, benchmark, method, cid = [component(r[k]) for k in ('model', 'benchmark', 'method', 'case_id')]
        rel = f'npz/{model}/{benchmark}/{method}/{cid}/seed{int(r["seed"])}.npz'
        size = src.stat().st_size
        group = model+'_'+benchmark
        row = {k: r[k] for k in ('sample_id', 'model', 'method', 'benchmark', 'category', 'source_id', 'case_id', 'seed', 'prompt')}
        row.update(npz_path=rel, npz_sha256=digest, npz_bytes=size,
            signal_sha256=metadata['request']['sha256'],
            video_package=group+'.zip', video_member=f'{group}/videos/{r["sample_id"]}.mp4',
            metadata_member=f'{group}/metadata/{r["sample_id"]}.json',
            height=metadata['height'], width=metadata['width'], frames=metadata['frames'], fps=metadata['fps'])
        public.append(row)
        tasks.append((src, payload/rel, digest, size))
        totals[group] += size
    if len({r['npz_path'] for r in public}) != expected:
        raise ValueError('Duplicate NPZ destination')
    state = dict(complete=False, phase='verifying_local_npz', identity=identity,
        expected=expected, checked=0, bytes_checked=0, total_bytes=sum(totals.values()), updated_utc=now())
    write(output/'progress.json', state)

    def check_link(task):
        src, dest, digest, size = task
        before = src.stat()
        if sha(src) != digest:
            raise ValueError('NPZ checksum mismatch: '+str(src))
        after = src.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('NPZ changed while verifying: '+str(src))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(src, dest)
        if not os.path.samefile(src, dest) or dest.stat().st_size != size:
            raise ValueError('Staged NPZ is not the checked source inode')
        return size

    with ThreadPoolExecutor(max_workers=2) as workers:
        for size in workers.map(check_link, tasks):
            state['checked'] += 1
            state['bytes_checked'] += size
            if state['checked'] % 100 == 0 or state['checked'] == expected:
                state['updated_utc'] = now()
                write(output/'progress.json', state)
                print(json.dumps(state), flush=True)
    group_counts = Counter(r['model']+'_'+r['benchmark'] for r in public)
    index_root = payload/'npz/index'
    index_root.mkdir(parents=True, exist_ok=True)
    for group in sorted(group_counts):
        subset = [r for r in public if r['model']+'_'+r['benchmark'] == group]
        subset.sort(key=lambda r: r['sample_id'])
        (index_root/(group+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in subset))
    sums = ''.join(f'{r["npz_sha256"]}  {r["npz_path"]}\n' for r in sorted(public, key=lambda r:r['npz_path']))
    (payload/'npz/SHA256SUMS').write_text(sums)
    inventory = dict(schema=1, inventory_complete=True, files=expected, bytes=sum(totals.values()),
        groups={g: dict(files=group_counts[g], bytes=totals[g]) for g in sorted(group_counts)},
        methods=job['methods'], source=identity, source_arrays='Original float32 generated video arrays',
        reencoded=False, checksums_verified=True, scope='Original completed five-method full I2V snapshot')
    write(payload/'npz/manifest.json', inventory)
    (payload/'npz/README.md').write_text(f'''# Original benchmark NPZ outputs

This inventory contains {expected:,} original compressed NPZ files
({inventory['bytes']:,} bytes), corresponding exactly to the existing four MP4
ZIP packages. It covers Edge and Nano, RBench 650 and PAI-Bench 1,044 scenes,
two fixed seeds and the original five methods. The newly requested four baseline
methods are a separate generation run and are not part of this snapshot.

No dtype conversion, resampling or re-encoding was applied. Each NPZ contains
the original `video` float32 array, shape (61, 192, 320, 3). Use NPZ arrays for
pixel-exact comparisons; MP4 compression changes pixel values.

`index/*.jsonl` maps sample IDs, methods, scenes and seeds to `npz_path`, file and
array SHA256 hashes, and the matching member in the existing video ZIP package.
All `npz_path` values are relative to the dataset repository root. ZIP member
paths are relative to their respective archives. `SHA256SUMS` also uses paths
relative to the dataset root. The upload is published in resumable batches;
compare the actual files against `manifest.json` before consuming a full copy.

```bash
hf download Concyclics/thunderworld-benchmark --repo-type dataset \\
  --include 'npz/**' --local-dir thunderworld-benchmark
cd thunderworld-benchmark
sha256sum -c npz/SHA256SUMS
```

Generation settings: 320×192, 61 frames at 24 FPS, 35 UniPC steps, CFG 6.0 and
flow shift 3.0. Full method definitions, prompts, images, receipts and logs are
in the existing video ZIPs. Recorded shared-GPU times are not performance
benchmarks. Upstream benchmark and model terms remain applicable.
''')
    write(output/'expected_files.json', {r['npz_path']: dict(sha256=r['npz_sha256'], bytes=r['npz_bytes']) for r in public})
    state.update(phase='local_verified', updated_utc=now())
    write(output/'progress.json', state)
    return payload, state


def publish(a):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.repo_info(a.repo, repo_type='dataset')
    write(a.output/'remote_before.json', dict(repo_id=a.repo, revision=info.sha, private=info.private))
    payload, state = prepare(a.source, a.output)
    # Preserve the existing card, adding a link to the separate NPZ inventory.
    original = Path(hf_hub_download(a.repo, 'README.md', repo_type='dataset',
        revision=info.sha, local_dir=a.output/'remote_card')).read_text()
    marker = '## Original floating-point NPZ outputs'
    if marker not in original:
        original += '\n'+marker+'\n\nThe original 33,880 NPZ results are available in '\
            '[`npz/`](tree/main/npz), with [portable indices and download instructions](npz/README.md). '\
            'This separate inventory totals 442.46 GB and is uploaded in resumable batches. '\
            'Check [`npz/manifest.json`](npz/manifest.json) and checksums before using a full copy.\n'
    (payload/'README.md').write_text(original)
    command = [str(Path(sys.executable).with_name('hf')), 'upload', a.repo, str(payload), '.',
        '--repo-type', 'dataset', '--commit-message', 'Add original full-benchmark NPZ outputs and portable indices']
    state.update(phase='uploading', repo_id=a.repo, command=command, updated_utc=now())
    write(a.output/'progress.json', state)
    env = os.environ.copy()
    env['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    with (a.output/'upload.log').open('ab') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    state.update(phase='verifying_remote', updated_utc=now())
    write(a.output/'progress.json', state)
    expected = json.loads((a.output/'expected_files.json').read_text())
    revision = api.repo_info(a.repo, repo_type='dataset').sha
    remote = {f.path: f for f in api.list_repo_tree(a.repo, path_in_repo='npz',
        repo_type='dataset', revision=revision, recursive=True) if f.path.endswith('.npz')}
    if set(remote) != set(expected):
        raise ValueError('Remote NPZ file set does not match the complete source snapshot')
    for path, value in expected.items():
        f = remote[path]
        if f.size != value['bytes'] or not f.lfs or f.lfs.sha256 != value['sha256']:
            raise ValueError('Remote size or SHA256 mismatch: '+path)
    state.update(complete=True, phase='complete', revision=revision, remote_verified=len(remote),
        url=f'https://huggingface.co/datasets/{a.repo}/tree/main/npz', updated_utc=now())
    write(a.output/'progress.json', state)
    print(json.dumps(state), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo', default='Concyclics/thunderworld-benchmark')
    a = p.parse_args()
    a.source, a.output = a.source.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'publisher.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            publish(a)
        except BaseException as exc:
            path = a.output/'progress.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            write(path, dict(state, complete=False, paused=True, error=repr(exc), updated_utc=now()))
            raise


if __name__ == '__main__':
    main()

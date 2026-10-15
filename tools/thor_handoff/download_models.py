"""Download the fixed benchmark model revisions and verify every source byte."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    from huggingface_hub import hf_hub_download
    root = args.root.resolve()
    manifest = root/'data/model_files.json'
    models = json.loads(manifest.read_text())
    (root/'models').mkdir(exist_ok=True)
    lock = (root/'models/.download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(complete=False, pid=os.getpid(), started=time.time(),
                 manifest_sha256=sha(manifest), models={})
    for folder, spec in models.items():
        dest = root/'models'/folder
        dest.mkdir(exist_ok=True)
        state['models'][folder] = dict(repo=spec['repo'], revision=spec['revision'],
                                      verified={}, complete=False)
        current = state['models'][folder]
        def fetch(item):
            name, expected = item
            path = dest/name
            if not path.resolve().is_relative_to(dest.resolve()):
                raise ValueError('Invalid model file path')
            if path.exists() and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']:
                return name, expected
            # Preserve our interrupted rsync file separately; HF downloads into
            # its own resumable .cache before atomically replacing the target.
            if path.exists():
                path.rename(path.with_name(path.name+'.rsync-partial'))
            last = None
            for attempt in range(4):
                try:
                    hf_hub_download(spec['repo'], name, revision=spec['revision'],
                                    local_dir=dest, token=False)
                    if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                        raise ValueError('Source model hash mismatch: '+name)
                    return name, expected
                except Exception as error:
                    last = error
                    if attempt < 3:
                        time.sleep(2 ** attempt)
            raise last
        write(root/'logs/model_download.json', state)
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                jobs = [pool.submit(fetch, item) for item in spec['files'].items()]
                for job in as_completed(jobs):
                    name, expected = job.result()
                    current['verified'][name] = expected
                    state['updated'] = time.time()
                    write(root/'logs/model_download.json', state)
                    print(json.dumps(dict(model=folder, verified=len(current['verified']),
                                          total=len(spec['files']), file=name)), flush=True)
            current['complete'] = True
            write(dest/'thunderworld_verified.json', dict(complete=True, repo=spec['repo'],
                  revision=spec['revision'], files=current['verified'], manifest_sha256=sha(manifest)))
            # Remove only interrupted copies made by this downloader after all
            # original files have been verified successfully.
            for name in spec['files']:
                partial = dest/(name+'.rsync-partial')
                if partial.exists():
                    partial.unlink()
            write(root/'logs/model_download.json', state)
        except Exception as error:
            state['error'] = repr(error)
            write(root/'logs/model_download.json', state)
            raise
    state['complete'] = True
    state['finished'] = time.time()
    write(root/'logs/model_download.json', state)


if __name__ == '__main__':
    main()

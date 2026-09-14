"""Download the private-repository handoff Release and verify tracked SHA256 hashes.

Requires an authenticated gh CLI. No model checkpoints or device profiles are downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def verify(folder,files):
    for name,record in files.items():
        path=folder/name
        if path.stat().st_size!=record['bytes']:raise ValueError(f'Size mismatch: {name}')
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=record['sha256']:raise ValueError(f'SHA256 mismatch: {name}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--core-only',action='store_true',help='Skip the optional 218 MiB real-operator panels')
    p.add_argument('--verify-only',action='store_true',help='Check an existing download without network access')
    a=p.parse_args()
    manifest=json.loads(Path(__file__).with_name('github_handoff').joinpath('release.json').read_text())
    files={name:record for name,record in manifest['files'].items() if not (a.core_only and name=='cutlass-5090-panels.tar.gz')}
    if not a.verify_only:
        a.output.mkdir(parents=True,exist_ok=False)
        command=['gh','release','download',manifest['tag'],'--repo',manifest['repository'],'--dir',str(a.output)]
        for name in files:command.extend(['--pattern',name])
        subprocess.run(command,check=True)
    verify(a.output,files)
    print(json.dumps(dict(verified=True,source_commit=manifest['source_commit'],files=list(files),
                         output=str(a.output.resolve())),indent=2))


if __name__=='__main__':main()

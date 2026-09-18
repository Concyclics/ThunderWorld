"""Authenticated GitHub Release download with pinned hashes and safe extraction."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from bundle import ROOT, relative, check_package
from baselines import sha


def extract(archive, destination):
    with tarfile.open(archive, 'r:gz') as stream:
        for item in stream.getmembers():
            relative(destination, item.name)
            if not (item.isfile() or item.isdir()):
                raise ValueError('Archive links and special files are not allowed')
        stream.extractall(destination, filter='data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', type=Path, default=Path(__file__).with_name('release.json'))
    args = parser.parse_args()
    lock = json.loads(args.release.read_text())
    if args.output.exists():raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(prefix='tw-fp4-release-') as folder:
        subprocess.run(['gh','release','download',lock['tag'],'--repo',lock['repository'],
                        '--dir',folder,'--pattern',lock['asset']], check=True)
        archive = Path(folder)/lock['asset']
        if archive.stat().st_size != lock['bytes'] or sha(archive) != lock['sha256']:
            raise ValueError('Release asset checksum mismatch')
        args.output.mkdir(parents=True)
        extract(archive, args.output)
        check_package(args.output)
        if sha(args.output/'package.json') != lock['package_sha256']:
            raise ValueError('Unexpected package manifest')
    print(str(args.output.resolve()))


if __name__ == '__main__':main()

"""Shared portable paths and content identities for public workflows."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'tools/contribution_study'
sys.path.insert(0, str(ROOT / 'packages/fp4-repair/src'))
sys.path.insert(0, str(STUDY))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def immutable(path, data):
    path = Path(path)
    if path.exists() and read(path) != data:
        raise ValueError(f'Changed frozen inputs: {path}. Use a new output directory.')
    write(path, data)

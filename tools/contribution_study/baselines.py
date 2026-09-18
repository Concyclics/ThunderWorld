"""Fetch official sources without installing dependencies or executing their code."""
import hashlib
import json
from pathlib import Path
import subprocess

SOURCES = {
    'arcquant': ('https://github.com/actypedef/ARCQuant.git', 'activation residual channels'),
    'sharq': ('https://github.com/actypedef/SharQ.git', 'sparse activation plus dense residual'),
    'fouroversix': ('https://github.com/mit-han-lab/fouroversix.git', 'adaptive FP4 block scaling'),
    'viditq': ('https://github.com/thu-nics/ViDiT-Q.git', 'layer and timestep sensitivity'),
    'sageattention': ('https://github.com/thu-ml/SageAttention.git', 'native FP4 attention'),
    'nunchaku': ('https://github.com/nunchux-ai/nunchaku.git', 'SVDQuant low-rank compensation'),
    'modelopt': ('https://github.com/NVIDIA/Model-Optimizer.git', 'NVFP4-AWQ W4A16'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def git(folder, *args):
    return subprocess.check_output(['git', '-C', str(folder), *args], text=True).strip()


def collect(root, names, source_lock=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'sources.lock.json'
    result = json.loads(lock.read_text()) if lock.exists() else dict(schema_version=1, sources={})
    pinned=json.loads(Path(source_lock).read_text())['sources'] if source_lock else {}
    for name in names:
        url, purpose = SOURCES[name]
        folder = root / name
        try:
            if not folder.exists():
                if name in pinned:
                    if pinned[name]['url']!=url:raise ValueError('Pinned repository URL mismatch')
                    folder.mkdir()
                    subprocess.run(['git','init',str(folder)],check=True)
                    subprocess.run(['git','-C',str(folder),'remote','add','origin',url],check=True)
                    subprocess.run(['git','-C',str(folder),'fetch','--depth=1','origin',pinned[name]['commit']],check=True)
                    subprocess.run(['git','-C',str(folder),'checkout','--detach','FETCH_HEAD'],check=True)
                else:subprocess.run(['git', 'clone', '--depth=1', url, str(folder)], check=True)
            if git(folder, 'remote', 'get-url', 'origin') != url:
                raise ValueError('Existing checkout has a different origin')
            commit = git(folder, 'rev-parse', 'HEAD')
            if name in pinned and commit!=pinned[name]['commit']:raise ValueError('Checkout differs from supplied source lock')
            old = result['sources'].get(name)
            if old and old.get('commit') and old['commit'] != commit:
                raise ValueError('Locked source changed; use a new collection directory')
            if git(folder, 'status', '--porcelain', '--untracked-files=no'):
                raise ValueError('Official source has local changes; keep adaptations outside checkout')
            files = git(folder, 'ls-files').splitlines()
            licenses = [f for f in files if Path(f).name.lower().startswith(('license', 'copying'))]
            dependencies = [f for f in files if Path(f).name in
                            ('pyproject.toml', 'requirements.txt', '.gitmodules', 'CMakeLists.txt')]
            result['sources'][name] = dict(url=url, commit=commit, purpose=purpose,
                path=str(folder.resolve()), source_type='official', adaptation_status='not_qualified',
                license_files={f: sha(folder / f) for f in licenses},
                redistribution_review='required' if not licenses else 'preserve upstream license',
                dependency_files={f: sha(folder / f) for f in dependencies},
                submodules=git(folder, 'submodule', 'status').splitlines(),
                performance_eligible=False, patches=[])
        except Exception as error:
            if name not in result['sources']:
                result['sources'][name] = dict(url=url, error=repr(error), adaptation_status='unavailable')
            write(lock, result)
            raise
        write(lock, result)
        print(json.dumps(dict(baseline=name, commit=commit)), flush=True)
    return result

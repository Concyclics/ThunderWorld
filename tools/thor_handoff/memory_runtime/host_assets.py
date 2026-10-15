"""Preparation-only lazy asset loading and explicit fixed-shape sealing."""
import gc
import hashlib
import ctypes
from pathlib import Path
import torch
from .cosmos.assets import expand, required_weight_fraction


class LazyAssets:
    def __init__(self, folder, manifest, config, policy):
        self.folder = Path(folder)
        self.manifest, self.config, self.policy = manifest, config, policy

    def __getitem__(self, name):
        entry = self.manifest['modules'][name]
        path = self.folder / entry['file']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry['sha256']:
            raise ValueError('Corrupt prepared asset: ' + name)
        asset = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        if self.config.precision == 'fp4':
            return expand(asset, required_weight_fraction(self.config, name, self.policy))
        return asset


def tensor_bytes(value):
    storages = {}
    def visit(v):
        if isinstance(v, torch.Tensor) and not v.is_cuda:
            storage = v.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(v, dict):
            for child in v.values(): visit(child)
        elif isinstance(v, (list, tuple)):
            for child in v: visit(child)
    visit(value)
    return sum(storages.values())


def release_host_assets(runner):
    """Seal all warmed shapes; constructing/exporting new assets then fails.

    Metadata used by eager dispatch is retained. All GPU weights, current-input
    selectors, masks and Graphs remain untouched. Call outside request timing.
    A new shape uses a new runner and reloads the original on-disk assets.
    """
    if not runner._warmed:
        raise RuntimeError('Warm complete requests before releasing preparation assets')
    before = 0; layers = 0
    for op in runner.engine.linears.values():
        if not hasattr(op, 'asset'):
            continue
        if not op.ops:
            raise RuntimeError('Unprepared projection: ' + op.name)
        before += tensor_bytes(op.asset)
        op.asset = {key: value for key, value in op.asset.items() if tensor_bytes(value) == 0}
        op._host_assets_released = True
        layers += 1
    runner._host_assets_released = True
    gc.collect()
    trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
    trimmed = None if trim is None else int(trim(0))
    return dict(released_host_tensor_bytes=before, projections=layers, malloc_trim_result=trimmed,
                scope='CPU asset tensors only; existing warmed shapes, eager and Graph supported; new shapes need a fresh runner')

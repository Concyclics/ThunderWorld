"""Shared calibration admission and process lock; never a timing eligibility test."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess


def free_memory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(i): int(m) for i, m in (line.split(',') for line in rows.splitlines())}


def select_shared_device(minimum_mib=12 * 1024):
    """Call before importing/initializing CUDA. Respect an explicit 0/3 selection."""
    available = free_memory()
    requested = os.environ.get('CUDA_VISIBLE_DEVICES')
    if requested is not None and requested not in ('0', '3'):
        raise ValueError('Shared calibration requires physical GPU0 or GPU3')
    eligible=[i for i in (0,3) if i in available]
    if not eligible:raise RuntimeError('No eligible GPU0/GPU3 found')
    if requested is not None and int(requested) not in eligible:
        raise RuntimeError(f'Requested GPU{requested} is not present')
    device = int(requested) if requested else max(eligible, key=lambda i: available[i])
    if available[device] < minimum_mib:
        raise RuntimeError(f'PAUSED: GPU{device} has {available[device]} MiB free; requires {minimum_mib}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    return dict(device=device, free_mib=available[device], minimum_mib=minimum_mib,
                resource_mode='shared-calibration', performance_eligible=False)


@contextmanager
def shared_worker(lock_path=None):
    """One experiment process across both eligible devices. Nonblocking, crash safe."""
    path = Path(lock_path or f'/tmp/thunderworld-shared-calibration-{os.getuid()}.lock')
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('PAUSED: another shared calibration worker owns the lock') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

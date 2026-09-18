"""Shared calibration admission and process lock; never a timing eligibility test."""
from contextlib import contextmanager, ExitStack
import fcntl
import os
from pathlib import Path
import subprocess


def free_memory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(i): int(m) for i, m in (line.split(',') for line in rows.splitlines())}


def allowed_devices():
    """Default historical 0/3 policy; an orchestrator may explicitly pin another set."""
    raw = os.environ.get('TW_SHARED_CALIBRATION_GPUS', '0,3')
    if not raw or any(not part.isdecimal() for part in raw.split(',')):
        raise ValueError('Expected comma-separated physical GPU indices')
    devices = tuple(int(part) for part in raw.split(','))
    if len(devices) != len(set(devices)):
        raise ValueError('Duplicate physical GPU indices')
    return devices


def select_shared_device(minimum_mib=12 * 1024):
    """Call before initializing CUDA; never replace an explicitly selected GPU."""
    available = free_memory()
    allowed = allowed_devices()
    requested = os.environ.get('CUDA_VISIBLE_DEVICES')
    if requested is not None and requested not in tuple(map(str, allowed)):
        raise ValueError(f'Shared calibration requires one physical GPU from {allowed}')
    eligible=[i for i in allowed if i in available]
    if not eligible:raise RuntimeError(f'No eligible physical GPU from {allowed} found')
    if requested is not None and int(requested) not in eligible:
        raise RuntimeError(f'Requested GPU{requested} is not present')
    device = int(requested) if requested else max(eligible, key=lambda i: available[i])
    if available[device] < minimum_mib:
        raise RuntimeError(f'PAUSED: GPU{device} has {available[device]} MiB free; requires {minimum_mib}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    return dict(device=device, free_mib=available[device], minimum_mib=minimum_mib,
                resource_mode='shared-calibration', performance_eligible=False,
                allowed_devices=list(allowed), lock_scope=os.environ.get('TW_SHARED_CALIBRATION_LOCK_SCOPE','global'))


@contextmanager
def shared_worker(lock_path=None):
    """Historical global exclusion, or explicit one-worker-per-physical-GPU mode.

    Device workers share the global gate, so they still exclude legacy global
    workers. A second exclusive lock prevents two workers using the same GPU.
    """
    path = Path(lock_path or f'/tmp/thunderworld-shared-calibration-{os.getuid()}.lock')
    scope = os.environ.get('TW_SHARED_CALIBRATION_LOCK_SCOPE','global')
    if scope not in ('global','device'):
        raise ValueError('Unknown shared calibration lock scope')
    device = os.environ.get('CUDA_VISIBLE_DEVICES')
    if scope == 'device' and device not in tuple(map(str,allowed_devices())):
        raise ValueError('Device lock requires one explicitly selected physical GPU')
    with ExitStack() as stack:
        locks = [(path, fcntl.LOCK_SH if scope=='device' else fcntl.LOCK_EX)]
        if scope=='device':locks.append((path.with_name(path.name+f'.gpu{device}'),fcntl.LOCK_EX))
        for target, kind in locks:
            handle = stack.enter_context(target.open('a+'))
            try:
                fcntl.flock(handle, kind | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('PAUSED: another shared calibration worker owns the lock') from exc
        yield

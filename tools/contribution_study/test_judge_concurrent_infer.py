import fcntl
import os
from pathlib import Path
import subprocess
import sys

import pytest

import judge_concurrent_infer as concurrent


def test_pool_lease_excludes_other_workers_and_preserves_device_independence(tmp_path, monkeypatch):
    def paths(gpu):
        return tmp_path/'global.lock', tmp_path/f'gpu{gpu}.lock'
    monkeypatch.setattr(concurrent, 'lock_paths', paths)
    with concurrent.device_leases([0]) as (global_fd, devices):
        with paths(0)[1].open('a+') as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with paths(0)[0].open('a+') as legacy:
            with pytest.raises(BlockingIOError):
                fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with concurrent.device_leases([1]):
            pass
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
        concurrent.verify_descriptors(0, global_fd, devices[0])
        with pytest.raises(ValueError):
            concurrent.verify_descriptors(0, devices[0], global_fd)
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
        with pytest.raises(ValueError):
            concurrent.verify_descriptors(0, global_fd, devices[0])
    with concurrent.device_leases([0]):
        pass


def test_child_receives_exact_device_lease_and_case_selection(tmp_path):
    cmd, env, fds = concurrent.child_command(tmp_path, 'edge', 'arcquant100',
        tmp_path/'out', 1, (40, {0:41, 1:42}), cases=['a', 'b'])
    assert fds == (40, 42)
    assert env['CUDA_VISIBLE_DEVICES'] == '1'
    assert env['TW_SHARED_CALIBRATION_GPUS'] == '0,1'
    assert cmd[-3:] == ['--cases', 'a', 'b']
    assert '--qualification' not in cmd


def test_inherited_lease_survives_coordinator_close_until_child_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(concurrent, 'lock_paths',
        lambda gpu:(tmp_path/'global.lock', tmp_path/f'gpu{gpu}.lock'))
    with concurrent.device_leases([0]) as (global_fd, devices):
        child = subprocess.Popen([sys.executable, '-c',
            'import os,sys;os.fstat(int(sys.argv[1]));print("ready",flush=True);sys.stdin.readline()',
            str(devices[0])], pass_fds=(global_fd, devices[0]),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert child.stdout.readline().strip() == 'ready'
    try:
        with (tmp_path/'gpu0.lock').open('a+') as another:
            with pytest.raises(BlockingIOError):
                fcntl.flock(another, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        child.communicate('\n', timeout=5)
    with concurrent.device_leases([0]):
        pass

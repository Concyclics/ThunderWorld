"""Receiver admission supports a single 5090 and never changes an explicit GPU."""
import pytest
from thunderworld_fp4.cosmos import resources


def test_single_gpu_auto_selection(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(resources,'free_memory',lambda:{0:24000})
    assert resources.select_shared_device()['device']==0


def test_explicit_missing_gpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    monkeypatch.setattr(resources,'free_memory',lambda:{0:24000})
    with pytest.raises(RuntimeError,match='not present'):resources.select_shared_device()


def test_explicit_gpu_does_not_switch_on_low_memory(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(resources,'free_memory',lambda:{0:1000,3:24000})
    with pytest.raises(RuntimeError,match='requires'):resources.select_shared_device()


def test_explicit_gpu1_requires_allowlist(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    monkeypatch.setattr(resources,'free_memory',lambda:{0:24000,1:16000})
    with pytest.raises(ValueError):resources.select_shared_device()
    monkeypatch.setenv('TW_SHARED_CALIBRATION_GPUS','0,1')
    assert resources.select_shared_device()['device']==1


def test_device_locks_allow_distinct_cards_but_exclude_same_and_global(tmp_path,monkeypatch):
    lock=tmp_path/'gpu.lock'
    monkeypatch.setenv('TW_SHARED_CALIBRATION_GPUS','0,1')
    monkeypatch.setenv('TW_SHARED_CALIBRATION_LOCK_SCOPE','device')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with resources.shared_worker(lock):
        with pytest.raises(RuntimeError,match='owns the lock'):
            with resources.shared_worker(lock):pass
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
        with resources.shared_worker(lock):pass
        monkeypatch.setenv('TW_SHARED_CALIBRATION_LOCK_SCOPE','global')
        with pytest.raises(RuntimeError,match='owns the lock'):
            with resources.shared_worker(lock):pass
    with resources.shared_worker(lock):
        monkeypatch.setenv('TW_SHARED_CALIBRATION_LOCK_SCOPE','device')
        with pytest.raises(RuntimeError,match='owns the lock'):
            with resources.shared_worker(lock):pass
    with resources.shared_worker(lock):pass

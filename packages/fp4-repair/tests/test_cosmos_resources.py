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

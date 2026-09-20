import json
import pytest

from baselines import sha, write
from judge_device_control import HERE, validate_policy, runtime_snapshot


def test_policy_accepts_qualified_subset_and_rejects_extra_gpu(tmp_path):
    job = dict(gpus=[0,1])
    write(tmp_path/'manifest.json', job)
    policy = dict(active_gpus=[0], job_sha256=sha(tmp_path/'manifest.json'), controller_sha256=sha(HERE))
    assert validate_policy(tmp_path, job, policy)['active_gpus'] == [0]
    for devices in ([], [0,0], [2], [True]):
        with pytest.raises(ValueError): validate_policy(tmp_path, job, dict(policy, active_gpus=devices))
    with pytest.raises(ValueError): validate_policy(tmp_path, job, dict(policy, job_sha256='changed'))
    with pytest.raises(ValueError): validate_policy(tmp_path, job, dict(policy, controller_sha256='changed'))


def test_single_gpu_status_preserves_all_completed_work(tmp_path):
    write(tmp_path/'queue.json', dict(tasks={'done_on_gpu1':dict(status='complete',gpu=1),
        'next_on_gpu0':dict(status='running',gpu=0)}))
    write(tmp_path/'edge/final/generation/method/case/result.json', dict(rows=[dict(seed=1),dict(seed=2)]))
    write(tmp_path/'edge/export_progress.json',dict(complete=False,completed=2))
    write(tmp_path/'gpu0/status.json',dict(phase='generation'))
    write(tmp_path/'gpu1/status.json',dict(phase='retired'))
    write(tmp_path/'device_policy.json',dict(active_gpus=[0]))
    job=dict(gpus=[0],models={'edge':dict(expected_videos=4)},
        tasks=[dict(id='done_on_gpu1',model='edge'),dict(id='next_on_gpu0',model='edge')])
    status=runtime_snapshot(tmp_path,job)
    assert status['active_gpus']==[0]
    assert set(status['workers'])=={'0'}
    assert status['models']['edge']['generated']==2
    assert not status['complete']

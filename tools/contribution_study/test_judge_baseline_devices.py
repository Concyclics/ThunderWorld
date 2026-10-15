from concurrent.futures import ThreadPoolExecutor

import pytest

from baselines import sha, write
import judge_baseline_devices as devices


def policy(tmp_path):
    write(tmp_path/'manifest.json', dict(gpus=[0, 1], methods=list(devices.METHODS)))
    return dict(gpus=[0, 2], slots_per_gpu=2, models=['nano'], secondary_models=['nano'],
        secondary_methods=list(devices.METHODS), numerical_qualification_required=True,
        job_sha256=sha(tmp_path/'manifest.json'), source_sha256=sha(devices.HERE),
        runtime_sha256=sha(devices.runtime.__file__))


def test_new_gpu_policy_preserves_frozen_manifest_and_rejects_unqualified_slots(tmp_path):
    value = policy(tmp_path)
    digest = sha(tmp_path/'manifest.json')
    job = dict(gpus=[0, 1], models=dict(edge={}, nano={}))
    assert devices.validate_policy(tmp_path, job, value)['gpus'] == [0, 2]
    assert sha(tmp_path/'manifest.json') == digest
    for change in [dict(gpus=[0, 0]), dict(slots_per_gpu=4), dict(models=['edge']),
                   dict(secondary_models=['edge']), dict(numerical_qualification_required=False),
                   dict(source_sha256='changed'), dict(job_sha256='changed')]:
        with pytest.raises(ValueError):
            devices.validate_policy(tmp_path, job, dict(value, **change))


def test_both_slots_on_both_cards_claim_nano_once_and_resume_partial_tasks(tmp_path):
    value = policy(tmp_path)
    write(tmp_path/'parallel_policy.json', value)
    tasks = [dict(id=f'nano{i}', model='nano', method='awq_reference') for i in range(4)]
    # The task resumes through the same path; existing files are not replaced
    # or rewritten when queue ownership moves to a new physical GPU.
    partial = tmp_path/'tasks/nano0/generation/case/result.json'
    write(partial, dict(complete=False, rows=[dict(seed=401001, outputs_sha256='saved')]))
    before = sha(partial)
    write(tmp_path/'queue.json', dict(job_sha256=value['job_sha256'],
        tasks={t['id']: dict(status='pending', attempts=[]) for t in tasks}))
    with ThreadPoolExecutor(4) as executor:
        claimed = list(executor.map(lambda pair: devices.claim(tmp_path, dict(tasks=tasks),
            value, *pair), [(0, 0), (0, 1), (2, 0), (2, 1)]))
    assert {t['id'] for t in claimed} == {t['id'] for t in tasks}
    assert sha(partial) == before
    assert devices.claim(tmp_path, dict(tasks=tasks), value, 2, 1) is None
    for gpu, slot in [(1, 0), (0, 2), (2, -1)]:
        with pytest.raises(ValueError):
            devices.claim(tmp_path, dict(tasks=tasks), value, gpu, slot)


def test_runtime_device_view_allows_gpu2_without_rewriting_input_identity(monkeypatch, tmp_path):
    value = policy(tmp_path)
    write(tmp_path/'parallel_policy.json', value)
    original = dict(gpus=[0, 1], models=dict(edge={}, nano={}), tasks=[])
    monkeypatch.setattr(devices.extension, 'verify_job', lambda root: original)
    digest = sha(tmp_path/'manifest.json')
    view = devices.view_job(tmp_path)
    assert view['gpus'] == [0, 2] and original['gpus'] == [0, 1]
    assert sha(tmp_path/'manifest.json') == digest

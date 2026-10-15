from concurrent.futures import ThreadPoolExecutor

import pytest

from baselines import sha, write
from judge_parallel_queue import allowed, claim, validate_admission


def test_four_slots_claim_distinct_tasks_with_secondary_method_and_model_limits(tmp_path):
    write(tmp_path/'manifest.json', dict(frozen=True))
    write(tmp_path/'parallel_policy.json', dict(gpus=[0,1]))
    tasks = [dict(id='nano', model='nano', method='b2_joint'),
             dict(id='serial_edge', model='edge', method='arcquant'),
             dict(id='parallel_a', model='edge', method='b2_joint'),
             dict(id='parallel_b', model='edge', method='b2_joint')]
    write(tmp_path/'queue.json', dict(job_sha256=sha(tmp_path/'manifest.json'),
        tasks={t['id']:dict(status='pending', attempts=[]) for t in tasks}))
    value = dict(secondary_methods=['b2_joint'])
    with ThreadPoolExecutor(4) as pool:
        result = list(pool.map(lambda pair:claim(tmp_path, dict(tasks=tasks), value, *pair),
                               [(0,0), (1,0), (0,1), (1,1)]))
    assert {r['id'] for r in result} == {t['id'] for t in tasks}
    assert all(r['model']=='edge' and r['method']=='b2_joint' for r in result[2:])
    assert claim(tmp_path, dict(tasks=tasks), value, 0, 0) is None


def test_unqualified_method_cannot_enter_secondary_slot():
    task = dict(model='nano', method='b2_joint')
    value = dict(secondary_methods=['b2_joint'])
    assert allowed(task, 0, value)
    assert not allowed(task, 1, value)
    job = dict(gpus=[0,1])
    good = dict(complete=True, methods={'b2_joint':dict(complete=True,
        candidate_for_live_pilot=True, all_source_hashes_match=True)})
    validate_admission(job, [0,1], 2, ['b2_joint'], good)
    for gpus, slots, methods, report in [([0,0],2,['b2_joint'],good),
            ([2],2,['b2_joint'],good), ([0],5,['b2_joint'],good),
            ([0],2,['arcquant'],good), ([0],2,[],good), ([0],2,['b2_joint'],None)]:
        with pytest.raises(ValueError):
            validate_admission(job, gpus, slots, methods, report)


def test_requested_four_process_policy_keeps_explicit_method_and_gpu_limits():
    job = dict(gpus=[0,1])
    validate_admission(job, [0,1], 4, ['arcquant','b2_joint'], admission='requested')
    with pytest.raises(ValueError):
        validate_admission(job, [0], 4, [], admission='requested')
    with pytest.raises(ValueError):
        validate_admission(job, [0], 4, ['unknown'], admission='requested')
    with pytest.raises(ValueError):
        validate_admission(job, [0], 5, ['arcquant'], admission='requested')
    with pytest.raises(ValueError):
        validate_admission(job, [2], 4, ['arcquant'], admission='requested')


def test_eight_workers_never_duplicate_a_task(tmp_path):
    write(tmp_path/'manifest.json', dict(frozen=True))
    write(tmp_path/'parallel_policy.json', dict(gpus=[0,1], slots_per_gpu=4))
    tasks = [dict(id='nano', model='nano', method='b2_joint'),
             *[dict(id=f'edge{i}', model='edge', method='b2_joint') for i in range(7)]]
    write(tmp_path/'queue.json', dict(job_sha256=sha(tmp_path/'manifest.json'),
        tasks={t['id']:dict(status='pending', attempts=[]) for t in tasks}))
    pairs = [(gpu, slot) for gpu in (0,1) for slot in range(4)]
    with ThreadPoolExecutor(8) as pool:
        selected = list(pool.map(lambda pair:claim(tmp_path, dict(tasks=tasks),
            dict(secondary_methods=['b2_joint']), *pair), pairs))
    assert len({t['id'] for t in selected}) == 8
    assert all(t['model']=='edge' for pair,t in zip(pairs, selected) if pair[1] > 0)

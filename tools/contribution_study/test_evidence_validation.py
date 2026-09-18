import json
import pytest
from baselines import sha, write
from evidence_validation import report


def fixture(root, candidate_l2):
    cases = []
    for benchmark in ('rbench', 'paibench'):
        for category in ('a', 'b'):
            for index in range(2):
                cases.append(dict(benchmark=benchmark, category=category,
                    case=dict(id=f'{benchmark}_{category}_{index}'), seeds=[2*len(cases)+1, 2*len(cases)+2]))
    methods = {name: dict(config=dict(compute_budget=2)) for name in ('bf16', 'fp8', 'rtn', 'candidate')}
    manifest = dict(role='validation', cases=cases, methods=methods,
        primary_comparisons=[dict(candidate='candidate', reference=r) for r in ('rtn', 'fp8')],
        statistics=dict(replicates=100, bootstrap_seed=17, family_size=4))
    write(root/'manifest.json', manifest)
    identity = dict(manifest_sha256=sha(root/'manifest.json'))
    write(root/'metrics/progress.json', dict(complete=True, identity=identity))
    for entry in cases:
        case = entry['case']['id']
        for name in methods:
            receipt = dict(complete=True, rows=[dict(seed=seed, performance_eligible=False, steps=35,
                debug='off', actual_mma_ratio=1.9 if name in ('rtn', 'candidate') else None,
                peak_allocated_bytes=1024) for seed in entry['seeds']])
            write(root/'generation'/name/case/'result.json', receipt)
        for name in ('fp8', 'rtn', 'candidate'):
            metrics = dict(lpips=.05 if name == 'fp8' else .09 if name == 'candidate' else .1,
                ssim_mean=.9, relative_l2=candidate_l2 if name == 'candidate' else .1, psnr_db=25.)
            write(root/'metrics'/name/(case+'.json'), dict(complete=True,
                provenance=dict(identity=identity,
                    candidate_sha256=sha(root/'generation'/name/case/'result.json'),
                    reference_sha256=sha(root/'generation/bf16'/case/'result.json')),
                rows=[dict(seed=seed, metrics=dict(video=metrics)) for seed in entry['seeds']]))


def test_significant_lpips_cannot_override_l2_guard(tmp_path):
    fixture(tmp_path, .11)
    result = report(tmp_path)
    assert result['final_eligible'] == []
    rtn = [r for r in result['contrasts'] if r['reference'] == 'rtn']
    assert all(r['metrics']['lpips']['familywise_ci'][1] < 0 and not r['passed'] for r in rtn)


def test_matched_budget_improvement_is_separate_from_fp8(tmp_path):
    fixture(tmp_path, .1)
    result = report(tmp_path)
    assert result['final_eligible'] == ['candidate']
    assert all(not r['passed'] for r in result['contrasts'] if r['reference'] == 'fp8')
    assert all(r['metrics']['lpips']['n_scenes'] == 4 for r in result['contrasts'])


def test_changed_reserved_seed_fails_closed(tmp_path):
    fixture(tmp_path, .1)
    file = tmp_path/'metrics/candidate/rbench_a_0.json'
    value = json.loads(file.read_text())
    value['rows'][0]['seed'] = 100
    write(file, value)
    with pytest.raises(ValueError, match='unpaired'):
        report(tmp_path)

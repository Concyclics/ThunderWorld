import json
from evidence_retest import emit


def test_emission_is_five_balanced_rounds_after_all_qualifications(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(frozen=True, jobs=[dict(name='bf16'), dict(name='awq')],
                                        rounds=5, ordering_seed=180918)))
    output = tmp_path/'commands'
    emit(manifest, output, '3')
    data = json.loads((output/'commands.json').read_text())
    rows = data['commands']
    assert [r['operation'] for r in rows[:4]] == ['prepare', 'prepare', 'verify', 'verify']
    assert len(rows) == 14
    assert all(r['environment']['CUDA_VISIBLE_DEVICES'] == '3' for r in rows)
    for index in range(5):
        assert {r['job'] for r in rows if r.get('round') == index} == {'bf16', 'awq'}
    assert list(output.iterdir()) == [output/'commands.json']

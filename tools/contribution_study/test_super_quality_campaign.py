import copy
import json
from types import SimpleNamespace

import pytest

import super_quality_campaign as campaign
from baselines import sha, write


def test_full_super_study_keeps_scope_and_uses_new_model_assets(tmp_path, monkeypatch):
    # Read the existing fixed method definitions; no GPU or historical output is
    # used. Empty manifests emulate independently prepared model-specific banks.
    root = tmp_path
    assets = root / 'super'
    cases = [dict(benchmark=b, category='category', seeds=[1, 2], image_sha256='sha',
                  case=dict(id=b+'__case', image='/input/'+b+'.png')) for b in ('rbench', 'paibench')]
    write(assets / 'cohort.json', dict(model='/official/Super', scope='independent calibration',
        cases=dict(development=cases)))
    write(root / 'full_cohort.json', dict(cases=cases, protocol=dict(steps=35)))
    candidates = [dict(name=n, assets=str(assets/n), config=dict(precision='fp4', budget_plan=str(assets/n/'plan.json')))
                  for n in ('b1.5_protected_n', 'b2_joint')]
    write(assets / 'ffn_guarded/candidates.json', dict(complete=True, candidates=candidates))
    for file in ('arc_build/build.json', 'arc_orders/manifest.json', 'awq_reference/manifest.json'):
        write(assets / file, dict(complete=True))
    campaign.build_studies(root)
    study = json.loads((assets / 'final/manifest.json').read_text())
    assert tuple(study['methods']) == campaign.METHODS
    assert study['model_name'] == 'super' and study['model'] == '/official/Super'
    assert study['cases'] == cases
    assert study['adapters']['arcquant']['fraction'] == .5
    assert study['adapters']['arcquant100']['fraction'] == 1
    assert study['adapters']['sharq_perf']['extra_fusion']
    for name in ('b1.5_protected_n', 'b2_joint'):
        assert study['methods'][name]['assets'] == str(assets/name)
        assert study['methods'][name]['runtime_options'] == {}
    awq = study['methods']['awq_reference']
    assert awq['baseline_adapter']['assets'] == str(assets/'awq_reference')
    assert not awq['storage_comparison_eligible']
    assert not awq['performance_eligible']
    assert study['methods']['fp8']['config']['precision'] == 'fp8'
    assert study['methods']['fp8']['config']['attention_backend'] == 'bf16'
    assert study['methods']['fp4_base']['config']['attention_backend'] == 'fp4'
    assert campaign.EXPECTED_VIDEOS == 30492


def test_official_weight_mismatch_rejected_on_resume(tmp_path):
    model = tmp_path/'weights'
    model.mkdir()
    file = model/'tensor'
    file.write_bytes(b'official')
    write(tmp_path/'model.lock.json', dict(model=str(model), files=[dict(path='tensor', bytes=8, sha256=sha(file))]))
    campaign.verify_model(tmp_path)
    file.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='checksum'):
        campaign.verify_model(tmp_path)

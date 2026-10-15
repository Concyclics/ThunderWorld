import copy
import json
from pathlib import Path

import pytest

import baseline_runtime
from baselines import sha, write
import judge_baseline_extension as extension


def cohort():
    protocol = dict(extension.EXPECTED_PROTOCOL, negative_prompt='original negative')
    return dict(protocol=protocol, cases=[dict(benchmark=b, seeds=[2*i+1, 2*i+2],
        case=dict(id=f'{b}_{i}', task='i2v', width=320, height=192, frames=61,
                  negative_prompt=protocol['negative_prompt']))
        for b, n in [('rbench', 650), ('paibench', 1044)] for i in range(n)])


def test_full_corpus_and_protocol_cannot_be_reduced_or_changed():
    original = cohort()
    extension.validate_cases(original)
    variants = []
    small = copy.deepcopy(original)
    small['cases'].pop()
    variants.append(small)
    duplicate = copy.deepcopy(original)
    duplicate['cases'][1]['case']['id'] = duplicate['cases'][0]['case']['id']
    variants.append(duplicate)
    wrong = copy.deepcopy(original)
    wrong['cases'][0]['case']['frames'] = 49
    variants.append(wrong)
    repeated = copy.deepcopy(original)
    repeated['cases'][0]['seeds'] = [1, 1]
    variants.append(repeated)
    cfg = copy.deepcopy(original)
    cfg['protocol']['guidance_scale'] = 5.0
    variants.append(cfg)
    for value in variants:
        with pytest.raises(ValueError):
            extension.validate_cases(value)


def methods():
    study = dict(methods={}, adapters={})
    for name in extension.METHODS:
        cfg = dict(debug='off', resource_mode='shared-calibration', execution='graph',
            compute_budget=None, budget_plan=None,
            precision={'fp8': 'fp8', 'fp4_base': 'fp4'}.get(name, 'bf16'),
            attention_backend='fp4' if name == 'fp4_base' else 'bf16',
            ffn=dict(activation=0, weight=0, cross=False),
            projection=dict(activation=0, weight=0, cross=False),
            attention=dict(default=dict(coverage=0, head_budgets=[], head_quotas=None,
                                        pv_head_quotas=None), rules=[]))
        study['methods'][name] = dict(config=cfg)
    study['methods']['awq_reference'].update(baseline_adapter=dict(method='awq'),
        performance_eligible=False, storage_comparison_eligible=False)
    return study


def test_no_hidden_repair_or_changed_baseline_semantics(monkeypatch):
    monkeypatch.setattr(baseline_runtime, 'verify', lambda *args: None)
    extension.validate_methods(methods())
    mutations = [
        ('fp4_base', lambda c: c['ffn'].update(weight=1)),
        ('fp4_base', lambda c: c.update(attention_backend='bf16')),
        ('fp4_base', lambda c: c['attention']['default'].update(coverage=.5)),
        ('fp4_base', lambda c: c['attention']['default'].update(pv_head_quotas=[1])),
        ('fp8', lambda c: c.update(precision='bf16')),
        ('awq_reference', lambda c: c.update(precision='fp4')),
        ('bf16', lambda c: c.update(execution='eager')),
    ]
    for name, mutate in mutations:
        study = methods()
        mutate(study['methods'][name]['config'])
        with pytest.raises(ValueError):
            extension.validate_methods(study)


def test_existing_children_use_extension_entrypoint_and_same_numeric_worker(monkeypatch, tmp_path):
    # Record original bindings so process-local configuration is restored here.
    monkeypatch.setattr(extension.frozen, 'METHODS', extension.frozen.METHODS)
    monkeypatch.setattr(extension.frozen, 'verify_job', extension.frozen.verify_job)
    monkeypatch.setattr(extension.pool, 'HERE', extension.pool.HERE)
    monkeypatch.setattr(extension.runtime, 'HERE', extension.runtime.HERE)
    extension.configure_existing_queue()
    assert extension.frozen.METHODS == extension.METHODS
    assert extension.frozen.verify_job is extension.verify_job
    command, env, fds = extension.runtime.child_command(tmp_path, 'nano', 'fp8',
        tmp_path/'generation', 1, (11, {1: 12}), qualification=True)
    assert command[1:3] == [str(extension.HERE), 'infer']
    assert '--qualification' in command
    assert env['CUDA_VISIBLE_DEVICES'] == '1'
    assert fds == (11, 12)
    assert extension.runtime.HERE.with_name('dataset_infer.py').is_file()
    extension.pool.validate_admission(dict(gpus=[0, 1]), [0, 1], 4,
        list(extension.METHODS), admission='requested')


def test_supplement_detects_modified_source_before_generation(tmp_path, monkeypatch):
    write(tmp_path/'manifest.json', dict(kind='full_I2V_four_baseline_extension',
        methods=list(extension.METHODS), source_sha256={'kernel.py': 'before'}))
    monkeypatch.setattr(extension, 'sources', lambda: {'kernel.py': 'after'})
    with pytest.raises(ValueError, match='frozen implementation'):
        extension.verify_job(tmp_path)

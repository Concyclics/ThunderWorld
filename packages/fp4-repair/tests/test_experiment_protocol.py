"""CPU checks for measurement provenance and interruption recovery."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load(name):
    path=Path(__file__).resolve().parents[3]/'tools/joint_inference'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_input_provenance_uses_content_and_prompt():
    fn=load('provenance').provenance
    run=dict(config={},model_identity={},targets=[],case=dict(id='x',prompt='original',image='/old/input.png'),image_sha256='abc')
    moved={**run,'case':{**run['case'],'image':'/new/input.png'}}
    assert fn(run)==fn(moved)
    changed={**run,'case':{**run['case'],'prompt':'changed'}}
    assert fn(run)!=fn(changed)
    assert fn(run)!=fn({**run,'image_sha256':'def'})


def test_resume_preserves_attempts_and_restarts_whole_round(tmp_path,monkeypatch):
    module=load('resume');path=tmp_path/'matrix.json';jobs=[]
    for repeat in range(2):
        for config in ('bf16','fp4'):
            output=str(tmp_path/f'{repeat}_{config}')
            jobs.append(dict(round=repeat,config=config,case='x',output=output,
                command=['python','infer','--output',output],exit_code=1 if repeat==1 and config=='fp4' else 0))
    original=dict(complete=False,suite='performance',device='0',jobs=jobs)
    path.write_text(json.dumps(original))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr('sys.argv',['resume',str(path)])
    monkeypatch.setattr(module,'idle_guard',lambda:None)
    calls=[]
    def run(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',run)
    module.main();updated=json.loads(path.read_text())
    assert updated['complete'] and len(calls)==2
    assert updated['jobs'][:2]==jobs[:2]
    for old,new in zip(jobs[2:],updated['jobs'][2:]):
        assert new['attempts'][0]==old
        assert new['output']!=old['output'] and new['exit_code']==0


def test_python_benchmark_rejects_competing_process_after_request(monkeypatch):
    import pytest
    from thunderworld_fp4.cosmos.runner import CosmosRunner
    checks=[];requests=[]
    def guard():
        checks.append(True)
        if len(checks)==3:raise RuntimeError('competing process')
    monkeypatch.setattr('thunderworld_fp4.cosmos.cli.idle_guard',guard)
    def infer(case,seed,*args):requests.append(seed);return {},dict(seed=seed)
    runner=SimpleNamespace(config=SimpleNamespace(debug='off',strict_tuning=True),
        warmup=lambda *args:None,warm_request=lambda *args:None,infer=infer)
    with pytest.raises(RuntimeError,match='competing process'):
        CosmosRunner.benchmark(runner,{},[1,2])
    assert requests==[1]


def test_cli_independent_policy_overrides(tmp_path):
    from thunderworld_fp4.cosmos.cli import policy_overrides
    from thunderworld_fp4.cosmos.settings import RunConfig, LinearSpec
    attention=tmp_path/'attention.json';projection=tmp_path/'projection.json'
    attention.write_text(json.dumps(dict(default=dict(coverage=.25,terms=2),rules=[
        dict(layers=[0],steps=[0,7],branches=['0'],
             config=dict(coverage=.5,terms=3,head_budgets=[[0,1],[1,0]]))])))
    projection.write_text(json.dumps(dict(activation=.75,weight=1,cross=True)))
    original=RunConfig()
    config=policy_overrides(original,attention_policy=attention,projection_config=projection,
                            ffn_policy=tmp_path/'ffn.json',ffn_policy_mode='calibrated')
    assert config.ffn==original.ffn and config.projection==LinearSpec(.75,1,True)
    assert config.ffn_policy==str(tmp_path/'ffn.json') and config.ffn_policy_mode=='calibrated'
    first=config.attention.resolve(layer=0,step=0,branch='0')
    assert first.head_counts(3,416)==(4,0,2)
    assert config.attention.resolve(layer=0,step=7,branch='0').coverage==.25

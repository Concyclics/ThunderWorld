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
    runner=SimpleNamespace(config=SimpleNamespace(debug='off',strict_tuning=True,resource_mode='exclusive'),
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


def test_final_freeze_binds_plan_and_control_assets(tmp_path,monkeypatch):
    import hashlib
    tools=Path(__file__).resolve().parents[3]/'tools/joint_inference'
    monkeypatch.syspath_prepend(str(tools))
    freeze=load('select_budget').freeze
    plan=tmp_path/'plan.json';plan.write_text('{"budget":1.25}')
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(budget_plan=str(plan))))
    assets=tmp_path/'assets';assets.mkdir();(assets/'manifest.json').write_text('{}')
    job=dict(name='control',control=True,config=str(config),assets=str(assets))
    frozen=freeze(job)
    assert frozen['budget_plan_sha256']==hashlib.sha256(plan.read_bytes()).hexdigest()
    assert frozen['assets_manifest_sha256']==hashlib.sha256(b'{}').hexdigest()
    plan.write_text('{"budget":1.5}')
    assert freeze(job)['budget_plan_sha256']!=frozen['budget_plan_sha256']


def test_compute_frontier_excludes_dominated_and_retains_tradeoff(monkeypatch):
    tools=Path(__file__).resolve().parents[3]/'tools/joint_inference'
    monkeypatch.syspath_prepend(str(tools))
    fn=load('final_compute_report').frontier
    rows=[dict(name='base',lpips=.3,mma=100,prepared_weight_bytes=100),
          dict(name='repair',lpips=.2,mma=125,prepared_weight_bytes=110),
          dict(name='dominated',lpips=.25,mma=130,prepared_weight_bytes=120)]
    assert fn(rows)==['base','repair']


def test_validation_retains_lower_error_control_even_if_candidate_passes_guardrails(tmp_path,monkeypatch):
    import hashlib
    from thunderworld_fp4.cosmos.cohorts import Cohort
    tools=Path(__file__).resolve().parents[3]/'tools/joint_inference'
    monkeypatch.syspath_prepend(str(tools))
    value=json.loads((tools/'cohorts/edge_i2v_compute_v1.json').read_text());value['budgets']=[1.25]
    cohort=Cohort(value);cohort_path=tmp_path/'cohort.json';cohort_path.write_text(json.dumps(value))
    jobs=[];rows=[];matrix_jobs=[]
    for name,candidate,error,ssim,l2 in [('control',False,.1,.8,.2),('candidate',True,.2,.9,.1)]:
        folder=tmp_path/name;folder.mkdir();(folder/'config.json').write_text('{}')
        (folder/'manifest.json').write_text('{}')
        (folder/'compute.json').write_text(json.dumps(dict(total_mma=125)))  # Older export metadata.
        (folder/'result.json').write_text(json.dumps(dict(compute_ledger=dict(total_mma=125,packing_traffic={},w1_support_union_bytes=1))))
        matrix_jobs.append(dict(config=name,output=str(folder)))
        jobs.append(dict(name=name,candidate=candidate,control=not candidate,budget=1.25,config=str(folder/'config.json'),assets=str(folder)))
        rows.extend(dict(config=name,seed=seed,metrics=dict(video=dict(lpips=error,ssim_mean=ssim,relative_l2=l2))) for seed in cohort.seeds['validation'])
    matrix=tmp_path/'matrix.json';matrix.write_text(json.dumps(dict(jobs=matrix_jobs)))
    quality=tmp_path/'quality.json';quality.write_text(json.dumps(dict(complete=True,suite='validation',cohort_sha256=cohort.fingerprint,rows=rows,
        matrix_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest())))
    manifest=tmp_path/'jobs.json';manifest.write_text(json.dumps(dict(jobs=jobs)))
    output=tmp_path/'frozen.json'
    monkeypatch.setattr('sys.argv',['select','--quality',str(quality),'--jobs',str(manifest),'--cohort',str(cohort_path),'--output',str(output)])
    load('select_budget').main()
    result=json.loads(output.read_text())
    assert result['decisions'][0]['selected']=='control'
    assert not result['decisions'][0]['optimized_candidate']
    assert result['jobs'][0]['assets_manifest_sha256']

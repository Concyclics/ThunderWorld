import json
import pytest
from thunderworld_fp4 import (RepairConfig, RepairPolicy, RepairRule, LinearRepairConfig,
                             InferenceConfig, PRESETS, preset, config_from_head_scores)

@pytest.mark.parametrize('name', PRESETS)
def test_presets(name):
    config=preset(name)
    assert RepairConfig.from_dict(json.loads(json.dumps(config.to_dict())))==config

@pytest.mark.parametrize('kwargs',[{'coverage':float('nan')},{'coverage':True},{'coverage':1.1},
    {'query_tile':128},{'terms':1},{'head_budgets':[(0,.2),(0,.5)]},{'head_budgets':[(-1,.2)]}])
def test_invalid(kwargs):
    with pytest.raises(ValueError):RepairConfig(**kwargs)

def test_independent_configs():
    c=InferenceConfig.from_dict({'ffn':{'coverage':1},'attention':{'coverage':.5,'head_budgets':[[1,0],[2,1]]},'attention_projection':{'coverage':.25}})
    assert c.ffn.output_counts(256,512)==(4,4)
    assert c.attention.head_counts(3,512)==(2,0,4)
    assert c.attention_projection.output_counts(256,512)==(1,1)
    assert InferenceConfig.from_dict(json.loads(json.dumps(c.to_dict())))==c
    assert c.attention.cost(3,512)['matrix_product_multiplier']==2

def test_head_finding_policy():
    high=config_from_head_scores([.2,.9,.9,.1],protected_count=1,coverage=.25,protected_coverage=1)
    assert high.head_counts(4,512)==(1,4,1,1)
    p=RepairPolicy(rules=(RepairRule(high,layers=(2,),steps=(3,5)),))
    assert p.resolve(layer=2,step=3)==high
    assert p.resolve(layer=2,step=5)==p.default
    with pytest.raises(ValueError):p.resolve(layer=2)
    with pytest.raises(ValueError):RepairConfig(head_budgets=((9,.5),)).head_counts(2,128)
    with pytest.raises(ValueError):LinearRepairConfig().output_counts(129,256)

def test_existing_head_atlas_adapter():
    from thunderworld_fp4 import policy_from_head_atlas
    atlas={'layers':[{'layer':0,'compiled_q_fp8_heads':[1,3]}, {'layer':2,'compiled_q_fp8_heads':[0]}]}
    policy=policy_from_head_atlas(atlas,coverage=.25,protected_coverage=1)
    assert policy.resolve(layer=0).head_counts(4,512)==(1,4,1,4)
    assert policy.resolve(layer=2).head_counts(4,512)==(4,1,1,1)
    assert policy.resolve(layer=1).head_counts(4,512)==(1,1,1,1)
    with pytest.raises(ValueError):policy_from_head_atlas({'layers':atlas['layers']*2})

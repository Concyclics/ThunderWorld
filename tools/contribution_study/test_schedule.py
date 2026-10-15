import pytest
from thunderworld_fp4 import RepairConfig
from schedule_common import attention_setting, resolve_attention, variants


def test_independent_pv_fractions_and_tail():
    cfg=resolve_attention(attention_setting([1,.5,0,.75],[0,1,.25,.5]))
    assert cfg.head_counts(4,129)==(2,1,0,2)
    assert cfg.pv_counts(4,129)==(0,2,1,1)
    assert cfg.head_counts(4,416)==(4,2,0,3)
    assert cfg.pv_counts(4,416)==(0,4,1,2)
    assert RepairConfig.from_dict(cfg.to_dict())==cfg
    with pytest.raises(ValueError):RepairConfig(pv_head_budgets=((0,.5),(0,.75)))
    with pytest.raises(ValueError):RepairConfig(pv_coverage=1.01)


def test_legacy_pv_inheritance_and_separate_override():
    cfg=RepairConfig(head_quotas=(0,2))
    assert cfg.pv_counts(2,129)==(0,2)
    assert RepairConfig(head_quotas=(0,2),pv_coverage=1).pv_counts(2,129)==(2,2)
    with pytest.raises(ValueError):resolve_attention(dict(attention_setting([1],[1]),full_attention=True))


def test_choices_cannot_expand_weight_bank():
    base=dict(a_blocks=8,w_blocks=[1,2,5,7],cross=False)
    rows=variants(base,2048,[7,1,5,2])
    assert rows['w_low']['w_blocks']==[1,5,7]
    assert rows['cross']['cross']
    assert all(set(row['w_blocks']).issubset(base['w_blocks']) for row in rows.values())
    assert rows['a_high']['a_blocks']==10


def test_joint_exchange_respects_every_shape_and_attention_floor():
    from schedule_allocate import allocate
    groups={'attention':[dict(label='base',kind='attention',loss=2.,costs=[4,8]),dict(label='low',kind='attention',loss=3.,costs=[2,4])],
            'ffn':[dict(label='base',kind='ffn',loss=5.,costs=[4,4]),dict(label='high',kind='ffn',loss=1.,costs=[6,7]),
                   dict(label='unsafe',kind='ffn',loss=0.,costs=[6,10])]}
    result=allocate(groups,[8,12],change_limit=[4,8],attention_floor=[2,4])
    assert result['totals']==[8,11]
    assert result['assignments']['ffn']['label']=='high'
    assert result['assignments']['attention']['label']=='low'
    held=allocate(groups,[8,12],change_limit=[4,8],attention_floor=[4,8])
    assert all(row['label']=='base' for row in held['assignments'].values())


def test_mixed_cfg_lengths_can_exceed_all_diagonal_bounds():
    from schedule_envelope import compile_envelope
    from thunderworld_fp4.cosmos.budget import PHASES
    plan=dict(schema_version=1,task='i2v',steps=35,phases=list(PHASES),budget=2.5,linears={},attention={})
    for branch in range(2):
        for ph in range(4):
            plan['linears'][f'{branch}|{ph}|projection']=dict(a_blocks=1 if branch==0 else 0,w_blocks=[0] if branch==0 else [],cross=branch==0)
            plan['attention'][f'{branch}|{ph}|0']=(dict(full_attention=True,qk_terms=4,pv_terms=4) if branch==0 else attention_setting([0.,0.],[0.,0.]))
    shapes=dict(linears={'projection':[960,128,128]},attention={'0':[960,2048,2,1]})
    for nt in range(8,17):assert compile_envelope(plan,shapes,(nt,nt))['ratio']==2.5
    with pytest.raises(ValueError,match='Compute budget exceeded'):compile_envelope(plan,shapes,(9,8))
    assert compile_envelope(plan,shapes,(8,9))['ratio']<2.5

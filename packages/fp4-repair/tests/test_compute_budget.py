import itertools
from dataclasses import replace
from fractions import Fraction
import pytest
from thunderworld_fp4.cosmos.budget import linear_cost,attention_cost,compile_request,RepairPlan
from thunderworld_fp4.cosmos.settings import LinearSpec,RunConfig
from thunderworld_fp4.cosmos.cohorts import Cohort
from thunderworld_fp4.config import RepairConfig


@pytest.mark.parametrize('m',[129,416])
@pytest.mark.parametrize('bm',[32,64,128])
def test_exact_and_dynamic_cross_upper_bound(m,bm):
    weights=(0,1,3,5,6)
    bound=linear_cost(m,256,512,bm=bm,a_blocks=2,w_blocks=weights,cross=True)
    for chosen in itertools.combinations(range(4),2):
        exact=linear_cost(m,256,512,bm=bm,a_blocks=2,w_blocks=weights,cross=True,
                          a_support=[chosen]*((m+127)//128))
        assert exact['total_mma']<=bound['total_mma']
    full=linear_cost(m,256,512,bm=bm,a_blocks=4,w_blocks=range(8),cross=True)
    assert full['total_mma']==4*full['base_mma']
    base=linear_cost(m,256,512,bm=bm)
    assert base['total_mma']==base['base_mma'] and base['padded_base_mma'] == 2*((m+bm-1)//bm*bm-m)*256*512


def test_gqa_independent_attention_and_selector_ledger():
    base=attention_cost(129,416,4,2,qk_counts=[0]*4,pv_counts=[0]*4)
    full=attention_cost(129,416,4,2,qk_counts=[4]*4,pv_counts=[4]*4,qk_terms=4,pv_terms=4)
    assert full['total_mma']==4*base['total_mma']
    partial=attention_cost(129,416,4,2,qk_counts=[1,0,2,4],pv_counts=[0,1,0,1])
    assert partial['mma_by_dtype']['bf16']['selector']==4*2*2*16*32*128
    assert partial['gqa_support_union_upper'][0]['k_blocks_upper']==2
    mask=[[[False]*4 for _ in range(2)] for _ in range(4)]
    exact=attention_cost(129,416,4,2,explicit_masks={'qk':mask,'pv':mask})
    assert exact['total_mma']==base['total_mma']


def test_request_budget_is_global_and_fails_closed():
    rows=[]
    for branch in range(2):
        for ph in range(4):
            rows.extend([{**linear_cost(129,128,128,a_blocks=1 if ph==0 else 0),'name':'linear','branch':branch,'phase':ph},
                         {**attention_cost(129,128,1,1),'name':'0','branch':branch,'phase':ph}])
    result=compile_request(rows,1.125,expected_modules=['linear'],expected_layers=['0'])
    assert result['feasible'] and result['rows'][0]['calls']==7
    with pytest.raises(ValueError,match='exceeded'):compile_request(rows,1.)
    with pytest.raises(ValueError):compile_request(rows[:-1],1.5)
    with pytest.raises(ValueError):compile_request(rows+rows[:1],1.5)


def test_integer_plan_spec_and_config_roundtrip():
    spec=LinearSpec(0,0,True,activation_blocks=1,weight_support=(0,2))
    assert spec.counts(256,256)==(1,2)
    with pytest.raises(ValueError):spec.counts(128,128)
    cfg=RepairConfig(head_quotas=(0,1),pv_head_quotas=(2,0),pv_terms=4)
    assert cfg.head_counts(2,129)==(0,1) and cfg.pv_counts(2,129)==(2,0)
    with pytest.raises(ValueError):cfg.pv_counts(2,128)
    run=RunConfig(resource_mode='shared-calibration',compute_budget=1.25,ffn=spec)
    assert RunConfig.from_dict(run.to_dict())==run


def test_cohort_disjoint_and_frozen():
    value=dict(schema_version=1,steps=35,phases=[0,7,18,28,35],
               seeds=dict(calibration=[1],trajectory=[2],validation=list(range(10,18)),final=list(range(20,36))))
    cohort=Cohort(value)
    with pytest.raises(ValueError):cohort.validate('final',[1])
    with pytest.raises(ValueError):cohort.validate('final',[20],complete=True)
    bad={**value,'seeds':{**value['seeds'],'trajectory':[1]}}
    with pytest.raises(ValueError):Cohort(bad)


def test_prescale_direction_and_mandatory_outliers():
    import torch
    from thunderworld_fp4.cosmos.calibration import prescale_candidates,outlier_mask,quantize_weight,decode
    from thunderworld_fp4.cosmos.assets import compact,expand
    torch.manual_seed(4)
    w=torch.randn(128,256).bfloat16();rms=torch.linspace(.1,5,256)
    for _,d in prescale_candidates(rms,w):
        assert d.min()>=1/16 and d.max()<=16
        x=torch.randn(3,256)
        torch.testing.assert_close((x*d)@(w.float()/d).T,x@w.float().T,atol=1e-4,rtol=1e-4)
    w[0,0]=100
    remove,mandatory=outlier_mask(w,torch.ones(256),rms,[0],2.)
    assert remove.reshape(128,16,16).sum(-1).max()<=1
    assert not remove[:,128:].any() and mandatory==(0,)
    c,s,g=quantize_weight(w,torch.ones(256),remove=remove)
    decoded=decode(c,s,g)
    assert decoded[0,0,0]==0 and decoded[1,0,0]>90
    asset=dict(codes=c,scales=s,factor=g,score=torch.ones(1,2),mandatory_w_blocks=mandatory)
    with pytest.raises(ValueError):compact(asset,[])
    restored=expand(compact(asset,[0]),[0])
    assert torch.equal(restored['codes'][0],c[0])
    with pytest.raises(ValueError):expand(compact(asset,[0]),[1])


def test_allocator_grow_prune_hard_cap_and_combination_loss():
    from thunderworld_fp4.cosmos.allocation import allocate
    groups={f'g{i}':[
        dict(id=0,base_mma=100,mma=100,loss=10.),
        dict(id=1,base_mma=100,mma=110,loss=8.),
        dict(id=2,base_mma=100,mma=125,loss=9.),  # antagonistic combination: dominated
        dict(id=3,base_mma=100,mma=150,loss=3.)] for i in range(4)}
    for direction in ('grow','prune'):
        for budget in (1.,1.125,1.25,1.5):
            result=allocate(groups,budget,direction=direction)
            assert result['total_mma']<=result['base_mma']*budget
            assert all(c['id']!=2 for c in result['assignments'].values())


def test_restricted_allocation_filters_before_pareto_pruning():
    from thunderworld_fp4.cosmos.allocation import allocate
    group=[dict(id=0,base_mma=100,mma=100,loss=10.,a_blocks=0,w_blocks=[]),
           dict(id=1,base_mma=100,mma=110,loss=8.,a_blocks=1,w_blocks=[]),
           dict(id=2,base_mma=100,mma=105,loss=7.,a_blocks=0,w_blocks=[0])]
    assert allocate({'op':group},1.125,preference='a')['assignments']['op']['id']==1


def test_attention_accounts_probability_quantization_separately_from_qkv_pack():
    from thunderworld_fp4.cosmos.budget import attention_cost
    # ACT2 PV repairs P only: no Q/K/V residual codes are needed, but online P1
    # quantization still executes for the selected padded MMA tiles.
    cost=attention_cost(129,384,4,2,pv_counts=[1,0,2,3],pv_terms=2)
    work=cost['non_mma']
    assert work['probability_residual_quantized_elements']==5*6*32*128
    assert work['probability_base_quantized_elements']==5*4*3*32*128
    assert work['qkv_residual_quantized_elements_upper']==0
    assert work['combine_output_elements']==129*4*128

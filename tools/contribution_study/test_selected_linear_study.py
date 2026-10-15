import copy
import pytest

from selected_linear_study import side_plan
from thunderworld_fp4.cosmos.budget import linear_cost, attention_cost, compile_request, PHASES


@pytest.mark.parametrize('m', [129,416,960])
@pytest.mark.parametrize('family', ['activation','weight'])
def test_full_side_cap_includes_tails_and_phase_cfg(m, family):
    name='layers.0.mlp_moe_gen.up_proj'
    source=dict(budget=1.5, linears={f'{b}|{p}|{name}':dict(a_blocks=1,w_blocks=[0],cross=False)
                                   for b in range(2) for p in range(4)})
    frozen=copy.deepcopy(source)
    bank=dict(modules={name:dict(shape=[256,384])})
    plan=side_plan(source,bank,family)
    assert source==frozen
    rows=[]
    for b in range(2):
        for p in range(4):
            spec=plan['linears'][f'{b}|{p}|{name}']
            r=linear_cost(m,256,384,**spec)
            assert r['total_mma']==2*r['base_mma']
            assert r['mma_by_dtype']['fp4']['a' if family=='weight' else 'w']==0
            rows.append(dict(r,name=name,phase=p,branch=b))
            a=attention_cost(m,416+b*128,4,2,qk_terms=2,pv_terms=2)
            assert a['total_mma']==a['base_mma']
            rows.append(dict(a,name='0',phase=p,branch=b))
    ledger=compile_request(rows,2.,expected_modules=[name],expected_layers=['0'])
    assert ledger['feasible'] and 1<ledger['ratio']<2
    assert sum(PHASES[i+1]-PHASES[i] for i in range(4))==35


def test_side_control_rejects_unsupported_tails():
    with pytest.raises(ValueError,match='tile'):
        side_plan(dict(linears={'0|0|x':{}}),dict(modules={'x':dict(shape=[129,256])}),'activation')

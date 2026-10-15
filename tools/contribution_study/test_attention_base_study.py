"""Independent request-count oracle and rejection of accidental attention repair."""
from types import SimpleNamespace
import pytest
from attention_base_runtime import compile_engine,compile_cost


@pytest.mark.parametrize('kind',['base','a','w','both25','both50'])
def test_current_scope_and_padded_mma(kind):
    masks={'base':1,'a':3,'w':17};budget=1 if kind=='base' else 1.5 if kind=='both25' else 2
    spec={'expansion':{'mask':masks.get(kind,19),'activation_planes':1 if kind in ('base','w') else 2}}
    if kind.startswith('both'):spec['sparse_refinement']={'numerator':1 if kind=='both25' else 2}
    shapes={};attention={};oracle_base=oracle_total=0
    for branch,m in enumerate((129,416)):
        shapes[str(branch)]=dict(module='test',branch=branch,m=m,n=256,k=512)
        attention[str(branch)]=dict(layer=0,branch=branch,q=[1,m,4,128],k=[1,513,2,128])
        linear=2*((m+127)//128*128)*256*512
        attn=2*2*((m+31)//32*32)*640*128*4
        oracle_base+=(linear+attn)*35;oracle_total+=(linear*budget+attn)*35
    engine=SimpleNamespace(branches=2,steps=35,targets=['test'],model=SimpleNamespace(layers=[None]),
        shapes=shapes,attention_shapes=attention,attention_base_spec=spec,config=SimpleNamespace(compute_budget=budget))
    ledger=compile_engine(engine)
    assert ledger['base_mma']==oracle_base and ledger['total_mma']==oracle_total
    assert ledger['linear_ratio']==budget and ledger['attention_ratio']==1
    assert ledger['ratio']<=budget
    for row in ledger['rows']:
        if row['kind']=='attention':
            assert row['qk_a_mma']==row['qk_w_mma']==row['pv_a_mma']==row['pv_w_mma']==0
    corrupted=[dict(r) for r in ledger['rows']]
    next(r for r in corrupted if r['kind']=='attention')['total_mma']+=1
    with pytest.raises(ValueError,match='Attention repair leaked'):compile_cost(corrupted,2.)

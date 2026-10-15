"""Frozen FP4 base attention with separately controlled linear A/W repairs.

No new GPU kernel: reuse the qualified parent expansion/sparse kernels. Attention
always packs one Q/K/V plane and executes one QK and one PV product per base tile.
"""
import copy
import json
from dataclasses import replace
from pathlib import Path
import types
from baselines import sha

HERE=Path(__file__).resolve().parent


def source_identity():
    from expansion_sparse_runtime import source_identity as sparse_identity
    return {**sparse_identity(),**{n:sha(HERE/n) for n in ('attention_base_runtime.py','attention_base_study.py')}}


def verify(study,name):
    lock=json.loads(Path(study['attention_base_lock']).read_text())
    if not lock['complete'] or lock['source_sha256']!=source_identity():
        raise ValueError('Changed attention-base study implementation')
    spec=study['methods'][name]
    if name=='bf16':
        import baseline_runtime
        return baseline_runtime._attention_base_original_verify(study,name)
    if spec.get('attention_expansion')!={'weight_planes':1,'activation_planes':1,'mask':1,'terms':1}:
        raise ValueError('Attention repair must be completely disabled')
    if 'sparse_refinement' in spec:
        from expansion_sparse_runtime import verify as check
    else:
        from expansion_runtime import verify as check
    check(study,name)


def compile_cost(rows,budget,steps=35,branches=2):
    if budget not in (1.,1.5,2.) or {r['branch'] for r in rows}!=set(range(branches)):
        raise ValueError('Invalid budget/CFG scope')
    base=sum(r['base_mma'] for r in rows)*steps
    total=sum(r['total_mma'] for r in rows)*steps
    linear=[r for r in rows if r['kind']=='linear'];attention=[r for r in rows if r['kind']=='attention']
    if not linear or not attention or any(r['total_mma']!=r['base_mma'] for r in attention):
        raise ValueError('Attention repair leaked into the ablation')
    lb=sum(r['base_mma'] for r in linear)*steps;lt=sum(r['total_mma'] for r in linear)*steps
    if total>base*budget or lt>lb*budget:raise ValueError('Budget exceeded')
    return dict(schema_version=1,steps=steps,branches=branches,budget=budget,linear_budget=budget,
        base_mma=base,total_mma=total,ratio=total/base,linear_base_mma=lb,linear_total_mma=lt,
        linear_ratio=lt/lb,attention_ratio=1.,feasible=True,selector_mma=0,cross_mma=0,
        mma_by_dtype={'native_nvfp4':total},performance_eligible=False,rows=rows,
        scope='C0 is padded base MMA for generation projection/FFN/QK/PV; common BF16 AR and non-MMA work excluded',
        labeling='1.5x/2x labels describe linear MMA caps; actual whole-request controlled-MMA ratios reported separately',
        non_mma='Packing, sorting, softmax, P quantization and split combine are additional costs; no speed claim')


def compile_engine(engine):
    from expansion_sparse_budget import linear_cost,attention_cost
    spec=engine.attention_base_spec;rows=[]
    for branch in range(engine.branches):
        for name in engine.targets:
            shapes={(r['m'],r['n'],r['k']) for r in engine.shapes.values() if r['module']==name and r['branch']==branch}
            if len(shapes)!=1:raise ValueError('Unobserved or ambiguous linear shape')
            sparse=spec.get('sparse_refinement');num=sparse['numerator'] if sparse else 0
            r=linear_cost(*next(iter(shapes)),num)
            if not sparse:
                mask=spec['expansion']['mask'];r['a_mma']=r['base_mma'] if mask&2 else 0
                r['w_mma']=r['base_mma'] if mask&16 else 0
                r['total_mma']=r['base_mma']+r['a_mma']+r['w_mma'];r['ratio']=r['total_mma']/r['base_mma']
                r['selector']='Full side or base; no selector'
                r['activation_pack_write_bytes']=r['activation_pack_write_bytes']//2*spec['expansion']['activation_planes']
                r['quota_per_side']=None
            rows.append(dict(r,name=name,branch=branch))
        for layer in range(len(engine.model.layers)):
            shapes={(tuple(r['q']),tuple(r['k'])) for r in engine.attention_shapes.values() if r['layer']==layer and r['branch']==branch}
            if len(shapes)!=1:raise ValueError('Unobserved or ambiguous attention shape')
            q,k=next(iter(shapes));r=attention_cost(q[1],k[1],q[2],k[2],0)
            r.update(full_residual_packing=False,gqa_packed_union_density=0.,selector='None; one-plane FP4 QK/PV',
                     qkv_pack_write_bytes=r['qkv_pack_write_bytes']//2)
            rows.append(dict(r,name=str(layer),branch=branch))
    engine.budget_ledger=compile_cost(rows,engine.config.compute_budget,engine.steps,engine.branches)
    return engine.budget_ledger


def make_runner(study,name):
    verify(study,name)
    if name=='bf16':
        import baseline_runtime
        return baseline_runtime._attention_base_original_make(study,name)
    spec=study['methods'][name]
    if 'sparse_refinement' in spec:
        from expansion_sparse_runtime import make_runner as make
        runner,adapter=make(study,name)
    else:
        from expansion_runtime import make_runner as make
        bootstrap=copy.deepcopy(study);bootstrap['methods'][name]['config']['compute_budget']=None
        runner,adapter=make(bootstrap,name)
        runner.config=replace(runner.config,compute_budget=spec['linear_budget'])
        runner.engine.config=runner.config
    from expansion_kernels import attention
    def attend(engine,layer,q,k,v):
        if engine.warming:
            engine.attention_shapes[layer,engine.branch,tuple(q.shape),tuple(k.shape)]=dict(
                q=list(q.shape),k=list(k.shape),layer=layer,branch=engine.branch,
                left_planes=1,right_planes=1,qk_terms=1,pv_terms=1,
                attention_repair=False,dense_base_blocks_preserved=True)
        return attention(q,k,v,1,1,1)
    runner.engine.attend=types.MethodType(attend,runner.engine)
    runner.engine.attention_base_spec=spec
    runner.engine.compile_budget=types.MethodType(compile_engine,runner.engine)
    adapter.update(method='fixed_base_fp4_attention',attention_expansion=spec['attention_expansion'],
        attention_sides='All Q/P and K/V corrections disabled; K/V remain online activations',
        attention_repair=False,linear_budget=spec['linear_budget'],source_sha256=source_identity(),
        scope='Only generation linear/FFN A/W repairs vary; all FP4 methods use identical base FP4 attention',
        performance_eligible=False)
    return runner,adapter


def install():
    import baseline_runtime
    baseline_runtime._attention_base_original_verify=baseline_runtime.verify
    baseline_runtime._attention_base_original_make=baseline_runtime.make_runner
    baseline_runtime.verify=verify;baseline_runtime.make_runner=make_runner

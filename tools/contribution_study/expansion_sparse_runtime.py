"""Install sparse supports while reusing the frozen expansion bank and quantizer."""
import json
import copy
from dataclasses import replace
from pathlib import Path
import types
import torch
from baselines import sha
import expansion_runtime as parent
import expansion_sparse_kernels as sparse
from expansion_sparse_budget import linear_cost,attention_cost,compile_request

HERE=Path(__file__).resolve().parent


def source_identity():
    return {**parent.source_identity(),**{n:sha(HERE/n) for n in (
        'expansion_sparse_kernels.py','expansion_sparse_budget.py','expansion_sparse_runtime.py',
        'expansion_sparse.py')}}


def verify(study,name):
    parent.verify(study,name)
    lock=json.loads(Path(study['sparse_implementation_lock']).read_text())
    if not lock['complete'] or lock['source_sha256']!=source_identity():
        raise ValueError('Sparse implementation not qualified or changed')
    for key in ('pytest','memcheck','racecheck'):
        entry=lock[key]
        if entry['exit_code'] or sha(entry['log'])!=entry['sha256']:
            raise ValueError('Changed qualification: '+key)
    spec=study['methods'][name]['sparse_refinement']
    if spec['budget']!=1+2*spec['numerator']/4 or spec['numerator'] not in (1,2):
        raise ValueError('Unexpected sparse quota definition')


def linear_forward(op,x):
    shape=x.shape;flat=x.reshape(-1,op.in_features).contiguous()
    if op.controller.warming:
        key=(op.name,op.controller.branch,len(flat))
        op.controller.shapes[str(key)]=dict(module=op.name,branch=op.controller.branch,
            m=len(flat),n=op.out_features,k=op.in_features)
    y=sparse.linear(flat,(op.codes,op.scales,op.factor),op.weight_mask,op.sensitivity,
                    op.numerator,prescale=op.prescale)
    if op.bias is not None:y=y+op.bias
    return y.reshape(*shape[:-1],op.out_features)


def compile_engine_budget(engine):
    rows=[]
    for branch in range(engine.branches):
        for name in engine.targets:
            observed=[r for r in engine.shapes.values() if r['module']==name and r['branch']==branch]
            shapes={(r['m'],r['n'],r['k']) for r in observed}
            if len(shapes)!=1:raise ValueError('Missing/ambiguous linear shape '+name)
            cost=linear_cost(*next(iter(shapes)),engine.sparse_numerator)
            rows.append(dict(cost,name=name,branch=branch))
        for layer in range(len(engine.model.layers)):
            observed=[r for r in engine.attention_shapes.values() if r['layer']==layer and r['branch']==branch]
            shapes={(tuple(r['q']),tuple(r['k'])) for r in observed}
            if len(shapes)!=1:raise ValueError('Missing/ambiguous attention shape')
            q,k=next(iter(shapes))
            cost=attention_cost(q[1],k[1],q[2],k[2],engine.sparse_numerator)
            rows.append(dict(cost,name=str(layer),branch=branch))
    engine.budget_ledger=compile_request(rows,engine.config.compute_budget,engine.steps,engine.branches)
    return engine.budget_ledger


@torch.inference_mode()
def make_runner(study,name):
    verify(study,name)
    # Parent first prepares BF16 modules before installing the frozen bank.
    # A budgeted configuration is valid only after that installation completes.
    bootstrap=copy.deepcopy(study)
    bootstrap['methods'][name]['config']['compute_budget']=None
    runner,adapter=parent.make_runner(bootstrap,name)
    runner.config=replace(runner.config,compute_budget=study['methods'][name]['sparse_refinement']['budget'])
    runner.engine.config=runner.config
    numerator=study['methods'][name]['sparse_refinement']['numerator']
    support={}
    for target,op in runner.engine.linears.items():
        wm,sensitivity=sparse.weight_support((op.codes,op.scales,op.factor),numerator)
        op.register_buffer('weight_mask',wm);op.register_buffer('sensitivity',sensitivity)
        op.numerator=numerator;op.forward=types.MethodType(linear_forward,op)
        support[target]=dict(mask=wm.cpu(),sensitivity=sensitivity.cpu())
    # Supports are determined from the unchanged offline bank before inference.
    support_file=Path(study['sparse_support_directory'])/(name+'.pt')
    support_file.parent.mkdir(parents=True,exist_ok=True)
    if support_file.exists():
        old=torch.load(support_file,weights_only=True,map_location='cpu')
        if old.keys()!=support.keys() or any(not torch.equal(old[n][k],support[n][k])
                for n in support for k in support[n]):raise ValueError('Changed offline support')
    else:torch.save(support,support_file)
    runner.engine.sparse_numerator=numerator
    runner.engine.compile_budget=types.MethodType(compile_engine_budget,runner.engine)
    def attend(engine,layer,q,k,v):
        if engine.warming:
            engine.attention_shapes[layer,engine.branch,tuple(q.shape),tuple(k.shape)]=dict(
                q=list(q.shape),k=list(k.shape),layer=layer,branch=engine.branch)
        return sparse.attention(q,k,v,numerator)
    runner.engine.attend=types.MethodType(attend,runner.engine)
    adapter.update(method='matched_sparse_bilateral',source_sha256=source_identity(),
        sparse_refinement=study['methods'][name]['sparse_refinement'],
        offline_support=dict(path=str(support_file),sha256=sha(support_file)),
        parent_quantizer_and_w0_unchanged=True,cross_terms=False,
        limitations=['Full residual planes retained and packed: no storage or packing-saving claim.',
            'MMA budget verified before Graph capture; non-MMA costs remain separate.',
            'Residual-energy support proxy fixed before observing this cohort; no calibration on these outputs.'])
    return runner,adapter


def install_worker_adapter():
    import baseline_runtime
    original_verify,original_make=baseline_runtime.verify,baseline_runtime.make_runner
    def checked(study,name):
        if 'sparse_refinement' in study['methods'][name]:return verify(study,name)
        return original_verify(study,name)
    def make(study,name):
        if 'sparse_refinement' in study['methods'][name]:return make_runner(study,name)
        return original_make(study,name)
    baseline_runtime.verify,baseline_runtime.make_runner=checked,make

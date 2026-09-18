"""Small Cosmos shape adapters around pinned official baseline implementations."""
import importlib
import sys
from pathlib import Path
import torch
from baselines import git,sha


def install_ar_repair(runner,terms):
    """AR prefill attribution; DM projections and attention remain BF16."""
    from types import SimpleNamespace
    from thunderworld_fp4.cosmos.linears import RepairLinear
    from thunderworld_fp4.cosmos.settings import LinearSpec
    from thunderworld_fp4.cosmos.tuning import LaunchTable
    from thunderworld_fp4.cosmos.budget import linear_cost
    records={}
    def record(name,shape,n,spec,op):
        m,k=shape
        records[f'{name}|{m}']=dict(name=name,shape=list(shape),output_features=n,
            issued_cost=linear_cost(m,n,k,a_blocks=k//128,w_blocks=list(range((n//128)*(k//128))),cross=terms==4),
            logical_bf16_flops=2*m*n*k)
    controller=SimpleNamespace(warming=True,table=LaunchTable(),linear_spec=lambda name,default:default,record_linear=record)
    model=runner.pipe.transformer;names=[]
    for name,module in list(model.named_modules()):
        if not isinstance(module,torch.nn.Linear):continue
        if '.mlp.' not in name and not any(name.endswith('.self_attn.'+suffix) for suffix in ('to_q','to_k','to_v','to_out')):continue
        if module.in_features%128 or module.out_features%128:raise ValueError('Unsupported AR projection '+name)
        model.set_submodule(name,RepairLinear(module,controller,name,LinearSpec(1.,1.,terms==4)))
        names.append(name)
    if not names:raise ValueError('No AR linears found')
    runner._contribution_ar_controller=controller
    return dict(method=f'ar{terms}',scope='AR projection/FFN attribution only; causal AR attention and all DM remain BF16. '
                'Same fresh AR-cache prefill on every request and CFG branch. Costs are per unique prefix shape, not multiplied by 35 DM steps.',
                targets=names,prefill_projection_costs=records,performance_eligible=False)


def load_sharq(root,commit):
    root=Path(root)
    if git(root,'rev-parse','HEAD')!=commit:raise ValueError('SharQ source version changed')
    if 'quantize' in sys.modules:
        loaded=Path(sys.modules['quantize'].__file__).resolve()
        if loaded!=root/'model/quantize.py':raise ValueError('Conflicting external quantize module; use a separate process')
    sys.path[:0]=[str(root/'model'),str(root)]
    return importlib.import_module('qLinearLayer').QLinearLayer


class SharQLinear(torch.nn.Module):
    def __init__(self,source,official,mode='SHARQ'):
        super().__init__()
        self.in_features=source.in_features;self.out_features=source.out_features
        self.official=official(source,quant_type=mode,extra_fusion=False)
        # Upstream stores several tensors as attributes, not registered buffers.
        # Register them so storage accounting includes every resident weight.
        for key,value in list(vars(self.official).items()):
            if isinstance(value,torch.Tensor):
                delattr(self.official,key);self.official.register_buffer(key,value)

    def forward(self,x):
        shape=x.shape
        return self.official(x.reshape(-1,self.in_features)).reshape(*shape[:-1],self.out_features)


def install_sharq(runner,root,commit,mode='SHARQ'):
    if runner.engine.cache or runner._warmed:raise ValueError('Install before graph capture/warmup')
    official=load_sharq(root,commit)
    for name in runner.engine.targets:
        source=runner.pipe.transformer.get_submodule(name)
        op=SharQLinear(source,official,mode)
        runner.pipe.transformer.set_submodule(name,op)
        runner.engine.linears[name]=op
    library=importlib.import_module('quantize').load_sharq_ops() if mode=='SHARQ' else None
    return dict(method=mode,commit=commit,execution_type='official_native_adapter' if mode=='SHARQ' else 'official_simulation_adapter',
                library=str(Path(library.__file__).resolve()) if library else None,library_sha256=sha(library.__file__) if library else None,
                changes=['Preserve arbitrary leading input dimensions','Register upstream tensor attributes for memory accounting'],
                extra_fusion=False,performance_eligible=False)


class ARCLinear(torch.nn.Module):
    """Official kernels; explicit eager-only adapter because alpha is host scalar."""
    def __init__(self,source,library,order,fraction):
        super().__init__()
        if not 0<=fraction<=1:raise ValueError('Invalid residual channel fraction')
        self.in_features=source.in_features;self.out_features=source.out_features
        self.count=int(self.in_features*fraction)//64*64
        self.library=library
        self.register_buffer('order',order.to(device=source.weight.device,dtype=torch.int16))
        scale=source.weight.detach().max().float()/2688.
        scale=torch.where(scale>0,scale,torch.ones_like(scale))
        codes,scales=library.reorder_quantize_w((source.weight.detach()/scale).bfloat16(),self.order,self.count)
        self.register_buffer('weight_codes',codes);self.register_buffer('weight_scales',scales)
        self.register_buffer('factor',scale)
        self.register_buffer('bias',source.bias.detach().clone() if source.bias is not None else None)

    def pack_input(self,x):
        scale=x.max().float()/2688.
        scale=torch.where(scale>0,scale,torch.ones_like(scale))
        codes,scales=self.library.reorder_quantize_x((x/scale).bfloat16(),self.order,self.count)
        return codes,scales,scale

    def forward(self,x):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Pinned ARCQuant exposes host alpha; graph execution is not qualified')
        shape=x.shape
        codes,scales,factor=self.pack_input(x.reshape(-1,self.in_features).contiguous())
        y=self.library.matmul(codes,self.weight_codes,scales,self.weight_scales,float(factor*self.factor))
        if self.bias is not None:y=y+self.bias
        return y.reshape(*shape[:-1],self.out_features)


def load_arc(build):
    import json
    from baselines import sha
    build=Path(build);manifest=json.loads((build/'build.json').read_text())
    library=build/'kernels/build/agemm.so'
    if not manifest['complete'] or sha(library)!=manifest['library_sha256']:
        raise ValueError('Unqualified ARC build provenance')
    sys.path.insert(0,str(library.parent))
    return importlib.import_module('agemm'),manifest


def install_arc(runner,build,capture,fraction):
    import json
    library,manifest=load_arc(build)
    capture=Path(capture);panels=json.loads((capture/'manifest.json').read_text())
    if panels['model_identity']!=runner.identity:raise ValueError('Calibration model mismatch')
    for name in runner.engine.targets:
        source=runner.pipe.transformer.get_submodule(name)
        peak=torch.zeros(source.in_features)
        rows=[r for r in panels['rows'] if r['kind']=='linear' and r['name']==name]
        if len(rows)!=16:raise ValueError('Incomplete ARC calibration')
        for row in rows:
            x=torch.load(capture/row['file'],weights_only=True)['x']
            peak=torch.maximum(peak,x.float().abs().amax(0))
        order=torch.argsort(peak,stable=True)
        op=ARCLinear(source,library,order,fraction)
        runner.pipe.transformer.set_submodule(name,op);runner.engine.linears[name]=op
    return dict(method='ARCQuant',source_commit=manifest['source_commit'],build=manifest,
        channel_fraction=fraction,channel_selection='Full-M max over 16 teacher calibration panels, ascending reorder',
        performance_eligible=False,execution_type='official_native_cosmos_adapter',
        limitations=['Eager-only host alpha API','Fraction is controlled, not upstream model-wide proportion search'],
        changes=['K9216 instantiation','Cosmos tensor shapes','Finite scale for all-zero/nonpositive-max input'])


class AWQReference(torch.nn.Module):
    def __init__(self,asset,bias=None):
        super().__init__()
        self.register_buffer('weight',asset['weight'].cuda())
        self.register_buffer('prescale',asset['prescale'].cuda())
        self.register_buffer('bias',None if bias is None else bias.detach().clone())
        self.out_features,self.in_features=self.weight.shape

    def forward(self,x):
        return torch.nn.functional.linear(x*self.prescale,self.weight,self.bias)


def install_awq(runner,assets):
    import json
    from baselines import sha
    assets=Path(assets);manifest=json.loads((assets/'manifest.json').read_text())
    if not manifest['complete'] or manifest['model_identity']!=runner.identity or set(manifest['modules'])!=set(runner.engine.targets):
        raise ValueError('Complete matching all-layer AWQ assets required')
    for name,entry in manifest['modules'].items():
        path=assets/entry['file']
        if sha(path)!=entry['sha256']:raise ValueError('Changed AWQ asset')
        source=runner.pipe.transformer.get_submodule(name)
        op=AWQReference(torch.load(path,weights_only=True),source.bias)
        runner.pipe.transformer.set_submodule(name,op);runner.engine.linears[name]=op
    return dict(method='NVFP4-AWQ-W4A16',assets_sha256=sha(assets/'manifest.json'),
                execution_type='offline_decoded_quality_reference',performance_eligible=False,
                storage_comparison_eligible=False)

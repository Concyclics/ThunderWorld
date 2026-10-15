"""Offline nested W4 assets and current-input native FP4 expansion adapters."""
import json
from dataclasses import replace
from pathlib import Path
import time
import types

import torch

from baselines import sha, write
from expansion_kernels import pack_linear, linear, attention

HERE=Path(__file__).resolve().parent


def source_identity():
    return {name:sha(HERE/name) for name in ('expansion_kernels.py','expansion_runtime.py','expansion_experiment.py')}


@torch.inference_mode()
def prepare_assets(study, destination):
    from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
    destination=Path(destination)
    if (destination/'manifest.json').exists():
        old=json.loads((destination/'manifest.json').read_text())
        if old['complete'] and old['kernel_sha256']==sha(HERE/'expansion_kernels.py'):
            for entry in old['modules'].values():
                if sha(destination/entry['file']) != entry['sha256']:raise ValueError('Changed offline weight asset')
            return old
        raise ValueError('Incomplete or changed offline assets; use a new asset directory')
    runner=CosmosRunner(study['model'],RunConfig.from_dict(study['methods']['bf16']['config']),
                        ar_cache=True,first_frame=True).prepare()
    destination.mkdir(parents=True,exist_ok=True)
    manifest=dict(complete=False,model_identity=runner.identity,planes=4,
        kernel_sha256=sha(HERE/'expansion_kernels.py'),modules={},targets=runner.engine.targets,
        prescale='identity',global_factor='full-matrix absmax / 2688; one for zero matrix',
        residual='FP32 normalized input minus actual E2M1 * E4M3 decoded plane, recursively',
        scale_floor=1/512,storage='Offline four-plane master bank; each run loads only its W prefix')
    write(destination/'manifest.json',manifest)
    for index,name in enumerate(runner.engine.targets):
        source=runner.pipe.transformer.get_submodule(name)
        w=source.weight.detach().contiguous()
        codes,scales,factor,_=pack_linear(w,4)
        asset=dict(codes=codes.cpu(),scales=scales.cpu(),factor=factor.cpu(),
            bias=None if source.bias is None else source.bias.detach().cpu(),shape=list(w.shape))
        path=destination/f'{index:04d}.pt'
        torch.save(asset,path)
        manifest['modules'][name]=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,shape=list(w.shape))
        if (index+1)%12==0:print(json.dumps(dict(event='offline_weights',completed=index+1,total=len(runner.engine.targets))),flush=True)
    manifest['complete']=True
    write(destination/'manifest.json',manifest)
    runner.engine.clear()
    return manifest


class ExpansionLinear(torch.nn.Module):
    def __init__(self, source, asset, controller, name, specification):
        super().__init__()
        self.in_features,self.out_features=source.in_features,source.out_features
        self.controller,self.name=controller,name
        self.wp=specification['weight_planes'];self.ap=specification['activation_planes'];self.mask=specification['mask']
        self.register_buffer('codes',asset['codes'][:self.wp].contiguous().cuda())
        self.register_buffer('scales',asset['scales'][:self.wp].contiguous().cuda())
        self.register_buffer('factor',asset['factor'].cuda())
        self.register_buffer('bias',None if asset['bias'] is None else asset['bias'].cuda())
        self.register_buffer('prescale',torch.ones(self.in_features,device='cuda'))

    def forward(self,x):
        shape=x.shape
        flat=x.reshape(-1,self.in_features).contiguous()
        if self.controller.warming:
            key=(self.name,self.controller.branch,len(flat))
            self.controller.shapes[str(key)]=dict(module=self.name,m=len(flat),n=self.out_features,k=self.in_features,
                branch=self.controller.branch,weight_planes=self.wp,activation_planes=self.ap,terms=self.mask.bit_count(),
                bm=128,bn=128,bk=128,backend='native_fp4_dot_scaled',dense_base_blocks_preserved=True)
        y=linear(flat,(self.codes,self.scales,self.factor),self.wp,self.ap,self.mask,prescale=self.prescale)
        if self.bias is not None:y=y+self.bias
        return y.reshape(*shape[:-1],self.out_features)


def verify(study,name):
    lock=json.loads(Path(study['implementation_lock']).read_text())
    if not lock['complete'] or lock['source_sha256'] != source_identity():
        raise ValueError('Expansion implementation is not frozen and qualified')
    bank=Path(study['expansion_assets'])
    if sha(bank/'manifest.json') != lock['weight_manifest_sha256']:
        raise ValueError('Offline expansion weights changed')
    specification=study['methods'][name]['expansion']
    wp,ap=specification['weight_planes'],specification['activation_planes']
    from expansion_kernels import term_mask
    expected=term_mask(wp,ap,specification.get('dual',False))
    if expected != specification['mask'] or expected.bit_count()!=specification['terms']:
        raise ValueError('Changed expansion term definition')


@torch.inference_mode()
def make_runner(study,name):
    from thunderworld_fp4.cosmos import CosmosRunner, RunConfig
    verify(study,name)
    specification=study['methods'][name]['expansion']
    # Build attention dispatch without invoking the built-in two-plane weight
    # quantizer. All generation linears are replaced from offline assets below.
    cfg=RunConfig.from_dict({**study['methods'][name]['config'],'precision':'fp4'})
    bootstrap=replace(cfg,precision='bf16',attention_only_attribution=True)
    runner=CosmosRunner(study['model'],bootstrap,ar_cache=True,first_frame=True).prepare()
    start=time.perf_counter()
    bank=Path(study['expansion_assets']);manifest=json.loads((bank/'manifest.json').read_text())
    if manifest['model_identity']!=runner.identity or set(manifest['modules'])!=set(runner.engine.targets):
        raise ValueError('Wrong model or projection scope for offline expansion weights')
    for target,entry in manifest['modules'].items():
        path=bank/entry['file']
        if sha(path)!=entry['sha256']:raise ValueError('Corrupted offline weight asset')
        asset=torch.load(path,weights_only=True,map_location='cpu')
        source=runner.pipe.transformer.get_submodule(target)
        op=ExpansionLinear(source,asset,runner.engine,target,specification)
        runner.pipe.transformer.set_submodule(target,op);runner.engine.linears[target]=op
    wp,ap,mask=(specification[field] for field in ('weight_planes','activation_planes','mask'))
    def attend(engine,layer,q,k,v):
        if engine.warming:
            engine.attention_shapes[layer,engine.branch,tuple(q.shape),tuple(k.shape)]=dict(
                q=list(q.shape),k=list(k.shape),layer=layer,branch=engine.branch,
                left_planes=ap,right_planes=wp,qk_terms=mask.bit_count(),pv_terms=mask.bit_count(),
                backend='native_fp4_streaming',dense_base_blocks_preserved=True)
        return attention(q,k,v,wp,ap,mask)
    runner.engine.attend=types.MethodType(attend,runner.engine)
    runner.config=cfg
    runner.engine.config=cfg
    torch.cuda.synchronize()
    runner.timings['offline_expansion_install_seconds']=time.perf_counter()-start
    adapter=dict(method='nested_nvfp4_expansion',execution_type='native_fp4_tensor_core',
        actual_generation_precision='NVFP4 E2M1/E4M3, FP32 accumulators, BF16 operator outputs',
        specification=specification,source_sha256=source_identity(),
        weight_manifest_sha256=sha(bank/'manifest.json'),targets=runner.engine.targets,
        scope='All generation projections and FFN; all noncausal generation QK and PV products. '
              'AR prefill/causal attention and non-GEMM operators retain the common BF16 path.',
        attention_sides='A planes apply to Q/P; W-plane count applies to current K/V, which remain online activations.',
        request_invariant_weights=True,current_input_activation_packing=True,
        packed_weight_bytes=sum(t.numel()*t.element_size() for op in runner.engine.linears.values()
                                for t in (op.codes,op.scales,op.factor)),
        performance_eligible=False,
        limitations=['Quality mechanism kernel; repeated on-chip P quantization bounds register use.',
                     'Fixed E4M3 scale floor and FP32 accumulation can limit higher-plane improvement.'])
    return runner,adapter


def install_worker_adapter():
    import baseline_runtime
    original_verify,original_make=baseline_runtime.verify,baseline_runtime.make_runner
    def checked(study,name):
        if 'expansion' in study['methods'][name]:return verify(study,name)
        return original_verify(study,name)
    def make(study,name):
        if 'expansion' in study['methods'][name]:return make_runner(study,name)
        return original_make(study,name)
    baseline_runtime.verify,baseline_runtime.make_runner=checked,make

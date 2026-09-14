"""Instance-local joint attention dispatch and current-input Transformer graphs."""
from dataclasses import replace
from collections import Counter
import json
from pathlib import Path
import types
import time
import torch
from ..runtime import _forward as fp4_attention
from .._fusion import activate_with_factor
from ..linear import _pack
from .linears import select_tiles
import triton
from .settings import TIERS, phase
from .linears import RepairLinear, FP8Linear
from .trees import signature, clone_tree, copy_tree


def attention_tuning_spec(config, mask):
    spec=config.to_dict()
    if mask is not None:
        # Configuration preparation only. An explicit mask overrides quotas
        # and uses a different kernel path, so it must not reuse a quota-only
        # timing entry or a profile measured for another support pattern.
        import hashlib
        spec['explicit_mask_sha256']=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
    return spec


def generation_targets(transformer):
    result=[]
    for i,layer in enumerate(transformer.layers):
        for name,child in layer.mlp_moe_gen.named_children():
            if isinstance(child,torch.nn.Linear):result.append(f'layers.{i}.mlp_moe_gen.{name}')
        for name in ('add_q_proj','add_k_proj','add_v_proj','to_add_out'):
            child=getattr(layer.self_attn,name)
            if not isinstance(child,torch.nn.Linear):raise TypeError(f'Unexpected generation projection: {name}')
            result.append(f'layers.{i}.self_attn.{name}')
    return result


class JointEngine:
    def __init__(self, transformer, config, table, assets=None):
        self.model=transformer;self.config=config;self.table=table
        self.cutlass_artifact=None
        if config.linear_backend=='cutlass-fused':
            from ..cutlass.artifact import verify_library
            self.cutlass_artifact=verify_library(config.cutlass_library)
            if config.cutlass_small_library:
                small=verify_library(config.cutlass_small_library)
                if self.cutlass_artifact['tile']!=[128,128,128] or small['tile']!=[128,64,128]:
                    raise ValueError('Dual-library dispatch requires N128 primary and N64 small-grid kernels')
                self.cutlass_artifact={'primary':self.cutlass_artifact,'small_grid':small}
        self.forward=transformer.forward
        self.warming=False;self.step=0;self.branch=0;self.steps=35;self.task='i2v'
        self.policy=json.loads(Path(config.ffn_policy).read_text()) if config.ffn_policy else None
        if self.policy and tuple(self.policy['phases'])!=(0,.2,.5,.8,1):
            raise ValueError('Unsupported FFN phase boundaries')
        from .budget import RepairPlan
        self.budget_plan=RepairPlan.load(config.budget_plan) if config.budget_plan else None
        if self.budget_plan and config.compute_budget!=self.budget_plan.value['budget']:
            raise ValueError('Compute budget and plan must agree')
        self.discovering=False;self.budget_ledger=None;self.attention_shapes={}
        self.targets=generation_targets(transformer);self.linears={};self.shapes={}
        self.cache={};self.pool=torch.cuda.graph_pool_handle();self.replays=Counter()
        self.attention_prepared={};self.trace_masks={};self.mask_history=[]
        self.explicit_masks={}
        self.pair_masks={}
        self._validated_domains=set()
        self.closed=False
        self.capture_attention=None
        self.graph_capture_seconds=0.
        self._schedule_cache={}
        if config.precision!='bf16':
            for name in self.targets:
                source=transformer.get_submodule(name)
                asset=None if assets is None else assets[name]
                spec=config.ffn if '.mlp_moe_gen.' in name else config.projection
                op=FP8Linear(source,asset) if config.precision=='fp8' else RepairLinear(source,self,name,spec,asset)
                transformer.set_submodule(name,op);self.linears[name]=op
        self._install_attention()
        self._install_domain_checks()
        if config.precision=='fp4' and config.fuse_ffn:self._install_fusion()
        transformer.forward=self

    def linear_spec(self,name,default):
        if self.budget_plan is not None:
            return self.budget_plan.linear(self.task,self.branch,phase(self.step,self.steps),name)
        if self.policy is None or '.mlp_moe_gen.' not in name:return default
        p=self.policy['policies'][self.config.ffn_policy_mode]
        parts=name.split('.')
        return TIERS[p['assignments'][f'{self.task}|{self.branch}|{phase(self.step,self.steps)}|{parts[1]}.{parts[-1]}']]

    def attention_config(self,layer,step,branch):
        cfg=self.config.attention.resolve(layer=layer,step=step,branch=str(branch))
        if self.budget_plan is not None:
            cfg=self.budget_plan.attention_config(self.task,branch,phase(step,self.steps),layer,cfg)
        return cfg

    def schedule(self,step):
        old=self.step;self.step=step
        try:
            linears=tuple((name,self.linear_spec(name,op.default).key)
                          for name,op in self.linears.items() if isinstance(op,RepairLinear))
            attention=tuple(self.attention_config(i,step,self.branch)
                            for i in range(len(self.model.layers))) if self.config.attention_backend=='fp4' else ()
            return (linears,attention)
        finally:self.step=old

    def record_linear(self,name,shape,n,spec,op):
        key=(self.task,self.branch,phase(self.step,self.steps),name,tuple(shape),spec.key)
        self.shapes[str(key)]=dict(module=name,m=int(shape[0]),n=n,k=int(shape[1]),
            branch=self.branch,phase=phase(self.step,self.steps),task=self.task,w_blocks=list(op.weight_ids),
            repair=spec.key,actual_a=op.na/(op.k//128),actual_w=op.wi.numel()/(op.n//128*(op.k//128)),
            backend=self.config.linear_backend,direct_scale_pack=getattr(op,'direct_scale_pack',False),
            launch=(dict(tile=[128,op.tile_n,128],schedule='TMA cooperative + sparse cp.async')
                    if self.config.linear_backend=='cutlass-fused' else dict(bm=op.bm,stages=op.num_stages)))

    def _install_attention(self):
        # Copy processor function globals for this instance; never patch the
        # process-global diffusers dispatch used by other model instances.
        for layer_id,layer in enumerate(self.model.layers):
            original=layer.self_attn.processor
            function=type(original).__call__
            if 'dispatch_attention_fn' not in function.__globals__:
                raise TypeError('Expected unfused Cosmos3 attention processor')
            dispatch=function.__globals__['dispatch_attention_fn']
            def local(q,k,v,_layer=layer_id,_original=dispatch,**kwargs):
                if kwargs.get('is_causal',False) or self.config.attention_backend=='bf16':
                    out=_original(q,k,v,**kwargs)
                    if self.capture_attention is not None and not self.warming and not kwargs.get('is_causal',False):
                        self.capture_attention(_layer,q,k,v,out)
                    return out
                if kwargs.get('attn_mask') is not None or kwargs.get('parallel_config') is not None:
                    raise ValueError('Joint FP4 supports unmasked single-device generation attention')
                return self.attend(_layer,q,k,v)
            globals_copy=dict(function.__globals__,dispatch_attention_fn=local)
            copied=types.FunctionType(function.__code__,globals_copy,function.__name__,function.__defaults__,function.__closure__)
            copied.__kwdefaults__=function.__kwdefaults__
            cls=type('JointCosmosProcessor',(type(original),),{'__call__':copied})
            processor=cls();processor.__dict__.update(original.__dict__)
            layer.self_attn.set_processor(processor)

    def attend(self,layer,q,k,v):
        cfg=self.attention_config(layer,self.step,self.branch)
        key=(layer,tuple(q.shape),tuple(k.shape),cfg,self.branch)
        explicit=self.explicit_masks.get((layer,self.branch))
        if self.warming and key not in self.attention_prepared:
            quotas=torch.tensor(cfg.head_counts(q.shape[2],k.shape[1]),device=q.device,dtype=torch.int32)
            if cfg.pv_head_quotas is not None or cfg.pv_terms is not None:
                quotas=(quotas,torch.tensor(cfg.pv_counts(q.shape[2],k.shape[1]),device=q.device,dtype=torch.int32))
            tune_key=self.table.key('attention',[list(q.shape),list(k.shape)],attention_tuning_spec(cfg,explicit))
            current=[cfg]
            def apply(v):current[0]=replace(cfg,**v)
            def fn():return fp4_attention(q,k,v,current[0],repair_mask=explicit,counts_tensor=quotas)[0]
            anchor=dict(query_tile=cfg.query_tile or 32,num_warps=cfg.num_warps,transpose_v=cfg.transpose_v,reload_q=cfg.reload_q)
            candidates=[anchor]+[dict(query_tile=b,num_warps=w,transpose_v=t,reload_q=r)
                for b in (16,32,64) for w in (4,8) for t in (False,True) for r in (False,True)]
            self.table.search(tune_key,candidates,apply,fn)
            launch=self.table.get(tune_key,anchor)
            self.attention_prepared[key]=(replace(cfg,**launch),quotas)
        if key not in self.attention_prepared:raise RuntimeError(f'Unwarmed attention layer {layer}')
        launch,quotas=self.attention_prepared[key]
        if self.warming:
            self.attention_shapes[self.task,self.branch,phase(self.step,self.steps),layer]=dict(
                q=list(q.shape),k=list(k.shape),config=launch,explicit=explicit)
        out,mask,held=fp4_attention(q,k,v,launch,repair_mask=explicit,counts_tensor=quotas,
                              return_mask=self.config.debug=='trace')
        if self.warming and launch.sparse_pack and held and isinstance(held[-1],dict):
            unions=held[-1]
            self.attention_shapes[self.task,self.branch,phase(self.step,self.steps),layer]['warmup_union_density']={
                name:float(value.float().mean().cpu()) for name,value in unions.items()}
        if self.capture_attention is not None and not self.warming:self.capture_attention(layer,q,k,v,out)
        if mask is not None:self.trace_masks[layer]=mask
        return out

    def _install_fusion(self):
        for layer in self.model.layers:
            mlp=layer.mlp_moe_gen
            # External calibrated assets may have a nonidentity down prescale.
            if not bool((mlp.down_proj.asset['prescale']==1).all()):continue
            shared=(self.config.share_pack and self.budget_plan is None and mlp.hidden_act=='silu' and
                mlp.gate_proj.asset.get('selector')!='residual_sensitivity' and
                mlp.up_proj.asset.get('selector')!='residual_sensitivity' and
                torch.equal(mlp.gate_proj.asset['prescale'],mlp.up_proj.asset['prescale']) and
                mlp.gate_proj.asset.get('activation_scale',(1.,1.))==(1.,1.) and
                mlp.up_proj.asset.get('activation_scale',(1.,1.))==(1.,1.))
            def forward(mod,x,_shared=shared):
                if mod.hidden_act=='relu2':z,factor=activate_with_factor(mod.up_proj(x))
                else:
                    packed=None
                    if _shared:
                        flat=x.reshape(-1,x.shape[-1]).contiguous();m,k=flat.shape
                        key=(m,k,mod.up_proj.name)
                        if self.warming and key not in self.pair_masks:
                            self.pair_masks[key]=torch.empty((triton.cdiv(m,128),k//128),device=x.device,dtype=torch.bool)
                        if not self.warming:
                            specs=[self.linear_spec(op.name,op.default) for op in (mod.gate_proj,mod.up_proj)]
                            na=max(__import__('math').ceil(s.activation*(k//128)) for s in specs)
                            d=mod.up_proj.ops[m,specs[1].key].d
                            union=self.pair_masks[key]
                            if 0<na<k//128:
                                select_tiles[(triton.cdiv(m,128),)](flat,d,union,m,k,triton.next_power_of_2(k//128),na,num_warps=4)
                            packed=_pack(flat,d,na>0,na==k//128,union)
                    z,factor=activate_with_factor(mod.act_fn(mod.gate_proj(x,packed=packed)),mod.up_proj(x,packed=packed))
                return mod.down_proj(z,factor=factor)
            mlp.forward=types.MethodType(forward,mlp)

    def _install_domain_checks(self):
        for mod in self.model.modules():
            if type(mod).__name__!='DomainAwareLinear':continue
            def domain_forward(mod,x,domain_id):
                ids=domain_id.to(device=x.device,dtype=torch.long).reshape(-1)
                if x.shape[0]!=len(ids):raise ValueError('Action token/domain shape mismatch')
                w=mod.fc(ids).view(len(ids),mod.input_size,mod.output_size)
                b=mod.bias(ids).view(len(ids),mod.output_size)
                if x.ndim==2:return torch.bmm(x.unsqueeze(1),w).squeeze(1)+b
                if x.ndim==3:return torch.bmm(x,w)+b.unsqueeze(1)
                raise ValueError('Unsupported action tensor rank')
            mod.forward=types.MethodType(domain_forward,mod)

    def validate_domains(self,kwargs):
        # Per-request validation happens once on first Transformer invocation.
        # Check both full and noisy domain uses from the pipeline's supplied IDs.
        for i,ids in enumerate(kwargs.get('action_domain_ids') or []):
            if i in self._validated_domains:continue
            limit=self.model.config.num_embodiment_domains
            if bool(((ids<0)|(ids>=limit)).any()):raise ValueError('Invalid action domain')
            self._validated_domains.add(i)

    def reset_request(self,steps,branches,task):
        self.steps,self.branches,self.task=steps,branches,task
        self.call=0;self.replays.clear();self.mask_history=[];self._validated_domains.clear()
        key=(steps,branches,task)
        if key not in self._schedule_cache:
            schedules={}
            for branch in range(branches):
                self.branch=branch
                for step in range(steps):schedules[branch,step]=self.schedule(step)
            self._schedule_cache[key]=schedules
        self._schedules=self._schedule_cache[key]
        self.branch=0;self.step=0

    def compile_budget(self):
        from .budget import linear_cost,attention_cost,compile_request
        rows=[]
        for branch in range(self.branches):
            for ph,start in enumerate((0,7,18,28)):
                for name,op in self.linears.items():
                    observed=[r for r in self.shapes.values() if r['task']==self.task and r['branch']==branch and r['module']==name]
                    shapes={(r['m'],r['n'],r['k']) for r in observed}
                    if len(shapes)!=1:raise ValueError('Budget requires one observed shape per operator and branch')
                    m,n,k=next(iter(shapes))
                    old_step,old_branch=self.step,self.branch;self.step,self.branch=start,branch
                    try:spec=self.linear_spec(name,op.default)
                    finally:self.step,self.branch=old_step,old_branch
                    if (m,spec.key) not in op.ops:raise ValueError('Unobserved repair schedule')
                    native=op.ops[m,spec.key]
                    cost=linear_cost(m,n,k,bm=native.bm,a_blocks=native.na,w_blocks=native.weight_ids,cross=spec.cross)
                    if self.config.linear_backend=='cutlass-fused':
                        cost['launch']['bn']=native.tile_n
                        cost['repair_block']=[128,128]
                        cost['execution_ctas']=triton.cdiv(m,128)*(n//native.tile_n)
                        cost['packing']['cutlass_scale_read_bytes']=0 if native.direct_scale_pack else triton.cdiv(m,16)*16*(k//16)
                        cost['packing']['cutlass_scale_conversion_launches']=0 if native.direct_scale_pack else 1
                        cost['packing']['cutlass_scale_write_bytes']=triton.cdiv(m,128)*128*(k//16)
                        cost['packing']['cutlass_alpha_write_bytes']=4
                        cost['cutlass_extra_weight_scale_bytes']=n*(k//16)
                        cost['cutlass_fp32_gmem_scratch_bytes']=0
                    sparse=op.asset.get('selector')=='residual_sensitivity' and 0<native.na<k//128
                    cost['packing']['implementation']='selected_codes' if sparse else 'legacy_full_plane'
                    cost['packing']['residual_write_bytes']=cost['packing']['selected_residual_write_bytes'] if sparse else cost['packing']['residual_allocation_bytes']
                    if sparse:cost['non_mma']['base_reencoded_elements']=cost['non_mma']['selected_residual_elements']
                    rows.append({**cost,'name':name,'branch':branch,'phase':ph})
                for layer in range(len(self.model.layers)):
                    cfg=self.attention_config(layer,start,branch)
                    observed=[r for (t,b,p,l),r in self.attention_shapes.items() if t==self.task and b==branch and l==layer]
                    shapes={(tuple(r['q']),tuple(r['k'])) for r in observed}
                    if len(shapes)!=1:raise ValueError('Budget requires one observed attention shape per branch')
                    q,k=next(iter(shapes))
                    key=(layer,q,k,cfg,branch)
                    if key not in self.attention_prepared:raise ValueError('Unobserved attention schedule')
                    launch,_=self.attention_prepared[key]
                    explicit=self.explicit_masks.get((layer,branch));masks=None
                    if explicit is not None:
                        mask=explicit.cpu().tolist();masks=dict(qk=mask,pv=mask)
                    cost=attention_cost(q[1],k[1],q[2],k[2],bm=launch.query_tile or 32,
                         qk_counts=cfg.head_counts(q[2],k[1]),pv_counts=cfg.pv_counts(q[2],k[1]),
                         qk_terms=cfg.terms,pv_terms=cfg.pv_terms or cfg.terms,explicit_masks=masks)
                    cost['packing_implementation']='gqa_union_selected_codes' if cfg.sparse_pack else 'full_residual_planes'
                    rows.append({**cost,'name':str(layer),'branch':branch,'phase':ph})
        if any(op.split_k!=1 for layer in self.linears.values() for op in getattr(layer,'ops',{}).values()):
            raise ValueError('Split-K requires its own budget/quality protocol')
        self.budget_ledger=compile_request(rows,self.config.compute_budget,steps=self.steps,branches=self.branches,
            expected_modules=self.targets,expected_layers=[str(i) for i in range(len(self.model.layers))])
        if self.budget_plan:self.budget_ledger['plan_sha256']=self.budget_plan.fingerprint
        return self.budget_ledger

    def clear(self):
        torch.cuda.synchronize();self.cache.clear();self.pool=torch.cuda.graph_pool_handle()

    def __call__(self,*args,**kwargs):
        if args:raise ValueError('Cosmos Transformer expects keyword arguments')
        self.step,self.branch=divmod(self.call,self.branches)
        if self.step>=self.steps:raise RuntimeError('Unexpected Transformer call count')
        self.validate_domains(kwargs)
        current=self._schedules[self.branch,self.step]
        basekey=(self.task,self.branch,signature(kwargs),torch.cuda.current_stream().cuda_stream)
        if self.warming:
            # Capture every distinct actual joint policy, not merely four FFN phases.
            schedules={}
            for step in range(self.steps):schedules.setdefault(self._schedules[self.branch,step],step)
            saved=self.step
            for schedule,step in schedules.items():
                self.step=step;key=(basekey,schedule)
                if key in self.cache:continue
                static=clone_tree(kwargs)
                for _ in range(2):self.forward(**static)
                if self.discovering:continue
                if self.config.execution=='graph':
                    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                    capture_start=time.perf_counter()
                    self.warming=False
                    try:
                        with torch.cuda.graph(graph,pool=self.pool):output=self.forward(**static)
                    finally:self.warming=True
                    torch.cuda.synchronize()
                    self.graph_capture_seconds+=time.perf_counter()-capture_start
                    self.cache[key]=(static,graph,output,dict(self.trace_masks))
                else:self.cache[key]=None
            self.step=saved
        if self.discovering:
            result=self.forward(**kwargs);self.call+=1
            return result
        key=(basekey,current)
        if key not in self.cache:raise RuntimeError('Unwarmed task/shape/joint policy')
        if self.config.execution=='graph':
            static,graph,output,masks=self.cache[key]
            copy_tree(static,kwargs);graph.replay();result=clone_tree(output)
        else:
            result=self.forward(**kwargs);masks=self.trace_masks
        if self.config.debug=='trace':self.mask_history.append({i:m.clone() for i,m in masks.items()})
        self.replays[self.branch,self.step]+=1
        self.call+=1
        return result

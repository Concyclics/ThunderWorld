"""Whole-transformer graphs keyed by phase/branch/task, with replay accounting.

Graphs execute serially, return owned outputs, and share an intermediate pool.
Static inputs are separate per entry and copied on every replay. No tensor
values are reused across steps. Clear before changing weights or task shapes.
"""
from collections import Counter
import math
import types
import torch
from dynamic import BOUNDS,phase
from graph import signature,clone_tree,copy_tree


class PhaseGraph:
    def __init__(self,forward,study,controller,modules,schedule_steps=35):
        self.forward=forward;self.study=study;self.controller=controller;self.modules=modules
        self.schedule_steps=schedule_steps;self.warming=False;self.cache={}
        self.pool=torch.cuda.graph_pool_handle();self.captures=0;self.reset_stats()
        model=getattr(forward,'__self__',None)
        self.domain_limit=getattr(getattr(model,'config',None),'num_embodiment_domains',None)
        if model is not None:
            for module in model.modules():
                if type(module).__name__!='DomainAwareLinear' or getattr(module,'_phase_graph_patched',False):continue
                original=module.forward
                def domain_forward(mod,x,domain_id,_original=original):
                    if not torch.cuda.is_current_stream_capturing():return _original(x,domain_id)
                    ids=domain_id.to(device=x.device,dtype=torch.long).reshape(-1)
                    if x.shape[0]!=ids.shape[0]:raise ValueError('Action domain/token shape mismatch')
                    weight=mod.fc(ids).view(ids.shape[0],mod.input_size,mod.output_size)
                    bias=mod.bias(ids).view(ids.shape[0],mod.output_size)
                    if x.ndim==2:return torch.bmm(x.unsqueeze(1),weight).squeeze(1)+bias
                    if x.ndim==3:return torch.bmm(x,weight)+bias.unsqueeze(1)
                    raise ValueError('Unsupported action input rank')
                module.forward=types.MethodType(domain_forward,module);module._phase_graph_patched=True

    def clear(self):
        torch.cuda.synchronize();self.cache.clear();self.pool=torch.cuda.graph_pool_handle()
        self.captures=0;self.reset_stats()

    def reset_stats(self):
        self.replays=0;self.phase_hits=Counter()

    def state(self):
        mods=[(m.calls,m.base_flops,m.selected_extra_flops,dict(m.tier_calls)) for m in self.modules()]
        return dict(self.study.counts),mods,self.controller.pack_hits

    def restore(self,state):
        counts,mods,hits=state;self.study.counts=counts
        for m,(calls,base,extra,tiers) in zip(self.modules(),mods):
            m.calls=calls;m.base_flops=base;m.selected_extra_flops=extra;m.tier_calls=tiers
        self.controller.pack_hits=hits

    def key(self,kwargs,ph):
        return (self.study.mode,self.controller.mode,self.controller.task,self.study.branch,ph,
                torch.cuda.current_stream().cuda_stream,signature(kwargs))

    def capture_phase(self,kwargs,ph):
        key=self.key(kwargs,ph)
        if key in self.cache:return
        if len(self.cache)>=8:raise RuntimeError('Clear phase graph cache before changing task/shape')
        saved=self.state();saved_step,saved_steps=self.study.step,self.study.steps
        old_warming=self.controller.warming
        self.study.steps=self.schedule_steps;self.study.step=math.ceil(self.schedule_steps*BOUNDS[ph])
        static=clone_tree(kwargs)
        try:
            self.controller.warming=True;self.forward(**static)
            self.controller.warming=False
            for _ in range(2):self.forward(**static)
            torch.cuda.synchronize();before=self.state()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool):output=self.forward(**static)
            after=self.state();delta_counts={k:after[0][k]-before[0][k] for k in before[0]}
            delta_mods=[]
            for a,b in zip(after[1],before[1]):
                tiers={k:a[3].get(k,0)-b[3].get(k,0) for k in a[3] if a[3].get(k,0)!=b[3].get(k,0)}
                delta_mods.append((a[0]-b[0],a[1]-b[1],a[2]-b[2],tiers))
            if self.controller.pack_cache:raise RuntimeError('Unconsumed shared pack during graph capture')
            self.cache[key]=(static,graph,output,delta_counts,delta_mods,after[2]-before[2]);self.captures+=1
        finally:
            self.restore(saved);self.study.step,self.study.steps=saved_step,saved_steps;self.controller.warming=old_warming

    @torch.inference_mode()
    def __call__(self,*args,**kwargs):
        if args:raise ValueError('PhaseGraph expects keyword inputs')
        # Keep the original range check outside capture on every call; embedding
        # lookup still consumes the current copied domain IDs inside the graph.
        for ids in kwargs.get('action_domain_ids') or []:
            if self.domain_limit is None or bool(((ids<0)|(ids>=self.domain_limit)).any()):
                raise ValueError('Action domain ID outside model range')
        dynamic=self.study.mode!='bf16' and self.controller.mode.startswith('auto')
        ph=phase(self.study.step,self.study.steps) if dynamic else 0
        if self.warming:
            for p in (range(4) if dynamic else [0]):self.capture_phase(kwargs,p)
        key=self.key(kwargs,ph)
        if key not in self.cache:raise RuntimeError('Unwarmed task/phase/branch/shape in timed request')
        static,graph,output,counts,mods,hits=self.cache[key]
        copy_tree(static,kwargs);graph.replay()
        for k,v in counts.items():self.study.counts[k]+=v
        for m,(calls,base,extra,tiers) in zip(self.modules(),mods):
            m.calls+=calls;m.base_flops+=base;m.selected_extra_flops+=extra
            for k,v in tiers.items():m.tier_calls[k]=m.tier_calls.get(k,0)+v
        self.controller.pack_hits+=hits;self.replays+=1;self.phase_hits[f'{self.study.branch}:{ph}']+=1
        return clone_tree(output)

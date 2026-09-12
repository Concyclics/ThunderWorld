"""Resident packed weights. All host support preparation is outside inference."""
import math
import torch
import triton
import triton.language as tl
from ..linear import _pack
from .._native import Native
from .._fusion import pack_with_factor


@triton.jit
def select_tiles(X,D,MASK,M:tl.constexpr,K:tl.constexpr,BT:tl.constexpr,NA:tl.constexpr):
    mb=tl.program_id(0);kt=tl.arange(0,BT)
    rows=tl.minimum(mb*128+tl.arange(0,8)*16,M-1)
    cols=kt[:,None]*128+tl.arange(0,8)[None,:]*16
    x=tl.load(X+rows[None,:,None]*K+cols[:,None,:],cols[:,None,:]<K,0).to(tl.float32)
    d=tl.load(D+cols,cols<K,0)
    x=(x*d[:,None,:]).to(tl.bfloat16).to(tl.float32)
    energy=tl.sum(tl.sum(x*x,2),1)
    energy=tl.where(kt<K//128,energy,-float('inf'))
    rank=tl.sum(((energy[None,:]>energy[:,None])|((energy[None,:]==energy[:,None])&(kt[None,:]<kt[:,None]))).to(tl.int32),1)
    tl.store(MASK+mb*(K//128)+kt,rank<NA,kt<K//128)


class FP8Linear(torch.nn.Module):
    """Historical E4M3 W8 per-tensor / A8 per-token native CUTLASS baseline."""
    def __init__(self, source, asset=None):
        super().__init__()
        from vllm import _custom_ops as ops
        self.in_features, self.out_features = source.in_features, source.out_features
        if asset is None:
            w, s = ops.scaled_fp8_quant(source.weight.detach().cuda().contiguous(), scale=None)
            asset = dict(weight=w.t(), scale=s,
                         bias=source.bias.detach() if source.bias is not None else None)
        self.register_buffer('weight', asset['weight'].cuda())
        self.register_buffer('weight_scale', asset['scale'].cuda())
        self.register_buffer('bias', None if asset['bias'] is None else asset['bias'].cuda())

    def export(self):
        return dict(weight=self.weight.cpu(), scale=self.weight_scale.cpu(),
                    bias=None if self.bias is None else self.bias.cpu())

    def forward(self, x):
        from vllm import _custom_ops as ops
        shape = x.shape
        a, s = ops.scaled_fp8_quant(x.reshape(-1,self.in_features).contiguous(),
                                  scale=None, use_per_token_if_dynamic=True)
        y = ops.cutlass_scaled_mm(a, self.weight, scale_a=s,
                                 scale_b=self.weight_scale, out_dtype=torch.bfloat16)
        if self.bias is not None: y = y+self.bias
        return y.reshape(*shape[:-1], self.out_features)


class RepairLinear(torch.nn.Module):
    def __init__(self, source, controller, name, spec, asset=None):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.controller, self.name, self.default = controller, name, spec
        n, k = self.out_features, self.in_features
        if n%128 or k%128: raise ValueError(f'Unsupported FP4 projection {name}: {n}x{k}')
        if asset is None:
            w=source.weight.detach().cuda().contiguous()
            d=torch.ones(k,device=w.device)
            c,s,g,_=_pack(w,d,True,True,w)
            # Residual magnitude is a deterministic offline fallback score.
            score=s[1].float().reshape(n//128,128,k//128,8).square().sum((1,3))
            asset=dict(codes=c.cpu(),scales=s.cpu(),factor=g.cpu(),prescale=d.cpu(),
                score=score.cpu(),bias=None if source.bias is None else source.bias.detach().cpu())
            if getattr(controller,'budget_plan',None) is not None:
                asset.update(selector='residual_sensitivity',activation_sensitivity=w.float().square().sum(0).cpu())
        self.asset=asset  # CPU preparation data; never transferred during replay.
        self.register_buffer('bias',None if asset['bias'] is None else asset['bias'].cuda())
        self.register_buffer('sensitivity',asset.get('activation_sensitivity',torch.ones(k)).cuda())
        self.ops={};self.bound=None;self.weight_bank=None

    def export(self): return self.asset

    def prepare_shape(self, m, spec):
        key=(m,spec.key)
        if key in self.ops:
            self.bound=self.ops[key]
            return
        n,k=self.out_features,self.in_features;mb=triton.cdiv(m,128)
        a=self.asset
        c,s,g,d=(a[x].cuda() for x in ('codes','scales','factor','prescale'))
        am=torch.ones((mb,k),device='cuda',dtype=torch.bool)
        scores=a['score'].flatten();na,nw=spec.counts(n,k)
        selected=torch.zeros_like(scores,dtype=torch.bool)
        ids=torch.argsort(scores,descending=True,stable=True)[:nw] if spec.weight_support is None else torch.tensor(spec.weight_support,dtype=torch.long)
        mandatory=set(a.get('mandatory_w_blocks',()))
        if not mandatory.issubset(ids.tolist()):raise ValueError('Policy disables mandatory outlier W1 support')
        selected[ids]=True
        wm=selected.reshape(n//128,k//128).repeat_interleave(128,0).repeat_interleave(128,1).cuda()
        native_cls=Native
        if getattr(getattr(self.controller,'config',None),'kernel_variant','optimized')=='legacy':
            from .._native_legacy import Native as native_cls
        op=native_cls(c,s,g,d,am,wm,bm=128,interleave=True,num_stages=2,cross=spec.cross)
        # W0 is shared across all M/A/cross variants; W1 across identical W support.
        shared_base=False
        for (old_m,old_spec),previous in self.ops.items():
            if not shared_base:
                for field in ('wc','ws','wg','d'):
                    setattr(op,field,getattr(previous,field))
                shared_base=True
            import json
            if json.loads(old_spec)['weight']==spec.weight and json.loads(old_spec).get('weight_support')==(list(spec.weight_support) if spec.weight_support is not None else None):
                for field in ('wr','wrs','wi','off'):
                    setattr(op,field,getattr(previous,field))
                break
        plan=getattr(self.controller,'budget_plan',None)
        if plan is not None and native_cls is Native:
            if self.weight_bank is None:
                union=sorted(set(i for key,row in plan.linears.items() if key.split('|',2)[-1]==self.name for i in row['w_blocks']))
                union_ids=torch.tensor(union,device=c.device,dtype=torch.long)
                wc=c[1].reshape(n//128,128,k//128,64).permute(0,2,1,3).reshape(-1,128,64)
                ws=s[1].view(torch.uint8).reshape(n//128,128,k//128,8).permute(0,2,1,3).reshape(-1,128,8)
                self.weight_bank=(wc[union_ids].contiguous(),ws[union_ids].contiguous().view(torch.float8_e4m3fn),{v:i for i,v in enumerate(union)})
            op.wr,op.wrs,positions=self.weight_bank
            op.wloc=torch.tensor([positions[i] for i in sorted(ids.tolist())],device=c.device,dtype=torch.int32)
            op.shared_bank=not op.dense_w
        op.na=na
        op.weight_ids=tuple(ids.tolist())
        op.am=op.union=torch.full((mb,k//128),op.na==k//128,device='cuda',dtype=torch.bool)
        self.ops[key]=op;self.bound=op

    def _compute(self, x, factor=None, packed=None):
        op=self.bound;m,k=x.shape
        sparse=self.asset.get('selector')=='residual_sensitivity' and 0<op.na<k//128
        if sparse:
            if packed is not None:raise ValueError('Residual sensitivity requires compatible shared selectors')
            from .._sparse_pack import pack_sparse
            base,res=self.asset.get('activation_scale',(1.,1.))
            packed=pack_sparse(x,op.d,self.sensitivity,op.am,op.na,base=base,res=res,
                               factor=factor if bool(self.asset.get('identity_prescale',False)) else None)
        elif 0<op.na<k//128:
            select_tiles[(triton.cdiv(m,128),)](x,op.d,op.am,m,k,
                triton.next_power_of_2(k//128),op.na,num_warps=4)
        if packed is not None:y=op.run_packed(x,packed)
        elif self.asset.get('activation_scale',(1.,1.))!=(1.,1.):
            from .._calibrated import pack
            base,res=self.asset['activation_scale']
            y=op.run_packed(x,pack(x,op.d,op.na>0,op.na==k//128,op.am,base=base,res=res))
        elif factor is None: y=op(x)
        else:
            packed=pack_with_factor(x,op.d,op.na>0,op.na==k//128,op.am,factor)
            y=op.run_packed(x,packed)
        return y if self.bias is None else y+self.bias

    def forward(self,x,*,factor=None,packed=None):
        shape=x.shape;flat=x.reshape(-1,self.in_features).contiguous()
        c=self.controller;spec=c.linear_spec(self.name,self.default)
        key=(len(flat),spec.key)
        if c.warming:
            self.prepare_shape(len(flat),spec)
            tune_spec=dict(repair=spec.key,activation_scale=self.asset.get('activation_scale',(1.,1.)),
                identity_prescale=bool((self.asset['prescale']==1).all()),
                kernel_variant=getattr(getattr(c,'config',None),'kernel_variant','optimized'))
            tune_key=c.table.key('linear',[len(flat),self.out_features,self.in_features],tune_spec)
            def apply(v):
                self.bound.bm=v['bm'];self.bound.num_stages=v['stages']
            c.table.search(tune_key,[dict(bm=b,stages=s) for b,s in
                [(128,2),(32,2),(32,3),(64,2),(64,3),(128,3)]],apply,lambda:self._compute(flat))
            apply(c.table.get(tune_key,dict(bm=128,stages=2)))
            c.record_linear(self.name,flat.shape,self.out_features,spec,self.bound)
        elif key not in self.ops: raise RuntimeError(f'Unwarmed projection {self.name}: {key}')
        else: self.bound=self.ops[key]
        return self._compute(flat,factor,packed).reshape(*shape[:-1],self.out_features)
